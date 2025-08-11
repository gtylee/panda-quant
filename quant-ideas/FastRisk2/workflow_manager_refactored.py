"""
Refactored workflow manager using Strategy pattern to separate product and approximator concerns.
This version is much cleaner and more maintainable than the original monolithic approach.
"""
import json
from datetime import date, datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import QuantLib as ql
import time
import abc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import sys

# Import the new handlers
from product_handlers import ProductHandlerFactory
from approximator_handlers import ApproximatorHandlerFactory

# Import existing classes and functions
from tff_approximator import TensorFunctionalForm
from base_pricer import PricerBase
from product_definitions import (
    ProductStaticBase, QuantLibBondStaticBase, CallableBondStaticBase,
    ConvertibleBondStaticBase, EuropeanOptionStatic, MBSPoolStatic
)
from registry.product_registry import create_product_static_from_dict
from scenario_generator import SimpleRandomScenarioGenerator


# --- JSON Serialization Helpers ---
def portfolio_json_serializer(obj):
    if isinstance(obj, (datetime, date)): return obj.isoformat()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, 'to_dict') and callable(obj.to_dict): return obj.to_dict()
    if isinstance(obj, ProductStaticBase): return obj.to_dict()
    if isinstance(obj, ql.Date):
        # QuantLib Date uses dayOfMonth()/month()/year()
        return date(int(obj.year()), int(obj.month()), int(obj.dayOfMonth())).isoformat()
    if isinstance(obj, ql.Calendar): return obj.name()
    if isinstance(obj, ql.DayCounter): return obj.name()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict):
        return {k: portfolio_json_serializer(v) for k, v in obj.items()}
    raise TypeError(f"Object of type {obj.__class__.__name__} ({obj}) is not JSON serializable by custom serializer")


def generate_portfolio_specs_for_serialization(
    holdings_data: list[dict],
    model_registry: dict,
    instrument_definitions_data_for_pricer_params: list[dict] = None
    ) -> list[dict]:
    portfolio_specs_for_json = []
    if instrument_definitions_data_for_pricer_params is None:
        instrument_definitions_data_for_pricer_params = []

    for holding in holdings_data:
        instrument_id = holding.get("instrument_id")
        if not instrument_id:
            print(f"   Skipping holding due to missing instrument_id: {holding}")
            continue

        if instrument_id in model_registry and not model_registry[instrument_id].get('error'):
            entry = model_registry[instrument_id]
            spec_item = {
                "client_id": holding.get("client_id"),
                "instrument_id": instrument_id,
                "num_holdings": holding.get("num_holdings"),
                "pricing_engine_type": entry["pricing_method"].lower(),
                "product_static_object": entry["product_static_dict"]
            }
            
            # Handle different pricing methods
            if entry["pricing_method"] in ['TFF', 'RBFI'] and 'model_dict' in entry:
                spec_item["direct_approximator_config"] = {
                    "model_dict": entry["model_dict"],
                    "raw_input_names": entry["raw_input_names"],
                    "normalization_params": entry["normalization_params"],
                    "option_feature_order": entry.get("option_feature_order", 0)
                }
                if 'fixed_pricer_params' in entry:
                    spec_item['pricer_params'] = entry['fixed_pricer_params']

            if entry["pricing_method"] == 'FULL':
                if 'pricer_params' in entry:
                     spec_item['pricer_params'] = entry['pricer_params']
                else:
                    original_instrument_spec = next(
                        (item for item in instrument_definitions_data_for_pricer_params
                         if item.get("instrument_id") == instrument_id),
                        None
                    )
                    if original_instrument_spec and 'pricer_params' in original_instrument_spec:
                         spec_item['pricer_params'] = original_instrument_spec['pricer_params']

            portfolio_specs_for_json.append(spec_item)
        else:
            print(f"   Skipping instrument '{instrument_id}' for JSON spec generation: not in valid model_registry or had an error.")
    return portfolio_specs_for_json


class PortfolioBase(abc.ABC):
    """Abstract base class for a portfolio of financial instruments."""
    
    def __init__(self):
        self.positions: list[dict] = []

    @abc.abstractmethod
    def add_position(self, *args, **kwargs):
        """Adds a position (instrument holding) to the portfolio."""
        pass

    @abc.abstractmethod
    def price_portfolio(self,
                        raw_market_scenarios: np.ndarray,
                        scenario_factor_names: list[str],
                        portfolio_rate_pillar_times: np.ndarray = None
                        ) -> np.ndarray:
        """Prices all instruments in the portfolio for given market scenarios."""
        pass


class Portfolio(PortfolioBase):
    """A portfolio where each instrument can be priced using either a pre-fitted
    approximator model or its original full pricer."""
    
    def __init__(self):
        super().__init__()
        self.approximator_model_cache: dict = {}
        # Cache structure:
        # { instrument_id: {
        #     'model': TensorFunctionalForm_object,
        #     'raw_input_names': list_of_names,
        #     'normalization_params': dict_of_norm_params,
        #     'option_feature_order': int
        #   }, ...
        # }

    def to_dict(self) -> dict:
        """Returns a dictionary representation of the portfolio."""
        return {
            'positions': [p.to_dict() if hasattr(p, 'to_dict') else p
                for p in self.positions],
            'approximator_model_cache': self.approximator_model_cache
        }

    def cache_approximator_model(self,
                        instrument_id: str,
                        model: TensorFunctionalForm,
                        raw_input_names: list[str],
                        normalization_params: dict,
                        option_feature_order: int = 0):
        """Explicitly caches a fitted approximator model and its associated parameters."""
        self.approximator_model_cache[instrument_id] = {
            'model': model,
            'raw_input_names': raw_input_names,
            'normalization_params': normalization_params,
            'option_feature_order': option_feature_order
        }

    def from_dict(self, portfolio_dict: dict):
        """Loads portfolio from dictionary representation."""
        self.positions = portfolio_dict.get('positions', [])
        self.approximator_model_cache = portfolio_dict.get('approximator_model_cache', {})

    def add_position(self,
                       instrument_id: str,
                       product_static: ProductStaticBase,
                       num_holdings: int = 1,
                       pricing_engine_type: str = 'tff',
                       direct_approximator_config: dict = None,
                       full_pricer_instance: PricerBase = None,
                       full_pricer_kwargs: dict = None):
        """Adds a position to the portfolio."""
        position = {
            'instrument_id': instrument_id,
            'product_static': product_static,
            'num_holdings': num_holdings,
            'pricing_engine_type': pricing_engine_type.lower()
        }

        if pricing_engine_type.lower() in ('tff', 'rbfi') and direct_approximator_config:
            position['direct_approximator_config'] = direct_approximator_config
        elif pricing_engine_type.lower() == 'full':
            position['full_pricer_instance'] = full_pricer_instance
            position['full_pricer_kwargs'] = full_pricer_kwargs or {}

        self.positions.append(position)

    def load_portfolio_from_specs(self, portfolio_specs: list[dict]):
        """Loads portfolio from specifications."""
        for spec in portfolio_specs:
            product_static = create_product_static_from_dict(spec['product_static_object'])
            
            direct_config = spec.get('direct_approximator_config')
            full_pricer = spec.get('full_pricer_instance')
            full_kwargs = spec.get('full_pricer_kwargs', {})
            
            self.add_position(
                instrument_id=spec['instrument_id'],
                product_static=product_static,
                num_holdings=spec['num_holdings'],
                pricing_engine_type=spec['pricing_engine_type'],
                direct_approximator_config=direct_config,
                full_pricer_instance=full_pricer,
                full_pricer_kwargs=full_kwargs
            )

    def price_portfolio(self,
                        raw_market_scenarios: np.ndarray,
                        scenario_factor_names: list[str],
                        portfolio_rate_pillar_times: np.ndarray = None
                        ) -> np.ndarray:
        """Prices all instruments in the portfolio for given market scenarios."""
        num_scenarios = raw_market_scenarios.shape[0]
        portfolio_values = np.zeros(num_scenarios)
        
        for position in self.positions:
            instrument_id = position['instrument_id']
            num_holdings = position['num_holdings']
            pricing_engine_type = position['pricing_engine_type']
            
            if pricing_engine_type in ('tff', 'rbfi') and 'direct_approximator_config' in position:
                # Use cached approximator model
                config = position['direct_approximator_config']
                model_dict = config['model_dict']
                raw_input_names = config['raw_input_names']
                normalization_params = config['normalization_params']
                option_feature_order = config.get('option_feature_order', 0)
                
                # Get scenario slice for this instrument's factors
                scenario_slice = self._get_scenario_slice(
                    raw_market_scenarios, scenario_factor_names, raw_input_names
                )
                
                # Create approximator and price
                from tff_approximator import TensorFunctionalForm
                from rbfi_approximator import RadialBasisFunctionInterpolator
                if pricing_engine_type == 'tff':
                    approximator = TensorFunctionalForm.from_dict(model_dict)
                elif pricing_engine_type == 'rbfi':
                    approximator = RadialBasisFunctionInterpolator.from_dict(model_dict)
                else:
                    raise ValueError(f"Unknown approximator type: {pricing_engine_type}")
                instrument_values = approximator(scenario_slice)
                
            elif pricing_engine_type == 'full' and 'full_pricer_instance' in position:
                # Use full pricer
                pricer = position['full_pricer_instance']
                kwargs = position['full_pricer_kwargs']
                
                # Get scenario slice for this instrument's factors
                if hasattr(pricer, 'get_required_factor_names'):
                    factor_names = pricer.get_required_factor_names(portfolio_rate_pillar_times)
                else:
                    factor_names = None
                scenario_slice = self._get_scenario_slice(
                    raw_market_scenarios, scenario_factor_names, 
                    factor_names
                )
                
                # Special handling for BlackScholesPricer
                from black_scholes_pricer import BlackScholesPricer
                if isinstance(pricer, BlackScholesPricer):
                    # scenario_slice shape: (n_scenarios, 2) for S0 and VOL
                    stock_price = scenario_slice[:, 0]
                    volatility = scenario_slice[:, 1]
                    risk_free_rate = kwargs.get('risk_free_rate', 0.025)
                    dividend_yield = kwargs.get('dividend_yield', 0.0)
                    instrument_values = pricer.price(
                        stock_price=stock_price,
                        volatility=volatility,
                        risk_free_rate=risk_free_rate,
                        dividend_yield=dividend_yield
                    )
                else:
                    instrument_values = pricer.price(
                        pillar_times=portfolio_rate_pillar_times,
                        market_scenario_data=scenario_slice,
                        **kwargs
                    )
                
            else:
                raise ValueError(f"Invalid pricing configuration for instrument {instrument_id}")
            
            portfolio_values += num_holdings * instrument_values
        
        return portfolio_values

    def _get_scenario_slice(self, all_scenarios, all_factor_names, target_factor_names):
        """Get scenario slice for specific factor names."""
        return get_scenario_slice_static(all_scenarios, all_factor_names, target_factor_names)


def get_scenario_slice_static(all_scenarios, all_factor_names, target_factor_names):
    """Static version of get_scenario_slice for use in worker processes."""
    if not target_factor_names:
        return all_scenarios
    
    # Find indices of target factors in all_factor_names
    target_indices = []
    for target_name in target_factor_names:
        try:
            idx = all_factor_names.index(target_name)
            target_indices.append(idx)
        except ValueError:
            print(f"Warning: Factor '{target_name}' not found in scenario factors")
    
    if not target_indices:
        return np.zeros((all_scenarios.shape[0], 0))
    
    return all_scenarios[:, target_indices]


def create_clean_processor_for_workers(iproc):
    """Create a clean version of the processor for worker processes."""
    # Extract only pickleable attributes
    clean_processor = {
        'global_valuation_date': iproc.global_valuation_date,
        'default_numeric_rate_tenors': iproc.default_numeric_rate_tenors,
        'default_g2_params': iproc.default_g2_params,
        'default_bs_risk_free_rate': iproc.default_bs_risk_free_rate,
        'default_bs_dividend_yield': iproc.default_bs_dividend_yield,
        'num_workers': iproc.num_workers,
        'n_scenarios_for_domain': iproc.n_scenarios_for_domain,
        'model_registry': iproc.model_registry,
        # Extract scenario generator data as clean dictionaries
        'base_rates_map': dict(getattr(iproc.scenario_generator, 'base_rates_map', {})),
        'base_s0_map': dict(getattr(iproc.scenario_generator, 'base_s0_map', {})),
        'base_vol_map': dict(getattr(iproc.scenario_generator, 'base_vol_map', {}))
    }
    
    return clean_processor

def process_single_instrument_worker_processpool(args_tuple):
    """Worker function for ProcessPoolExecutor that recreates objects from clean data."""
    from datetime import date
    from scenario_generator import SimpleRandomScenarioGenerator
    from registry.product_registry import create_product_static_from_dict
    from product_handlers import ProductHandlerFactory
    from approximator_handlers import ApproximatorHandlerFactory
    
    (
        instrument_spec, global_market_scenarios, global_factor_names, ql_val_date_iso,
        clean_processor_data
    ) = args_tuple
    
    val_d_worker = date.fromisoformat(ql_val_date_iso)
    import QuantLib as ql
    ql.Settings.instance().evaluationDate = ql.Date(val_d_worker.day, val_d_worker.month, val_d_worker.year)
    
    # Recreate scenario generator from clean data
    scenario_generator_worker = SimpleRandomScenarioGenerator(
        base_rates_map=clean_processor_data['base_rates_map'],
        base_s0_map=clean_processor_data['base_s0_map'],
        base_vol_map=clean_processor_data['base_vol_map'],
        random_seed=42
    )
    
    instrument_id = instrument_spec.get('instrument_id')
    product_type_str = instrument_spec.get('product_type')
    params = instrument_spec.get('params', {})
    pricing_preference = instrument_spec.get('pricing_preference', 'FULL').upper()
    
    if hasattr(params, 'to_dict'):
        params = params.to_dict()
    
    registry_entry = {'instrument_id': instrument_id, 'pricing_method': pricing_preference}
    
    if 'valuation_date' not in params:
        params['valuation_date'] = val_d_worker
    if 'product_type' not in params:
        params['product_type'] = product_type_str
    
    try:
        product_static_object = create_product_static_from_dict(params)
        registry_entry['product_static_dict'] = product_static_object.to_dict()
        
        product_handler = ProductHandlerFactory.get_handler_by_product_static(product_static_object)
        pricer_params = instrument_spec.get('pricer_params', {})
        pricer_template = product_handler.create_pricer(product_static_object, pricer_params)
        
        if 'pricer_params' in instrument_spec:
            registry_entry['pricer_params'] = instrument_spec['pricer_params']
        
        if pricing_preference in ['TFF', 'RBFI']:
            approximator_handler = ApproximatorHandlerFactory.get_handler(pricing_preference)
            tff_config_from_spec = instrument_spec.get('tff_config', {})
            factory_behavior_params = tff_config_from_spec.copy()
            factory_behavior_params['fixed_cb_params'] = instrument_spec.get('pricer_params', {})
            factory_behavior_params['fixed_bs_params'] = instrument_spec.get('pricer_params', {})
            factory_behavior_params['fixed_mbs_params'] = instrument_spec.get('pricer_params', {})
            
            tff_inputs = product_handler.get_tff_factors(
                product_static=product_static_object,
                scenario_generator=scenario_generator_worker,
                default_numeric_rate_tenors=clean_processor_data['default_numeric_rate_tenors'],
                tff_behavior_params=factory_behavior_params,
                instrument_pricer_params=instrument_spec.get('pricer_params', {})
            )
            
            scenarios_for_this_approximator = get_scenario_slice_static(
                global_market_scenarios, global_factor_names, tff_inputs["tff_input_raw_factor_names"]
            )
            
            config_params = {
                'n_train': tff_config_from_spec.get('n_train', 64),
                'n_test': tff_config_from_spec.get('n_test', 8),
                'seed': instrument_spec.get('seed', 42),
                'parallel_workers': False,
                'option_feature_order': tff_inputs["option_feature_order"],
                'order': tff_config_from_spec.get("order", 2),
                'fixed_pricer_params_for_tff_training': tff_inputs["fixed_pricer_params_for_tff_training"]
            }
            
            calibration_result = approximator_handler.calibrate(
                pricer_template=pricer_template,
                tff_input_raw_factor_names=tff_inputs["tff_input_raw_factor_names"],
                tff_input_raw_base_values=tff_inputs["tff_input_raw_base_values"],
                product_static_params_for_worker=product_static_object.to_dict(),
                pricer_config_for_worker=tff_inputs["pricer_config_for_worker"],
                actual_rate_pillars=tff_inputs["actual_rate_pillars"],
                scenarios_for_this_approximator=scenarios_for_this_approximator,
                config_params=config_params
            )
            
            prefix = 'tff' if pricing_preference == 'TFF' else 'rbfi'
            registry_entry.update({
                f'{prefix}_model_dict': calibration_result['model_dict'],
                f'{prefix}_raw_input_names': calibration_result['raw_input_names'],
                f'{prefix}_normalization_params': calibration_result['normalization_params'],
                f'{prefix}_option_feature_order': calibration_result['option_feature_order'],
                f'{prefix}_rmse': calibration_result['rmse'],
                f'{prefix}_fit_time_seconds': calibration_result['fit_time_seconds'],
                f'{prefix}_base_value': calibration_result['base_value'],
                f'{prefix}_base_approximator_value': calibration_result['base_approximator_value']
            })
            
            if calibration_result['fixed_pricer_params']:
                registry_entry[f'{prefix}_fixed_pricer_params'] = calibration_result['fixed_pricer_params']
            
            registry_entry.update({
                'model_dict': calibration_result['model_dict'],
                'raw_input_names': calibration_result['raw_input_names'],
                'normalization_params': calibration_result['normalization_params'],
                'option_feature_order': calibration_result['option_feature_order'],
                'rmse': calibration_result['rmse'],
                'fit_time_seconds': calibration_result['fit_time_seconds'],
                'base_value': calibration_result['base_value'],
                'base_approximator_value': calibration_result['base_approximator_value']
            })
            
            if calibration_result['fixed_pricer_params']:
                registry_entry['fixed_pricer_params'] = calibration_result['fixed_pricer_params']
                
        elif pricing_preference == 'FULL':
            # For FULL pricing, we need to be careful about what we store
            # Don't store the pricer template directly as it may contain QuantLib objects
            registry_entry['full_pricing_method'] = 'FULL'
            # Store only the product static dict and pricer params, not the actual pricer object
            
    except Exception as e:
        print(f"    ERROR processing {instrument_id}: {e}")
        registry_entry.update({'error': str(e), 'pricing_method': 'ERROR'})
    
    return instrument_id, registry_entry

def process_instruments_parallel_processpool(instrument_definitions, global_market_scenarios, global_factor_names, 
                                           global_valuation_date_iso, clean_processor_data, num_workers, batch_size):
    """Process instruments in parallel using ProcessPoolExecutor with clean data."""
    from concurrent.futures import ProcessPoolExecutor
    from tqdm import tqdm
    
    # Prepare arguments for each instrument
    worker_args_list = []
    for spec in instrument_definitions:
        # Ensure we have clean data
        spec_for_worker = spec.copy()
        if 'params' in spec_for_worker and hasattr(spec_for_worker['params'], 'to_dict'):
            spec_for_worker['params'] = spec_for_worker['params'].to_dict()
        
        args = (
            spec_for_worker, global_market_scenarios, global_factor_names, global_valuation_date_iso,
            clean_processor_data
        )
        worker_args_list.append(args)
    
    # Process instruments in parallel using ProcessPoolExecutor
    model_registry = {}
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_instrument_worker_processpool, args) for args in worker_args_list]
        
        for future in tqdm(futures, desc="Processing Instruments"):
            instrument_id, registry_entry = future.result()
            model_registry[instrument_id] = registry_entry
    
    return model_registry

class RefactoredInstrumentProcessor:
    """Refactored instrument processor using Strategy pattern."""
    
    def __init__(self, scenario_generator: SimpleRandomScenarioGenerator,
                 global_valuation_date: date,
                 default_numeric_rate_tenors: np.ndarray = None,
                 default_g2_params = None,
                 default_bs_risk_free_rate: float = 0.025,
                 default_bs_dividend_yield: float = 0.0,
                 parallel_workers: int = None,
                 n_scenarios_for_domain: int = 1000):
        
        self.scenario_generator = scenario_generator
        self.global_valuation_date = global_valuation_date
        self.default_numeric_rate_tenors = default_numeric_rate_tenors
        self.default_g2_params = default_g2_params
        self.default_bs_risk_free_rate = default_bs_risk_free_rate
        self.default_bs_dividend_yield = default_bs_dividend_yield
        self.num_workers = parallel_workers if parallel_workers else 0
        self.n_scenarios_for_domain = n_scenarios_for_domain
        
        self.model_registry = {}

    def _process_single_instrument_spec(self, args_tuple):
        """Process a single instrument specification."""
        instrument_spec, global_market_scenarios, global_factor_names, ql_val_date_iso, base_rates_map, base_s0_map, base_vol_map, default_numeric_rate_tenors = args_tuple

        val_d_worker = date.fromisoformat(ql_val_date_iso)
        ql.Settings.instance().evaluationDate = ql.Date(val_d_worker.day, val_d_worker.month, val_d_worker.year)

        # Recreate scenario generator in worker process to avoid pickling issues
        from scenario_generator import SimpleRandomScenarioGenerator
        scenario_generator_worker = SimpleRandomScenarioGenerator(
            base_rates_map=base_rates_map,
            base_s0_map=base_s0_map,
            base_vol_map=base_vol_map,
            random_seed=42
        )

        instrument_id = instrument_spec.get('instrument_id')
        product_type_str = instrument_spec.get('product_type')
        params = instrument_spec.get('params', {})
        pricing_preference = instrument_spec.get('pricing_preference', 'FULL').upper()

        # Ensure params is a dict (should always be true now)
        if hasattr(params, 'to_dict'):
            params = params.to_dict()

        registry_entry = {'instrument_id': instrument_id, 'pricing_method': pricing_preference}
        
        if 'valuation_date' not in params: 
            params['valuation_date'] = val_d_worker
        if 'product_type' not in params: 
            params['product_type'] = product_type_str

        try:
            # Create product static object from dict
            product_static_object = create_product_static_from_dict(params)
            registry_entry['product_static_dict'] = product_static_object.to_dict()
            
            # Get product handler
            product_handler = ProductHandlerFactory.get_handler_by_product_static(product_static_object)
            
            # Create pricer template (recreate in worker to avoid pickling issues)
            pricer_params = instrument_spec.get('pricer_params', {})
            pricer_template = product_handler.create_pricer(product_static_object, pricer_params)
            
            if 'pricer_params' in instrument_spec: 
                registry_entry['pricer_params'] = instrument_spec['pricer_params']

            # Handle different pricing preferences
            if pricing_preference in ['TFF', 'RBFI']:
                # Get approximator handler
                approximator_handler = ApproximatorHandlerFactory.get_handler(pricing_preference)
                
                # Get TFF factors from product handler
                tff_config_from_spec = instrument_spec.get('tff_config', {})
                factory_behavior_params = tff_config_from_spec.copy()
                factory_behavior_params['fixed_cb_params'] = instrument_spec.get('pricer_params', {})
                factory_behavior_params['fixed_bs_params'] = instrument_spec.get('pricer_params', {})
                factory_behavior_params['fixed_mbs_params'] = instrument_spec.get('pricer_params', {})
                
                tff_inputs = product_handler.get_tff_factors(
                    product_static=product_static_object,
                    scenario_generator=scenario_generator_worker,
                    default_numeric_rate_tenors=default_numeric_rate_tenors,
                    tff_behavior_params=factory_behavior_params,
                    instrument_pricer_params=instrument_spec.get('pricer_params', {})
                )
                
                # Get scenario slice for this approximator
                scenarios_for_this_approximator = self._get_scenario_slice(
                    global_market_scenarios, global_factor_names, 
                    tff_inputs["tff_input_raw_factor_names"]
                )
                
                # Calibrate approximator (pass product_static and pricer_params instead of pricer_template)
                config_params = {
                    'n_train': tff_config_from_spec.get('n_train', 64),
                    'n_test': tff_config_from_spec.get('n_test', 8),
                    'seed': instrument_spec.get('seed', 42),
                    'parallel_workers': False,
                    'option_feature_order': tff_inputs["option_feature_order"],
                    'order': tff_config_from_spec.get("order", 2),
                    'fixed_pricer_params_for_tff_training': tff_inputs["fixed_pricer_params_for_tff_training"]
                }
                
                calibration_result = approximator_handler.calibrate(
                    pricer_template=pricer_template,  # This is recreated in worker process
                    tff_input_raw_factor_names=tff_inputs["tff_input_raw_factor_names"],
                    tff_input_raw_base_values=tff_inputs["tff_input_raw_base_values"],
                    product_static_params_for_worker=product_static_object.to_dict(),
                    pricer_config_for_worker=tff_inputs["pricer_config_for_worker"],
                    actual_rate_pillars=tff_inputs["actual_rate_pillars"],
                    scenarios_for_this_approximator=scenarios_for_this_approximator,
                    config_params=config_params
                )
                
                # Patch: store under tff_* or rbfi_* keys as well as generic keys
                prefix = 'tff' if pricing_preference == 'TFF' else 'rbfi'
                registry_entry.update({
                    f'{prefix}_model_dict': calibration_result['model_dict'],
                    f'{prefix}_raw_input_names': calibration_result['raw_input_names'],
                    f'{prefix}_normalization_params': calibration_result['normalization_params'],
                    f'{prefix}_option_feature_order': calibration_result['option_feature_order'],
                    f'{prefix}_rmse': calibration_result['rmse'],
                    f'{prefix}_fit_time_seconds': calibration_result['fit_time_seconds'],
                    f'{prefix}_base_value': calibration_result['base_value'],
                    f'{prefix}_base_approximator_value': calibration_result['base_approximator_value']
                })
                if calibration_result['fixed_pricer_params']:
                    registry_entry[f'{prefix}_fixed_pricer_params'] = calibration_result['fixed_pricer_params']
                # Also keep generic keys for backward compatibility
                registry_entry.update({
                    'model_dict': calibration_result['model_dict'],
                    'raw_input_names': calibration_result['raw_input_names'],
                    'normalization_params': calibration_result['normalization_params'],
                    'option_feature_order': calibration_result['option_feature_order'],
                    'rmse': calibration_result['rmse'],
                    'fit_time_seconds': calibration_result['fit_time_seconds'],
                    'base_value': calibration_result['base_value'],
                    'base_approximator_value': calibration_result['base_approximator_value']
                })
                if calibration_result['fixed_pricer_params']:
                    registry_entry['fixed_pricer_params'] = calibration_result['fixed_pricer_params']
                    
            elif pricing_preference == 'FULL':
                registry_entry['full_pricer_template'] = pricer_template
                
        except Exception as e:
            print(f"    ERROR processing {instrument_id}: {e}")
            registry_entry.update({'error': str(e), 'pricing_method': 'ERROR'})

        return instrument_id, registry_entry

    def _process_batch(self, args_list: list[tuple]) -> dict:
        """Process a batch of instrument specifications."""
        batch_registry = {}
        for args in args_list:
            instrument_id, registry_entry = self._process_single_instrument_spec(args)
            if instrument_id:
                batch_registry[instrument_id] = registry_entry
        return batch_registry

    def process_instruments(self, instrument_definitions, global_market_scenarios, global_factor_names, batch_size=4):
        """Process instrument definitions to build model registry."""
        print(f"Processing {len(instrument_definitions)} instrument definitions...")
        
        if self.num_workers > 1:
            # Create clean processor data for ProcessPoolExecutor
            clean_processor_data = create_clean_processor_for_workers(self)
            
            # Use ProcessPoolExecutor with clean data
            self.model_registry = process_instruments_parallel_processpool(
                instrument_definitions, global_market_scenarios, global_factor_names,
                self.global_valuation_date.isoformat(), clean_processor_data, self.num_workers, batch_size
            )
        else:
            # Sequential processing
            for spec in instrument_definitions:
                args = (
                    spec, global_market_scenarios, global_factor_names, self.global_valuation_date.isoformat(),
                    create_clean_processor_for_workers(self)
                )
                instrument_id, registry_entry = process_single_instrument_worker_processpool(args)
                self.model_registry[instrument_id] = registry_entry
        
        return self.model_registry

    def save_model_registry(self, filepath: str):
        """Save model registry to file."""
        with open(filepath, 'w') as f:
            json.dump(self.model_registry, f, indent=4, default=portfolio_json_serializer)

    @classmethod
    def load_model_registry(cls, filepath: str) -> dict:
        """Load model registry from file."""
        with open(filepath, 'r') as f:
            return json.load(f)

    def _get_scenario_slice(self, all_scenarios, all_factor_names, target_factor_names):
        """Get scenario slice for target factors."""
        return get_scenario_slice_static(all_scenarios, all_factor_names, target_factor_names)


class RefactoredPortfolioBuilder:
    """Refactored portfolio builder."""
    
    def __init__(self, model_registry: dict = None):
        self.model_registry = model_registry or {}
        self.uncalculated_instruments = []

    def build_portfolios_from_specs(self, portfolio_specs_list: list[dict],
                                   global_valuation_date: date,
                                   default_g2_params=None,
                                   default_bs_rfr: float = 0.025, default_bs_div: float = 0.0
                                   ) -> dict[str, Portfolio]:
        """Build portfolios from specifications."""
        print(f"Building portfolios from {len(portfolio_specs_list)} detailed specifications...")
        
        client_portfolios = {}
        
        for spec in portfolio_specs_list:
            client_id = spec.get('client_id')
            instrument_id = spec.get('instrument_id')
            pricing_engine_type = spec.get('pricing_engine_type', 'tff').lower()
            if pricing_engine_type not in ('tff', 'full', 'rbfi'):
                print(f"  ERROR adding position '{instrument_id}' to portfolio for '{client_id}': Unsupported pricing_engine_type: {pricing_engine_type}. Choose 'tff', 'rbfi', or 'full'.")
                self.uncalculated_instruments.append(instrument_id)
                continue
            
            if client_id not in client_portfolios:
                client_portfolios[client_id] = Portfolio()
            
            portfolio = client_portfolios[client_id]
            
            # Check if instrument exists in model registry
            if instrument_id in self.model_registry and not self.model_registry[instrument_id].get('error'):
                registry_entry = self.model_registry[instrument_id]
                
                # Reconstruct product static
                product_static = create_product_static_from_dict(registry_entry['product_static_dict'])
                
                # Add position based on pricing method
                if pricing_engine_type in ('tff', 'rbfi'):
                    prefix = 'tff' if pricing_engine_type == 'tff' else 'rbfi'
                    model_dict = registry_entry.get(f'{prefix}_model_dict')
                    raw_input_names = registry_entry.get(f'{prefix}_raw_input_names')
                    normalization_params = registry_entry.get(f'{prefix}_normalization_params')
                    option_feature_order = registry_entry.get(f'{prefix}_option_feature_order')
                    
                    portfolio.add_position(
                        instrument_id=instrument_id,
                        product_static=product_static,
                        num_holdings=spec['num_holdings'],
                        pricing_engine_type=pricing_engine_type,
                        direct_approximator_config={
                            "model_dict": model_dict,
                            "raw_input_names": raw_input_names,
                            "normalization_params": normalization_params,
                            "option_feature_order": option_feature_order
                        }
                    )
                    
                elif pricing_engine_type == 'full':
                    # Create new pricer instance for full pricing
                    from product_handlers import ProductHandlerFactory
                    product_handler = ProductHandlerFactory.get_handler_by_product_static(product_static)
                    pricer_params = registry_entry.get('pricer_params', {})
                    full_pricer_instance = product_handler.create_pricer(product_static, pricer_params)
                    
                    # Handle BlackScholesPricer parameters correctly
                    from product_definitions import EuropeanOptionStatic
                    if isinstance(product_static, EuropeanOptionStatic):
                        # Set up BlackScholesPricer kwargs with correct parameter names
                        rfr = pricer_params.get('bs_risk_free_rate', 0.025)
                        div = pricer_params.get('bs_dividend_yield', 0.0)
                        full_pricer_kwargs = {
                            'risk_free_rate': rfr,
                            'dividend_yield': div
                        }
                    else:
                        full_pricer_kwargs = pricer_params
                    
                    portfolio.add_position(
                        instrument_id=instrument_id,
                        product_static=product_static,
                        num_holdings=spec['num_holdings'],
                        pricing_engine_type='full',
                        full_pricer_instance=full_pricer_instance,
                        full_pricer_kwargs=full_pricer_kwargs
                    )
            else:
                print(f"  WARNING: TFF spec for '{instrument_id}' missing direct_approximator_config and not found as TFF in registry. Fallback to FULL.")
                self.uncalculated_instruments.append(instrument_id)
        
        print(f"Finished building {len(client_portfolios)} portfolios from detailed specs.")
        return client_portfolios


class RefactoredPortfolioAnalytics:
    """Refactored portfolio analytics."""
    
    def __init__(self,
                 client_portfolios: dict[str, Portfolio],
                 global_market_scenarios: np.ndarray,
                 global_factor_names: list[str],
                 numeric_rate_tenors: np.ndarray,
                 scenario_generator_for_base_values: SimpleRandomScenarioGenerator):
        
        self.client_portfolios = client_portfolios
        self.global_market_scenarios = global_market_scenarios
        self.global_factor_names = global_factor_names
        self.numeric_rate_tenors = numeric_rate_tenors
        self.scenario_generator_for_base_values = scenario_generator_for_base_values

    def calculate_base_portfolio_values(self) -> dict[str, float]:
        """Calculate base portfolio values."""
        base_values = {}
        for client_id, portfolio in self.client_portfolios.items():
            base_values[client_id] = self.calculate_base_value(portfolio)
        return base_values

    def run_var_analysis(self, var_percentiles: list[float] = None):
        """Run VaR analysis on portfolios."""
        if var_percentiles is None:
            var_percentiles = [1.0, 5.0]
        
        var_results = {}
        
        for client_id, portfolio in self.client_portfolios.items():
            # Price portfolio for all scenarios
            portfolio_prices = portfolio.price_portfolio(
                self.global_market_scenarios,
                self.global_factor_names,
                self.numeric_rate_tenors
            )
            
            # Calculate VaR
            var_values = {}
            for percentile in var_percentiles:
                var_value = np.percentile(portfolio_prices, percentile)
                var_values[f'var_{percentile}%'] = var_value
            
            var_results[client_id] = var_values
        
        return var_results

    def calculate_base_value(self, portfolio):
        """Calculate base value for a portfolio."""
        # Use first scenario as base scenario
        base_scenario = self.global_market_scenarios[0:1]
        base_prices = portfolio.price_portfolio(
            base_scenario,
            self.global_factor_names,
            self.numeric_rate_tenors
        )
        return base_prices[0]


class HybridVaRWorkflow:
    """
    Streamlined workflow for running hybrid VaR analysis with multiple approximators.
    Consolidates instrument processing, portfolio building, pricing, and VaR calculation.
    """
    
    def __init__(self, 
                 valuation_date: date,
                 tenors: np.ndarray = None,
                 default_g2_params: tuple = None,
                 default_bs_rfr: float = 0.025,
                 default_bs_div: float = 0.01,
                 random_seed: int = 42,
                 workers: int = None):
        """
        Initialize the hybrid VaR workflow.
        
        Args:
            valuation_date: Valuation date for the analysis
            tenors: Interest rate tenors (default: [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
            default_g2_params: Default G2++ model parameters for callable bonds
            default_bs_rfr: Default Black-Scholes risk-free rate
            default_bs_div: Default Black-Scholes dividend yield
            random_seed: Random seed for reproducibility
            workers: Number of parallel workers for instrument processing (None for sequential)
        """
        self.valuation_date = valuation_date
        self.tenors = tenors if tenors is not None else np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
        self.default_g2_params = default_g2_params if default_g2_params is not None else (0.01, 0.003, 0.015, 0.006, -0.75)
        self.default_bs_rfr = default_bs_rfr
        self.default_bs_div = default_bs_div
        self.random_seed = random_seed
        self.workers = workers
        
        # Performance tracking
        self.performance_metrics = {}
        
    def save_model_registry(self, model_registry: dict, filepath: str):
        """Save the model registry to a JSON file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(model_registry, f, indent=4, default=portfolio_json_serializer)
        print(f"Model registry saved to: {filepath}")
        
    def load_model_registry(self, filepath: str) -> dict:
        """Load the model registry from a JSON file."""
        import json
        with open(filepath, 'r') as f:
            model_registry = json.load(f)
        print(f"Model registry loaded from: {filepath}")
        return model_registry
        
    def run_hybrid_var_analysis(self,
                               instrument_definitions: list[dict],
                               holdings_data: list[dict],
                               var_scenario_generator: SimpleRandomScenarioGenerator,
                               domain_scenario_generator: SimpleRandomScenarioGenerator,
                               num_var_scenarios: int = 1000,
                               n_domain_scenarios: int = 2000,
                               n_fitting_samples: int = 50,
                               hybrid_critical_percentile: float = 0.02,
                               approximators: tuple = ("TFF", "RBFI"),
                               var_percentile: float = 0.01,
                               batch_size: int = 2,
                               workers: int = None,
                               save_models: bool = False,
                               model_save_path: str = "calibrated_models.json",
                               load_models: bool = False,
                               model_load_path: str = "calibrated_models.json",
                               use_full_reval_for_hybrid: bool = True) -> dict:
        """
        Run complete hybrid VaR analysis.
        
        Args:
            instrument_definitions: List of instrument definitions
            holdings_data: List of holdings data with client_id, instrument_id, num_holdings
            var_scenario_generator: Scenario generator for VaR calculation
            domain_scenario_generator: Scenario generator for approximator training
            num_var_scenarios: Number of scenarios for VaR calculation
            n_domain_scenarios: Number of scenarios for approximator training
            n_fitting_samples: Number of samples for approximator fitting
            hybrid_critical_percentile: Percentile threshold for hybrid approach
            approximators: Tuple of approximator types to use ("TFF", "RBFI")
            var_percentile: VaR percentile (default: 0.01 for 1% VaR)
            batch_size: Batch size for instrument processing
            workers: Number of parallel workers (overrides constructor setting if provided)
            save_models: Whether to save calibrated models to file
            model_save_path: Path to save calibrated models
            load_models: Whether to load calibrated models from file
            model_load_path: Path to load calibrated models from
            use_full_reval_for_hybrid: Whether to use full revaluation for critical scenarios in hybrid approach
            
        Returns:
            Dictionary with results for each method (Full, TFF, RBFI, Hybrid) and performance metrics
        """
        # Use provided workers or fall back to constructor setting
        workers_to_use = workers if workers is not None else self.workers
        
        # Reset performance metrics
        self.performance_metrics = {
            'total_start_time': time.time(),
            'scenario_generation': {},
            'full_revaluation': {},
            'approximator_calibration': {},
            'approximator_pricing': {},
            'hybrid_pricing': {}
        }
        
        print(f"--- Hybrid VaR Analysis ---")
        print(f"Approximators: {approximators}")
        print(f"VaR Percentile: {var_percentile:.1%}")
        print(f"Hybrid Critical Percentile: {hybrid_critical_percentile:.1%}")
        print(f"Workers: {workers_to_use if workers_to_use is not None else 'Sequential'}")
        print(f"Save Models: {save_models}")
        print(f"Load Models: {load_models}")
        
        # Generate scenarios using provided generators
        scenario_start = time.time()
        var_scenarios, var_factor_names = var_scenario_generator.generate_scenarios(num_var_scenarios)
        domain_scenarios, domain_factor_names = domain_scenario_generator.generate_scenarios(n_domain_scenarios)
        self.performance_metrics['scenario_generation']['elapsed_time'] = time.time() - scenario_start
        self.performance_metrics['scenario_generation']['var_scenarios'] = num_var_scenarios
        self.performance_metrics['scenario_generation']['domain_scenarios'] = n_domain_scenarios
        
        # Full revaluation path
        print("\n--- Full Revaluation ---")
        full_results = self._run_full_revaluation(
            instrument_definitions, holdings_data, domain_scenarios, domain_factor_names,
            var_scenarios, var_factor_names, var_percentile, batch_size, domain_scenario_generator, workers_to_use
        )
        
        # Approximator paths
        approx_results = {}
        for approx in approximators:
            print(f"\n--- {approx} Approximator Path ---")
            approx_results[approx] = self._run_approximator_path(
                approx, instrument_definitions, holdings_data, domain_scenarios, domain_factor_names,
                var_scenarios, var_factor_names, var_percentile, n_fitting_samples, domain_scenario_generator, 
                workers_to_use, save_models, model_save_path, load_models, model_load_path
            )
        
        # Hybrid paths
        for approx in approximators:
            print(f"\n--- Hybrid ({approx} + Full) Path ---")
            approx_results[approx]["hybrid"] = self._run_hybrid_path(
                approx, approx_results[approx], full_results, var_scenarios, var_factor_names,
                var_percentile, hybrid_critical_percentile, full_results["portfolio_obj"], use_full_reval_for_hybrid
            )
        
        # Calculate total time
        self.performance_metrics['total_elapsed_time'] = time.time() - self.performance_metrics['total_start_time']
        
        # Compile results
        results = {
            "full": full_results,
            "approximators": approx_results,
            "summary": self._create_summary_table(full_results, approx_results, var_percentile),
            "performance_metrics": self.performance_metrics
        }
        
        return results
    
    def _run_full_revaluation(self, instrument_definitions, holdings_data, domain_scenarios, 
                             domain_factor_names, var_scenarios, var_factor_names, 
                             var_percentile, batch_size, scenario_generator, workers):
        """Run full revaluation path."""
        print("\n--- Full Revaluation ---")
        time_start = time.time()
        
        # Instrument processing phase
        processing_start = time.time()
        iproc = RefactoredInstrumentProcessor(
            scenario_generator=scenario_generator,
            global_valuation_date=self.valuation_date,
            default_numeric_rate_tenors=self.tenors,
            default_g2_params=self.default_g2_params,
            default_bs_risk_free_rate=self.default_bs_rfr,
            default_bs_dividend_yield=self.default_bs_div,
            parallel_workers=workers,
            n_scenarios_for_domain=len(domain_scenarios)
        )
        
        # Process instruments
        model_registry = iproc.process_instruments(
            instrument_definitions, domain_scenarios, domain_factor_names, batch_size
        )
        processing_time = time.time() - processing_start
        
        # Portfolio building phase
        portfolio_build_start = time.time()
        portfolio_specs = generate_portfolio_specs_for_serialization(holdings_data, model_registry, instrument_definitions)
        builder = RefactoredPortfolioBuilder(model_registry)
        portfolio_dict = builder.build_portfolios_from_specs(
            portfolio_specs, self.valuation_date, self.default_g2_params,
            self.default_bs_rfr, self.default_bs_div
        )
        portfolio_build_time = time.time() - portfolio_build_start
        
        # Pricing phase
        pricing_start = time.time()
        portfolio_obj = portfolio_dict["HybridClient"]
        portfolio_values = portfolio_obj.price_portfolio(var_scenarios, var_factor_names, self.tenors)
        pricing_time = time.time() - pricing_start
        
        base_value = portfolio_values[0]
        losses = base_value - portfolio_values
        var_value = self._calculate_var(losses, var_percentile)
        
        time_end = time.time()
        elapsed_time = time_end - time_start
        
        # Track performance metrics
        self.performance_metrics['full_revaluation'] = {
            'elapsed_time': elapsed_time,
            'instrument_processing_time': processing_time,
            'portfolio_build_time': portfolio_build_time,
            'pricing_time': pricing_time,
            'n_instruments': len(instrument_definitions),
            'n_scenarios': len(var_scenarios)
        }
        
        print(f"Full {var_percentile*100:.1f}% VaR: {var_value:,.2f}. Base Value: {base_value:,.2f}")
        print(f"Total elapsed time: {elapsed_time:.2f}s")
        print(f"  Instrument processing: {processing_time:.2f}s")
        print(f"  Portfolio building: {portfolio_build_time:.2f}s")
        print(f"  Pricing: {pricing_time:.2f}s")
        
        return {
            "var_value": var_value,
            "base_value": base_value,
            "elapsed_time": elapsed_time,
            "portfolio_obj": portfolio_obj,
            "processing_time": processing_time,
            "portfolio_build_time": portfolio_build_time,
            "pricing_time": pricing_time
        }
    
    def _run_approximator_path(self, approximator, instrument_definitions, holdings_data,
                              domain_scenarios, domain_factor_names, var_scenarios, 
                              var_factor_names, var_percentile, n_fitting_samples, scenario_generator, workers,
                              save_models, model_save_path, load_models, model_load_path):
        """Run approximator path (TFF or RBFI)."""
        start_time = time.time()
        
        # Update instrument definitions for this approximator
        approx_instrument_defs = [dict(d, pricing_preference=approximator) for d in instrument_definitions]
        
        # Create approximator-specific file paths
        if save_models:
            base_path = model_save_path.replace('.json', '')
            approx_save_path = f"{base_path}_{approximator.lower()}.json"
        if load_models:
            base_path = model_load_path.replace('.json', '')
            approx_load_path = f"{base_path}_{approximator.lower()}.json"
        
        # Check if we should load pre-calibrated models
        if load_models:
            try:
                model_registry = self.load_model_registry(approx_load_path)
                print(f"  Loaded pre-calibrated {approximator} models from {approx_load_path}")
                calibration_time = 0.0  # No calibration time when loading
            except FileNotFoundError:
                print(f"  Model file {approx_load_path} not found, calibrating new models...")
                load_models = False
        
        # Calibrate models if not loading
        if not load_models:
            calibration_start = time.time()
            
            # Process instruments
            iproc = RefactoredInstrumentProcessor(
                scenario_generator,
                self.valuation_date, self.tenors, self.default_g2_params, 
                self.default_bs_rfr, self.default_bs_div, workers, n_fitting_samples
            )
            
            model_registry = iproc.process_instruments(
                approx_instrument_defs, domain_scenarios, domain_factor_names, 1
            )
            
            calibration_time = time.time() - calibration_start
            
            # Save models if requested
            if save_models:
                self.save_model_registry(model_registry, approx_save_path)
        
        # Track calibration performance
        self.performance_metrics['approximator_calibration'][approximator] = {
            'elapsed_time': calibration_time,
            'n_instruments': len(instrument_definitions),
            'n_fitting_samples': n_fitting_samples
        }
        
        # Build portfolio
        portfolio_build_start = time.time()
        portfolio_specs = generate_portfolio_specs_for_serialization(
            holdings_data, model_registry, approx_instrument_defs
        )
        
        builder = RefactoredPortfolioBuilder(model_registry)
        portfolios = builder.build_portfolios_from_specs(
            portfolio_specs, self.valuation_date, self.default_g2_params, 
            self.default_bs_rfr, self.default_bs_div
        )
        
        # Price portfolio
        portfolio_obj = portfolios["HybridClient"]
        pricing_start = time.time()
        portfolio_values = portfolio_obj.price_portfolio(var_scenarios, var_factor_names, self.tenors)
        pricing_time = time.time() - pricing_start
        
        # Calculate VaR
        base_value = portfolio_values[0]
        losses = base_value - portfolio_values
        var_value = self._calculate_var(losses, var_percentile)
        
        elapsed_time = time.time() - start_time
        
        # Track pricing performance
        self.performance_metrics['approximator_pricing'][approximator] = {
            'elapsed_time': pricing_time,
            'n_scenarios': len(var_scenarios),
            'portfolio_build_time': time.time() - portfolio_build_start
        }
        
        print(f"{approximator} {var_percentile:.1%} VaR: {var_value:,.2f}. Base Value: {base_value:,.2f}")
        print(f"Total elapsed time: {elapsed_time:.2f}s")
        if not load_models:
            print(f"  Calibration time: {calibration_time:.2f}s")
        print(f"  Pricing time: {pricing_time:.2f}s")
        
        return {
            "base_value": base_value,
            "var_value": var_value,
            "losses": losses,
            "portfolio_values": portfolio_values,
            "elapsed_time": elapsed_time,
            "calibration_time": calibration_time if not load_models else 0.0,
            "pricing_time": pricing_time,
            "model_registry": model_registry
        }
    
    def _run_hybrid_path(self, approximator, approx_results, full_results, var_scenarios, 
                        var_factor_names, var_percentile, hybrid_critical_percentile, portfolio_obj_full, use_full_reval_for_hybrid):
        """Run hybrid path (approximator + full revaluation for critical scenarios)."""
        start_time = time.time()
        
        losses_approx = approx_results["losses"]
        base_value_full = full_results["base_value"]
        portfolio_values_approx = approx_results["portfolio_values"]
        
        # Identify critical scenarios (worst scenarios based on approximator)
        sorted_losses_approx = np.sort(losses_approx)
        sorted_indices_approx = np.argsort(losses_approx)
        N = len(sorted_losses_approx)
        idx_threshold = max(0, int(np.ceil(hybrid_critical_percentile * N)) - 1)
        critical_idx = sorted_indices_approx[:idx_threshold]
        
        if len(critical_idx) == 0:
            print("No critical scenarios identified. Using approximator VaR.")
            var_value_hybrid = approx_results["var_value"]
            full_reval_time = 0.0
        else:
            if use_full_reval_for_hybrid:
                # Start with approximator values for all scenarios
                hybrid_portfolio_values = portfolio_values_approx.copy()
                
                # Full revaluation for critical scenarios only
                full_reval_start = time.time()
                critical_scenarios = var_scenarios[critical_idx]
                portfolio_values_full_critical = portfolio_obj_full.price_portfolio(
                    critical_scenarios, var_factor_names, self.tenors
                )
                full_reval_time = time.time() - full_reval_start
                
                # Replace approximator values with full revaluation for critical scenarios
                hybrid_portfolio_values[critical_idx] = portfolio_values_full_critical
                
                # Calculate VaR on the combined distribution
                losses_hybrid = base_value_full - hybrid_portfolio_values
                var_value_hybrid = self._calculate_var(losses_hybrid, var_percentile)
                
                print(f"  Critical scenarios: {len(critical_idx)} out of {N} ({len(critical_idx)/N:.1%})")
                print(f"  Using full revaluation for critical scenarios")
            else:
                # Use only approximator results (no full revaluation)
                var_value_hybrid = approx_results["var_value"]
                full_reval_time = 0.0
                print(f"  Critical scenarios: {len(critical_idx)} out of {N} ({len(critical_idx)/N:.1%})")
                print(f"  Using approximator results only (no full revaluation)")
        
        elapsed_time = time.time() - start_time
        
        # Track hybrid performance
        self.performance_metrics['hybrid_pricing'][approximator] = {
            'elapsed_time': elapsed_time,
            'full_revaluation_time': full_reval_time,
            'n_critical_scenarios': len(critical_idx),
            'n_total_scenarios': N,
            'used_full_reval': use_full_reval_for_hybrid
        }
        
        print(f"Hybrid ({approximator}) {var_percentile:.1%} VaR: {var_value_hybrid:,.2f}. Base Value: {base_value_full:,.2f}")
        print(f"Total elapsed time: {elapsed_time:.2f}s")
        if len(critical_idx) > 0 and use_full_reval_for_hybrid:
            print(f"  Full revaluation time: {full_reval_time:.2f}s")
        
        return {
            "var_value": var_value_hybrid,
            "base_value": base_value_full,
            "elapsed_time": elapsed_time,
            "full_revaluation_time": full_reval_time,
            "n_critical_scenarios": len(critical_idx),
            "used_full_reval": use_full_reval_for_hybrid
        }
    
    def _calculate_var(self, losses: np.ndarray, percentile: float) -> float:
        """Calculate VaR from loss distribution."""
        sorted_losses = np.sort(losses)
        N = len(sorted_losses)
        idx = max(0, int(np.ceil(percentile * N)) - 1)
        return -sorted_losses[idx]
    
    def _create_summary_table(self, full_results, approx_results, var_percentile):
        """Create summary table of results."""
        print(f"\n--- VaR Comparison Summary ---")
        print(f"{'Method':<12} | {'Base Value':<12} | {f'{var_percentile:.1%} VaR':<12} | {'Hybrid VaR':<12}")
        print("-" * 55)
        
        # Format full results
        full_base = f"{full_results['base_value']:,.0f}"
        full_var = f"{full_results['var_value']:,.0f}"
        print(f"{'Full':<12} | {full_base:<12} | {full_var:<12} | {'-':<12}")
        
        # Format approximator results
        for approx, results in approx_results.items():
            base_value = f"{results['base_value']:,.0f}"
            var_value = f"{results['var_value']:,.0f}"
            hybrid_var = results.get("hybrid", {}).get("var_value", None)
            if hybrid_var is not None:
                hybrid_var_str = f"{hybrid_var:,.0f}"
            else:
                hybrid_var_str = "-"
            print(f"{approx:<12} | {base_value:<12} | {var_value:<12} | {hybrid_var_str:<12}")
        
        return {
            "full": full_results,
            "approximators": {k: {"base_value": v["base_value"], "var_value": v["var_value"], 
                                 "hybrid_var": v.get("hybrid", {}).get("var_value", None)} 
                             for k, v in approx_results.items()}
        }
    
    def display_performance_metrics(self):
        """Display detailed performance metrics in a formatted way."""
        metrics = self.performance_metrics
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*60)
        
        # Total time
        total_time = metrics.get('total_elapsed_time', 0)
        print(f"Total Analysis Time: {total_time:.2f}s")
        
        # Scenario generation
        scenario_metrics = metrics.get('scenario_generation', {})
        if scenario_metrics:
            print(f"\nScenario Generation:")
            print(f"  Time: {scenario_metrics.get('elapsed_time', 0):.2f}s")
            print(f"  VaR Scenarios: {scenario_metrics.get('var_scenarios', 0):,}")
            print(f"  Domain Scenarios: {scenario_metrics.get('domain_scenarios', 0):,}")
        
        # Full revaluation
        full_metrics = metrics.get('full_revaluation', {})
        if full_metrics:
            print(f"\nFull Revaluation:")
            print(f"  Total Time: {full_metrics.get('elapsed_time', 0):.2f}s")
            print(f"  Instruments: {full_metrics.get('n_instruments', 0)}")
            print(f"  Scenarios: {full_metrics.get('n_scenarios', 0):,}")
        
        # Approximator calibration - consolidated
        calib_metrics = metrics.get('approximator_calibration', {})
        if calib_metrics:
            print(f"\nApproximator Calibration:")
            total_calib_time = sum(calib_data.get('elapsed_time', 0) for calib_data in calib_metrics.values())
            print(f"  Total Time: {total_calib_time:.2f}s")
            print(f"  Instruments: {next(iter(calib_metrics.values())).get('n_instruments', 0)}")
            print(f"  Fitting Samples: {next(iter(calib_metrics.values())).get('n_fitting_samples', 0)}")
        
        # Approximator pricing - consolidated
        pricing_metrics = metrics.get('approximator_pricing', {})
        if pricing_metrics:
            print(f"\nApproximator Pricing:")
            total_pricing_time = sum(pricing_data.get('elapsed_time', 0) for pricing_data in pricing_metrics.values())
            print(f"  Total Time: {total_pricing_time:.2f}s")
            print(f"  Scenarios: {next(iter(pricing_metrics.values())).get('n_scenarios', 0):,}")
        
        # Hybrid pricing - consolidated
        hybrid_metrics = metrics.get('hybrid_pricing', {})
        if hybrid_metrics:
            print(f"\nHybrid Pricing:")
            total_hybrid_time = sum(hybrid_data.get('elapsed_time', 0) for hybrid_data in hybrid_metrics.values())
            total_full_reval_time = sum(hybrid_data.get('full_revaluation_time', 0) for hybrid_data in hybrid_metrics.values())
            print(f"  Total Time: {total_hybrid_time:.2f}s")
            print(f"  Full Revaluation Time: {total_full_reval_time:.2f}s")
            print(f"  Critical Scenarios: {next(iter(hybrid_metrics.values())).get('n_critical_scenarios', 0)}")
            print(f"  Total Scenarios: {next(iter(hybrid_metrics.values())).get('n_total_scenarios', 0):,}")
        
        # Speedup analysis
        if full_metrics and pricing_metrics:
            print(f"\nSpeedup Analysis:")
            full_time = full_metrics.get('elapsed_time', 0)
            for approx in pricing_metrics.keys():
                approx_time = pricing_metrics[approx].get('elapsed_time', 0)
                if approx_time > 0:
                    speedup = full_time / approx_time
                    print(f"  {approx} vs Full: {speedup:.1f}x faster")
        
        print("="*60) 