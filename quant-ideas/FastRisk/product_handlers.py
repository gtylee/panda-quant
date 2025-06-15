"""
Product-specific handlers using Strategy pattern to separate concerns.
Each product type has its own handler that encapsulates all product-specific logic.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import date
import inspect
import sys

from product_definitions import (
    ProductStaticBase, QuantLibBondStaticBase, CallableBondStaticBase,
    ConvertibleBondStaticBase, EuropeanOptionStatic, MBSPoolStatic
)
from base_pricer import PricerBase
from black_scholes_pricer import BlackScholesPricer
from quantlib_bond_pricer import QuantLibBondPricer
from fast_bond_pricer import FastBondPricer
from mbs_pricer import MBSPricer
from prepayment_models import ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel


def product_handler(product_type: str):
    """
    Decorator to mark a class as a product handler for a specific product type.
    
    Args:
        product_type: The product type string this handler handles
        
    Usage:
        @product_handler("VanillaBond")
        class VanillaBondHandler(ProductHandler):
            pass
    """
    def decorator(cls):
        if not issubclass(cls, ProductHandler):
            raise ValueError(f"Class {cls.__name__} must inherit from ProductHandler")
        
        # Add the product type as a class attribute
        cls._product_type = product_type
        return cls
    return decorator


class ProductHandler(ABC):
    """Abstract base class for product-specific handlers."""
    
    @abstractmethod
    def create_pricer(self, product_static: ProductStaticBase, pricer_params: Dict[str, Any]) -> PricerBase:
        """Create a pricer instance for this product type."""
        pass
    
    @abstractmethod
    def get_tff_factors(self, product_static: ProductStaticBase, 
                       scenario_generator, default_numeric_rate_tenors: np.ndarray,
                       tff_behavior_params: Dict[str, Any],
                       instrument_pricer_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get TFF factors and configuration for this product type."""
        pass
    
    @abstractmethod
    def get_product_type(self) -> str:
        """Return the product type string."""
        pass


@product_handler("EuropeanOption")
class EuropeanOptionHandler(ProductHandler):
    """Handler for European Option products."""
    
    def get_product_type(self) -> str:
        return "EuropeanOption"
    
    def create_pricer(self, product_static: EuropeanOptionStatic, pricer_params: Dict[str, Any]) -> PricerBase:
        pricer = BlackScholesPricer(product_static)
        pricer._default_price_kwargs = {
            'risk_free_rate': pricer_params.get('bs_risk_free_rate', 0.025),
            'dividend_yield': pricer_params.get('bs_dividend_yield', 0.01)
        }
        return pricer
    
    def get_tff_factors(self, product_static: EuropeanOptionStatic,
                       scenario_generator, default_numeric_rate_tenors: np.ndarray,
                       tff_behavior_params: Dict[str, Any],
                       instrument_pricer_params: Dict[str, Any]) -> Dict[str, Any]:
        
        if not product_static.underlying_symbol or not product_static.currency:
            raise ValueError("EuropeanOptionStatic needs 'underlying_symbol' and 'currency'.")
        
        s0_fn = f"{product_static.currency}_{product_static.underlying_symbol}_S0"
        vol_fn = f"{product_static.currency}_{product_static.underlying_symbol}_VOL"
        
        raw_names = [s0_fn, vol_fn]
        raw_base_values = [
            self._get_base_value(scenario_generator, s0_fn),
            self._get_base_value(scenario_generator, vol_fn)
        ]
        
        opt_feature_order = tff_behavior_params.get('option_feature_order', 0)
        pricer_cfg_worker = {
            'bs_pricer_config': {
                'risk_free_rate': instrument_pricer_params.get('bs_risk_free_rate'),
                'dividend_yield': instrument_pricer_params.get('bs_dividend_yield', 0.0)
            }
        }
        
        if pricer_cfg_worker['bs_pricer_config']['risk_free_rate'] is None:
            raise ValueError("Missing 'bs_risk_free_rate' in pricer_params for EuropeanOption.")
        
        return {
            "tff_input_raw_factor_names": raw_names,
            "tff_input_raw_base_values": np.array(raw_base_values),
            "fixed_pricer_params_for_tff_training": {},
            "option_feature_order": opt_feature_order,
            "pricer_config_for_worker": pricer_cfg_worker,
            "actual_rate_pillars": np.array([])
        }
    
    def _get_base_value(self, scenario_generator, factor_name: str) -> float:
        """Helper to get base value from scenario generator's maps."""
        for map_name in ['base_rates_map', 'base_s0_map', 'base_vol_map', 'base_credit_spread_points_map']:
            map_obj = getattr(scenario_generator, map_name, {})
            if factor_name in map_obj:
                return map_obj[factor_name]
        raise ValueError(f"Base value for TFF factor '{factor_name}' not found in scenario_generator's configured base maps.")


@product_handler("VanillaBond")
class VanillaBondHandler(ProductHandler):
    """Handler for Vanilla Bond products."""
    
    def get_product_type(self) -> str:
        return "VanillaBond"
    
    def create_pricer(self, product_static: QuantLibBondStaticBase, pricer_params: Dict[str, Any]) -> PricerBase:
        if pricer_params.get('pricer_type_preference', 'QuantLib').upper() == 'FAST':
            return FastBondPricer(product_static)
        return QuantLibBondPricer(product_static, method='discount')
    
    def get_tff_factors(self, product_static: QuantLibBondStaticBase,
                       scenario_generator, default_numeric_rate_tenors: np.ndarray,
                       tff_behavior_params: Dict[str, Any],
                       instrument_pricer_params: Dict[str, Any]) -> Dict[str, Any]:
        
        return self._get_bond_tff_factors(
            product_static, scenario_generator, default_numeric_rate_tenors,
            tff_behavior_params, instrument_pricer_params, method='discount'
        )
    
    def _get_bond_tff_factors(self, product_static: QuantLibBondStaticBase,
                             scenario_generator, default_numeric_rate_tenors: np.ndarray,
                             tff_behavior_params: Dict[str, Any],
                             instrument_pricer_params: Dict[str, Any],
                             method: str) -> Dict[str, Any]:
        
        if not product_static.currency or not product_static.index_stub:
            raise ValueError(f"{product_static.__class__.__name__} needs 'currency' and a non-empty 'index_stub'.")
        
        if default_numeric_rate_tenors is None or default_numeric_rate_tenors.size == 0:
            raise ValueError("default_numeric_rate_tenors needed for TFF setup.")
        
        # Rate factors
        rate_factor_names = [f"{product_static.currency}_{product_static.index_stub}_{t:.2f}Y" 
                           for t in default_numeric_rate_tenors]
        base_rate_vals = [self._get_base_value(scenario_generator, name) for name in rate_factor_names]
        
        raw_names = rate_factor_names.copy()
        raw_base_values = base_rate_vals.copy()
        actual_pillars = self._parse_numeric_pillars_from_factor_names(rate_factor_names)
        
        pricer_cfg_worker = {'bond_pricer_config': {'method': method}}
        
        # Credit spread factors
        if hasattr(product_static, 'credit_spread_curve_name') and product_static.credit_spread_curve_name:
            cs_curve_name = product_static.credit_spread_curve_name
            cs_factor_names = [f"{cs_curve_name}_{t:.2f}Y" for t in default_numeric_rate_tenors]
            base_cs_vals = [self._get_base_value(scenario_generator, name) for name in cs_factor_names]
            raw_names.extend(cs_factor_names)
            raw_base_values.extend(base_cs_vals)
        
        return {
            "tff_input_raw_factor_names": raw_names,
            "tff_input_raw_base_values": np.array(raw_base_values),
            "fixed_pricer_params_for_tff_training": {},
            "option_feature_order": 0,
            "pricer_config_for_worker": pricer_cfg_worker,
            "actual_rate_pillars": actual_pillars
        }
    
    def _get_base_value(self, scenario_generator, factor_name: str) -> float:
        """Helper to get base value from scenario generator's maps."""
        for map_name in ['base_rates_map', 'base_s0_map', 'base_vol_map', 'base_credit_spread_points_map']:
            map_obj = getattr(scenario_generator, map_name, {})
            if factor_name in map_obj:
                return map_obj[factor_name]
        raise ValueError(f"Base value for TFF factor '{factor_name}' not found in scenario_generator's configured base maps.")
    
    def _parse_numeric_pillars_from_factor_names(self, factor_names: List[str]) -> np.ndarray:
        """Parse numeric pillars from factor names."""
        pillars = []
        for name in factor_names:
            if '_' in name and name.endswith('Y'):
                try:
                    pillar_str = name.split('_')[-1].replace('Y', '')
                    pillars.append(float(pillar_str))
                except ValueError:
                    continue
        return np.array(pillars)


@product_handler("CallableBond")
class CallableBondHandler(VanillaBondHandler):
    """Handler for Callable Bond products."""
    
    def get_product_type(self) -> str:
        return "CallableBond"
    
    def create_pricer(self, product_static: CallableBondStaticBase, pricer_params: Dict[str, Any]) -> PricerBase:
        grid_steps = pricer_params.get('g2_grid_steps', 32)
        return QuantLibBondPricer(product_static, method='g2', grid_steps=grid_steps)
    
    def get_tff_factors(self, product_static: CallableBondStaticBase,
                       scenario_generator, default_numeric_rate_tenors: np.ndarray,
                       tff_behavior_params: Dict[str, Any],
                       instrument_pricer_params: Dict[str, Any]) -> Dict[str, Any]:
        
        result = self._get_bond_tff_factors(
            product_static, scenario_generator, default_numeric_rate_tenors,
            tff_behavior_params, instrument_pricer_params, method='g2'
        )
        
        # Add G2-specific configuration
        result["pricer_config_for_worker"]['bond_pricer_config']['grid_steps'] = \
            instrument_pricer_params.get('g2_grid_steps', 32)
        
        if instrument_pricer_params.get('g2_params'):
            result["fixed_pricer_params_for_tff_training"]['g2_params'] = \
                instrument_pricer_params['g2_params']
        
        return result


@product_handler("ConvertibleBond")
class ConvertibleBondHandler(VanillaBondHandler):
    """Handler for Convertible Bond products."""
    
    def get_product_type(self) -> str:
        return "ConvertibleBond"
    
    def create_pricer(self, product_static: ConvertibleBondStaticBase, pricer_params: Dict[str, Any]) -> PricerBase:
        engine_steps = pricer_params.get('conv_engine_steps', 128)
        return QuantLibBondPricer(product_static, method='convertible_binomial', 
                                 convertible_engine_steps=engine_steps)
    
    def get_tff_factors(self, product_static: ConvertibleBondStaticBase,
                       scenario_generator, default_numeric_rate_tenors: np.ndarray,
                       tff_behavior_params: Dict[str, Any],
                       instrument_pricer_params: Dict[str, Any]) -> Dict[str, Any]:
        
        result = self._get_bond_tff_factors(
            product_static, scenario_generator, default_numeric_rate_tenors,
            tff_behavior_params, instrument_pricer_params, method='convertible_binomial'
        )
        
        # Add convertible-specific configuration
        result["pricer_config_for_worker"]['bond_pricer_config']['convertible_engine_steps'] = \
            instrument_pricer_params.get('conv_engine_steps', 128)
        
        if not product_static.underlying_symbol:
            raise ValueError("Convertible needs 'underlying_symbol'.")
        
        # Add underlying stock factor
        s0_fn_cb = f"{product_static.currency}_{product_static.underlying_symbol}_S0"
        if s0_fn_cb not in result["tff_input_raw_factor_names"]:
            result["tff_input_raw_factor_names"].append(s0_fn_cb)
            result["tff_input_raw_base_values"] = np.append(
                result["tff_input_raw_base_values"], 
                self._get_base_value(scenario_generator, s0_fn_cb)
            )
        
        conv_all_dynamic = tff_behavior_params.get('convertible_tff_market_inputs_as_factors', False)
        
        if conv_all_dynamic:
            # Add all convertible factors dynamically
            div_fn = f"{product_static.currency}_{product_static.underlying_symbol}_DIVYIELD"
            vol_fn = f"{product_static.currency}_{product_static.underlying_symbol}_EQVOL"
            cs_fn_engine = f"{product_static.currency}_{product_static.underlying_symbol}_CS"
            
            new_factors = []
            new_base_values = []
            
            for factor_name in [div_fn, vol_fn, cs_fn_engine]:
                if factor_name not in result["tff_input_raw_factor_names"]:
                    new_factors.append(factor_name)
                    new_base_values.append(self._get_base_value(scenario_generator, factor_name))
            
            result["tff_input_raw_factor_names"].extend(new_factors)
            result["tff_input_raw_base_values"] = np.append(
                result["tff_input_raw_base_values"], new_base_values
            )
        else:
            # Use fixed parameters
            fixed_cb_p = tff_behavior_params.get('fixed_cb_params', {})
            fixed_params = {
                'dividend_yield': fixed_cb_p.get('dividend_yield'),
                'equity_volatility': fixed_cb_p.get('equity_volatility'),
                'credit_spread': fixed_cb_p.get('credit_spread')
            }
            
            if any(v is None for k, v in fixed_params.items() if k in ['dividend_yield', 'equity_volatility', 'credit_spread']):
                raise ValueError(f"Missing fixed CB params (div,eq_vol,cs) when S0 is dynamic but others fixed. Got: {fixed_cb_p}")
            
            result["fixed_pricer_params_for_tff_training"].update(fixed_params)
        
        return result


@product_handler("MBSPool")
class MBSPoolHandler(VanillaBondHandler):
    """Handler for MBS Pool products."""
    
    def get_product_type(self) -> str:
        return "MBSPool"
    
    def create_pricer(self, product_static: MBSPoolStatic, pricer_params: Dict[str, Any]) -> PricerBase:
        prepayment_model_type = product_static.prepayment_model_type
        prepayment_rate_param = product_static.prepayment_rate_param
        prepayment_model_instance = None
        
        if prepayment_model_type == "CPR":
            prepayment_model_instance = ConstantCPRModel(prepayment_rate_param)
        elif prepayment_model_type == "PSA":
            prepayment_model_instance = PSAModel(prepayment_rate_param)
        elif prepayment_model_type == "RefiIncentive":
            refi_A = pricer_params.get('refi_A')
            refi_B = pricer_params.get('refi_B')
            refi_C = pricer_params.get('refi_C')
            refi_D = pricer_params.get('refi_D')
            if all(p is not None for p in [refi_A, refi_B, refi_C, refi_D]):
                prepayment_model_instance = RefiIncentivePrepaymentModel(refi_A, refi_B, refi_C, refi_D)
            else:
                prepayment_model_instance = RefiIncentivePrepaymentModel()
        else:
            raise ValueError(f"Unsupported prepayment_model_type: {prepayment_model_type} for MBS.")
        
        return MBSPricer(product_static, prepayment_model=prepayment_model_instance)
    
    def get_tff_factors(self, product_static: MBSPoolStatic,
                       scenario_generator, default_numeric_rate_tenors: np.ndarray,
                       tff_behavior_params: Dict[str, Any],
                       instrument_pricer_params: Dict[str, Any]) -> Dict[str, Any]:
        
        result = self._get_bond_tff_factors(
            product_static, scenario_generator, default_numeric_rate_tenors,
            tff_behavior_params, instrument_pricer_params, method='discount'
        )
        
        # Add MBS-specific configuration
        result["pricer_config_for_worker"]['mbs_pricer_config'] = {
            'prepayment_model_type': product_static.prepayment_model_type,
            'prepayment_rate_param': product_static.prepayment_rate_param
        }
        
        # Add MBS-specific fixed parameters
        fixed_mbs_p = tff_behavior_params.get('fixed_mbs_params', {})
        mbs_fixed_params = {}
        
        for param in ['fixed_market_mortgage_rate_for_prepay', 'refi_A', 'refi_B', 'refi_C', 'refi_D']:
            if param in fixed_mbs_p:
                mbs_fixed_params[param] = fixed_mbs_p[param]
        
        result["fixed_pricer_params_for_tff_training"].update(mbs_fixed_params)
        
        return result


@product_handler("Futures")
class FuturesHandler(ProductHandler):
    """Handler for Futures products (example of adding a new product type)."""
    
    def get_product_type(self) -> str:
        return "Futures"
    
    def create_pricer(self, product_static: ProductStaticBase, pricer_params: Dict[str, Any]) -> PricerBase:
        """Create a pricer for futures (placeholder implementation)."""
        # In a real implementation, you would create an actual futures pricer
        # For now, we'll return None as a placeholder
        print(f"Creating futures pricer for {product_static}")
        return None
    
    def get_tff_factors(self, product_static: ProductStaticBase,
                       scenario_generator, default_numeric_rate_tenors: np.ndarray,
                       tff_behavior_params: Dict[str, Any],
                       instrument_pricer_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get TFF factors for futures products."""
        
        # Example implementation for futures
        if not hasattr(product_static, 'underlying_symbol') or not product_static.underlying_symbol:
            raise ValueError("Futures product needs 'underlying_symbol'.")
        
        # For futures, we typically need spot price and interest rate factors
        s0_fn = f"{product_static.currency}_{product_static.underlying_symbol}_S0"
        rate_fn = f"{product_static.currency}_LIBOR_3M"  # Example rate factor
        
        raw_names = [s0_fn, rate_fn]
        raw_base_values = [
            self._get_base_value(scenario_generator, s0_fn),
            self._get_base_value(scenario_generator, rate_fn)
        ]
        
        return {
            "tff_input_raw_factor_names": raw_names,
            "tff_input_raw_base_values": np.array(raw_base_values),
            "fixed_pricer_params_for_tff_training": {},
            "option_feature_order": 0,
            "pricer_config_for_worker": {
                'futures_pricer_config': {
                    'pricing_method': 'cost_of_carry'
                }
            },
            "actual_rate_pillars": np.array([])
        }
    
    def _get_base_value(self, scenario_generator, factor_name: str) -> float:
        """Helper to get base value from scenario generator's maps."""
        for map_name in ['base_rates_map', 'base_s0_map', 'base_vol_map', 'base_credit_spread_points_map']:
            map_obj = getattr(scenario_generator, map_name, {})
            if factor_name in map_obj:
                return map_obj[factor_name]
        # Return a default value for demonstration
        return 100.0


class ProductHandlerFactory:
    """Factory for creating product handlers using attribute-based discovery."""
    
    _handlers = None  # Will be populated on first access
    
    @classmethod
    def _discover_handlers(cls) -> Dict[str, ProductHandler]:
        """
        Discover all product handlers in the current module by scanning for classes
        with the @product_handler decorator.
        """
        if cls._handlers is not None:
            return cls._handlers
        
        handlers = {}
        current_module = sys.modules[__name__]
        
        # Scan all classes in the current module
        for name, obj in inspect.getmembers(current_module, inspect.isclass):
            # Check if the class has the _product_type attribute (set by decorator)
            if hasattr(obj, '_product_type') and issubclass(obj, ProductHandler):
                product_type = obj._product_type
                handlers[product_type] = obj()
        
        cls._handlers = handlers
        return handlers
    
    @classmethod
    def get_handler(cls, product_type: str) -> ProductHandler:
        """Get handler for a specific product type."""
        handlers = cls._discover_handlers()
        if product_type not in handlers:
            available_types = list(handlers.keys())
            raise ValueError(f"Unsupported product type: {product_type}. Available types: {available_types}")
        return handlers[product_type]
    
    @classmethod
    def get_handler_by_product_static(cls, product_static: ProductStaticBase) -> ProductHandler:
        """Get handler based on product static object type."""
        handlers = cls._discover_handlers()
        
        # Map product static types to handler types
        type_mapping = {
            EuropeanOptionStatic: "EuropeanOption",
            CallableBondStaticBase: "CallableBond",
            ConvertibleBondStaticBase: "ConvertibleBond",
            MBSPoolStatic: "MBSPool",
            QuantLibBondStaticBase: "VanillaBond",  # Default for vanilla bonds
        }
        
        for static_type, handler_type in type_mapping.items():
            if isinstance(product_static, static_type):
                if handler_type in handlers:
                    return handlers[handler_type]
        
        raise ValueError(f"Unsupported product static type: {type(product_static)}")
    
    @classmethod
    def get_available_product_types(cls) -> List[str]:
        """Get list of all available product types."""
        handlers = cls._discover_handlers()
        return list(handlers.keys())
    
    @classmethod
    def register_handler(cls, product_type: str, handler: ProductHandler):
        """
        Register a new product handler manually (for backward compatibility).
        Note: This is not needed when using the @product_handler decorator.
        """
        if cls._handlers is None:
            cls._handlers = {}
        cls._handlers[product_type] = handler
    
    @classmethod
    def clear_cache(cls):
        """Clear the handler cache to force re-discovery."""
        cls._handlers = None 