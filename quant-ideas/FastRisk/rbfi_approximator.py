"""
Contains classes and functions for Radial Basis Function Interpolation (RBFI) approximation.
RBFICalibrate follows the same pattern as TensorFunctionalFormCalibrate.
Uses Gaussian kernels for interpolation.
"""
import numpy as np
from scipy.stats.qmc import LatinHypercube, Sobol, scale
from scipy.spatial.distance import cdist
from concurrent.futures import ProcessPoolExecutor
import QuantLib as ql
from datetime import date, datetime
import warnings

from product_definitions import (
    ProductStaticBase, QuantLibBondStaticBase, CallableBondStaticBase,
    ConvertibleBondStaticBase, EuropeanOptionStatic, MBSPoolStatic,
    _parse_date_input # For worker
)
# Updated pricer imports
from base_pricer import PricerBase
from quantlib_bond_pricer import QuantLibBondPricer
from black_scholes_pricer import BlackScholesPricer
from mbs_pricer import MBSPricer

from prepayment_models import ( # For worker with MBS
    ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel
)

from features_generator import FeatureGenerator, engineer_option_features, normalize_features
# Input the utility functions for feature engineering and normalization from tff_approximator
from tff_approximator import (
    _parse_date_input, _parse_numeric_pillars_from_factor_names
)

def _price_one_scenario_for_rbfi(worker_args: tuple) -> float:
    """
    Worker function for RBFI pricing - identical to TFF worker function.
    """
    (product_static_params_dict, pricer_config_for_worker,
     factor_names_for_rbfi, single_market_scenario_data,
     valuation_date_for_worker_iso, price_kwargs_dict) = worker_args

    valuation_date_for_worker = _parse_date_input(valuation_date_for_worker_iso)
    ql_val_date = ql.Date(valuation_date_for_worker.day, valuation_date_for_worker.month, valuation_date_for_worker.year)
    ql.Settings.instance().evaluationDate = ql_val_date

    product_type = product_static_params_dict['product_type']
    actual_rate_pillars_for_worker = np.asarray(product_static_params_dict.get('actual_rate_pillars', []), dtype=float)

    current_static_params = product_static_params_dict.copy()
    current_static_params['valuation_date'] = valuation_date_for_worker

    if product_type == 'VanillaBond': product_static_obj = QuantLibBondStaticBase.from_dict(current_static_params)
    elif product_type == 'CallableBond': product_static_obj = CallableBondStaticBase.from_dict(current_static_params)
    elif product_type == 'ConvertibleBond': product_static_obj = ConvertibleBondStaticBase.from_dict(current_static_params)
    elif product_type == 'EuropeanOption': product_static_obj = EuropeanOptionStatic.from_dict(current_static_params)
    elif product_type == 'MBSPool': product_static_obj = MBSPoolStatic.from_dict(current_static_params)
    else: raise ValueError(f"Unknown product type for RBFI worker: {product_type}")

    if product_type in ['VanillaBond', 'CallableBond', 'ConvertibleBond']:
        pricer_instance = QuantLibBondPricer(product_static_obj, **pricer_config_for_worker.get('bond_pricer_config',{}))
        market_data_for_ql_pricer = np.array([single_market_scenario_data])

        credit_spread_pillars = price_kwargs_dict.pop('credit_spread_pillar_times', actual_rate_pillars_for_worker)

        price_result_array = pricer_instance.price(
            pillar_times=actual_rate_pillars_for_worker,
            market_scenario_data=market_data_for_ql_pricer,
            credit_spread_pillar_times=credit_spread_pillars,
            **price_kwargs_dict
        )
        return price_result_array[0]

    elif product_type == 'MBSPool':
        mbs_static_obj: MBSPoolStatic = product_static_obj

        prepay_model_type = mbs_static_obj.prepayment_model_type
        prepay_rate_param = mbs_static_obj.prepayment_rate_param
        prepayment_model_instance = None

        if prepay_model_type == "CPR":
            prepayment_model_instance = ConstantCPRModel(prepay_rate_param)
        elif prepay_model_type == "PSA":
            prepayment_model_instance = PSAModel(prepay_rate_param)
        elif prepay_model_type == "RefiIncentive":
            refi_A = price_kwargs_dict.get('refi_A')
            refi_B = price_kwargs_dict.get('refi_B')
            refi_C = price_kwargs_dict.get('refi_C')
            refi_D = price_kwargs_dict.get('refi_D')
            if all(p is not None for p in [refi_A, refi_B, refi_C, refi_D]):
                 prepayment_model_instance = RefiIncentivePrepaymentModel(refi_A, refi_B, refi_C, refi_D)
            else:
                 prepayment_model_instance = RefiIncentivePrepaymentModel()
        else:
            raise ValueError(f"Unsupported prepayment_model_type: {prepay_model_type} for MBS RBFI worker.")

        pricer_instance = MBSPricer(mbs_static_obj, prepayment_model=prepayment_model_instance)
        market_data_for_mbs_pricer = np.array([single_market_scenario_data])

        price_result_array = pricer_instance.price(
            pillar_times_rf=actual_rate_pillars_for_worker,
            market_scenario_data=market_data_for_mbs_pricer,
            credit_spread_pillar_times=actual_rate_pillars_for_worker if mbs_static_obj.credit_spread_curve_name else None,
            **price_kwargs_dict
        )
        return price_result_array[0]

    elif product_type == 'EuropeanOption':
        bs_cfg = pricer_config_for_worker.get('bs_pricer_config', {})
        pricer_instance = BlackScholesPricer(product_static_obj)
        return pricer_instance.price(
            stock_price=single_market_scenario_data[0],
            volatility=single_market_scenario_data[1],
            risk_free_rate=bs_cfg['risk_free_rate'],
            dividend_yield=bs_cfg.get('dividend_yield', 0.0)
        )
    raise ValueError(f"Pricer path failed for product type: {product_type}")


class RadialBasisFunctionInterpolator:
    """
    Radial Basis Function Interpolator using Gaussian kernels.
    """
    def __init__(self, centers: np.ndarray, weights: np.ndarray, length_scales: np.ndarray):
        """
        Initialize RBFI model.
        
        Args:
            centers: Training points (N_centers, D)
            weights: RBF weights (N_centers,)
            length_scales: Gaussian kernel length scales (D,)
        """
        self.centers = np.asarray(centers, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.length_scales = np.asarray(length_scales, dtype=float)
        
        if self.centers.ndim != 2:
            raise ValueError("Centers must be 2D array")
        if self.weights.ndim != 1:
            raise ValueError("Weights must be 1D array")
        if len(self.weights) != len(self.centers):
            raise ValueError("Number of weights must match number of centers")
        
        self.n_centers, self.D = self.centers.shape
        
        if len(self.length_scales) != self.D:
            raise ValueError(f"Length scales dimension {len(self.length_scales)} must match feature dimension {self.D}")

    def _gaussian_kernel(self, x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Compute Gaussian RBF kernel values.
        
        Args:
            x: Query points (N_query, D) or (D,)
            centers: Center points (N_centers, D)
            
        Returns:
            Kernel values (N_query, N_centers) or (N_centers,)
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Scaled distance computation
        scaled_centers = centers / self.length_scales[np.newaxis, :]
        scaled_x = x / self.length_scales[np.newaxis, :]
        
        # Compute squared distances
        distances_sq = cdist(scaled_x, scaled_centers, metric='sqeuclidean')
        
        # Gaussian kernel: exp(-0.5 * distance^2)
        kernel_values = np.exp(-0.5 * distances_sq)
        
        return kernel_values.squeeze() if x.shape[0] == 1 else kernel_values

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate RBFI at query points.
        
        Args:
            x: Query points (N_query, D) or (D,)
            
        Returns:
            Predictions (N_query,) or scalar
        """
        x = np.asarray(x, dtype=float)
        
        if x.ndim == 1:
            if x.shape[0] != self.D:
                raise ValueError(f"Input dimension {x.shape[0]} != model dimension {self.D}")
            kernel_vals = self._gaussian_kernel(x, self.centers)
            return float(np.dot(kernel_vals, self.weights))
        
        elif x.ndim == 2:
            if x.shape[1] != self.D:
                raise ValueError(f"Input shape {x.shape}, expected (N, {self.D})")
            kernel_vals = self._gaussian_kernel(x, self.centers)
            return kernel_vals @ self.weights
        
        else:
            raise ValueError(f"Input must be 1D or 2D, got ndim={x.ndim}")

    def to_dict(self) -> dict:
        """Export RBFI model to dictionary."""
        return {
            'centers': self.centers.tolist(),
            'weights': self.weights.tolist(),
            'length_scales': self.length_scales.tolist(),
            'n_centers': self.n_centers,
            'D': self.D
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'RadialBasisFunctionInterpolator':
        """Load RBFI model from dictionary."""
        required_keys = ['centers', 'weights', 'length_scales']
        if not all(k in data for k in required_keys):
            raise ValueError(f"Missing keys in RBFI data dict. Required: {required_keys}")
        
        return cls(
            centers=np.array(data['centers'], dtype=float),
            weights=np.array(data['weights'], dtype=float),
            length_scales=np.array(data['length_scales'], dtype=float)
        )


class RBFICalibrate:
    """
    Radial Basis Function Interpolation Calibrator - follows TFF pattern.
    """
    def __init__(
        self,
        pricer_template: PricerBase,
        rbfi_input_raw_factor_names: list[str],
        rbfi_input_raw_base_values: np.ndarray,
        product_static_params_for_worker: dict,
        pricer_config_for_worker: dict,
        feature_generation: FeatureGenerator = None,
        actual_rate_pillars: np.ndarray = None
    ):
        self.pricer_template = pricer_template
        self.product_static: ProductStaticBase = pricer_template.product_static

        self.rbfi_input_raw_factor_names = rbfi_input_raw_factor_names
        self.rbfi_input_raw_base_values = rbfi_input_raw_base_values

        self.product_static_params_for_worker = product_static_params_for_worker
        self.pricer_config_for_worker = pricer_config_for_worker
        self.actual_rate_pillars = actual_rate_pillars if actual_rate_pillars is not None else np.array([])

        val_date_from_params = self.product_static_params_for_worker.get('valuation_date')
        if isinstance(val_date_from_params, str):
            self.valuation_date_for_ql_settings_in_worker = date.fromisoformat(val_date_from_params)
        elif isinstance(val_date_from_params, date):
            self.valuation_date_for_ql_settings_in_worker = val_date_from_params
        elif self.product_static and hasattr(self.product_static, 'valuation_date_py'):
             self.valuation_date_for_ql_settings_in_worker = self.product_static.valuation_date_py
        else:
            raise TypeError("valuation_date in product_static_params_for_worker must be an ISO string or date object, or available on product_static.")

        self.product_type_str = self.product_static_params_for_worker.get('product_type')
        if not self.product_type_str:
            if isinstance(self.product_static, EuropeanOptionStatic): self.product_type_str = 'EuropeanOption'
            elif isinstance(self.product_static, CallableBondStaticBase): self.product_type_str = 'CallableBond'
            elif isinstance(self.product_static, ConvertibleBondStaticBase): self.product_type_str = 'ConvertibleBond'
            elif isinstance(self.product_static, MBSPoolStatic): self.product_type_str = 'MBSPool'
            elif isinstance(self.product_static, QuantLibBondStaticBase): self.product_type_str = 'VanillaBond'
            else: raise TypeError(f"Cannot determine product_type_str for RBFICalibrate from {type(self.product_static)}")

        if self.product_type_str in ['VanillaBond', 'CallableBond', 'ConvertibleBond', 'MBSPool']:
            if self.actual_rate_pillars.size == 0 and self.rbfi_input_raw_factor_names:
                 print(f"Warning: 'actual_rate_pillars' was empty for {self.product_type_str}. Parsing from RBFI input names.")
                 self.actual_rate_pillars = _parse_numeric_pillars_from_factor_names(self.rbfi_input_raw_factor_names)
            self.product_static_params_for_worker['actual_rate_pillars'] = self.actual_rate_pillars.tolist()

        if not self.rbfi_input_raw_factor_names or self.rbfi_input_raw_base_values.size == 0:
            raise RuntimeError(f"RBFI input factors/base values not set for {self.product_type_str}")
        if len(self.rbfi_input_raw_factor_names) != len(self.rbfi_input_raw_base_values):
            raise RuntimeError(f"Mismatch RBFI factor names ({len(self.rbfi_input_raw_factor_names)}) and base values ({len(self.rbfi_input_raw_base_values)}).")

        if feature_generation is not None:
            if not isinstance(feature_generation, FeatureGenerator):
                raise TypeError(f"feature_generation must be a FeatureGenerator instance, got {type(feature_generation)}.")
            self.feature_generation = feature_generation
        else:
            self.feature_generation = None

    def sample_and_fit(
        self, full_market_scenarios_for_rbfi_factors: np.ndarray,
        n_train: int = 64, n_test: int = 8,
        random_seed: int = 0, sampling_method: str = 'sobol', parallel_workers: int = None,
        option_feature_order: int = 0, 
        length_scale_method: str = 'auto',  # 'auto', 'fixed', or array
        fixed_length_scale: float = 1.0,
        regularization: float = 1e-8,
        **price_kwargs
    ) -> tuple[RadialBasisFunctionInterpolator, np.ndarray, np.ndarray, float, dict, float, float]:
        """
        Sample training points and fit RBFI model.
        
        Args:
            full_market_scenarios_for_rbfi_factors: Full scenario domain (N_scenarios, D)
            n_train: Number of training points
            n_test: Number of test points
            random_seed: Random seed
            sampling_method: 'sobol' or 'uniform'
            parallel_workers: Number of parallel workers (unused for now)
            option_feature_order: Feature order for options
            length_scale_method: How to set length scales ('auto', 'fixed')
            fixed_length_scale: Fixed length scale value
            regularization: Regularization for matrix inversion
            **price_kwargs: Additional pricing arguments
            
        Returns:
            (fitted_rbfi, test_inputs, test_true_prices, rmse, normalization_params, base_value, base_rbfi_value)
        """
        rng_np = np.random.default_rng(random_seed)
        num_rbfi_factors = len(self.rbfi_input_raw_factor_names)

        if full_market_scenarios_for_rbfi_factors.ndim != 2 or \
           full_market_scenarios_for_rbfi_factors.shape[1] != num_rbfi_factors:
            raise ValueError(f"Shape error for scenarios. Expected (N, {num_rbfi_factors}), got {full_market_scenarios_for_rbfi_factors.shape}. Factors: {self.rbfi_input_raw_factor_names}")

        domain_min, domain_max = np.min(full_market_scenarios_for_rbfi_factors, axis=0), np.max(full_market_scenarios_for_rbfi_factors, axis=0)

        # 1) Sample training points
        train_rbfi_inputs_raw = None
        if sampling_method in ['sobol', 'uniform']:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*power of 2.*")
                sampler = Sobol(d=num_rbfi_factors, scramble=True, seed=random_seed)
                train_rbfi_inputs_raw = scale(sampler.random(n=n_train-1), domain_min, domain_max)
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}. Available methods are 'sobol' and 'uniform'.")

        # Ensure base scenario is included as first training point
        if self.rbfi_input_raw_base_values.ndim == 1:
            if self.rbfi_input_raw_base_values.shape[0] != num_rbfi_factors:
                raise ValueError(f"Base values shape {self.rbfi_input_raw_base_values.shape} does not match factor names length {num_rbfi_factors}.")
            train_rbfi_inputs_raw = np.vstack([self.rbfi_input_raw_base_values, train_rbfi_inputs_raw])

        # 2) Price all training scenarios
        worker_args_list = [(self.product_static_params_for_worker,
                             self.pricer_config_for_worker,
                             self.rbfi_input_raw_factor_names,
                             train_rbfi_inputs_raw[i],
                             self.valuation_date_for_ql_settings_in_worker.isoformat(),
                             price_kwargs) for i in range(n_train)]

        train_prices = np.array([_price_one_scenario_for_rbfi(args) for args in worker_args_list])

        if train_prices.ndim == 0 and n_train == 1: 
            train_prices = np.array([train_prices])
        if train_prices.shape[0] != n_train: 
            raise ValueError(f"Shape of train_prices {train_prices.shape} != n_train {n_train}")

        # 3) Feature engineering
        if self.feature_generation is not None:
            # Use custom feature generator
            feat_vals_train, feat_names = self.feature_generation.create_features()
            feat_normed, means, stds = normalize_features(feat_vals_train)
            rbfi_inputs_for_fitting = feat_normed
            normalization_params = {
                'means': means.tolist(),
                'stds': stds.tolist(),
                'engineered_feature_names': feat_names,
                'is_engineered': True
            }
        else:
            # Fallback: raw factors or built-in option featurization
            rbfi_inputs_for_fitting = train_rbfi_inputs_raw
            normalization_params = {
                'means': None, 'stds': None,
                'engineered_feature_names': self.rbfi_input_raw_factor_names,
                'is_engineered': False
            }
            if self.product_type_str == 'EuropeanOption' and option_feature_order > 0:
                eng_vals, eng_names = engineer_option_features(
                    train_rbfi_inputs_raw[:,0], train_rbfi_inputs_raw[:,1],
                    order=option_feature_order
                )
                feat_normed, means, stds = normalize_features(eng_vals)
                rbfi_inputs_for_fitting = feat_normed
                normalization_params = {
                    'means': means.tolist(),
                    'stds': stds.tolist(),
                    'engineered_feature_names': eng_names,
                    'is_engineered': True
                }

        # 4) Determine length scales
        D_eff = rbfi_inputs_for_fitting.shape[1]
        if length_scale_method == 'auto':
            # Use silverman's rule of thumb
            data_ranges = np.ptp(rbfi_inputs_for_fitting, axis=0)
            length_scales = np.maximum(data_ranges / (2.0 * np.sqrt(D_eff)), 0.2)  # Prevent too small scales
        elif length_scale_method == 'ptp':
            # Estimate length scales from data spread
            data_ranges = np.ptp(rbfi_inputs_for_fitting, axis=0)  # Peak-to-peak (max-min)
            length_scales = np.maximum(data_ranges / 4.0, 0.2)  # Prevent too small scales
        elif length_scale_method == 'fixed':
            length_scales = np.full(D_eff, fixed_length_scale)
        elif isinstance(length_scale_method, (list, np.ndarray)):
            length_scales = np.asarray(length_scale_method, dtype=float)
            if len(length_scales) != D_eff:
                raise ValueError(f"Length scales array length {len(length_scales)} != feature dimension {D_eff}")
        else:
            raise ValueError(f"Unknown length_scale_method: {length_scale_method}")

        # 5) Build and solve RBF system
        # Compute kernel matrix K
        rbfi = RadialBasisFunctionInterpolator(
            centers=rbfi_inputs_for_fitting,
            weights=np.zeros(n_train),  # Temporary
            length_scales=length_scales
        )
        
        K = rbfi._gaussian_kernel(rbfi_inputs_for_fitting, rbfi_inputs_for_fitting)
        
        # Add regularization to diagonal
        K_reg = K + regularization * np.eye(n_train)
        
        # Solve for weights: K * weights = prices
        try:
            weights = np.linalg.solve(K_reg, train_prices)
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError(f"RBFI solve failed: {e}")

        # Create fitted RBFI model
        fitted_rbfi = RadialBasisFunctionInterpolator(
            centers=rbfi_inputs_for_fitting,
            weights=weights,
            length_scales=length_scales
        )

        # 6) Generate test scenarios and evaluate
        test_idx = rng_np.choice(full_market_scenarios_for_rbfi_factors.shape[0], size=n_test-1, replace=False)
        test_idx = np.insert(test_idx, 0, 0)  # Ensure first scenario is included
        test_rbfi_inputs_raw = full_market_scenarios_for_rbfi_factors[test_idx]
        
        test_worker_args = [(self.product_static_params_for_worker, self.pricer_config_for_worker,
             self.rbfi_input_raw_factor_names, test_rbfi_inputs_raw[i],
             self.valuation_date_for_ql_settings_in_worker.isoformat(), price_kwargs) for i in range(n_test)]
        
        test_true_prices = np.array([_price_one_scenario_for_rbfi(args) for args in test_worker_args])

        # Apply same feature logic to test set
        if self.feature_generation is not None:
            feat_vals_test, _ = self.feature_generation.create_features()
            test_inputs_eval, _, _ = normalize_features(
                feat_vals_test,
                np.array(normalization_params['means']),
                np.array(normalization_params['stds'])
            )
        else:
            test_inputs_eval = test_rbfi_inputs_raw
            if self.product_type_str == 'EuropeanOption' and normalization_params.get('is_engineered', False):
                eng_vals_test, _ = engineer_option_features(
                    test_rbfi_inputs_raw[:,0], test_rbfi_inputs_raw[:,1],
                    order=option_feature_order
                )
                test_inputs_eval, _, _ = normalize_features(
                    eng_vals_test,
                    np.array(normalization_params['means']),
                    np.array(normalization_params['stds'])
                )

        test_pred_prices = fitted_rbfi(test_inputs_eval)
        
        # Get base values
        base_value = test_true_prices[0]
        base_rbfi_value = fitted_rbfi(rbfi_inputs_for_fitting[0])
        
        if test_true_prices.ndim == 0 and n_test == 1: 
            test_true_prices = np.array([test_true_prices])
        if test_pred_prices.ndim == 0 and n_test == 1: 
            test_pred_prices = np.array([test_pred_prices])
        if test_true_prices.shape != test_pred_prices.shape: 
            raise ValueError(f"Shape mismatch test prices: true {test_true_prices.shape}, pred {test_pred_prices.shape}")

        rmse = np.sqrt(np.mean((test_true_prices - test_pred_prices)**2))
        
        return fitted_rbfi, test_rbfi_inputs_raw, test_true_prices, rmse, normalization_params, base_value, base_rbfi_value