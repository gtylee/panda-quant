"""
Approximator-specific handlers using Strategy pattern to separate concerns.
Each approximator type has its own handler that encapsulates all approximator-specific logic.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import inspect
import sys

from tff_approximator import TensorFunctionalFormCalibrate
from rbfi_approximator import RBFICalibrate


def approximator_handler(approximator_type: str):
    """
    Decorator to mark a class as an approximator handler for a specific approximator type.
    
    Args:
        approximator_type: The approximator type string this handler handles
        
    Usage:
        @approximator_handler("TFF")
        class TFFHandler(ApproximatorHandler):
            pass
    """
    def decorator(cls):
        if not issubclass(cls, ApproximatorHandler):
            raise ValueError(f"Class {cls.__name__} must inherit from ApproximatorHandler")
        
        # Add the approximator type as a class attribute
        cls._approximator_type = approximator_type
        return cls
    return decorator


class ApproximatorHandler(ABC):
    """Abstract base class for approximator-specific handlers."""
    
    @abstractmethod
    def create_calibrator(self, pricer_template, factor_names: List[str], 
                         base_values: np.ndarray, product_static_params: Dict[str, Any],
                         pricer_config: Dict[str, Any], actual_rate_pillars: np.ndarray = None) -> Any:
        """Create a calibrator instance for this approximator type."""
        pass
    
    @abstractmethod
    def get_approximator_type(self) -> str:
        """Return the approximator type string."""
        pass
    
    @abstractmethod
    def get_training_config(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get training configuration for this approximator type."""
        pass


@approximator_handler("TFF")
class TFFHandler(ApproximatorHandler):
    """Handler for TFF (TensorFlow Financial) approximator."""
    
    def get_approximator_type(self) -> str:
        return "TFF"
    
    def create_calibrator(self, pricer_template, factor_names: List[str], 
                         base_values: np.ndarray, product_static_params: Dict[str, Any],
                         pricer_config: Dict[str, Any], actual_rate_pillars: np.ndarray = None) -> TensorFunctionalFormCalibrate:
        """Create a TFF calibrator instance."""
        return TensorFunctionalFormCalibrate(
            pricer_template=pricer_template,
            tff_input_raw_factor_names=factor_names,
            tff_input_raw_base_values=base_values,
            product_static_params_for_worker=product_static_params,
            pricer_config_for_worker=pricer_config,
            actual_rate_pillars=actual_rate_pillars
        )
    
    def get_training_config(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get TFF-specific training configuration."""
        return {
            'n_train': training_params.get('tff_n_train', 64),
            'n_test': training_params.get('tff_n_test', 8),
            'random_seed': training_params.get('tff_random_seed', 42),
            'sampling_method': training_params.get('tff_sampling_method', 'sobol'),
            'parallel_workers': training_params.get('tff_parallel_workers', False),
            'option_feature_order': training_params.get('tff_option_feature_order', 0),
            'order': training_params.get('tff_order', 2),
            'use_early_stopping': training_params.get('tff_use_early_stopping', True),
            'use_dropout': training_params.get('tff_use_dropout', True),
            'use_regularization': training_params.get('tff_use_regularization', True)
        }

    def calibrate(self, pricer_template, tff_input_raw_factor_names, tff_input_raw_base_values,
                  product_static_params_for_worker, pricer_config_for_worker, actual_rate_pillars,
                  scenarios_for_this_approximator, config_params):
        # Recreate pricer template in worker process to avoid pickling issues
        from product_handlers import ProductHandlerFactory
        from product_definitions import reconstruct_product_static
        
        # Reconstruct product static and create new pricer template
        product_static = reconstruct_product_static(product_static_params_for_worker)
        product_handler = ProductHandlerFactory.get_handler_by_product_static(product_static)
        
        # Extract pricer params from pricer_config_for_worker
        pricer_params = {}
        if 'bs_pricer_config' in pricer_config_for_worker:
            pricer_params.update(pricer_config_for_worker['bs_pricer_config'])
        if 'bond_pricer_config' in pricer_config_for_worker:
            pricer_params.update(pricer_config_for_worker['bond_pricer_config'])
        if 'mbs_pricer_config' in pricer_config_for_worker:
            pricer_params.update(pricer_config_for_worker['mbs_pricer_config'])
        
        # Create new pricer template
        new_pricer_template = product_handler.create_pricer(product_static, pricer_params)
        
        calibrator = self.create_calibrator(
            new_pricer_template, tff_input_raw_factor_names, tff_input_raw_base_values,
            product_static_params_for_worker, pricer_config_for_worker, actual_rate_pillars
        )
        # Fit the model
        tff_model, X_test, y_test, rmse, normalization_params, fit_time, base_value = calibrator.sample_and_fit(
            scenarios_for_this_approximator,
            n_train=config_params.get('n_train', 64),
            n_test=config_params.get('n_test', 8),
            random_seed=config_params.get('seed', 42),
            sampling_method=config_params.get('sampling_method', 'sobol'),
            option_feature_order=config_params.get('option_feature_order', 0),
            order=config_params.get('order', 2),
            **config_params.get('fixed_pricer_params_for_tff_training', {})
        )
        return {
            'model_dict': tff_model.to_dict(),
            'raw_input_names': tff_input_raw_factor_names,
            'normalization_params': normalization_params,
            'option_feature_order': config_params.get('option_feature_order', 0),
            'rmse': rmse,
            'fit_time_seconds': fit_time,
            'base_value': base_value,
            'base_approximator_value': base_value,
            'fixed_pricer_params': config_params.get('fixed_pricer_params_for_tff_training', {})
        }


@approximator_handler("RBFI")
class RBFIHandler(ApproximatorHandler):
    """Handler for RBFI (Radial Basis Function Interpolation) approximator."""
    
    def get_approximator_type(self) -> str:
        return "RBFI"
    
    def create_calibrator(self, pricer_template, factor_names: List[str], 
                         base_values: np.ndarray, product_static_params: Dict[str, Any],
                         pricer_config: Dict[str, Any], actual_rate_pillars: np.ndarray = None) -> RBFICalibrate:
        """Create an RBFI calibrator instance."""
        return RBFICalibrate(
            pricer_template=pricer_template,
            rbfi_input_raw_factor_names=factor_names,
            rbfi_input_raw_base_values=base_values,
            product_static_params_for_worker=product_static_params,
            pricer_config_for_worker=pricer_config,
            actual_rate_pillars=actual_rate_pillars
        )
    
    def get_training_config(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get RBFI-specific training configuration."""
        return {
            'n_train': training_params.get('rbfi_n_train', 64),
            'n_test': training_params.get('rbfi_n_test', 8),
            'random_seed': training_params.get('rbfi_random_seed', 42),
            'sampling_method': training_params.get('rbfi_sampling_method', 'sobol'),
            'parallel_workers': training_params.get('rbfi_parallel_workers', False),
            'option_feature_order': training_params.get('rbfi_option_feature_order', 0),
            'length_scale_method': training_params.get('rbfi_length_scale_method', 'auto'),
            'fixed_length_scale': training_params.get('rbfi_fixed_length_scale', 1.0),
            'regularization': training_params.get('rbfi_regularization', 1e-8),
            'use_smoothing': training_params.get('rbfi_use_smoothing', False),
            'optimize_parameters': training_params.get('rbfi_optimize_parameters', False)
        }

    def calibrate(self, pricer_template, tff_input_raw_factor_names, tff_input_raw_base_values,
                  product_static_params_for_worker, pricer_config_for_worker, actual_rate_pillars,
                  scenarios_for_this_approximator, config_params):
        # Recreate pricer template in worker process to avoid pickling issues
        from product_handlers import ProductHandlerFactory
        from product_definitions import reconstruct_product_static
        
        # Reconstruct product static and create new pricer template
        product_static = reconstruct_product_static(product_static_params_for_worker)
        product_handler = ProductHandlerFactory.get_handler_by_product_static(product_static)
        
        # Extract pricer params from pricer_config_for_worker
        pricer_params = {}
        if 'bs_pricer_config' in pricer_config_for_worker:
            pricer_params.update(pricer_config_for_worker['bs_pricer_config'])
        if 'bond_pricer_config' in pricer_config_for_worker:
            pricer_params.update(pricer_config_for_worker['bond_pricer_config'])
        if 'mbs_pricer_config' in pricer_config_for_worker:
            pricer_params.update(pricer_config_for_worker['mbs_pricer_config'])
        
        # Create new pricer template
        new_pricer_template = product_handler.create_pricer(product_static, pricer_params)
        
        calibrator = self.create_calibrator(
            new_pricer_template, tff_input_raw_factor_names, tff_input_raw_base_values,
            product_static_params_for_worker, pricer_config_for_worker, actual_rate_pillars
        )
        # Fit the model
        rbfi_model, X_test, y_test, rmse, normalization_params, fit_time, base_value = calibrator.sample_and_fit(
            scenarios_for_this_approximator,
            n_train=config_params.get('n_train', 64),
            n_test=config_params.get('n_test', 8),
            random_seed=config_params.get('seed', 42),
            sampling_method=config_params.get('sampling_method', 'sobol'),
            option_feature_order=config_params.get('option_feature_order', 0),
            order=config_params.get('order', 2),
            **config_params.get('fixed_pricer_params_for_tff_training', {})
        )
        return {
            'model_dict': rbfi_model.to_dict(),
            'raw_input_names': tff_input_raw_factor_names,
            'normalization_params': normalization_params,
            'option_feature_order': config_params.get('option_feature_order', 0),
            'rmse': rmse,
            'fit_time_seconds': fit_time,
            'base_value': base_value,
            'base_approximator_value': base_value,
            'fixed_pricer_params': config_params.get('fixed_pricer_params_for_tff_training', {})
        }


@approximator_handler("NeuralNetwork")
class NeuralNetworkHandler(ApproximatorHandler):
    """Handler for Neural Network approximator (placeholder for future implementation)."""
    
    def get_approximator_type(self) -> str:
        return "NeuralNetwork"
    
    def create_calibrator(self, pricer_template, factor_names: List[str], 
                         base_values: np.ndarray, product_static_params: Dict[str, Any],
                         pricer_config: Dict[str, Any], actual_rate_pillars: np.ndarray = None):
        """Create a Neural Network calibrator instance (placeholder)."""
        raise NotImplementedError("Neural Network approximator not yet implemented")
    
    def get_training_config(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get Neural Network-specific training configuration (placeholder)."""
        return {
            'learning_rate': training_params.get('nn_learning_rate', 0.001),
            'batch_size': training_params.get('nn_batch_size', 64),
            'epochs': training_params.get('nn_epochs', 200),
            'validation_split': training_params.get('nn_validation_split', 0.2),
            'early_stopping_patience': training_params.get('nn_early_stopping_patience', 15),
            'dropout_rate': training_params.get('nn_dropout_rate', 0.3),
            'regularization_factor': training_params.get('nn_regularization_factor', 0.001),
            'activation_function': training_params.get('nn_activation_function', 'relu'),
            'optimizer': training_params.get('nn_optimizer', 'adam'),
            'use_early_stopping': training_params.get('nn_use_early_stopping', True),
            'use_dropout': training_params.get('nn_use_dropout', True),
            'use_regularization': training_params.get('nn_use_regularization', True),
            'use_batch_normalization': training_params.get('nn_use_batch_normalization', False)
        }


class ApproximatorHandlerFactory:
    """Factory for creating approximator handlers using attribute-based discovery."""
    
    _handlers = None  # Will be populated on first access
    
    @classmethod
    def _discover_handlers(cls) -> Dict[str, ApproximatorHandler]:
        """
        Discover all approximator handlers in the current module by scanning for classes
        with the @approximator_handler decorator.
        """
        if cls._handlers is not None:
            return cls._handlers
        
        handlers = {}
        current_module = sys.modules[__name__]
        
        # Scan all classes in the current module
        for name, obj in inspect.getmembers(current_module, inspect.isclass):
            # Check if the class has the _approximator_type attribute (set by decorator)
            if hasattr(obj, '_approximator_type') and issubclass(obj, ApproximatorHandler):
                approximator_type = obj._approximator_type
                handlers[approximator_type] = obj()
        
        cls._handlers = handlers
        return handlers
    
    @classmethod
    def get_handler(cls, approximator_type: str) -> ApproximatorHandler:
        """Get handler for a specific approximator type."""
        handlers = cls._discover_handlers()
        if approximator_type not in handlers:
            available_types = list(handlers.keys())
            raise ValueError(f"Unsupported approximator type: {approximator_type}. Available types: {available_types}")
        return handlers[approximator_type]
    
    @classmethod
    def get_available_approximator_types(cls) -> List[str]:
        """Get list of all available approximator types."""
        handlers = cls._discover_handlers()
        return list(handlers.keys())
    
    @classmethod
    def register_handler(cls, approximator_type: str, handler: ApproximatorHandler):
        """
        Register a new approximator handler manually (for backward compatibility).
        Note: This is not needed when using the @approximator_handler decorator.
        """
        if cls._handlers is None:
            cls._handlers = {}
        cls._handlers[approximator_type] = handler
    
    @classmethod
    def clear_cache(cls):
        """Clear the handler cache to force re-discovery."""
        cls._handlers = None 