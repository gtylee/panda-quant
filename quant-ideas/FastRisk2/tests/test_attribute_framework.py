"""
Test script to demonstrate the attribute-based framework for handlers.
"""
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from product_handlers import ProductHandlerFactory
from approximator_handlers import ApproximatorHandlerFactory


def test_product_handler_discovery():
    """Test that product handlers are automatically discovered."""
    print("=== Testing Product Handler Discovery ===")
    
    # Get available product types
    available_types = ProductHandlerFactory.get_available_product_types()
    print(f"Available product types: {available_types}")
    
    # Test getting handlers for each type
    for product_type in available_types:
        try:
            handler = ProductHandlerFactory.get_handler(product_type)
            print(f"✓ Successfully got handler for {product_type}: {handler.__class__.__name__}")
        except Exception as e:
            print(f"✗ Failed to get handler for {product_type}: {e}")
    
    print()


def test_approximator_handler_discovery():
    """Test that approximator handlers are automatically discovered."""
    print("=== Testing Approximator Handler Discovery ===")
    
    # Get available approximator types
    available_types = ApproximatorHandlerFactory.get_available_approximator_types()
    print(f"Available approximator types: {available_types}")
    
    # Test getting handlers for each type
    for approximator_type in available_types:
        try:
            handler = ApproximatorHandlerFactory.get_handler(approximator_type)
            print(f"✓ Successfully got handler for {approximator_type}: {handler.__class__.__name__}")
        except Exception as e:
            print(f"✗ Failed to get handler for {approximator_type}: {e}")
    
    print()


def test_handler_creation():
    """Test creating handlers with parameters."""
    print("=== Testing Handler Creation ===")
    
    # Test TFF approximator creation
    try:
        tff_handler = ApproximatorHandlerFactory.get_handler("TFF")
        tff_config = tff_handler.get_training_config({
            'tff_n_train': 128,
            'tff_n_test': 16,
            'tff_random_seed': 123,
            'tff_sampling_method': 'latin_hypercube'
        })
        print(f"✓ TFF training config: {tff_config}")
    except Exception as e:
        print(f"✗ TFF handler creation failed: {e}")
    
    # Test RBFI approximator creation
    try:
        rbfi_handler = ApproximatorHandlerFactory.get_handler("RBFI")
        rbfi_config = rbfi_handler.get_training_config({
            'rbfi_n_train': 100,
            'rbfi_length_scale_method': 'fixed',
            'rbfi_fixed_length_scale': 0.5
        })
        print(f"✓ RBFI training config: {rbfi_config}")
    except Exception as e:
        print(f"✗ RBFI handler creation failed: {e}")
    
    print()


def test_adding_new_handler():
    """Test adding a new handler dynamically."""
    print("=== Testing Dynamic Handler Addition ===")
    
    # Test adding a custom product handler
    try:
        from product_handlers import ProductHandler
        
        class CustomProductHandler(ProductHandler):
            def get_product_type(self) -> str:
                return "CustomProduct"
            
            def create_pricer(self, product_static, pricer_params):
                # Placeholder implementation
                return None
            
            def get_tff_factors(self, product_static, scenario_generator, 
                              default_numeric_rate_tenors, tff_behavior_params, 
                              instrument_pricer_params):
                # Placeholder implementation
                return {}
        
        # Register the custom handler
        ProductHandlerFactory.register_handler("CustomProduct", CustomProductHandler())
        
        # Verify it's available
        available_types = ProductHandlerFactory.get_available_product_types()
        if "CustomProduct" in available_types:
            print("✓ Custom product handler successfully registered")
        else:
            print("✗ Custom product handler not found in available types")
            
    except Exception as e:
        print(f"✗ Custom handler registration failed: {e}")
    
    print()


def test_error_handling():
    """Test error handling for unsupported types."""
    print("=== Testing Error Handling ===")
    
    # Test unsupported product type
    try:
        ProductHandlerFactory.get_handler("UnsupportedProduct")
        print("✗ Should have raised an error for unsupported product type")
    except ValueError as e:
        print(f"✓ Correctly caught error for unsupported product type: {e}")
    
    # Test unsupported approximator type
    try:
        ApproximatorHandlerFactory.get_handler("UnsupportedApproximator")
        print("✗ Should have raised an error for unsupported approximator type")
    except ValueError as e:
        print(f"✓ Correctly caught error for unsupported approximator type: {e}")
    
    print()


def test_decorator_validation():
    """Test that the decorator properly validates classes."""
    print("=== Testing Decorator Validation ===")
    
    try:
        from product_handlers import product_handler
        
        # This should fail because it doesn't inherit from ProductHandler
        @product_handler("InvalidProduct")
        class InvalidHandler:
            pass
        
        print("✗ Should have raised an error for invalid handler class")
    except ValueError as e:
        print(f"✓ Correctly caught decorator validation error: {e}")
    
    print()


if __name__ == "__main__":
    print("Testing Attribute-Based Handler Framework\n")
    
    test_product_handler_discovery()
    test_approximator_handler_discovery()
    test_handler_creation()
    test_adding_new_handler()
    test_error_handling()
    test_decorator_validation()
    
    print("=== Framework Test Complete ===")


