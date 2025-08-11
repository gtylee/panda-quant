"""
Example demonstrating how to add a new product handler using the attribute-based framework.
"""
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from product_handlers import ProductHandler, product_handler, ProductHandlerFactory
from product_definitions import ProductStaticBase
from base_pricer import PricerBase
import numpy as np
from typing import Dict, List, Any


# Example: Adding a new "Futures" product handler
@product_handler("Futures")
class FuturesHandler(ProductHandler):
    """Handler for Futures products."""
    
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


def demonstrate_new_handler():
    """Demonstrate how the new handler is automatically discovered."""
    print("=== Demonstrating New Handler Discovery ===\n")
    
    # Clear the cache to force re-discovery (in case the module was already loaded)
    ProductHandlerFactory.clear_cache()
    
    # Get available product types - should now include "Futures"
    available_types = ProductHandlerFactory.get_available_product_types()
    print(f"Available product types: {available_types}")
    
    # Check if our new handler is available
    if "Futures" in available_types:
        print("✓ Futures handler successfully discovered!")
        
        # Get the handler
        futures_handler = ProductHandlerFactory.get_handler("Futures")
        print(f"✓ Got handler: {futures_handler.__class__.__name__}")
        
        # Test the handler methods
        print(f"✓ Product type: {futures_handler.get_product_type()}")
        
        # Test creating a mock product static object
        class MockFuturesStatic(ProductStaticBase):
            def __init__(self):
                self.currency = "USD"
                self.underlying_symbol = "SPX"
        
        mock_product = MockFuturesStatic()
        
        # Test TFF factors generation
        try:
            # Create a mock scenario generator
            class MockScenarioGenerator:
                def __init__(self):
                    self.base_s0_map = {"USD_SPX_S0": 4500.0}
                    self.base_rates_map = {"USD_LIBOR_3M": 0.05}
            
            mock_generator = MockScenarioGenerator()
            
            tff_factors = futures_handler.get_tff_factors(
                product_static=mock_product,
                scenario_generator=mock_generator,
                default_numeric_rate_tenors=np.array([0.25, 0.5, 1.0]),
                tff_behavior_params={},
                instrument_pricer_params={}
            )
            
            print("✓ TFF factors generated successfully:")
            print(f"  - Factor names: {tff_factors['tff_input_raw_factor_names']}")
            print(f"  - Base values: {tff_factors['tff_input_raw_base_values']}")
            
        except Exception as e:
            print(f"✗ TFF factors generation failed: {e}")
        
    else:
        print("✗ Futures handler not found in available types")
    
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    demonstrate_new_handler() 