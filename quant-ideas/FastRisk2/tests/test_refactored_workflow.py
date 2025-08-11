"""
Test the refactored workflow manager to ensure it maintains the same functionality
as the original while being more maintainable and following separation of concerns.
"""
import unittest
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from scenario_generator import SimpleRandomScenarioGenerator
from workflow_manager_refactored import (
    RefactoredInstrumentProcessor, RefactoredPortfolioBuilder, RefactoredPortfolioAnalytics,
    generate_portfolio_specs_for_serialization
)
from product_handlers import ProductHandlerFactory
from approximator_handlers import ApproximatorHandlerFactory


class TestRefactoredWorkflow(unittest.TestCase):
    """Test the refactored workflow manager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.val_date = date(2025, 5, 18)
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        self.DEMO_CURRENCY = "USD"
        self.DEMO_RATE_INDEX_STUB = "IR"
        
        # Create scenario generator
        self.scenario_gen = SimpleRandomScenarioGenerator(
            base_rates_map={f"{self.DEMO_CURRENCY}_{self.DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t * 0.001 
                           for t in self.tenors},
            base_s0_map={f"{self.DEMO_CURRENCY}_STOCK_S0": 100.0},
            base_vol_map={f"{self.DEMO_CURRENCY}_STOCK_VOL": 0.25},
            random_seed=42
        )
    
    def test_product_handler_factory(self):
        """Test that product handler factory works correctly."""
        # Test getting handlers by product type
        vanilla_handler = ProductHandlerFactory.get_handler("VanillaBond")
        self.assertEqual(vanilla_handler.get_product_type(), "VanillaBond")
        
        callable_handler = ProductHandlerFactory.get_handler("CallableBond")
        self.assertEqual(callable_handler.get_product_type(), "CallableBond")
        
        # Test error for unknown product type
        with self.assertRaises(ValueError):
            ProductHandlerFactory.get_handler("UnknownProduct")
    
    def test_approximator_handler_factory(self):
        """Test that approximator handler factory works correctly."""
        # Test getting handlers by approximator type
        tff_handler = ApproximatorHandlerFactory.get_handler("TFF")
        self.assertEqual(tff_handler.get_approximator_type(), "TFF")
        
        rbfi_handler = ApproximatorHandlerFactory.get_handler("RBFI")
        self.assertEqual(rbfi_handler.get_approximator_type(), "RBFI")
        
        full_handler = ApproximatorHandlerFactory.get_handler("FULL")
        self.assertEqual(full_handler.get_approximator_type(), "FULL")
        
        # Test error for unknown approximator type
        with self.assertRaises(ValueError):
            ApproximatorHandlerFactory.get_handler("UnknownApproximator")
    
    def test_refactored_instrument_processor_creation(self):
        """Test that refactored instrument processor can be created."""
        processor = RefactoredInstrumentProcessor(
            scenario_generator=self.scenario_gen,
            global_valuation_date=self.val_date,
            default_numeric_rate_tenors=self.tenors,
            default_g2_params=(0.01, 0.003, 0.015, 0.006, -0.75),
            default_bs_risk_free_rate=0.025,
            default_bs_dividend_yield=0.01,
            parallel_workers=0,
            n_scenarios_for_domain=50
        )
        
        self.assertIsNotNone(processor)
        self.assertEqual(processor.global_valuation_date, self.val_date)
        self.assertTrue(np.array_equal(processor.default_numeric_rate_tenors, self.tenors))
        self.assertEqual(processor.num_workers, 0)
    
    def test_refactored_instrument_processing(self):
        """Test that refactored instrument processor can process instruments."""
        processor = RefactoredInstrumentProcessor(
            scenario_generator=self.scenario_gen,
            global_valuation_date=self.val_date,
            default_numeric_rate_tenors=self.tenors,
            parallel_workers=0,
            n_scenarios_for_domain=10
        )
        
        # Create test instrument definitions
        instrument_definitions = [
            {
                "instrument_id": "TEST_VANILLA_BOND",
                "product_type": "VanillaBond",
                "pricing_preference": "TFF",
                "params": {
                    "valuation_date": self.val_date.isoformat(),
                    "maturity_date": (self.val_date + relativedelta(years=5)).isoformat(),
                    "coupon_rate": 0.03,
                    "face_value": 100.0,
                    "currency": self.DEMO_CURRENCY,
                    "index_stub": self.DEMO_RATE_INDEX_STUB,
                    "freq": 2
                },
                "tff_config": {"n_train": 8, "n_test": 2}
            },
            {
                "instrument_id": "TEST_EUROPEAN_OPTION",
                "product_type": "EuropeanOption",
                "pricing_preference": "TFF",
                "params": {
                    "valuation_date": self.val_date.isoformat(),
                    "expiry_date": (self.val_date + relativedelta(years=1)).isoformat(),
                    "strike_price": 105.0,
                    "option_type": "call",
                    "currency": self.DEMO_CURRENCY,
                    "underlying_symbol": "STOCK"
                },
                "pricer_params": {"bs_risk_free_rate": 0.025, "bs_dividend_yield": 0.01},
                "tff_config": {"n_train": 8, "n_test": 2}
            }
        ]
        
        # Generate scenarios
        scenarios, factor_names = self.scenario_gen.generate_scenarios(10)
        
        # Process instruments
        model_registry = processor.process_instruments(
            instrument_definitions, scenarios, factor_names
        )
        
        # Verify results
        self.assertIn("TEST_VANILLA_BOND", model_registry)
        self.assertIn("TEST_EUROPEAN_OPTION", model_registry)
        
        # Check that instruments were processed successfully
        vanilla_entry = model_registry["TEST_VANILLA_BOND"]
        self.assertEqual(vanilla_entry["pricing_method"], "TFF")
        self.assertFalse(vanilla_entry.get("error", False))
        self.assertIn("model_dict", vanilla_entry)
        
        option_entry = model_registry["TEST_EUROPEAN_OPTION"]
        self.assertEqual(option_entry["pricing_method"], "TFF")
        self.assertFalse(option_entry.get("error", False))
        self.assertIn("model_dict", option_entry)
    
    def test_refactored_portfolio_builder(self):
        """Test that refactored portfolio builder works correctly."""
        # First create a model registry
        processor = RefactoredInstrumentProcessor(
            scenario_generator=self.scenario_gen,
            global_valuation_date=self.val_date,
            default_numeric_rate_tenors=self.tenors,
            parallel_workers=0,
            n_scenarios_for_domain=10
        )
        
        instrument_definitions = [
            {
                "instrument_id": "TEST_BOND",
                "product_type": "VanillaBond",
                "pricing_preference": "TFF",
                "params": {
                    "valuation_date": self.val_date.isoformat(),
                    "maturity_date": (self.val_date + relativedelta(years=5)).isoformat(),
                    "coupon_rate": 0.03,
                    "face_value": 100.0,
                    "currency": self.DEMO_CURRENCY,
                    "index_stub": self.DEMO_RATE_INDEX_STUB,
                    "freq": 2
                },
                "tff_config": {"n_train": 8, "n_test": 2}
            }
        ]
        
        scenarios, factor_names = self.scenario_gen.generate_scenarios(10)
        model_registry = processor.process_instruments(
            instrument_definitions, scenarios, factor_names
        )
        
        # Create portfolio builder
        builder = RefactoredPortfolioBuilder(model_registry)
        
        # Create portfolio specs
        portfolio_specs = generate_portfolio_specs_for_serialization(
            holdings_data=[{"client_id": "TestClient", "instrument_id": "TEST_BOND", "num_holdings": 100}],
            model_registry=model_registry,
            instrument_definitions_data_for_pricer_params=instrument_definitions
        )
        
        # Build portfolios
        portfolios = builder.build_portfolios_from_specs(
            portfolio_specs, self.val_date
        )
        
        # Verify results
        self.assertIn("TestClient", portfolios)
        portfolio = portfolios["TestClient"]
        self.assertEqual(len(portfolio.positions), 1)
        self.assertEqual(portfolio.positions[0]["instrument_id"], "TEST_BOND")
        self.assertEqual(portfolio.positions[0]["num_holdings"], 100)
    
    def test_refactored_portfolio_analytics(self):
        """Test that refactored portfolio analytics works correctly."""
        # Create a simple portfolio
        from workflow import Portfolio
        
        portfolio = Portfolio()
        portfolios = {"TestClient": portfolio}
        
        # Create analytics
        scenarios, factor_names = self.scenario_gen.generate_scenarios(5)
        analytics = RefactoredPortfolioAnalytics(
            client_portfolios=portfolios,
            global_market_scenarios=scenarios,
            global_factor_names=factor_names,
            numeric_rate_tenors=self.tenors,
            scenario_generator_for_base_values=self.scenario_gen
        )
        
        self.assertIsNotNone(analytics)
        
        # Test base value calculation
        base_values = analytics.calculate_base_portfolio_values()
        self.assertIn("TestClient", base_values)
        
        # Test VaR analysis
        var_results = analytics.run_var_analysis(var_percentiles=[1.0, 5.0])
        self.assertIn("TestClient", var_results)
        self.assertIn("var_1.0%", var_results["TestClient"])
        self.assertIn("var_5.0%", var_results["TestClient"])


class TestProductHandlers(unittest.TestCase):
    """Test individual product handlers."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.val_date = date(2025, 5, 18)
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        self.DEMO_CURRENCY = "USD"
        self.DEMO_RATE_INDEX_STUB = "IR"
        
        self.scenario_gen = SimpleRandomScenarioGenerator(
            base_rates_map={f"{self.DEMO_CURRENCY}_{self.DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t * 0.001 
                           for t in self.tenors},
            base_s0_map={f"{self.DEMO_CURRENCY}_STOCK_S0": 100.0},
            base_vol_map={f"{self.DEMO_CURRENCY}_STOCK_VOL": 0.25},
            random_seed=42
        )
    
    def test_vanilla_bond_handler(self):
        """Test vanilla bond handler."""
        handler = ProductHandlerFactory.get_handler("VanillaBond")
        
        # Create product static
        from product_definitions import QuantLibBondStaticBase
        product_static = QuantLibBondStaticBase(
            valuation_date=self.val_date,
            maturity_date=self.val_date + relativedelta(years=5),
            coupon_rate=0.03,
            face_value=100.0,
            currency=self.DEMO_CURRENCY,
            index_stub=self.DEMO_RATE_INDEX_STUB
        )
        
        # Test pricer creation
        pricer = handler.create_pricer(product_static, {})
        self.assertIsNotNone(pricer)
        
        # Test TFF factors
        tff_factors = handler.get_tff_factors(
            product_static, self.scenario_gen, self.tenors, {}, {}
        )
        
        self.assertIn("tff_input_raw_factor_names", tff_factors)
        self.assertIn("tff_input_raw_base_values", tff_factors)
        self.assertIn("pricer_config_for_worker", tff_factors)
        
        # Check that rate factors are included
        rate_factors = tff_factors["tff_input_raw_factor_names"]
        self.assertTrue(any("USD_IR_" in factor for factor in rate_factors))
    
    def test_european_option_handler(self):
        """Test European option handler."""
        handler = ProductHandlerFactory.get_handler("EuropeanOption")
        
        # Create product static
        from product_definitions import EuropeanOptionStatic
        product_static = EuropeanOptionStatic(
            valuation_date=self.val_date,
            expiry_date=self.val_date + relativedelta(years=1),
            strike_price=105.0,
            option_type="call",
            currency=self.DEMO_CURRENCY,
            underlying_symbol="STOCK"
        )
        
        # Test pricer creation
        pricer = handler.create_pricer(product_static, {"bs_risk_free_rate": 0.025})
        self.assertIsNotNone(pricer)
        
        # Test TFF factors
        tff_factors = handler.get_tff_factors(
            product_static, self.scenario_gen, self.tenors, {}, {"bs_risk_free_rate": 0.025}
        )
        
        self.assertIn("tff_input_raw_factor_names", tff_factors)
        self.assertIn("tff_input_raw_base_values", tff_factors)
        
        # Check that stock factors are included
        factor_names = tff_factors["tff_input_raw_factor_names"]
        self.assertTrue(any("STOCK_S0" in factor for factor in factor_names))
        self.assertTrue(any("STOCK_VOL" in factor for factor in factor_names))


class TestApproximatorHandlers(unittest.TestCase):
    """Test individual approximator handlers."""
    
    def test_tff_handler(self):
        """Test TFF handler."""
        handler = ApproximatorHandlerFactory.get_handler("TFF")
        self.assertEqual(handler.get_approximator_type(), "TFF")
    
    def test_rbfi_handler(self):
        """Test RBFI handler."""
        handler = ApproximatorHandlerFactory.get_handler("RBFI")
        self.assertEqual(handler.get_approximator_type(), "RBFI")
    
    def test_full_handler(self):
        """Test full pricer handler."""
        handler = ApproximatorHandlerFactory.get_handler("FULL")
        self.assertEqual(handler.get_approximator_type(), "FULL")
        
        # Test calibration (should return empty structure)
        result = handler.calibrate(
            pricer_template=None,
            tff_input_raw_factor_names=[],
            tff_input_raw_base_values=np.array([]),
            product_static_params_for_worker={},
            pricer_config_for_worker={},
            actual_rate_pillars=np.array([]),
            scenarios_for_this_approximator=np.array([]),
            config_params={}
        )
        
        self.assertIn("model_dict", result)
        self.assertIn("rmse", result)
        self.assertEqual(result["rmse"], 0.0)
    
    
if __name__ == '__main__':
    unittest.main()


