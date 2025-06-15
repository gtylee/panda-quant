"""
Regression tests based on demo files to ensure functionality remains intact during refactoring.
These tests validate the core workflows and expected outputs from the demo files.
"""
import unittest
import numpy as np
import json
import tempfile
import os
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import time
import io
import sys

# Import demo functions
from demo_main import run_demonstration
from demo_calibration import run_calibration_demo, generate_instrument_definitions
from demo_compare import run_comparison_demo
from demo_rbfi import run_callable_bond_rbfi_demo

# Import core classes for validation
from workflow_manager import (
    InstrumentProcessor, PortfolioBuilder, PortfolioAnalytics,
    generate_portfolio_specs_for_serialization, portfolio_json_serializer
)
from scenario_generator import SimpleRandomScenarioGenerator


class TestDemoMain(unittest.TestCase):
    def test_demo_main_runs(self):
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_demonstration(enable_parallel_tff_fitting=False)
            output = captured.getvalue()
            self.assertIn("End of Demonstration", output)
            self.assertNotIn("ERROR", output)
        finally:
            sys.stdout = sys.__stdout__


class TestDemoRBFI(unittest.TestCase):
    def test_rbfi_demo_runs(self):
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_callable_bond_rbfi_demo(num_callable_bonds=2, num_var_scenarios=5, n_domain_scenarios=10, n_fitting_samples=8, random_seed=42)
            output = captured.getvalue()
            self.assertIn("RBFI", output)
            self.assertNotIn("ERROR", output)
        finally:
            sys.stdout = sys.__stdout__


class TestDemoCalibration(unittest.TestCase):
    def test_calibration_demo_runs(self):
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_calibration_demo(num_instruments_to_generate=5, num_workers=1, batch_size=2, random_seed=42)
            output = captured.getvalue()
            self.assertIn("TFF Calibration Demo", output)
            self.assertNotIn("ERROR", output)
        finally:
            sys.stdout = sys.__stdout__


class TestDemoCompare(unittest.TestCase):
    def test_compare_demo_runs(self):
        captured = io.StringIO()
        sys.stdout = captured
        try:
            run_comparison_demo(num_instruments_to_generate=3, num_var_scenarios=10, n_tff_domain_scenarios=20, n_tff_fitting_samples=10, num_tff_workers=1, tff_batch_size=2, random_seed=42)
            output = captured.getvalue()
            self.assertIn("Comparison Demo: Full vs. TFF Pricing", output)
            self.assertNotIn("ERROR", output)
        finally:
            sys.stdout = sys.__stdout__


class TestDemoMainRegression(unittest.TestCase):
    """Regression tests for demo_main.py functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.val_date = date(2025, 5, 18)
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        self.DEMO_CURRENCY = "USD"
        self.DEMO_RATE_INDEX_STUB = "IR"
        
    def test_demo_main_basic_execution(self):
        """Test that demo_main runs without errors and produces expected outputs."""
        # Capture output to validate
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Run demo with minimal settings for speed
            run_demonstration(
                enable_parallel_tff_fitting=False,
                use_hardcoded_g2_params=True
            )
            
            output = captured_output.getvalue()
            
            # Validate expected output patterns
            self.assertIn("FastRiskDemo with Workflow Manager", output)
            self.assertIn("Step 1: Instrument Processing", output)
            self.assertIn("Step 2: Portfolio Construction", output)
            self.assertIn("Step 3: Portfolio Pricing", output)
            self.assertIn("End of Demonstration", output)
            
            # Check for no critical errors
            self.assertNotIn("ERROR:", output)
            self.assertNotIn("CRITICAL:", output)
            
        finally:
            # Restore stdout
            sys.stdout = sys.__stdout__
    
    def test_instrument_processor_creation(self):
        """Test InstrumentProcessor creation with demo parameters."""
        scenario_gen = SimpleRandomScenarioGenerator(
            base_rates_map={f"{self.DEMO_CURRENCY}_{self.DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t * 0.001 
                           for t in self.tenors},
            base_s0_map={},
            base_vol_map={},
            random_seed=42
        )
        
        processor = InstrumentProcessor(
            scenario_generator=scenario_gen,
            global_valuation_date=self.val_date,
            default_numeric_rate_tenors=self.tenors,
            default_g2_params=(0.01, 0.003, 0.015, 0.006, -0.75),
            default_bs_risk_free_rate=0.025,
            default_bs_dividend_yield=0.01,
            parallel_workers_tff=False,
            n_scenarios_for_tff_domain=50
        )
        
        self.assertIsNotNone(processor)
        self.assertEqual(processor.global_valuation_date, self.val_date)
        self.assertTrue(np.array_equal(processor.default_numeric_rate_tenors, self.tenors))
    
    def test_portfolio_builder_creation(self):
        """Test PortfolioBuilder creation and basic functionality."""
        builder = PortfolioBuilder()
        self.assertIsNotNone(builder)
        self.assertFalse(builder.uncalculated_instruments)


class TestDemoCalibrationRegression(unittest.TestCase):
    """Regression tests for demo_calibration.py functionality."""
    
    def test_instrument_definitions_generation(self):
        """Test that instrument definitions are generated correctly."""
        val_date = date(2025, 5, 18)
        num_instruments = 10
        
        definitions = generate_instrument_definitions(num_instruments, val_date)
        
        # Validate structure
        self.assertEqual(len(definitions), num_instruments)
        
        for definition in definitions:
            # Check required fields
            self.assertIn('instrument_id', definition)
            self.assertIn('product_type', definition)
            self.assertIn('pricing_preference', definition)
            self.assertIn('params', definition)
            
            # Check product types are valid
            valid_types = ['VanillaBond', 'CallableBond', 'EuropeanOption', 'ConvertibleBond']
            self.assertIn(definition['product_type'], valid_types)
            
            # Check pricing preferences are valid
            valid_prefs = ['TFF', 'FULL']
            self.assertIn(definition['pricing_preference'], valid_prefs)
    
    def test_calibration_demo_basic_execution(self):
        """Test that calibration demo runs without errors."""
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Run with minimal settings for speed
            run_calibration_demo(
                num_instruments_to_generate=5,
                num_workers=1,
                batch_size=2,
                random_seed=42
            )
            
            output = captured_output.getvalue()
            
            # Validate expected output patterns
            self.assertIn("TFF Calibration Demo", output)
            self.assertIn("Instrument Processing", output)
            self.assertIn("Created", output)
            
            # Check for no critical errors
            self.assertNotIn("ERROR:", output)
            self.assertNotIn("CRITICAL:", output)
            
        finally:
            sys.stdout = sys.__stdout__


class TestDemoCompareRegression(unittest.TestCase):
    """Regression tests for demo_compare.py functionality."""
    
    def test_comparison_demo_basic_execution(self):
        """Test that comparison demo runs without errors."""
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            # Run with minimal settings for speed
            run_comparison_demo(
                num_instruments_to_generate=3,
                num_var_scenarios=10,
                n_tff_domain_scenarios=20,
                n_tff_fitting_samples=10,
                num_tff_workers=1,
                tff_batch_size=2,
                random_seed=42
            )
            
            output = captured_output.getvalue()
            
            # Validate expected output patterns
            self.assertIn("Comparison Demo: Full vs. TFF Pricing", output)
            self.assertIn("Path 1: Full Valuation", output)
            self.assertIn("Path 2: TFF Valuation", output)
            self.assertIn("Comparison Summary", output)
            
            # Check for no critical errors
            self.assertNotIn("ERROR:", output)
            self.assertNotIn("CRITICAL:", output)
            
        finally:
            sys.stdout = sys.__stdout__


class TestWorkflowManagerRegression(unittest.TestCase):
    """Regression tests for workflow_manager.py core functionality."""
    
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
            base_s0_map={},
            base_vol_map={},
            random_seed=42
        )
    
    def test_portfolio_json_serialization(self):
        """Test JSON serialization of portfolio specifications."""
        # Create sample portfolio spec
        sample_spec = {
            "client_id": "TestClient",
            "instrument_id": "TEST_BOND_001",
            "num_holdings": 100,
            "pricing_engine_type": "tff",
            "product_static_object": {
                "product_type": "VanillaBond",
                "valuation_date": self.val_date.isoformat(),
                "maturity_date": (self.val_date + relativedelta(years=5)).isoformat(),
                "coupon_rate": 0.03,
                "face_value": 100.0,
                "currency": self.DEMO_CURRENCY
            }
        }
        
        # Test serialization
        try:
            json_str = json.dumps(sample_spec, default=portfolio_json_serializer)
            self.assertIsInstance(json_str, str)
            
            # Test deserialization
            deserialized = json.loads(json_str)
            self.assertEqual(deserialized["client_id"], "TestClient")
            self.assertEqual(deserialized["instrument_id"], "TEST_BOND_001")
            
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")
    
    def test_portfolio_analytics_creation(self):
        """Test PortfolioAnalytics creation and basic functionality."""
        # Create empty portfolios dict
        portfolios = {}
        
        # Generate scenarios
        scenarios, factor_names = self.scenario_gen.generate_scenarios(5)
        
        analytics = PortfolioAnalytics(
            client_portfolios=portfolios,
            global_market_scenarios=scenarios,
            global_factor_names=factor_names,
            numeric_rate_tenors=self.tenors,
            scenario_generator_for_base_values=self.scenario_gen
        )
        
        self.assertIsNotNone(analytics)
        self.assertEqual(analytics.numeric_rate_tenors.shape, self.tenors.shape)


class TestScenarioGeneratorRegression(unittest.TestCase):
    """Regression tests for scenario generation functionality."""
    
    def test_scenario_generator_creation(self):
        """Test scenario generator creation and basic functionality."""
        base_rates_map = {"USD_IR_1.00Y": 0.025, "USD_IR_5.00Y": 0.03}
        base_s0_map = {"USD_STOCK_S0": 100.0}
        base_vol_map = {"USD_STOCK_VOL": 0.25}
        
        generator = SimpleRandomScenarioGenerator(
            base_rates_map=base_rates_map,
            base_s0_map=base_s0_map,
            base_vol_map=base_vol_map,
            random_seed=42
        )
        
        self.assertIsNotNone(generator)
    
    def test_scenario_generation(self):
        """Test that scenarios are generated correctly."""
        base_rates_map = {"USD_IR_1.00Y": 0.025, "USD_IR_5.00Y": 0.03}
        base_s0_map = {"USD_STOCK_S0": 100.0}
        base_vol_map = {"USD_STOCK_VOL": 0.25}
        
        generator = SimpleRandomScenarioGenerator(
            base_rates_map=base_rates_map,
            base_s0_map=base_s0_map,
            base_vol_map=base_vol_map,
            random_seed=42
        )
        
        num_scenarios = 10
        scenarios, factor_names = generator.generate_scenarios(num_scenarios)
        
        # Validate output
        self.assertEqual(scenarios.shape[0], num_scenarios)
        self.assertEqual(scenarios.shape[1], len(factor_names))
        self.assertGreater(len(factor_names), 0)
        
        # Check that all expected factors are present
        expected_factors = list(base_rates_map.keys()) + list(base_s0_map.keys()) + list(base_vol_map.keys())
        for factor in expected_factors:
            self.assertIn(factor, factor_names)


class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests to ensure no significant performance degradation."""
    
    def test_instrument_processing_performance(self):
        """Test that instrument processing performance is within acceptable bounds."""
        val_date = date(2025, 5, 18)
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        
        # Create minimal scenario generator
        scenario_gen = SimpleRandomScenarioGenerator(
            base_rates_map={"USD_IR_1.00Y": 0.025},
            base_s0_map={},
            base_vol_map={},
            random_seed=42
        )
        
        # Create minimal instrument definitions
        instrument_definitions = [
            {
                "instrument_id": "TEST_BOND_001",
                "product_type": "VanillaBond",
                "pricing_preference": "TFF",
                "params": {
                    "valuation_date": val_date.isoformat(),
                    "maturity_date": (val_date + relativedelta(years=5)).isoformat(),
                    "coupon_rate": 0.03,
                    "face_value": 100.0,
                    "currency": "USD",
                    "index_stub": "IR",
                    "freq": 2
                },
                "tff_config": {"n_train": 8, "n_test": 2}
            }
        ]
        
        # Generate scenarios
        scenarios, factor_names = scenario_gen.generate_scenarios(10)
        
        # Measure processing time
        start_time = time.time()
        
        processor = InstrumentProcessor(
            scenario_generator=scenario_gen,
            global_valuation_date=val_date,
            default_numeric_rate_tenors=tenors,
            parallel_workers_tff=False,
            n_scenarios_for_tff_domain=10
        )
        
        model_registry = processor.process_instruments(
            instrument_definitions, scenarios, factor_names
        )
        
        processing_time = time.time() - start_time
        
        # Validate performance (should complete within 30 seconds for this simple case)
        self.assertLess(processing_time, 30.0, 
                       f"Instrument processing took {processing_time:.2f}s, expected < 30s")
        
        # Validate output
        self.assertIn("TEST_BOND_001", model_registry)
        self.assertFalse(model_registry["TEST_BOND_001"].get("error", False))


class TestErrorHandlingRegression(unittest.TestCase):
    """Regression tests for error handling functionality."""
    
    def test_invalid_instrument_definition(self):
        """Test that invalid instrument definitions are handled gracefully."""
        val_date = date(2025, 5, 18)
        tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
        
        scenario_gen = SimpleRandomScenarioGenerator(
            base_rates_map={"USD_IR_1.00Y": 0.025},
            base_s0_map={},
            base_vol_map={},
            random_seed=42
        )
        
        # Invalid instrument definition (missing required fields)
        invalid_definitions = [
            {
                "instrument_id": "INVALID_BOND",
                "product_type": "VanillaBond",
                # Missing params
            }
        ]
        
        scenarios, factor_names = scenario_gen.generate_scenarios(5)
        
        processor = InstrumentProcessor(
            scenario_generator=scenario_gen,
            global_valuation_date=val_date,
            default_numeric_rate_tenors=tenors,
            parallel_workers_tff=False,
            n_scenarios_for_tff_domain=5
        )
        
        # Should not raise an exception, but should handle the error gracefully
        model_registry = processor.process_instruments(
            invalid_definitions, scenarios, factor_names
        )
        
        # Check that the invalid instrument is marked as having an error
        self.assertIn("INVALID_BOND", model_registry)
        self.assertTrue(model_registry["INVALID_BOND"].get("error", False))
    
    def test_missing_instrument_in_portfolio(self):
        """Test that missing instruments in portfolio are handled gracefully."""
        builder = PortfolioBuilder()
        
        # Create portfolio specs with missing instrument
        portfolio_specs = [
            {
                "client_id": "TestClient",
                "instrument_id": "MISSING_INSTRUMENT",
                "num_holdings": 100,
                "pricing_engine_type": "tff",
                "product_static_object": {
                    "product_type": "VanillaBond",
                    "valuation_date": date(2025, 5, 18).isoformat(),
                    "maturity_date": date(2030, 5, 18).isoformat(),
                    "coupon_rate": 0.03,
                    "face_value": 100.0,
                    "currency": "USD"
                }
            }
        ]
        
        # Capture output to check for fallback warning
        captured_output = io.StringIO()
        sys.stdout = captured_output
        try:
            portfolios = builder.build_portfolios_from_specs(
                portfolio_specs, date(2025, 5, 18)
            )
        finally:
            sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        self.assertIn("Fallback to FULL", output)


if __name__ == '__main__':
    # Create a test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestDemoMain,
        TestDemoCalibration,
        TestDemoCompare,
        TestDemoRBFI,
        TestDemoMainRegression,
        TestDemoCalibrationRegression,
        TestDemoCompareRegression,
        TestWorkflowManagerRegression,
        TestScenarioGeneratorRegression,
        TestPerformanceRegression,
        TestErrorHandlingRegression
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")
    
    # Exit with appropriate code
    exit(0 if result.wasSuccessful() else 1) 