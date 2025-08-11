import unittest
import numpy as np

from scenario_generator import SimpleRandomScenarioGenerator


class TestScenarioGeneratorExtended(unittest.TestCase):
    def setUp(self):
        self.gen = SimpleRandomScenarioGenerator(
            base_rates_map={
                'USD_GENERIC_IR_0.25Y': 0.02,
                'USD_GENERIC_IR_0.50Y': 0.021,
            },
            rate_factor_shock_std_dev_map={
                'USD_GENERIC_IR_0.25Y': 0.001,
                'USD_GENERIC_IR_0.50Y': 0.0015,
            },
            base_s0_map={
                'USD_AAPL_S0': 100.0,
                'USD_AAPL_DIVYIELD': 0.01,
                'USD_AAPL_CS': 0.002,
            },
            s0_shock_config_map={
                'USD_AAPL_S0': ('normal_relative', 0.02),
                'USD_AAPL_DIVYIELD': ('normal_absolute', 0.002),
                'USD_AAPL_CS': ('normal_absolute', 0.0005),
            },
            base_vol_map={'USD_AAPL_VOL': 0.25},
            vol_shock_config_map={'USD_AAPL_VOL': ('normal_relative', 0.10)},
            base_inflation_map={'USD_INFLATION': 0.02},
            inflation_shock_std_dev_map={'USD_INFLATION': 0.0005},
            random_seed=42,
        )

    def test_generate_all_configured_factors(self):
        scenarios, names = self.gen.generate_scenarios(10)
        self.assertEqual(scenarios.shape[0], 10)
        # At least all configured factors should be present
        for key in ['USD_GENERIC_IR_0.25Y', 'USD_AAPL_S0', 'USD_AAPL_VOL', 'USD_INFLATION']:
            self.assertIn(key, names)
        # First row equals base for rate and s0/vol since shocks[0]=0
        idx_ir = names.index('USD_GENERIC_IR_0.25Y')
        idx_s0 = names.index('USD_AAPL_S0')
        idx_vol = names.index('USD_AAPL_VOL')
        self.assertAlmostEqual(scenarios[0, idx_ir], 0.02, places=10)
        self.assertAlmostEqual(scenarios[0, idx_s0], 100.0, places=10)
        self.assertAlmostEqual(scenarios[0, idx_vol], 0.25, places=10)

    def test_non_negative_constraints(self):
        scenarios, names = self.gen.generate_scenarios(100)
        idx_cs = names.index('USD_AAPL_CS')
        idx_vol = names.index('USD_AAPL_VOL')
        self.assertTrue(np.all(scenarios[:, idx_cs] >= 0.0))
        self.assertTrue(np.all(scenarios[:, idx_vol] > 0.0))


if __name__ == '__main__':
    unittest.main()


