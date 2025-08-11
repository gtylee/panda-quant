import unittest
import numpy as np

from scenario_generator import SimpleRandomScenarioGenerator


class TestScenarioGenerator(unittest.TestCase):
    def test_generate_selected_factors(self):
        gen = SimpleRandomScenarioGenerator(
            base_rates_map={"USD_IR_1.00Y": 0.02},
            base_s0_map={"USD_AAPL_S0": 100.0},
            base_vol_map={"USD_AAPL_VOL": 0.25},
            random_seed=1,
        )
        factors = ["USD_IR_1.00Y", "USD_AAPL_S0", "USD_AAPL_VOL"]
        X, names = gen.generate_scenarios(5, target_factor_names=factors)
        self.assertEqual(names, factors)
        self.assertEqual(X.shape, (5, 3))

    def test_empty_when_no_config(self):
        gen = SimpleRandomScenarioGenerator(random_seed=1)
        X, names = gen.generate_scenarios(4)
        self.assertEqual(len(names), 0)
        self.assertEqual(X.shape, (4, 0))


if __name__ == '__main__':
    unittest.main()


