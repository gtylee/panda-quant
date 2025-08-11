import unittest
import numpy as np

from workflow import get_scenario_slice_static


class TestScenarioSliceUtils(unittest.TestCase):
    def test_slice_by_names(self):
        all_names = ["A", "B", "C", "D"]
        X = np.arange(20, dtype=float).reshape(5, 4)
        sub = get_scenario_slice_static(X, all_names, ["C", "A"])
        self.assertEqual(sub.shape, (5, 2))
        self.assertTrue(np.allclose(sub[:, 0], X[:, 2]))
        self.assertTrue(np.allclose(sub[:, 1], X[:, 0]))

    def test_missing_factor_returns_empty_cols(self):
        all_names = ["A", "B"]
        X = np.arange(10, dtype=float).reshape(5, 2)
        sub = get_scenario_slice_static(X, all_names, ["Z"])  # none match
        self.assertEqual(sub.shape, (5, 0))


if __name__ == '__main__':
    unittest.main()


