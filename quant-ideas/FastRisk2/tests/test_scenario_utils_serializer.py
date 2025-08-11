import unittest
import numpy as np
from datetime import date, datetime
import QuantLib as ql

from workflow_manager_refactored import get_scenario_slice_static, portfolio_json_serializer


class TestScenarioUtilsAndSerializer(unittest.TestCase):
    def test_scenario_slice_empty(self):
        X = np.arange(10).reshape(5, 2)
        names = ['A', 'B']
        sub = get_scenario_slice_static(X, names, [])
        self.assertTrue((sub == X).all())

    def test_serializer_quantlib_and_numpy(self):
        d = date(2025, 1, 2)
        qld = ql.Date(d.day, d.month, d.year)
        arr = np.array([1.0, 2.0], dtype=np.float64)
        payload = {
            'date': d,
            'qld': qld,
            'arr': arr,
            'nested': {'x': np.float32(1.5)},
        }
        # Should not raise
        import json
        s = json.dumps(payload, default=portfolio_json_serializer)
        self.assertIn('"date"', s)


if __name__ == '__main__':
    unittest.main()



