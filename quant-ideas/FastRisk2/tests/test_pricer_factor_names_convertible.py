import unittest
import numpy as np
from datetime import date

from product_definitions import ConvertibleBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer


class TestConvertibleFactorNamesPlumbing(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 1, 2)
        self.cb = ConvertibleBondStaticBase(
            valuation_date=self.val_date,
            issue_date=date(2024, 1, 2),
            maturity_date=date(2030, 1, 2),
            coupon_rate=0.02,
            conversion_ratio=1.0,
            face_value=100.0,
            freq=2,
            currency='USD',
            index_stub='GENERIC_IR',
            underlying_symbol='AAPL',
        )
        self.pillars = np.array([0.25, 0.50, 1.00], dtype=float)

    def test_required_factor_names_include_equity(self):
        pricer = QuantLibBondPricer(self.cb, method='discount')
        names = pricer.get_required_factor_names(rate_pillars=self.pillars)
        for key in ['USD_GENERIC_IR_0.25Y', 'USD_AAPL_S0', 'USD_AAPL_DIVYIELD', 'USD_AAPL_VOL', 'USD_AAPL_CS']:
            self.assertIn(key, names)

    def test_price_scenarios_auto_append_equity_cols(self):
        pricer = QuantLibBondPricer(self.cb, method='discount')
        # Order must be S0, DIVYIELD, VOL, CS to map correctly
        names = [f'USD_GENERIC_IR_{t:.2f}Y' for t in self.pillars] + ['USD_AAPL_S0', 'USD_AAPL_DIVYIELD', 'USD_AAPL_VOL']
        scenarios = np.array([
            [0.02, 0.021, 0.022, 100.0, 0.01, 0.25],
            [0.022, 0.023, 0.024, 101.0, 0.012, 0.30],
        ])
        prices = pricer.price_scenarios(scenarios, names, rate_pillars=self.pillars, credit_spread=0.002)
        self.assertEqual(prices.shape, (2,))


if __name__ == '__main__':
    unittest.main()


