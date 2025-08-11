import unittest
import numpy as np
from datetime import date

from product_definitions import CallableBondStaticBase, ConvertibleBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer


class TestCallableBondG2(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 1, 2)
        self.callable = CallableBondStaticBase(
            valuation_date=self.val_date,
            maturity_date=date(2032, 1, 2),
            coupon_rate=0.035,
            face_value=100.0,
            freq=2,
            currency='USD',
            index_stub='GENERIC_IR',
            call_dates=[date(2028, 1, 2), date(2029, 1, 2), date(2030, 1, 2)],
            call_prices=[100.0, 100.0, 100.0],
        )
        self.pillars = np.array([0.25, 0.50, 1.00, 2.00, 5.00, 10.00], dtype=float)
        self.scenarios = np.vstack([
            np.array([0.02, 0.021, 0.022, 0.024, 0.026, 0.028]),
            np.array([0.019, 0.020, 0.021, 0.023, 0.025, 0.027])
        ])
        # Reasonable G2 params (a, sigma, b, eta, rho)
        self.g2_params = (0.1, 0.01, 0.3, 0.01, -0.75)

    def test_callable_g2_prices_run(self):
        pricer = QuantLibBondPricer(self.callable, method='g2', grid_steps=100)
        prices = pricer.price(self.pillars, self.scenarios, g2_params=self.g2_params)
        self.assertEqual(prices.shape, (self.scenarios.shape[0],))
        self.assertTrue(np.all(np.isfinite(prices)))


class TestConvertibleBond(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 1, 2)
        self.cb = ConvertibleBondStaticBase(
            valuation_date=self.val_date,
            issue_date=date(2024, 1, 2),
            maturity_date=date(2030, 1, 2),
            coupon_rate=0.02,
            conversion_ratio=1.5,
            face_value=100.0,
            freq=2,
            currency='USD',
            index_stub='GENERIC_IR',
            underlying_symbol='AAPL',
        )
        self.pillars = np.array([0.25, 0.50, 1.00, 2.00, 5.00, 10.00], dtype=float)
        self.rf_scenarios = np.vstack([
            np.array([0.02, 0.021, 0.022, 0.024, 0.026, 0.028]),
            np.array([0.019, 0.020, 0.021, 0.023, 0.025, 0.027])
        ])

    def test_convertible_prices_run(self):
        pricer = QuantLibBondPricer(self.cb, method='discount', convertible_engine_steps=64)
        # For CB, we pass s0, dividend yield, vol and a credit spread for engine
        prices = pricer.price(
            self.pillars,
            self.rf_scenarios,
            s0_val=100.0,
            dividend_yield=0.01,
            equity_volatility=0.30,
            credit_spread=0.005,
        )
        self.assertEqual(prices.shape, (self.rf_scenarios.shape[0],))
        self.assertTrue(np.all(np.isfinite(prices)))


if __name__ == '__main__':
    unittest.main()


