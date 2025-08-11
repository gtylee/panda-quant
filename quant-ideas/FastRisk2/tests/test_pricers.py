import unittest
import numpy as np
from datetime import date

from product_definitions import EuropeanOptionStatic
from black_scholes_pricer import BlackScholesPricer


class TestBlackScholesPricer(unittest.TestCase):
    def setUp(self):
        self.opt = EuropeanOptionStatic(
            valuation_date=date(2025, 1, 2),
            expiry_date=date(2026, 1, 2),
            strike_price=100.0,
            option_type='call',
            currency='USD',
            underlying_symbol='AAPL'
        )
        self.pricer = BlackScholesPricer(self.opt)

    def test_scalar_price(self):
        price = self.pricer.price(
            stock_price=100.0, volatility=0.20, risk_free_rate=0.02, dividend_yield=0.0
        )
        self.assertIsInstance(price, float)
        self.assertGreaterEqual(price, 0.0)

    def test_vectorized_price(self):
        S = np.array([90.0, 100.0, 110.0])
        V = np.array([0.15, 0.20, 0.25])
        prices = self.pricer.price(stock_price=S, volatility=V, risk_free_rate=0.02, dividend_yield=0.0)
        self.assertEqual(prices.shape, S.shape)
        self.assertTrue(np.all(prices >= 0.0))

    def test_price_scenarios(self):
        factor_names = ['USD_AAPL_S0', 'USD_AAPL_VOL']
        scenarios = np.array([
            [100.0, 0.2],
            [105.0, 0.22],
        ])
        prices = self.pricer.price_scenarios(scenarios, factor_names, risk_free_rate=0.02, dividend_yield=0.0)
        self.assertEqual(prices.shape[0], scenarios.shape[0])


if __name__ == '__main__':
    unittest.main()


