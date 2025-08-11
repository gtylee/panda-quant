import unittest
import numpy as np
from datetime import date

from workflow import Portfolio
from product_definitions import EuropeanOptionStatic
from black_scholes_pricer import BlackScholesPricer


class TestPortfolioPricingPaths(unittest.TestCase):
    def setUp(self):
        self.factor_names = ['USD_AAPL_S0', 'USD_AAPL_VOL']
        self.scenarios = np.array([
            [100.0, 0.20],
            [101.0, 0.22],
            [ 98.0, 0.18],
        ])

    def test_full_pricing_option(self):
        opt = EuropeanOptionStatic(
            valuation_date=date(2025, 1, 2),
            expiry_date=date(2026, 1, 2),
            strike_price=100.0,
            option_type='call',
            currency='USD',
            underlying_symbol='AAPL'
        )
        pricer = BlackScholesPricer(opt)

        p = Portfolio()
        p.add_position(
            instrument_id='OPT1',
            product_static=opt,
            num_holdings=10,
            pricing_engine_type='full',
            full_pricer_instance=pricer,
            full_pricer_kwargs={'risk_free_rate': 0.02, 'dividend_yield': 0.0}
        )

        values = p.price_portfolio(self.scenarios, self.factor_names)
        self.assertEqual(values.shape, (self.scenarios.shape[0],))
        self.assertTrue(np.all(np.isfinite(values)))


if __name__ == '__main__':
    unittest.main()


