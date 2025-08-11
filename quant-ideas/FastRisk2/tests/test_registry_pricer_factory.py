import unittest
from datetime import date

from product_definitions import EuropeanOptionStatic
from registry.pricer_factory import create_pricer


class TestPricerFactory(unittest.TestCase):
    def test_create_black_scholes_pricer(self):
        static = EuropeanOptionStatic(
            valuation_date=date(2025, 1, 2),
            expiry_date=date(2026, 1, 2),
            strike_price=100.0,
            option_type='call',
            currency='USD',
            underlying_symbol='AAPL'
        )
        pricer = create_pricer(static, {
            'pricer_module_name': 'black_scholes_pricer',
            'pricer_class_name': 'BlackScholesPricer',
            'pricer_params': {}
        })
        # price a simple scenario
        import numpy as np
        prices = pricer.price_scenarios(
            raw_market_scenarios=np.array([[100.0, 0.2]]),
            scenario_factor_names=['USD_AAPL_S0', 'USD_AAPL_VOL'],
            risk_free_rate=0.02,
            dividend_yield=0.0,
        )
        self.assertEqual(prices.shape, (1,))


if __name__ == '__main__':
    unittest.main()



