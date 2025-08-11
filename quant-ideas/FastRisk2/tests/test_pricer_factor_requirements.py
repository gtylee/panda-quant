import unittest
import numpy as np
from datetime import date

from product_definitions import QuantLibBondStaticBase, ConvertibleBondStaticBase, EuropeanOptionStatic
from quantlib_bond_pricer import QuantLibBondPricer
from black_scholes_pricer import BlackScholesPricer


class TestFactorRequirements(unittest.TestCase):
    def test_bond_required_factor_names_strict(self):
        bond = QuantLibBondStaticBase(
            valuation_date=date(2025, 1, 2),
            maturity_date=date(2030, 1, 2),
            coupon_rate=0.03,
            face_value=100.0,
            freq=2,
            currency='USD',
            index_stub='GENERIC_IR',
        )
        pricer = QuantLibBondPricer(bond)
        pillars = np.array([0.50, 1.00])
        # Intentionally wrong names
        names = ["USD_IR_0.50Y", "WRONG"]
        X = np.array([[0.02, 0.03]])
        with self.assertRaises(ValueError):
            pricer.price_scenarios(X, names, rate_pillars=pillars, strict_factor_names=True)

        # Non-strict falls back to first N columns
        pricer.price_scenarios(X, names, rate_pillars=pillars, strict_factor_names=False)

    def test_convertible_optional_factors(self):
        cb = ConvertibleBondStaticBase(
            valuation_date=date(2025, 1, 2),
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
        pricer = QuantLibBondPricer(cb)
        pillars = np.array([1.00])
        names = ["USD_GENERIC_IR_1.00Y", "USD_AAPL_S0", "USD_AAPL_DIVYIELD", "USD_AAPL_VOL", "USD_AAPL_CS"]
        X = np.array([[0.02, 100.0, 0.01, 0.3, 0.005]])
        # price_scenarios should accept with scenario-driven optional inputs; price() will read them
        pricer.price_scenarios(X, names, rate_pillars=pillars)

    def test_option_required_factors(self):
        opt = EuropeanOptionStatic(
            valuation_date=date(2025, 1, 2),
            expiry_date=date(2026, 1, 2),
            strike_price=100.0,
            option_type='call',
            currency='USD',
            underlying_symbol='AAPL'
        )
        pricer = BlackScholesPricer(opt)
        req = pricer.get_required_factor_names()
        self.assertIn("USD_AAPL_S0", req)
        self.assertIn("USD_AAPL_VOL", req)


if __name__ == '__main__':
    unittest.main()


