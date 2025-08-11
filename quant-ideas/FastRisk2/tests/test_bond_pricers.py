import unittest
from datetime import date
import numpy as np
import QuantLib as ql

from product_definitions import QuantLibBondStaticBase, FloatingRateBondStaticBase, InflationLinkedBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer


class TestQuantLibEnumMappings(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 5, 18)
        self.maturity = date(2030, 5, 18)
        self.tenors = np.array([0.5, 1.0, 2.0, 3.0, 5.0], dtype=float)
        self.rf = np.array([0.02, 0.022, 0.025, 0.028, 0.03], dtype=float)

    def _price_ok(self, static_kwargs):
        static = QuantLibBondStaticBase(
            valuation_date=self.val_date,
            maturity_date=self.maturity,
            coupon_rate=0.03,
            face_value=100.0,
            **static_kwargs
        )
        pricer = QuantLibBondPricer(static, method='discount')
        price = pricer.price(self.tenors, self.rf)
        self.assertTrue(np.isfinite(price[0]))

    def test_calendars_supported(self):
        cases = [
            {'calendar': 'TARGET'},
            {'calendar': 'us_federalreserve', 'currency': 'USD'},
            {'calendar': 'UnitedStates/NYSE', 'currency': 'USD'},
            {'calendar': 'Null'},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self._price_ok(kwargs)

    def test_day_count_supported(self):
        cases = [
            {'day_count': 'Actual/Actual (ISDA)'},
            {'day_count': 'actual365fixed'},
            {'day_count': 'actual360'},
            {'day_count': 'thirty360'},
            {'day_count': 'Thirty/360 European'},
            {'day_count': 'Thirty/360 Italian'},
            {'day_count': 'Business/252'},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self._price_ok(kwargs)

    def test_business_conventions_supported(self):
        cases = [
            {'business_convention': 'following'},
            {'business_convention': 'modifiedfollowing'},
            {'business_convention': 'preceding'},
            {'business_convention': 'modifiedpreceding'},
            {'business_convention': 'unadjusted'},
            {'business_convention': 'halfmonthmodifiedfollowing'},
            {'business_convention': 'nearest'},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                self._price_ok(kwargs)


class TestNewBondTypes(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 5, 18)
        self.maturity = date(2030, 5, 18)
        self.tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0], dtype=float)
        self.rf = np.array([0.02, 0.021, 0.022, 0.024, 0.026], dtype=float)

    def test_floating_rate_bond_pricing(self):
        frn_static = FloatingRateBondStaticBase(
            valuation_date=self.val_date,
            maturity_date=self.maturity,
            coupon_rate=0.0,
            face_value=100.0,
            freq=4,
            index_tenor_months=3,
            spread=0.001,
        )
        pricer = QuantLibBondPricer(frn_static, method='discount')
        price = pricer.price(self.tenors, self.rf)
        self.assertTrue(np.isfinite(price[0]))

    def test_inflation_linked_bond_pricing(self):
        ilb_static = InflationLinkedBondStaticBase(
            valuation_date=self.val_date,
            maturity_date=self.maturity,
            coupon_rate=0.02,
            face_value=100.0,
            freq=2,
            inflation_factor_name='USD_INFLATION',
            inflation_compounding='continuous',
            inflation_lag_months=3
        )
        pricer = QuantLibBondPricer(ilb_static, method='discount')
        price = pricer.price(
            self.tenors, self.rf,
            credit_spread_pillar_times=None,
            inflation_rate=0.02
        )
        self.assertTrue(np.isfinite(price[0]))

    def test_inflation_curve_piecewise(self):
        # Vary maturity and inflation curve shapes
        maturities_years = [2, 5, 10]
        for years in maturities_years:
            with self.subTest(years=years):
                mat = date(self.val_date.year + years, self.val_date.month, self.val_date.day)
                ilb_static = InflationLinkedBondStaticBase(
                    valuation_date=self.val_date,
                    maturity_date=mat,
                    coupon_rate=0.015,
                    face_value=100.0,
                    freq=2,
                    inflation_factor_name='USD_INFLATION',
                    inflation_compounding='continuous',
                    inflation_lag_months=2
                )
                pricer = QuantLibBondPricer(ilb_static, method='discount')
                # Define a simple piecewise inflation curve (increasing then flat)
                infl_pillars = np.array([1.0, 3.0, 7.0], dtype=float)
                infl_rates = np.array([0.015, 0.02, 0.022], dtype=float)
                # perturb market rates for variety
                rf = self.rf * (1.0 + 0.1*(years/10))
                price = pricer.price(
                    self.tenors, rf,
                    inflation_rate=0.02, # base used when beyond curve
                    inflation_curve_pillars=infl_pillars,
                    inflation_curve_rates=infl_rates
                )[0]
                self.assertTrue(np.isfinite(price))


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
from datetime import date

from product_definitions import QuantLibBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer
from fast_bond_pricer import FastBondPricer


class TestVanillaBondPricers(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 1, 2)
        self.bond = QuantLibBondStaticBase(
            valuation_date=self.val_date,
            maturity_date=date(2030, 1, 2),
            coupon_rate=0.03,
            face_value=100.0,
            freq=2,
            currency='USD',
            index_stub='GENERIC_IR',
        )
        self.pillars = np.array([0.50, 1.00, 2.00, 5.00, 10.00], dtype=float)
        # Build three scenarios with mild slope/twist
        base = np.array([0.02, 0.022, 0.024, 0.026, 0.028])
        self.scenarios = np.vstack([
            base,
            base + 0.001,
            base + np.array([0.000, 0.0005, 0.001, -0.0005, -0.001])
        ])

    def test_fast_vs_quantlib_price(self):
        ql_pricer = QuantLibBondPricer(self.bond, method='discount')
        fast_pricer = FastBondPricer(self.bond)

        ql_prices = ql_pricer.price(self.pillars, self.scenarios)
        fast_prices = fast_pricer.price(self.pillars, self.scenarios)

        self.assertEqual(ql_prices.shape, fast_prices.shape)
        # They should be within a few cents given same curve and cashflows
        self.assertTrue(np.allclose(ql_prices, fast_prices, atol=0.05))

    def test_price_scenarios_factor_mapping(self):
        factor_names = [f"USD_GENERIC_IR_{t:.2f}Y" for t in self.pillars]
        raw = self.scenarios.copy()
        ql_pricer = QuantLibBondPricer(self.bond, method='discount')

        direct = ql_pricer.price(self.pillars, raw)
        mapped = ql_pricer.price_scenarios(raw, factor_names, rate_pillars=self.pillars)

        self.assertTrue(np.allclose(direct, mapped, atol=1e-8))


if __name__ == '__main__':
    unittest.main()


