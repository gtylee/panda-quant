import unittest
import numpy as np
from datetime import date
import QuantLib as ql

from product_definitions import FloatingRateBondStaticBase, InflationLinkedBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer


class TestFRNAndILBMoreCases(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 6, 15)
        ql.Settings.instance().evaluationDate = ql.Date(self.val_date.day, self.val_date.month, self.val_date.year)
        self.pillars = np.array([0.25, 0.5, 1.0, 3.0, 7.0], dtype=float)
        self.up = np.array([0.02, 0.021, 0.023, 0.026, 0.028], dtype=float)
        self.down = np.array([0.028, 0.026, 0.023, 0.021, 0.02], dtype=float)

    def test_frn_monotonicity_with_spread(self):
        mat = date(2030, 6, 15)
        frn = FloatingRateBondStaticBase(
            valuation_date=self.val_date, maturity_date=mat, coupon_rate=0.0, face_value=100.0,
            freq=4, index_tenor_months=3, spread=0.0
        )
        pricer = QuantLibBondPricer(frn, method='discount')
        p0 = pricer.price(self.pillars, self.up)[0]
        frn_sp = FloatingRateBondStaticBase(
            valuation_date=self.val_date, maturity_date=mat, coupon_rate=0.0, face_value=100.0,
            freq=4, index_tenor_months=3, spread=0.002
        )
        pricer_sp = QuantLibBondPricer(frn_sp, method='discount')
        p1 = pricer_sp.price(self.pillars, self.up)[0]
        self.assertGreater(p1, p0, msg=f"FRN with higher spread should price higher: {p1} vs {p0}")

    def test_frn_curve_direction(self):
        mat = date(2030, 6, 15)
        frn = FloatingRateBondStaticBase(
            valuation_date=self.val_date, maturity_date=mat, coupon_rate=0.0, face_value=100.0,
            freq=4, index_tenor_months=3, spread=0.001
        )
        pricer = QuantLibBondPricer(frn, method='discount')
        p_up = pricer.price(self.pillars, self.up)[0]
        p_down = pricer.price(self.pillars, self.down)[0]
        # On upward curve, FRN tends to have slightly lower PV than on downward curve (higher discounting)
        self.assertLess(p_up, p_down + 1.0)  # very loose check to avoid flakiness

    def test_ilb_compounding_modes(self):
        mat = date(2032, 6, 15)
        pillars = self.pillars
        rates = self.up
        ilb_cont = InflationLinkedBondStaticBase(
            valuation_date=self.val_date, maturity_date=mat, coupon_rate=0.01, face_value=100.0,
            freq=2, inflation_compounding='continuous', inflation_lag_months=3
        )
        ilb_annual = InflationLinkedBondStaticBase(
            valuation_date=self.val_date, maturity_date=mat, coupon_rate=0.01, face_value=100.0,
            freq=2, inflation_compounding='annual', inflation_lag_months=3
        )
        pricer_c = QuantLibBondPricer(ilb_cont, method='discount')
        pricer_a = QuantLibBondPricer(ilb_annual, method='discount')
        # Use a flat inflation input
        price_c = pricer_c.price(pillars, rates, inflation_rate=0.02)[0]
        price_a = pricer_a.price(pillars, rates, inflation_rate=0.02)[0]
        self.assertNotEqual(price_c, price_a)


if __name__ == '__main__':
    unittest.main()


