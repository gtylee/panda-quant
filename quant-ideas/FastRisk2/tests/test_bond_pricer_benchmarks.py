import unittest
import numpy as np
import QuantLib as ql
from datetime import date

from product_definitions import FloatingRateBondStaticBase, InflationLinkedBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer


class TestBondPricerBenchmarks(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 5, 18)
        ql.Settings.instance().evaluationDate = ql.Date(self.val_date.day, self.val_date.month, self.val_date.year)
        # Base rate curve pillars and rates
        self.pillars = np.array([0.25, 0.5, 1.0, 2.0, 3.0], dtype=float)
        self.flat = np.array([0.022, 0.022, 0.022, 0.022, 0.022], dtype=float)
        self.up = np.array([0.02, 0.021, 0.022, 0.024, 0.026], dtype=float)
        self.down = np.array([0.026, 0.024, 0.022, 0.021, 0.02], dtype=float)
        # Discount curve builder
        self.dc = ql.Actual365Fixed()
        self.cal = ql.TARGET()

    def build_curve(self, rates: np.ndarray) -> ql.YieldTermStructureHandle:
        based = ql.Settings.instance().evaluationDate
        dates = ql.DateVector(); dates.push_back(based)
        for t in self.pillars:
            dates.push_back(based + ql.Period(int(round(t*365)), ql.Days))
        curve = ql.ZeroCurve(dates, [rates[0]] + list(rates), self.dc, self.cal, ql.Linear(), ql.Continuous, ql.Annual)
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def test_frn_against_quantlib(self):
        maturities = [2, 5, 10]
        curves = [self.flat, self.up, self.down]
        for years in maturities:
            for rates in curves:
                with self.subTest(years=years, curve=rates.tolist()):
                    mat = date(self.val_date.year + years, self.val_date.month, self.val_date.day)
                    frn_static = FloatingRateBondStaticBase(
                        valuation_date=self.val_date, maturity_date=mat,
                        coupon_rate=0.0, face_value=100.0, freq=4,
                        index_tenor_months=3, spread=0.001
                    )
                    pricer = QuantLibBondPricer(frn_static, method='discount')
                    price_model = pricer.price(self.pillars, rates)[0]

                    # Build QL FRN with same schedule and index
                    handle = self.build_curve(rates)
                    tenor = ql.Period(3, ql.Months)
                    ibor = ql.IborIndex("GENERIC_IBOR", tenor, 0, ql.USDCurrency(), self.cal, ql.Following, False, frn_static.day_count_ql, handle)
                    frn = ql.FloatingRateBond(
                        frn_static.settlement_days,
                        frn_static.face_value,
                        frn_static.schedule,
                        ibor,
                        frn_static.day_count_ql,
                        ql.Following,
                        fixingDays=0,
                        spreads=[frn_static.spread]
                    )
                    eng = ql.DiscountingBondEngine(handle)
                    frn.setPricingEngine(eng)
                    price_ql = frn.NPV()
                    diff = abs(price_model - price_ql)
                    self.assertLess(diff, 0.05, msg=f"FRN diff too large: {diff} (model={price_model}, ql={price_ql})")

    def test_ilb_against_manual_piecewise(self):
        # Piecewise inflation curve
        infl_pillars = np.array([1.0, 3.0, 7.0], dtype=float)
        infl_rates = np.array([0.015, 0.02, 0.022], dtype=float)
        maturities = [2, 5, 10]
        rates = self.up
        handle = self.build_curve(rates)
        for years in maturities:
            with self.subTest(years=years):
                mat = date(self.val_date.year + years, self.val_date.month, self.val_date.day)
                ilb_static = InflationLinkedBondStaticBase(
                    valuation_date=self.val_date, maturity_date=mat,
                    coupon_rate=0.015, face_value=100.0, freq=2,
                    inflation_factor_name='USD_INFLATION', inflation_compounding='continuous', inflation_lag_months=2
                )
                pricer = QuantLibBondPricer(ilb_static, method='discount')
                price_model = pricer.price(
                    self.pillars, rates,
                    inflation_rate=0.02,
                    inflation_curve_pillars=infl_pillars,
                    inflation_curve_rates=infl_rates
                )[0]

                # Manual reconstruction matching pricer logic
                eval_d = ql.Settings.instance().evaluationDate
                dc = ilb_static.day_count_ql
                sched = ilb_static.schedule
                lag_years = ilb_static.inflation_lag_months/12.0
                def infl_factor(t_years: float) -> float:
                    t_eff = max(t_years, 0.0)
                    pillars = infl_pillars
                    rates_i = infl_rates
                    order = np.argsort(pillars)
                    pillars = pillars[order]; rates_arr = rates_i[order]
                    acc = 0.0; last_t = 0.0
                    for p, r in zip(pillars, rates_arr):
                        seg_end = min(t_eff, p)
                        if seg_end > last_t:
                            dt = seg_end - last_t
                            acc += r * dt
                            last_t = seg_end
                        if last_t >= t_eff:
                            break
                    if last_t < t_eff:
                        r_last = rates_arr[-1]
                        dt = t_eff - last_t
                        acc += r_last * dt
                    return float(np.exp(acc))

                pv_manual = 0.0
                ts = handle
                dates = list(sched)
                for i in range(1, len(dates)):
                    d2 = dates[i]; d1 = dates[i-1]
                    accrual = dc.yearFraction(d1, d2)
                    t = dc.yearFraction(eval_d, d2) - lag_years
                    scale = infl_factor(t)
                    cpn = ilb_static.coupon_rate * accrual * ilb_static.face_value * scale
                    pv_manual += cpn * ts.discount(d2)
                t_end = dc.yearFraction(eval_d, dates[-1]) - lag_years
                scale_end = infl_factor(t_end)
                pv_manual += ilb_static.face_value * scale_end * ts.discount(dates[-1])

                diff = abs(price_model - pv_manual)
                self.assertLess(diff, 0.02, msg=f"ILB diff too large: {diff} (model={price_model}, manual={pv_manual})")


if __name__ == '__main__':
    unittest.main()


