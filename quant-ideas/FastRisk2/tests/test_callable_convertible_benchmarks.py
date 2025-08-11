import unittest
from datetime import date
import numpy as np
import QuantLib as ql

from product_definitions import CallableBondStaticBase, ConvertibleBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer


class TestCallableConvertibleBenchmarks(unittest.TestCase):
    def setUp(self):
        self.val_date = date(2025, 5, 18)
        ql.Settings.instance().evaluationDate = ql.Date(self.val_date.day, self.val_date.month, self.val_date.year)
        self.dc = ql.Actual365Fixed(); self.cal = ql.TARGET()
        self.pillars = np.array([0.25, 0.5, 1.0, 2.0, 3.0], dtype=float)
        self.up = np.array([0.02, 0.021, 0.022, 0.024, 0.026], dtype=float)

    def build_curve(self, rates: np.ndarray) -> ql.YieldTermStructureHandle:
        base = ql.Settings.instance().evaluationDate
        dates = ql.DateVector(); dates.push_back(base)
        for t in self.pillars:
            dates.push_back(base + ql.Period(int(round(float(t)*365)), ql.Days))
        curve = ql.ZeroCurve(dates, [float(rates[0])] + list(map(float, rates)), self.dc, self.cal, ql.Linear(), ql.Continuous, ql.Annual)
        curve.enableExtrapolation()
        return ql.YieldTermStructureHandle(curve)

    def test_callable_call_to_worst_benchmark(self):
        mat = date(2030, 5, 18)
        call_dates = [date(2027, 5, 18).isoformat(), date(2028, 5, 18).isoformat(), date(2029, 5, 18).isoformat()]
        call_prices = [102.0, 101.0, 100.5]
        callable_static = CallableBondStaticBase(
            valuation_date=self.val_date, maturity_date=mat, coupon_rate=0.035,
            face_value=100.0, freq=2,
            call_dates=call_dates, call_prices=call_prices,
            calendar='target', day_count='Actual/Actual (ISDA)'
        )
        pricer = QuantLibBondPricer(callable_static, method='g2', grid_steps=32)
        g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
        model_price = pricer.price(self.pillars, self.up, g2_params=g2_params)[0]

        # Call-to-worst approximation: PV coupons to each call date + call redemption, take min across calls and final maturity
        handle = self.build_curve(self.up)
        sched = callable_static.schedule
        dc = callable_static.day_count_ql
        eval_d = ql.Settings.instance().evaluationDate
        dates = list(sched)
        # Helper to PV cashflows until stop_date with redemption amount red
        def pv_to(date_stop: ql.Date, red: float) -> float:
            pv = 0.0
            for i in range(1, len(dates)):
                d1 = dates[i-1]; d2 = dates[i]
                if d2 > date_stop:
                    break
                accrual = dc.yearFraction(d1, d2)
                cpn = callable_static.coupon_rate * accrual * callable_static.face_value
                pv += cpn * handle.discount(d2)
                if d2 == date_stop:
                    pv += red * handle.discount(d2)
            return pv
        # Build list of PVs at each call date and final maturity
        ql_call_dates = [ql.Date(d.day, d.month, d.year) for d in [date.fromisoformat(cd) for cd in call_dates]]
        pv_list = []
        for d, cp in zip(ql_call_dates, call_prices):
            pv_list.append(pv_to(d, cp))
        pv_list.append(pv_to(dates[-1], callable_static.face_value))
        ctw_price = min(pv_list)

        diff = abs(model_price - ctw_price)
        # Allow a moderate tolerance as G2 includes rate vol; CTW is static PV rule
        self.assertLess(diff, 2.0, msg=f"Callable CTW benchmark diff {diff:.4f} too large (model={model_price:.4f}, ctw={ctw_price:.4f})")

    def test_convertible_decomposition_benchmark(self):
        mat = date(2030, 5, 18)
        conv_static = ConvertibleBondStaticBase(
            valuation_date=self.val_date, issue_date=date(2024, 11, 18), maturity_date=mat,
            coupon_rate=0.02, conversion_ratio=1.0, face_value=100.0, freq=2,
            calendar='target', day_count='Actual/Actual (ISDA)'
        )
        pricer = QuantLibBondPricer(conv_static, method='discount', grid_steps=128)
        s0 = 100.0; div = 0.01; vol = 0.25; cs = 0.002
        model_price = pricer.price(self.pillars, self.up, s0_val=s0, dividend_yield=div, equity_volatility=vol, credit_spread=cs)[0]

        # Decomposition: PV of straight bond (with credit spread) + conversion option (European) via Black-Scholes
        # Build credit-adjusted curve by parallel shift
        rf_plus_cs = self.up + cs
        handle_cs = self.build_curve(rf_plus_cs)
        # Straight bond PV
        sched = conv_static.schedule
        dc = conv_static.day_count_ql
        dates = list(sched)
        pv_bond = 0.0
        for i in range(1, len(dates)):
            d1 = dates[i-1]; d2 = dates[i]
            accrual = dc.yearFraction(d1, d2)
            cpn = conv_static.coupon_rate * accrual * conv_static.face_value
            pv_bond += cpn * handle_cs.discount(d2)
        pv_bond += conv_static.face_value * handle_cs.discount(dates[-1])

        # Conversion option: European call on equity with strike = face/conversion_ratio
        mat_t = dc.yearFraction(ql.Settings.instance().evaluationDate, dates[-1])
        r = float(np.mean(self.up))
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(s0)),
            ql.YieldTermStructureHandle(ql.FlatForward(ql.Settings.instance().evaluationDate, div, dc)),
            ql.YieldTermStructureHandle(ql.FlatForward(ql.Settings.instance().evaluationDate, r, dc)),
            ql.BlackVolTermStructureHandle(ql.BlackConstantVol(ql.Settings.instance().evaluationDate, self.cal, vol, dc))
        )
        strike = conv_static.face_value / conv_static.conversion_ratio
        payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
        exercise = ql.EuropeanExercise(dates[-1])
        option = ql.VanillaOption(payoff, exercise)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        pv_option = conv_static.conversion_ratio * option.NPV()

        decomposed = pv_bond + pv_option
        diff = abs(model_price - decomposed)
        # Allow loose tolerance since convertible pricing is path-dependent in general; this is a coarse decomposition
        self.assertLess(diff, 5.0, msg=f"Convertible decomposition diff {diff:.4f} too large (model={model_price:.4f}, decomp={decomposed:.4f})")


if __name__ == '__main__':
    unittest.main()


