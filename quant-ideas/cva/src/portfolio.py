from __future__ import annotations
import QuantLib as ql
from dataclasses import dataclass
from typing import Callable, List

ZCBFunc = Callable[[float], float]  # expects T -> P(t,T) with a captured t


@dataclass
class PayerParSwap:
    notional: float
    maturity_years: int
    fixed_frequency: ql.Frequency = ql.Annual
    float_frequency: ql.Frequency = ql.Semiannual
    fixed_daycount: ql.DayCounter = ql.Thirty360(ql.Thirty360.BondBasis)
    float_daycount: ql.DayCounter = ql.Actual365Fixed()
    fixed_rate: float | None = None

    fixed_payment_times: List[float] | None = None
    fixed_accruals: List[float] | None = None
    T_end: float | None = None

    def build_from_curve(self, ts: ql.YieldTermStructureHandle) -> None:
        ref = ts.referenceDate()
        dc_time = ts.dayCounter()
        cal = ql.TARGET()
        start_date = ref
        end_date = ref + ql.Period(self.maturity_years, ql.Years)
        fixed_schedule = ql.Schedule(
            start_date, end_date, ql.Period(self.fixed_frequency), cal,
            ql.Following, ql.Following, ql.DateGeneration.Forward, False
        )
        self.fixed_payment_times = []
        self.fixed_accruals = []
        for i in range(1, len(fixed_schedule)):
            pay_date = fixed_schedule[i]
            prev_date = fixed_schedule[i - 1]
            t_pay = dc_time.yearFraction(ref, pay_date)
            self.fixed_payment_times.append(t_pay)
            accr = self.fixed_daycount.yearFraction(prev_date, pay_date)
            self.fixed_accruals.append(accr)
        self.T_end = self.fixed_payment_times[-1]
        # Par fixed rate using time-based discounts
        A0 = sum(alpha * ts.discount(Tj) for alpha, Tj in zip(self.fixed_accruals, self.fixed_payment_times))
        P0T = ts.discount(self.T_end)
        if self.fixed_rate is None:
            self.fixed_rate = (1.0 - P0T) / max(1e-12, A0)

    def pv_at(self, t: float, zcb_at_t: ZCBFunc) -> float:
        if t >= (self.T_end or 0.0):
            return 0.0
        fixed_pv = 0.0
        A_t = 0.0
        for tau, Tj in zip(self.fixed_accruals or [], self.fixed_payment_times or []):
            if Tj > t:
                P_t_Tj = zcb_at_t(Tj)
                A_t += tau * P_t_Tj
                fixed_pv += (self.fixed_rate or 0.0) * tau * P_t_Tj
        if A_t <= 0.0:
            return 0.0
        P_t_Tend = zcb_at_t(self.T_end or t)
        fwd_par = (1.0 - P_t_Tend) / A_t
        float_pv = fwd_par * A_t
        return self.notional * (float_pv - fixed_pv)


@dataclass
class ParBond:
    notional: float
    maturity_years: int
    coupon_frequency: ql.Frequency = ql.Annual
    daycount: ql.DayCounter = ql.Thirty360(ql.Thirty360.BondBasis)
    coupon_rate: float | None = None

    payment_times: List[float] | None = None
    accruals: List[float] | None = None
    T_end: float | None = None

    def build_from_curve(self, ts: ql.YieldTermStructureHandle) -> None:
        ref = ts.referenceDate()
        dc_time = ts.dayCounter()
        cal = ql.TARGET()
        start_date = ref
        end_date = ref + ql.Period(self.maturity_years, ql.Years)
        sched = ql.Schedule(
            start_date, end_date, ql.Period(self.coupon_frequency), cal,
            ql.Following, ql.Following, ql.DateGeneration.Forward, False
        )
        self.payment_times = []
        self.accruals = []
        for i in range(1, len(sched)):
            pay_date = sched[i]
            prev = sched[i - 1]
            t_pay = dc_time.yearFraction(ref, pay_date)
            self.payment_times.append(t_pay)
            accr = self.daycount.yearFraction(prev, pay_date)
            self.accruals.append(accr)
        self.T_end = self.payment_times[-1]
        A0 = sum(alpha * ts.discount(Tj) for alpha, Tj in zip(self.accruals, self.payment_times))
        P0T = ts.discount(self.T_end)
        if self.coupon_rate is None:
            self.coupon_rate = (1.0 - P0T) / max(1e-12, A0)

    def pv_at(self, t: float, zcb_at_t: ZCBFunc) -> float:
        if t >= (self.T_end or 0.0):
            return 0.0
        pv = 0.0
        for tau, Tj in zip(self.accruals or [], self.payment_times or []):
            if Tj > t:
                pv += (self.coupon_rate or 0.0) * tau * zcb_at_t(Tj)
        pv += 1.0 * zcb_at_t(self.T_end or t)
        return self.notional * pv


@dataclass
class Portfolio:
    instruments: List[object]

    def pv_at(self, t: float, zcb_at_t: ZCBFunc) -> float:
        return sum(inst.pv_at(t, zcb_at_t) for inst in self.instruments)


def build_linear_portfolio(ts: ql.YieldTermStructureHandle, include_bonds: bool = False) -> Portfolio:
    n = 10_000_000.0
    instruments: List[object] = []
    for m in [5, 10, 20]:
        s = PayerParSwap(notional=n, maturity_years=m)
        s.build_from_curve(ts)
        instruments.append(s)
    if include_bonds:
        for m in [2, 5, 10]:
            b = ParBond(notional=n, maturity_years=m)
            b.build_from_curve(ts)
            instruments.append(b)
    return Portfolio(instruments=instruments)
