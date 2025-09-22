import QuantLib as ql
from typing import List, Tuple


def build_curve(flat_rate: float = 0.02,
                day_count: ql.DayCounter = ql.Actual365Fixed(),
                calendar: ql.Calendar = ql.TARGET(),
                settlement_days: int = 2) -> ql.YieldTermStructureHandle:
    """Build a flat discount curve and return a handle."""
    today = calendar.adjust(ql.Date.todaysDate())
    ql.Settings.instance().evaluationDate = today

    curve = ql.FlatForward(today, flat_rate, day_count)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def _parse_leg(token: str) -> ql.Period:
    unit = token[-1].upper()
    value = int(token[:-1])
    if unit == 'D':
        return ql.Period(value, ql.Days)
    if unit == 'W':
        return ql.Period(value, ql.Weeks)
    if unit == 'M':
        return ql.Period(value, ql.Months)
    if unit == 'Y':
        return ql.Period(value, ql.Years)
    raise ValueError(f"Unrecognized period token: {token}")


def calendar_advance(d: ql.Date, period: ql.Period, cal: ql.Calendar | None = None) -> ql.Date:
    if cal is None:
        cal = ql.TARGET()
    return cal.advance(d, period, ql.Following)


def make_time_grid(spec: str = '1M_10Y_3M_30Y',
                   reference_date: ql.Date | None = None,
                   day_count: ql.DayCounter = ql.Actual365Fixed()) -> List[float]:
    """Build a piecewise step time grid from a compact spec.

    Example: '1M_10Y_3M_30Y' => 1M steps to 10Y, then 3M to 30Y
    Returns strictly positive times (year fractions).
    """
    if reference_date is None:
        reference_date = ql.Settings.instance().evaluationDate

    tokens = spec.split('_')
    if len(tokens) % 2 != 0:
        raise ValueError("Grid spec must alternate STEP_HORIZON pairs, e.g., '1M_10Y_3M_30Y'")

    pairs = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

    dates: list[ql.Date] = []
    last_horizon = reference_date
    for step_tok, horiz_tok in pairs:
        step = _parse_leg(step_tok)
        horizon = _parse_leg(horiz_tok)
        end_date = reference_date + horizon

        cursor = last_horizon
        while cursor < end_date:
            cursor = calendar_advance(cursor, step)
            if cursor > end_date:
                break
            dates.append(cursor)
        last_horizon = end_date

    dates = sorted(set(dates))
    times = [day_count.yearFraction(reference_date, d) for d in dates]
    times = [t for t in times if t > 0]
    return times


# ---- Shaped curves for realism ----

def build_zero_curve(points: List[Tuple[float, float]],
                     day_count: ql.DayCounter = ql.Actual365Fixed()) -> ql.YieldTermStructureHandle:
    """Build a zero curve from (t_years, zero_rate) points (cont comp)."""
    ref = ql.Settings.instance().evaluationDate
    dates: list[ql.Date] = []
    rates: list[float] = []
    last = ref
    for t, r in sorted(points, key=lambda x: x[0]):
        if t <= 0:
            continue
        d = ref + ql.Period(int(round(365 * t)), ql.Days)
        if d <= last:
            continue
        dates.append(d)
        rates.append(r)
        last = d
    curve = ql.ZeroCurve(dates, rates, day_count)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def build_upward_curve() -> ql.YieldTermStructureHandle:
    pts = [(0.5, 0.010), (2.0, 0.015), (5.0, 0.020), (10.0, 0.025), (30.0, 0.040)]
    return build_zero_curve(pts)


def build_inverted_curve() -> ql.YieldTermStructureHandle:
    pts = [(0.5, 0.040), (2.0, 0.030), (5.0, 0.025), (10.0, 0.020), (30.0, 0.015)]
    return build_zero_curve(pts)


def build_humped_curve() -> ql.YieldTermStructureHandle:
    pts = [(0.5, 0.020), (2.0, 0.022), (5.0, 0.030), (10.0, 0.025), (30.0, 0.020)]
    return build_zero_curve(pts)
