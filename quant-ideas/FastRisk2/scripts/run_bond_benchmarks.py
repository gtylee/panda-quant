#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import numpy as np
import QuantLib as ql

from product_definitions import FloatingRateBondStaticBase, InflationLinkedBondStaticBase, CallableBondStaticBase, ConvertibleBondStaticBase
from quantlib_bond_pricer import QuantLibBondPricer


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_zero_curve(val_date: date, pillars: np.ndarray, rates: np.ndarray) -> ql.YieldTermStructureHandle:
    ql.Settings.instance().evaluationDate = ql.Date(val_date.day, val_date.month, val_date.year)
    dc = ql.Actual365Fixed(); cal = ql.TARGET()
    base_d = ql.Settings.instance().evaluationDate
    dates = ql.DateVector(); dates.push_back(base_d)
    for t in pillars:
        dates.push_back(base_d + ql.Period(int(round(float(t)*365)), ql.Days))
    curve = ql.ZeroCurve(dates, [float(rates[0])] + list(map(float, rates)), dc, cal, ql.Linear(), ql.Continuous, ql.Annual)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def build_curve_library(pillars: np.ndarray) -> dict[str, np.ndarray]:
    # Create a richer set of curve shapes
    flat = np.full_like(pillars, 0.022, dtype=float)
    upward = np.array([0.018, 0.020, 0.022, 0.025, 0.028][: len(pillars)], dtype=float)
    downward = np.array([0.030, 0.027, 0.024, 0.022, 0.020][: len(pillars)], dtype=float)
    steepener = upward + np.array([0.002, 0.001, 0.000, -0.001, -0.002][: len(pillars)], dtype=float)
    flattener = upward + np.array([0.000, -0.0005, -0.0010, -0.0015, -0.002][: len(pillars)], dtype=float)
    hump = upward + np.array([0.000, 0.0008, 0.0015, 0.0005, -0.0005][: len(pillars)], dtype=float)
    inverted = np.flip(upward)
    return {
        'flat': flat,
        'upward': upward,
        'downward': downward,
        'steepener': steepener,
        'flattener': flattener,
        'hump': hump,
        'inverted': inverted,
    }


def frn_benchmark(val_date: date, pillars: np.ndarray, curves: dict) -> list[dict]:
    results = []
    maturities = [1, 2, 3, 5, 7, 10]
    tenors = [3, 6]
    spreads = [0.0, 0.001, 0.005]
    for curve_name, rates in curves.items():
        for years in maturities:
            for tenor_m in tenors:
                for spr in spreads:
                    mat = date(val_date.year + years, val_date.month, val_date.day)
                    frn_static = FloatingRateBondStaticBase(
                        valuation_date=val_date, maturity_date=mat,
                        coupon_rate=0.0, face_value=100.0, freq=12 // (tenor_m if tenor_m in (3, 6) else 3),
                        index_tenor_months=tenor_m, spread=spr
                    )
                    model = QuantLibBondPricer(frn_static, method='discount').price(pillars, rates)[0]
                    handle = build_zero_curve(val_date, pillars, rates)
                    tenor = ql.Period(int(tenor_m), ql.Months)
                    ibor = ql.IborIndex("GENERIC_IBOR", tenor, 0, ql.USDCurrency(), ql.TARGET(), ql.Following, False, frn_static.day_count_ql, handle)
                    frn = ql.FloatingRateBond(
                        frn_static.settlement_days, frn_static.face_value,
                        frn_static.schedule, ibor, frn_static.day_count_ql, ql.Following,
                        fixingDays=0, spreads=[frn_static.spread]
                    )
                    frn.setPricingEngine(ql.DiscountingBondEngine(handle))
                    ref = frn.NPV()
                    diff_abs = float(model - ref)
                    diff_bps = (diff_abs / ref) * 10000 if ref != 0 else 0.0
                    results.append({
                        'product': 'FRN', 'maturity_years': years, 'curve': curve_name,
                        'index_tenor_m': tenor_m, 'spread': spr,
                        'model_price': float(model), 'ref_price': float(ref),
                        'diff_abs': float(diff_abs), 'diff_bps': float(diff_bps)
                    })
    return results


def ilb_benchmark(val_date: date, pillars: np.ndarray, curves: dict) -> list[dict]:
    results = []
    maturities = [2, 5, 10]
    infl_pillars = np.array([1.0, 3.0, 7.0], dtype=float)
    infl_curves = {
        'infl_up': np.array([0.015, 0.020, 0.022], dtype=float),
        'infl_flat': np.array([0.018, 0.018, 0.018], dtype=float),
        'infl_down': np.array([0.022, 0.020, 0.017], dtype=float),
    }
    comp_modes = ['continuous', 'annual']
    lags = [0, 2, 3]
    coupons = [0.00, 0.015]
    ql.Settings.instance().evaluationDate = ql.Date(val_date.day, val_date.month, val_date.year)
    for curve_name, rates in curves.items():
        handle = build_zero_curve(val_date, pillars, rates)
        for years in maturities:
            for comp in comp_modes:
                for lag_m in lags:
                    for cpn in coupons:
                        mat = date(val_date.year + years, val_date.month, val_date.day)
                        ilb_static = InflationLinkedBondStaticBase(
                            valuation_date=val_date, maturity_date=mat,
                            coupon_rate=cpn, face_value=100.0, freq=2,
                            inflation_factor_name='USD_INFLATION', inflation_compounding=comp, inflation_lag_months=lag_m
                        )
                        pricer = QuantLibBondPricer(ilb_static, method='discount')
                        # choose an inflation curve
                        for infl_name, infl_rates in infl_curves.items():
                            model = pricer.price(
                                pillars, rates,
                                inflation_rate=float(infl_rates[-1]),
                                inflation_curve_pillars=infl_pillars,
                                inflation_curve_rates=infl_rates
                            )[0]
                            # manual piecewise calc mirroring pricer
                            eval_d = ql.Settings.instance().evaluationDate
                            dc = ilb_static.day_count_ql
                            sched = ilb_static.schedule
                            lag_years = ilb_static.inflation_lag_months/12.0
                            def infl_factor(t_years: float) -> float:
                                t_eff = max(t_years, 0.0)
                                pillars_i = infl_pillars
                                rates_i = infl_rates
                                order = np.argsort(pillars_i)
                                pillars_s = pillars_i[order]; rates_s = rates_i[order]
                                acc = 0.0; last_t = 0.0
                                for p, r in zip(pillars_s, rates_s):
                                    seg_end = min(t_eff, p)
                                    if seg_end > last_t:
                                        dt = seg_end - last_t
                                        if comp == 'continuous':
                                            acc += float(r) * dt
                                        else:
                                            acc += float(np.log(1.0 + float(r))) * dt
                                        last_t = seg_end
                                    if last_t >= t_eff:
                                        break
                                if last_t < t_eff:
                                    r_last = float(rates_s[-1])
                                    dt = t_eff - last_t
                                    if comp == 'continuous':
                                        acc += r_last * dt
                                    else:
                                        acc += float(np.log(1.0 + r_last)) * dt
                                return float(np.exp(acc))
                            pv_manual = 0.0
                            dates = list(sched)
                            for i in range(1, len(dates)):
                                d2 = dates[i]; d1 = dates[i-1]
                                accrual = dc.yearFraction(d1, d2)
                                t = dc.yearFraction(eval_d, d2) - lag_years
                                scale = infl_factor(t)
                                cpn_amt = ilb_static.coupon_rate * accrual * ilb_static.face_value * scale
                                pv_manual += cpn_amt * handle.discount(d2)
                            t_end = dc.yearFraction(eval_d, dates[-1]) - lag_years
                            scale_end = infl_factor(t_end)
                            pv_manual += ilb_static.face_value * scale_end * handle.discount(dates[-1])
                            diff_abs = float(model - pv_manual)
                            diff_bps = (diff_abs / pv_manual) * 10000 if pv_manual != 0 else 0.0
                            results.append({
                                'product': 'ILB', 'maturity_years': years, 'curve': curve_name,
                                'infl_curve': infl_name, 'infl_comp': comp, 'infl_lag_m': lag_m, 'coupon_rate': cpn,
                                'model_price': float(model), 'ref_price': float(pv_manual),
                                'diff_abs': float(diff_abs), 'diff_bps': float(diff_bps)
                            })
    return results


def write_report_md(path: str, frn_rows: list[dict], ilb_rows: list[dict], callable_rows: list[dict], convertible_rows: list[dict]):
    lines = []
    lines.append("# Bond Pricer Benchmarks\n")
    lines.append("\n## Floating-Rate Bonds vs QuantLib\n")
    lines.append("| Mat (Y) | Curve | Tenor (m) | Spread | Model | QL Ref | Diff | Diff (bps) |\n")
    lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|\n")
    for r in frn_rows:
        lines.append(f"| {r['maturity_years']} | {r['curve']} | {r.get('index_tenor_m','')} | {r.get('spread','')} | {r['model_price']:.4f} | {r['ref_price']:.4f} | {r['diff_abs']:.4f} | {r['diff_bps']:.2f} |\n")
    lines.append("\n## Inflation-Linked Bonds vs Manual Piecewise\n")
    lines.append("| Mat (Y) | Curve | InflCurve | Comp | Lag (m) | Coupon | Model | Manual | Diff | Diff (bps) |\n")
    lines.append("|---:|:---|:---|:---|---:|---:|---:|---:|---:|---:|\n")
    for r in ilb_rows:
        lines.append(f"| {r['maturity_years']} | {r['curve']} | {r.get('infl_curve','')} | {r.get('infl_comp','')} | {r.get('infl_lag_m','')} | {r.get('coupon_rate','')} | {r['model_price']:.4f} | {r['ref_price']:.4f} | {r['diff_abs']:.4f} | {r['diff_bps']:.2f} |\n")
    lines.append("\n## Callable Bonds (G2) vs Call-To-Worst\n")
    lines.append("| Mat (Y) | Curve | Model | CTW Ref | Diff | Diff (bps) |\n")
    lines.append("|---:|:---|---:|---:|---:|---:|\n")
    for r in callable_rows:
        lines.append(f"| {r['maturity_years']} | {r['curve']} | {r['model_price']:.4f} | {r['ref_price']:.4f} | {r['diff_abs']:.4f} | {r['diff_bps']:.2f} |\n")
    lines.append("\n## Convertible Bonds vs Decomposition\n")
    lines.append("| Mat (Y) | Curve | S0 | Vol | Div | CS | Model | Decomp Ref | Diff | Diff (bps) |\n")
    lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in convertible_rows:
        lines.append(f"| {r['maturity_years']} | {r['curve']} | {r.get('s0','')} | {r.get('vol','')} | {r.get('div','')} | {r.get('credit_spread','')} | {r['model_price']:.4f} | {r['ref_price']:.4f} | {r['diff_abs']:.4f} | {r['diff_bps']:.2f} |\n")
    with open(path, 'w') as f:
        f.writelines(lines)


def write_report_csv(path: str, rows: list[dict]):
    import csv
    fieldnames = [
        'product', 'maturity_years', 'curve',
        # FRN extras
        'index_tenor_m', 'spread',
        # ILB extras
        'infl_curve', 'infl_comp', 'infl_lag_m', 'coupon_rate',
        # Callable extras
        'g2_steps', 'g2_params',
        # Convertible extras
        's0', 'vol', 'div', 'credit_spread',
        # Common
        'model_price', 'ref_price', 'diff_abs', 'diff_bps'
    ]
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def callable_benchmark(val_date: date, pillars: np.ndarray, curves: dict) -> list[dict]:
    rows = []
    maturities = [5, 10]
    call_schedules = [
        # (years, prices)
        ([2, 3, 4], [102.0, 101.0, 100.5]),
        ([3, 5, 7], [101.0, 100.5, 100.0]),
    ]
    g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    for curve_name, rates in curves.items():
        for years in maturities:
            for yrs_list, prices_list in call_schedules:
                mat = date(val_date.year + years, val_date.month, val_date.day)
                # ensure call dates strictly before maturity
                call_dates_py = [date(val_date.year + y, val_date.month, val_date.day) for y in yrs_list]
                call_pairs = [(d, p) for d, p in zip(call_dates_py, prices_list) if d < mat]
                if not call_pairs:
                    # skip schedule if no valid call dates
                    continue
                call_dates_py = [d for d, _ in call_pairs]
                prices_list = [p for _, p in call_pairs]
                callable_static = CallableBondStaticBase(
                    valuation_date=val_date, maturity_date=mat, coupon_rate=0.035,
                    face_value=100.0, freq=2,
                    call_dates=[d.isoformat() for d in call_dates_py], call_prices=prices_list,
                    calendar='target', day_count='Actual/Actual (ISDA)'
                )
                model = QuantLibBondPricer(callable_static, method='g2', grid_steps=64).price(pillars, rates, g2_params=g2_params)[0]

                handle = build_zero_curve(val_date, pillars, rates)
                dc = callable_static.day_count_ql
                sched = callable_static.schedule
                dates = list(sched)
                def map_to_schedule_stop_date(call_dt_py: date) -> ql.Date | None:
                    call_dt = ql.Date(call_dt_py.day, call_dt_py.month, call_dt_py.year)
                    # choose first schedule date >= call date
                    for d in dates[1:]:  # skip schedule start
                        if d >= call_dt:
                            return d
                    return None

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
                pv_list = []
                for d_py, cp in zip(call_dates_py, prices_list):
                    stop = map_to_schedule_stop_date(d_py)
                    if stop is None:
                        continue
                    pv_list.append(pv_to(stop, cp))
                pv_list.append(pv_to(dates[-1], callable_static.face_value))
                ref = min(pv_list)
                diff_abs = float(model - ref)
                diff_bps = (diff_abs / ref) * 10000 if ref != 0 else 0.0
                rows.append({
                    'product': 'CALLABLE', 'maturity_years': years, 'curve': curve_name,
                    'model_price': float(model), 'ref_price': float(ref), 'diff_abs': diff_abs, 'diff_bps': diff_bps,
                    'g2_steps': 64, 'g2_params': str(g2_params)
                })
    return rows


def convertible_benchmark(val_date: date, pillars: np.ndarray, curves: dict) -> list[dict]:
    rows = []
    maturities = [5, 7]
    vols = [0.20, 0.30]
    s0s = [90.0, 100.0, 110.0]
    divs = [0.00, 0.01]
    css = [0.001, 0.003]
    for curve_name, rates in curves.items():
        for years in maturities:
            mat = date(val_date.year + years, val_date.month, val_date.day)
            conv_static = ConvertibleBondStaticBase(
                valuation_date=val_date, issue_date=date(val_date.year - 1, val_date.month, val_date.day), maturity_date=mat,
                coupon_rate=0.02, conversion_ratio=1.0, face_value=100.0, freq=2,
                calendar='target', day_count='Actual/Actual (ISDA)'
            )
            for s0 in s0s:
                for vol in vols:
                    for div in divs:
                        for cs in css:
                            model = QuantLibBondPricer(conv_static, method='discount', grid_steps=128).price(pillars, rates, s0_val=s0, dividend_yield=div, equity_volatility=vol, credit_spread=cs)[0]
                            handle_rf = build_zero_curve(val_date, pillars, rates)
                            handle_cs = build_zero_curve(val_date, pillars, rates + cs)
                            dc = conv_static.day_count_ql; cal = ql.TARGET()
                            eval_d = ql.Settings.instance().evaluationDate
                            dates = list(conv_static.schedule)
                            pv_bond = 0.0
                            for i in range(1, len(dates)):
                                d1 = dates[i-1]; d2 = dates[i]
                                accrual = dc.yearFraction(d1, d2)
                                cpn = conv_static.coupon_rate * accrual * conv_static.face_value
                                if d2 > eval_d:
                                    pv_bond += cpn * handle_cs.discount(d2)
                            if dates[-1] > eval_d:
                                pv_bond += conv_static.face_value * handle_cs.discount(dates[-1])
                            # Equity option
                            r = float(np.mean(rates))
                            process = ql.BlackScholesMertonProcess(
                                ql.QuoteHandle(ql.SimpleQuote(s0)),
                                ql.YieldTermStructureHandle(ql.FlatForward(eval_d, div, dc)),
                                ql.YieldTermStructureHandle(ql.FlatForward(eval_d, r, dc)),
                                ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_d, cal, vol, dc))
                            )
                            strike = conv_static.face_value / conv_static.conversion_ratio
                            payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
                            exercise = ql.EuropeanExercise(dates[-1])
                            option = ql.VanillaOption(payoff, exercise)
                            option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
                            pv_option = conv_static.conversion_ratio * option.NPV()
                            ref = pv_bond + pv_option
                            diff_abs = float(model - ref)
                            diff_bps = (diff_abs / ref) * 10000 if ref != 0 else 0.0
                            rows.append({
                                'product': 'CONVERTIBLE', 'maturity_years': years, 'curve': curve_name,
                                's0': s0, 'vol': vol, 'div': div, 'credit_spread': cs,
                                'model_price': float(model), 'ref_price': float(ref), 'diff_abs': diff_abs, 'diff_bps': diff_bps
                            })
    return rows


def main():
    val_date = date(2025, 5, 18)
    pillars = np.array([0.25, 0.5, 1.0, 2.0, 3.0], dtype=float)
    curves = build_curve_library(pillars)

    frn_rows = frn_benchmark(val_date, pillars, curves)
    ilb_rows = ilb_benchmark(val_date, pillars, {k: v for k, v in curves.items() if k in ('flat','upward','downward')})
    callable_rows = callable_benchmark(val_date, pillars, {k: v for k, v in curves.items() if k in ('flat','upward','downward')})
    convertible_rows = convertible_benchmark(val_date, pillars, {k: v for k, v in curves.items() if k in ('flat','upward')})

    out_dir = os.path.join('output', 'benchmarks')
    ensure_dir(out_dir)
    md_path = os.path.join(out_dir, 'bond_benchmark_report.md')
    csv_path = os.path.join(out_dir, 'bond_benchmark_report.csv')
    write_report_md(md_path, frn_rows, ilb_rows, callable_rows, convertible_rows)
    write_report_csv(csv_path, frn_rows + ilb_rows + callable_rows + convertible_rows)
    print(f"Benchmark report written to:\n - {md_path}\n - {csv_path}")


if __name__ == '__main__':
    main()


