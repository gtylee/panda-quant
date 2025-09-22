from __future__ import annotations
import argparse
import numpy as np
import QuantLib as ql

from market import build_curve, make_time_grid
from credit import build_credit_curve
from portfolio import build_linear_portfolio
from models import CalibratedHW, CalibratedG2, simulate_hw, simulate_g2, calibration_fit_rows_hw, calibration_fit_rows_g2
from models import calibrate_hw_sigma, calibrate_g2_sigmas
from exposure import epe_profile_from_model, epe_profile_from_model_with_ratio_mult, epe_profile_from_states_with_ratio_mult, bucket_ratio_multiplier
from cva import cva_unilateral, compare_series, ee_profile_stats, print_summary, write_csvs, write_calibration_csv


def _build_bumped_curve_from_times(base_ts: ql.YieldTermStructureHandle, times: list[float], shift_bp: float, bstart: float, bend: float) -> ql.YieldTermStructureHandle:
    ref = base_ts.referenceDate()
    dc = base_ts.dayCounter()
    # Build knot times, exclude <= 0 and dedupe with rounding
    knots = sorted(set([round(t, 10) for t in times if t > 0.0] + [round(bstart, 10), round(bend, 10)]))
    # Remove any zeros/negatives and ensure strictly increasing
    knots = [t for t in knots if t > 0.0]
    dates = [ref]
    discounts = [1.0]
    shift = shift_bp * 1e-4
    last_date = ref
    for t in knots:
        d = ref + ql.Period(int(round(365 * t)), ql.Days)
        if d <= last_date:
            continue
        # Bumped DF at 0->t with local shift over overlap
        overlap = max(0.0, min(t, bend) - max(0.0, bstart))
        mult = np.exp(-shift * overlap)
        dates.append(d)
        discounts.append(float(base_ts.discount(t) * mult))
        last_date = d
    curve = ql.DiscountCurve(dates, discounts, dc)
    curve.enableExtrapolation()
    return ql.YieldTermStructureHandle(curve)


def run(seed: int = 42, n_paths: int = 20000, grid: str = '1M_10Y_3M_30Y', include_bonds: bool = False):
    ts = build_curve(flat_rate=0.02)
    ql.Settings.instance().evaluationDate = ts.referenceDate()

    cc = build_credit_curve(flat_hazard=0.015)  # 150 bps
    lgd = 0.60

    port = build_linear_portfolio(ts, include_bonds=include_bonds)

    atm = 0.20
    hw = CalibratedHW.from_atm_level(ts, atm_vol=atm, a=0.03)
    g2 = CalibratedG2.from_atm_level(ts, atm_vol=atm, a=0.03, b=0.10, rho=-0.75)

    # Dump calibration parameters
    write_calibration_csv([{
        'model': 'HW1F', 'a': hw.a, 'sigma': hw.sigma,
        'b': None, 'rho': None, 'eta': None,
        'atm_vol_input': atm
    }, {
        'model': 'G2++', 'a': g2.a, 'sigma': g2.sigma,
        'b': g2.b, 'rho': g2.rho, 'eta': g2.eta,
        'atm_vol_input': atm
    }], 'output/calibration_params.csv')

    times = make_time_grid(grid, ts.referenceDate())

    epe_hw = epe_profile_from_model(port, ts, hw, simulate_hw, times, n_paths, seed)
    epe_g2 = epe_profile_from_model(port, ts, g2, simulate_g2, times, n_paths, seed)

    cva_hw = cva_unilateral(epe_hw, times, ts, cc, lgd)
    cva_g2 = cva_unilateral(epe_g2, times, ts, cc, lgd)

    report = {
        'CVA_HW': float(cva_hw), 'CVA_G2': float(cva_g2),
        'CVA_abs_diff': float(cva_hw - cva_g2),
        'CVA_rel_diff_%': float(100.0 * (cva_hw - cva_g2) / max(1e-12, cva_hw))
    }

    stats = compare_series(epe_hw, epe_g2)
    ee_hw = ee_profile_stats(epe_hw, times)
    ee_g2 = ee_profile_stats(epe_g2, times)

    pass_cva = abs(report['CVA_rel_diff_%']) <= 1.0 and abs(report['CVA_abs_diff']) <= 0.00005 * 10_000_000.0
    pass_epe = stats['MAPE_%'] <= 1.0 and stats['MAX_abs_%'] <= 2.0 and abs(ee_hw['EE_mean'] - ee_g2['EE_mean']) <= 0.01 * max(1e-12, ee_hw['EE_mean'])

    print_summary(report, stats, pass_cva, pass_epe)
    write_csvs(epe_hw, epe_g2, times, report)

    hw_rows = calibration_fit_rows_hw(hw, atm)
    g2_rows = calibration_fit_rows_g2(g2, atm)
    write_calibration_csv(hw_rows, 'output/calibration_hw.csv')
    write_calibration_csv(g2_rows, 'output/calibration_g2.csv')

    # --- Small-path attribution ---
    small_paths = max(1024, n_paths // 16)
    from models import CalibratedG2 as _CalG2
    g2_aligned = _CalG2(ts, a=g2.a, b=g2.b, rho=g2.rho, sigma=hw.sigma, eta=1e-12)
    epe_g2_aligned = epe_profile_from_model(port, ts, g2_aligned, simulate_g2, times, small_paths, seed+123)
    cva_g2_aligned = cva_unilateral(epe_g2_aligned, times, ts, cc, lgd)

    cva_diff = float(cva_hw - cva_g2)
    calib_component = float(cva_g2_aligned - cva_g2)
    model_component = float(cva_hw - cva_g2_aligned)

    write_calibration_csv([{
        'CVA_HW': float(cva_hw), 'CVA_G2': float(cva_g2), 'CVA_diff': cva_diff,
        'calibration_component': calib_component, 'model_component': model_component,
        'small_paths': small_paths
    }], 'output/cva_attribution.csv')

    # --- Bucketed DV01-style deltas (reuse states across buckets) ---
    X_hw = simulate_hw(hw, times, small_paths, seed+777, True)
    X_g2, Y_g2 = simulate_g2(g2, times, small_paths, seed+777, True)

    buckets = [
        (0.0, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
        (5.0, 10.0),
        (10.0, 30.0),
    ]
    shift_bp = 1.0
    rows = []
    rows_test = []
    for bstart, bend in buckets:
        mult = bucket_ratio_multiplier(shift_bp, bstart, bend)
        epe_hw_up = epe_profile_from_states_with_ratio_mult(port, hw, times, X_hw, None, mult)
        epe_g2_up = epe_profile_from_states_with_ratio_mult(port, g2, times, X_g2, Y_g2, mult)
        cva_hw_up = cva_unilateral(epe_hw_up, times, ts, cc, lgd)
        cva_g2_up = cva_unilateral(epe_g2_up, times, ts, cc, lgd)
        delta_hw_ratio = float(cva_hw_up - cva_hw)
        delta_g2_ratio = float(cva_g2_up - cva_g2)
        rows.append({
            'bucket_startY': bstart,
            'bucket_endY': bend,
            'delta_CVA_HW_per_bp': delta_hw_ratio,
            'delta_CVA_G2_per_bp': delta_g2_ratio
        })
        ts_bumped = _build_bumped_curve_from_times(ts, times, shift_bp, bstart, bend)
        hw_b = CalibratedHW(ts_bumped, a=hw.a, sigma=hw.sigma)
        g2_b = CalibratedG2(ts_bumped, a=g2.a, b=g2.b, rho=g2.rho, sigma=g2.sigma, eta=g2.eta)
        epe_hw_up_b = epe_profile_from_states_with_ratio_mult(port, hw_b, times, X_hw, None, None)
        epe_g2_up_b = epe_profile_from_states_with_ratio_mult(port, g2_b, times, X_g2, Y_g2, None)
        cva_hw_up_b = cva_unilateral(epe_hw_up_b, times, ts_bumped, cc, lgd)
        cva_g2_up_b = cva_unilateral(epe_g2_up_b, times, ts_bumped, cc, lgd)
        rows_test.append({
            'bucket_startY': bstart,
            'bucket_endY': bend,
            'delta_CVA_HW_per_bp_ratio': delta_hw_ratio,
            'delta_CVA_HW_per_bp_bumped': float(cva_hw_up_b - cva_hw),
            'delta_CVA_G2_per_bp_ratio': delta_g2_ratio,
            'delta_CVA_G2_per_bp_bumped': float(cva_g2_up_b - cva_g2)
        })
    write_calibration_csv(rows, 'output/cva_bucketed_delta.csv')
    write_calibration_csv(rows_test, 'output/cva_bucketed_delta_test.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVA equivalence: HW1F vs G2++ on linear portfolio')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--paths', type=int, default=20000)
    parser.add_argument('--grid', type=str, default='1M_10Y_3M_30Y')
    parser.add_argument('--include-bonds', action='store_true', help='Include par bonds in the portfolio')
    args = parser.parse_args()
    run(seed=args.seed, n_paths=args.paths, grid=args.grid, include_bonds=args.include_bonds)
