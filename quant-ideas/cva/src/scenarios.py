from __future__ import annotations
import QuantLib as ql
import numpy as np
from typing import List, Dict, Tuple

from market import build_curve, make_time_grid
from credit import build_credit_curve
from portfolio import build_linear_portfolio, PayerParSwap, ParBond, Portfolio
from models import CalibratedHW, CalibratedG2, simulate_g2
from exposure import epe_profile_from_model, epe_paths_from_model, mtm_paths_from_model
from cva import cva_unilateral, compare_series


def build_slope_portfolio(ts: ql.YieldTermStructureHandle) -> Portfolio:
    short_m = 2
    long_m = 30
    n_short = -20_000_000.0  # receiver 2Y
    n_long = 10_000_000.0    # payer 30Y
    s_short = PayerParSwap(notional=n_short, maturity_years=short_m)
    s_short.build_from_curve(ts)
    s_long = PayerParSwap(notional=n_long, maturity_years=long_m)
    s_long.build_from_curve(ts)
    return Portfolio(instruments=[s_short, s_long])


def _pathwise_cva(expo_paths: np.ndarray, times: List[float], ts: ql.YieldTermStructureHandle, hazard: float, lgd: float) -> np.ndarray:
    from credit import CreditCurve
    cc = build_credit_curve(hazard)
    t = np.array(times, dtype=float)
    dPD = cc.dPD(times)  # shape (n_times,)
    df0 = np.array([ts.discount(float(ti)) for ti in t])
    weights = df0 * dPD * lgd  # (n_times,)
    # CVA per path = sum_j exposure_path[j] * weight[j]
    return expo_paths @ weights


def run_scenarios(out_path: str = 'output/scenario_summary.csv',
                  seeds: List[int] | None = None,
                  path_count: int = 1024,
                  grid: str = '3M_10Y') -> None:
    import pandas as pd

    if seeds is None:
        seeds = [9]

    rows: List[Dict[str, float]] = []
    rows_ci: List[Dict[str, float]] = []

    cases = [
        {'name': 'SwapsOnly_ATM20_Hz150', 'portfolio': 'swaps', 'atm': 0.20, 'hazard': 0.015},
        {'name': 'SwapsOnly_ATM15_Hz150', 'portfolio': 'swaps', 'atm': 0.15, 'hazard': 0.015},
        {'name': 'SwapsOnly_ATM25_Hz150', 'portfolio': 'swaps', 'atm': 0.25, 'hazard': 0.015},
        {'name': 'SwapsPlusBonds_ATM20_Hz150', 'portfolio': 'swaps_bonds', 'atm': 0.20, 'hazard': 0.015},
        {'name': 'SwapsOnly_ATM20_Hz100', 'portfolio': 'swaps', 'atm': 0.20, 'hazard': 0.010},
        {'name': 'SwapsOnly_ATM20_Hz200', 'portfolio': 'swaps', 'atm': 0.20, 'hazard': 0.020},
        {'name': 'SlopeSteepener_ATM20_Hz150', 'portfolio': 'slope', 'atm': 0.20, 'hazard': 0.015},
    ]

    ts = build_curve(0.02)
    ql.Settings.instance().evaluationDate = ts.referenceDate()

    for case in cases:
        for seed in seeds:
            cc = build_credit_curve(case['hazard'])
            lgd = 0.60
            if case['portfolio'] == 'swaps':
                port = build_linear_portfolio(ts, include_bonds=False)
            elif case['portfolio'] == 'swaps_bonds':
                port = build_linear_portfolio(ts, include_bonds=True)
            elif case['portfolio'] == 'slope':
                port = build_slope_portfolio(ts)
            else:
                port = build_linear_portfolio(ts, include_bonds=False)

            times = make_time_grid(grid, ts.referenceDate())

            hw = CalibratedHW.from_atm_level(ts, atm_vol=case['atm'], a=0.03)
            g2 = CalibratedG2.from_atm_level(ts, atm_vol=case['atm'], a=0.03, b=0.10, rho=-0.75)

            epe_hw = epe_profile_from_model(port, ts, hw, simulate_g2.__globals__['simulate_hw'], times, path_count, seed)
            epe_g2 = epe_profile_from_model(port, ts, g2, simulate_g2, times, path_count, seed)

            cva_hw = cva_unilateral(epe_hw, times, ts, cc, lgd)
            cva_g2 = cva_unilateral(epe_g2, times, ts, cc, lgd)

            # Aligned G2 (sigma=HW sigma, eta≈0)
            g2_aligned = CalibratedG2(ts, a=g2.a, b=g2.b, rho=g2.rho, sigma=hw.sigma, eta=1e-12)
            epe_g2a = epe_profile_from_model(port, ts, g2_aligned, simulate_g2, times, max(512, path_count//2), seed+123)
            cva_g2a = cva_unilateral(epe_g2a, times, ts, cc, lgd)

            diffs = compare_series(epe_hw, epe_g2)
            total = float(cva_hw - cva_g2)
            calib_component = float(cva_g2a - cva_g2)
            model_component = float(cva_hw - cva_g2a)

            rows.append({
                'case': case['name'], 'seed': seed, 'paths': path_count, 'grid': grid,
                'CVA_HW': float(cva_hw), 'CVA_G2': float(cva_g2), 'CVA_diff': total,
                'CVA_calib_component': calib_component, 'CVA_model_component': model_component,
                'EPE_MAPE_%': diffs['MAPE_%'], 'EPE_MAX_%': diffs['MAX_abs_%']
            })

            # CI/SE with common random numbers
            expo_hw_paths = epe_paths_from_model(port, ts, hw, simulate_g2.__globals__['simulate_hw'], times, path_count, seed)
            expo_g2_paths = epe_paths_from_model(port, ts, g2, simulate_g2, times, path_count, seed)
            cva_hw_paths = _pathwise_cva(expo_hw_paths, times, ts, case['hazard'], lgd)
            cva_g2_paths = _pathwise_cva(expo_g2_paths, times, ts, case['hazard'], lgd)
            diff_paths = cva_hw_paths - cva_g2_paths
            se_hw = float(np.std(cva_hw_paths, ddof=1) / np.sqrt(path_count))
            se_g2 = float(np.std(cva_g2_paths, ddof=1) / np.sqrt(path_count))
            se_diff = float(np.std(diff_paths, ddof=1) / np.sqrt(path_count))
            rows_ci.append({
                'case': case['name'], 'seed': seed, 'paths': path_count, 'grid': grid,
                'CVA_HW': float(cva_hw), 'SE_HW': se_hw, 'CVA_G2': float(cva_g2), 'SE_G2': se_g2,
                'CVA_diff': total, 'SE_diff': se_diff, '%diff_vs_G2': float(100.0 * total / max(1e-12, cva_g2))
            })

    import pandas as pd
    pd.DataFrame(rows).to_csv(out_path, index=False)
    pd.DataFrame(rows_ci).to_csv('output/scenario_summary_with_ci.csv', index=False)


def run_grid_sensitivity(out_path: str = 'output/grid_sensitivity.csv',
                         grid_list: List[str] | None = None,
                         seed: int = 9,
                         paths: int = 4096) -> None:
    import pandas as pd
    if grid_list is None:
        grid_list = ['1M_10Y', '3M_10Y', '6M_10Y']
    ts = build_curve(0.02)
    ql.Settings.instance().evaluationDate = ts.referenceDate()
    cc = build_credit_curve(0.015)
    lgd = 0.60
    port = build_linear_portfolio(ts, include_bonds=False)
    hw = CalibratedHW.from_atm_level(ts, 0.20)
    g2 = CalibratedG2.from_atm_level(ts, 0.20, a=0.03, b=0.10, rho=-0.75)
    rows: List[Dict[str, float]] = []
    for grid in grid_list:
        times = make_time_grid(grid, ts.referenceDate())
        epe_hw = epe_profile_from_model(port, ts, hw, simulate_g2.__globals__['simulate_hw'], times, paths, seed)
        epe_g2 = epe_profile_from_model(port, ts, g2, simulate_g2, times, paths, seed)
        cva_hw = cva_unilateral(epe_hw, times, ts, cc, lgd)
        cva_g2 = cva_unilateral(epe_g2, times, ts, cc, lgd)
        rows.append({'grid': grid, 'CVA_HW': float(cva_hw), 'CVA_G2': float(cva_g2), 'CVA_diff': float(cva_hw - cva_g2)})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_convergence(case_name: str = 'SwapsOnly_ATM20_Hz150',
                    out_path: str = 'output/convergence_SwapsOnly_ATM20_Hz150.csv',
                    seeds: List[int] | None = None,
                    path_counts: List[int] | None = None,
                    grid: str = '3M_10Y') -> None:
    import pandas as pd
    if seeds is None:
        seeds = [7]
    if path_counts is None:
        path_counts = [1024, 2048, 4096, 8192]
    ts = build_curve(0.02)
    ql.Settings.instance().evaluationDate = ts.referenceDate()
    # decode case
    include_bonds = False
    atm = 0.20
    hazard = 0.015
    if 'SwapsPlusBonds' in case_name:
        include_bonds = True
    if 'ATM15' in case_name:
        atm = 0.15
    if 'ATM25' in case_name:
        atm = 0.25
    if 'Hz100' in case_name:
        hazard = 0.010
    if 'Hz200' in case_name:
        hazard = 0.020

    cc = build_credit_curve(hazard)
    lgd = 0.60
    if 'SlopeSteepener' in case_name:
        port = build_slope_portfolio(ts)
    else:
        port = build_linear_portfolio(ts, include_bonds=include_bonds)

    rows: List[Dict[str, float]] = []
    for paths in path_counts:
        for seed in seeds:
            times = make_time_grid(grid, ts.referenceDate())
            hw = CalibratedHW.from_atm_level(ts, atm)
            g2 = CalibratedG2.from_atm_level(ts, atm, a=0.03, b=0.10, rho=-0.75)
            epe_hw = epe_profile_from_model(port, ts, hw, simulate_g2.__globals__['simulate_hw'], times, paths, seed)
            epe_g2 = epe_profile_from_model(port, ts, g2, simulate_g2, times, paths, seed)
            cva_hw = cva_unilateral(epe_hw, times, ts, cc, lgd)
            cva_g2 = cva_unilateral(epe_g2, times, ts, cc, lgd)
            rows.append({'paths': paths, 'seed': seed, 'CVA_HW': float(cva_hw), 'CVA_G2': float(cva_g2), 'CVA_diff': float(cva_hw - cva_g2)})
    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_dva_mr_sweep(out_path: str = 'output/dva_mr_sweep.csv',
                      a_values: List[float] | None = None,
                      paths: int = 4096,
                      grid: str = '3M_10Y',
                      seed: int = 7,
                      include_bonds: bool = False) -> None:
    """Sweep HW mean reversion a over [a_values] and report DVA across the sweep.
    DVA analogue computed as sum DF*ENE*dPD*LGD on the same credit curve (our default likelihood).
    """
    import pandas as pd
    if a_values is None:
        a_values = [0.0, 0.01, 0.03, 0.05, 0.10]
    ts = build_curve(0.02)
    ql.Settings.instance().evaluationDate = ts.referenceDate()
    times = make_time_grid(grid, ts.referenceDate())
    lgd = 0.60
    cc = build_credit_curve(0.015)
    port = build_linear_portfolio(ts, include_bonds=include_bonds)

    rows: List[Dict[str, float]] = []

    for a in a_values:
        a_eff = max(a, 1e-6)
        atm = 0.20
        hw = CalibratedHW.from_atm_level(ts, atm_vol=atm, a=a_eff)
        # Pathwise MTM
        mtm_paths = mtm_paths_from_model(port, ts, hw, simulate_g2.__globals__['simulate_hw'], times, paths, seed)
        # ENE profile and DVA
        ene = np.mean(np.maximum(-mtm_paths, 0.0), axis=0)
        # DVA = sum DF * ENE * dPD * LGD
        t = np.array(times, dtype=float)
        dPD = cc.dPD(times)
        df0 = np.array([ts.discount(float(ti)) for ti in t])
        dva = float(np.sum(df0 * ene * dPD) * lgd)
        rows.append({'a': a, 'a_used': a_eff, 'DVA': dva})

    pd.DataFrame(rows).to_csv(out_path, index=False)


def run_cva_mr_sweep(out_path: str = 'output/cva_mr_sweep.csv',
                      a_values: List[float] | None = None,
                      paths: int = 4096,
                      grid: str = '3M_10Y',
                      seed: int = 7,
                      include_bonds: bool = False) -> None:
    import pandas as pd
    if a_values is None:
        a_values = [0.0, 0.01, 0.03, 0.05, 0.10]
    ts = build_curve(0.02)
    ql.Settings.instance().evaluationDate = ts.referenceDate()
    times = make_time_grid(grid, ts.referenceDate())
    lgd = 0.60
    cc = build_credit_curve(0.015)
    port = build_linear_portfolio(ts, include_bonds=include_bonds)

    rows: List[Dict[str, float]] = []

    for a in a_values:
        a_eff = max(a, 1e-6)
        atm = 0.20
        hw = CalibratedHW.from_atm_level(ts, atm_vol=atm, a=a_eff)
        epe = epe_profile_from_model(port, ts, hw, simulate_g2.__globals__['simulate_hw'], times, paths, seed)
        cva = cva_unilateral(epe, times, ts, cc, lgd)
        rows.append({'a': a, 'a_used': a_eff, 'CVA': float(cva)})

    pd.DataFrame(rows).to_csv(out_path, index=False)
