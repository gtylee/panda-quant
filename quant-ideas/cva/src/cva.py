from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
import QuantLib as ql

from credit import CreditCurve


def cva_unilateral(epe: np.ndarray,
                   times: List[float],
                   ts: ql.YieldTermStructureHandle,
                   cc: CreditCurve,
                   lgd: float) -> float:
    t = np.array(times, dtype=float)
    dPD = cc.dPD(times)
    # Discount factors DF(0,t)
    try:
        df0 = np.array([ts.discount(ti) for ti in t])
    except TypeError:
        ref = ts.referenceDate()
        df0 = np.array([ts.discount(ref + ql.Period(int(round(365 * ti)), ql.Days)) for ti in t])
    return float(np.sum(df0 * epe * dPD) * lgd)


def compare_series(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    denom = np.maximum(eps, np.maximum(np.abs(a), np.abs(b)))
    abs_pct = np.abs(a - b) / denom
    mape = float(np.mean(abs_pct) * 100.0)
    max_abs = float(np.max(abs_pct) * 100.0)
    return {"MAPE_%": mape, "MAX_abs_%": max_abs}


def ee_profile_stats(epe: np.ndarray, times: List[float], t1: float = 1.0, t2: float = 10.0) -> Dict[str, float]:
    t = np.array(times, dtype=float)
    mask = (t >= t1) & (t <= t2)
    if not np.any(mask):
        return {"EE_mean": float(np.mean(epe)), "EE_std": float(np.std(epe))}
    segment = epe[mask]
    return {"EE_mean": float(np.mean(segment)), "EE_std": float(np.std(segment))}


def print_summary(report: Dict[str, float], epe_stats: Dict[str, float], pass_cva: bool, pass_epe: bool) -> None:
    print("\n==== CVA Model Equivalence Report ====")
    print(f"CVA_HW: {report['CVA_HW']:.6f}")
    print(f"CVA_G2: {report['CVA_G2']:.6f}")
    print(f"CVA_abs_diff: {report['CVA_abs_diff']:.6f}")
    print(f"CVA_rel_diff_%: {report['CVA_rel_diff_%']:.4f}%")
    print(f"EPE MAPE%: {epe_stats['MAPE_%']:.4f}%  |  EPE MAX abs%: {epe_stats['MAX_abs_%']:.4f}%")
    print(f"PASS_CVA: {'PASS' if pass_cva else 'FAIL'}")
    print(f"PASS_EPE: {'PASS' if pass_epe else 'FAIL'}")


def write_csvs(epe_hw: np.ndarray, epe_g2: np.ndarray, times: List[float], report: Dict[str, float], out_dir: str = 'output') -> None:
    import os
    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame({"time": times, "EPE_HW": epe_hw, "EPE_G2": epe_g2})
    df.to_csv(os.path.join(out_dir, 'epe_profiles.csv'), index=False)
    pd.DataFrame([report]).to_csv(os.path.join(out_dir, 'cva_report.csv'), index=False)


def write_calibration_csv(rows: List[dict], path: str) -> None:
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
