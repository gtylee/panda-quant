from __future__ import annotations
from typing import List, Callable
import numpy as np
import QuantLib as ql

from portfolio import Portfolio, PayerParSwap, ParBond
from models import CalibratedHW, CalibratedG2, simulate_hw, simulate_g2


RatioMultiplier = Callable[[float, np.ndarray], np.ndarray]  # (t, T_vector)->multiplier per T


def _hw_base_terms(model: CalibratedHW, t: float, T: np.ndarray, ratio_mult: RatioMultiplier | None = None) -> tuple[np.ndarray, np.ndarray]:
    h = np.maximum(0.0, T - t)
    a = model.a
    sigma = model.sigma
    B = np.where(a > 1e-12, (1.0 - np.exp(-a * h)) / a, h)
    term1 = h
    term2 = np.where(a > 1e-12, (2.0 / a) * (1.0 - np.exp(-a * h)), 0.0)
    term3 = np.where(a > 1e-12, (1.0 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * h)), 0.0)
    Var = (sigma * sigma / (a * a)) * (term1 - term2 + term3)
    ratio = np.array([model.ts.discount(float(Ti)) / model.ts.discount(float(t)) for Ti in T])
    base = ratio * np.exp(0.5 * Var)
    if ratio_mult is not None:
        base = base * ratio_mult(t, T)
    return B, base


def _g2_base_terms(model: CalibratedG2, t: float, T: np.ndarray, ratio_mult: RatioMultiplier | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.maximum(0.0, T - t)
    a, b, rho = model.a, model.b, model.rho
    sigma, eta = model.sigma, model.eta
    Bx = np.where(a > 1e-12, (1.0 - np.exp(-a * h)) / a, h)
    By = np.where(b > 1e-12, (1.0 - np.exp(-b * h)) / b, h)

    def V_single(kappa: float, vol: float) -> np.ndarray:
        if kappa <= 1e-12:
            return (vol * vol) * (h ** 3) / 3.0
        term1 = h
        term2 = (2.0 / kappa) * (1.0 - np.exp(-kappa * h))
        term3 = (1.0 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * h))
        return (vol * vol / (kappa * kappa)) * (term1 - term2 + term3)

    if a <= 1e-12 or b <= 1e-12:
        Cov = np.zeros_like(h)
    else:
        term = h - (1.0 - np.exp(-a * h)) / a - (1.0 - np.exp(-b * h)) / b + (1.0 - np.exp(-(a + b) * h)) / (a + b)
        Cov = (rho * sigma * eta / (a * b)) * term

    Var = V_single(a, sigma) + V_single(b, eta) + 2.0 * Cov
    ratio = np.array([model.ts.discount(float(Ti)) / model.ts.discount(float(t)) for Ti in T])
    base = ratio * np.exp(0.5 * Var)
    if ratio_mult is not None:
        base = base * ratio_mult(t, T)
    return Bx, By, base


def epe_profile_from_model(port: Portfolio,
                           ts: ql.YieldTermStructureHandle,
                           model,
                           simulate_fn: Callable,
                           times: List[float],
                           n_paths: int,
                           seed: int = 42,
                           antithetic: bool = True) -> np.ndarray:
    return epe_profile_from_model_with_ratio_mult(port, ts, model, simulate_fn, times, n_paths, seed, antithetic, ratio_mult=None)


def epe_profile_from_model_with_ratio_mult(port: Portfolio,
                                           ts: ql.YieldTermStructureHandle,
                                           model,
                                           simulate_fn: Callable,
                                           times: List[float],
                                           n_paths: int,
                                           seed: int = 42,
                                           antithetic: bool = True,
                                           ratio_mult: RatioMultiplier | None = None) -> np.ndarray:
    # Simulate and delegate to states-based API
    if isinstance(model, CalibratedHW):
        X = simulate_fn(model, times, n_paths, seed, antithetic)
        Y = None
    elif isinstance(model, CalibratedG2):
        X, Y = simulate_fn(model, times, n_paths, seed, antithetic)
    else:
        raise TypeError("Unsupported model type")
    return epe_profile_from_states_with_ratio_mult(port, model, times, X, Y, ratio_mult)


def epe_profile_from_states_with_ratio_mult(port: Portfolio,
                                            model,
                                            times: List[float],
                                            X: np.ndarray,
                                            Y: np.ndarray | None,
                                            ratio_mult: RatioMultiplier | None) -> np.ndarray:
    t = np.array(times, dtype=float)
    n_paths = X.shape[0]
    epe = np.zeros_like(t)
    instruments = port.instruments

    for j, tj in enumerate(t):
        xj = X[:, j]
        yj = Y[:, j] if Y is not None else None

        pv_paths = np.zeros(n_paths, dtype=float)

        for inst in instruments:
            if isinstance(inst, PayerParSwap):
                T_all = np.array([Ti for Ti in (inst.fixed_payment_times or []) if Ti > tj], dtype=float)
                if T_all.size == 0:
                    continue
                tau = np.array([tau for tau, Ti in zip(inst.fixed_accruals or [], inst.fixed_payment_times or []) if Ti > tj], dtype=float)
                if isinstance(model, CalibratedHW):
                    B, base = _hw_base_terms(model, tj, T_all, ratio_mult)
                    Z = np.exp(-np.outer(xj, B)) * base
                    B_end, base_end = _hw_base_terms(model, tj, np.array([float(inst.T_end)]), ratio_mult)
                    P_end = np.exp(-xj * B_end[0]) * base_end[0]
                else:
                    Bx, By, base = _g2_base_terms(model, tj, T_all, ratio_mult)
                    Z = np.exp(-np.outer(xj, Bx) - np.outer(yj, By)) * base
                    Bx_end, By_end, base_end = _g2_base_terms(model, tj, np.array([float(inst.T_end)]), ratio_mult)
                    P_end = np.exp(-xj * Bx_end[0] - (yj if yj is not None else 0.0) * By_end[0]) * base_end[0]

                A_vec = Z @ tau
                float_pv = 1.0 - P_end
                fixed_pv = (inst.fixed_rate or 0.0) * A_vec
                pv_paths += inst.notional * (float_pv - fixed_pv)

            elif isinstance(inst, ParBond):
                T_all = np.array([Ti for Ti in (inst.payment_times or []) if Ti > tj], dtype=float)
                if T_all.size == 0:
                    continue
                tau = np.array([tau for tau, Ti in zip(inst.accruals or [], inst.payment_times or []) if Ti > tj], dtype=float)
                if isinstance(model, CalibratedHW):
                    B, base = _hw_base_terms(model, tj, T_all, ratio_mult)
                    Z = np.exp(-np.outer(xj, B)) * base
                    B_end, base_end = _hw_base_terms(model, tj, np.array([float(inst.T_end)]), ratio_mult)
                    P_end = np.exp(-xj * B_end[0]) * base_end[0]
                else:
                    Bx, By, base = _g2_base_terms(model, tj, T_all, ratio_mult)
                    Z = np.exp(-np.outer(xj, Bx) - np.outer(yj, By)) * base
                    Bx_end, By_end, base_end = _g2_base_terms(model, tj, np.array([float(inst.T_end)]), ratio_mult)
                    P_end = np.exp(-xj * Bx_end[0] - (yj if yj is not None else 0.0) * By_end[0]) * base_end[0]

                coupon_pv = (inst.coupon_rate or 0.0) * (Z @ tau)
                pv_paths += inst.notional * (coupon_pv + P_end)

        epe[j] = float(np.mean(np.maximum(pv_paths, 0.0)))

    return epe


def bucket_ratio_multiplier(shift_bp: float, bucket_start: float, bucket_end: float) -> RatioMultiplier:
    shift = shift_bp * 1e-4  # convert bp to absolute rate
    def mult(t: float, T: np.ndarray) -> np.ndarray:
        # overlap length between (t, T] and (bucket_start, bucket_end]
        start = np.maximum(t, bucket_start)
        end = np.minimum(T, bucket_end)
        overlap = np.maximum(0.0, end - start)
        return np.exp(-shift * overlap)
    return mult


def epe_paths_from_model(port: Portfolio,
                         ts: ql.YieldTermStructureHandle,
                         model,
                         simulate_fn: Callable,
                         times: List[float],
                         n_paths: int,
                         seed: int = 42,
                         antithetic: bool = True) -> np.ndarray:
    """Return exposures per path per time bucket: shape (n_paths, n_times)."""
    t = np.array(times, dtype=float)
    if isinstance(model, CalibratedHW):
        X = simulate_fn(model, times, n_paths, seed, antithetic)
        Y = None
    elif isinstance(model, CalibratedG2):
        X, Y = simulate_fn(model, times, n_paths, seed, antithetic)
    else:
        raise TypeError("Unsupported model type")

    exposures = np.zeros((X.shape[0], len(t)), dtype=float)
    instruments = port.instruments

    for j, tj in enumerate(t):
        xj = X[:, j]
        yj = Y[:, j] if Y is not None else None

        pv_paths = np.zeros(X.shape[0], dtype=float)

        for inst in instruments:
            if isinstance(inst, PayerParSwap):
                T_all = np.array([Ti for Ti in (inst.fixed_payment_times or []) if Ti > tj], dtype=float)
                if T_all.size == 0:
                    continue
                tau = np.array([tau for tau, Ti in zip(inst.fixed_accruals or [], inst.fixed_payment_times or []) if Ti > tj], dtype=float)
                if isinstance(model, CalibratedHW):
                    B, base = _hw_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, B)) * base
                    B_end, base_end = _hw_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * B_end[0]) * base_end[0]
                else:
                    Bx, By, base = _g2_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, Bx) - np.outer(yj, By)) * base
                    Bx_end, By_end, base_end = _g2_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * Bx_end[0] - (yj if yj is not None else 0.0) * By_end[0]) * base_end[0]
                A_vec = Z @ tau
                float_pv = 1.0 - P_end
                fixed_pv = (inst.fixed_rate or 0.0) * A_vec
                pv_paths += inst.notional * (float_pv - fixed_pv)

            elif isinstance(inst, ParBond):
                T_all = np.array([Ti for Ti in (inst.payment_times or []) if Ti > tj], dtype=float)
                if T_all.size == 0:
                    continue
                tau = np.array([tau for tau, Ti in zip(inst.accruals or [], inst.payment_times or []) if Ti > tj], dtype=float)
                if isinstance(model, CalibratedHW):
                    B, base = _hw_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, B)) * base
                    B_end, base_end = _hw_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * B_end[0]) * base_end[0]
                else:
                    Bx, By, base = _g2_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, Bx) - np.outer(yj, By)) * base
                    Bx_end, By_end, base_end = _g2_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * Bx_end[0] - (yj if yj is not None else 0.0) * By_end[0]) * base_end[0]
                coupon_pv = (inst.coupon_rate or 0.0) * (Z @ tau)
                pv_paths += inst.notional * (coupon_pv + P_end)

        exposures[:, j] = np.maximum(pv_paths, 0.0)

    return exposures


def mtm_paths_from_model(port: Portfolio,
                         ts: ql.YieldTermStructureHandle,
                         model,
                         simulate_fn: Callable,
                         times: List[float],
                         n_paths: int,
                         seed: int = 42,
                         antithetic: bool = True) -> np.ndarray:
    """Return pathwise portfolio MTM per time bucket: shape (n_paths, n_times).
    Positive = exposure to counterparty; negative = our liability (for DVA).
    """
    t = np.array(times, dtype=float)
    if isinstance(model, CalibratedHW):
        X = simulate_fn(model, times, n_paths, seed, antithetic)
        Y = None
    elif isinstance(model, CalibratedG2):
        X, Y = simulate_fn(model, times, n_paths, seed, antithetic)
    else:
        raise TypeError("Unsupported model type")

    mtm = np.zeros((X.shape[0], len(t)), dtype=float)
    instruments = port.instruments

    for j, tj in enumerate(t):
        xj = X[:, j]
        yj = Y[:, j] if Y is not None else None

        pv_paths = np.zeros(X.shape[0], dtype=float)

        for inst in instruments:
            if isinstance(inst, PayerParSwap):
                T_all = np.array([Ti for Ti in (inst.fixed_payment_times or []) if Ti > tj], dtype=float)
                if T_all.size == 0:
                    continue
                tau = np.array([tau for tau, Ti in zip(inst.fixed_accruals or [], inst.fixed_payment_times or []) if Ti > tj], dtype=float)
                if isinstance(model, CalibratedHW):
                    B, base = _hw_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, B)) * base
                    B_end, base_end = _hw_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * B_end[0]) * base_end[0]
                else:
                    Bx, By, base = _g2_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, Bx) - np.outer(yj, By)) * base
                    Bx_end, By_end, base_end = _g2_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * Bx_end[0] - (yj if yj is not None else 0.0) * By_end[0]) * base_end[0]
                A_vec = Z @ tau
                float_pv = 1.0 - P_end
                fixed_pv = (inst.fixed_rate or 0.0) * A_vec
                pv_paths += inst.notional * (float_pv - fixed_pv)

            elif isinstance(inst, ParBond):
                T_all = np.array([Ti for Ti in (inst.payment_times or []) if Ti > tj], dtype=float)
                if T_all.size == 0:
                    continue
                tau = np.array([tau for tau, Ti in zip(inst.accruals or [], inst.payment_times or []) if Ti > tj], dtype=float)
                if isinstance(model, CalibratedHW):
                    B, base = _hw_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, B)) * base
                    B_end, base_end = _hw_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * B_end[0]) * base_end[0]
                else:
                    Bx, By, base = _g2_base_terms(model, tj, T_all, None)
                    Z = np.exp(-np.outer(xj, Bx) - np.outer(yj, By)) * base
                    Bx_end, By_end, base_end = _g2_base_terms(model, tj, np.array([float(inst.T_end)]), None)
                    P_end = np.exp(-xj * Bx_end[0] - (yj if yj is not None else 0.0) * By_end[0]) * base_end[0]
                coupon_pv = (inst.coupon_rate or 0.0) * (Z @ tau)
                pv_paths += inst.notional * (coupon_pv + P_end)

        mtm[:, j] = pv_paths

    return mtm
