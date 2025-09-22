from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict

import numpy as np
import QuantLib as ql
from scipy import optimize as opt
from scipy.stats import qmc, norm


def _build_helpers(ts: ql.YieldTermStructureHandle, flat_atm_vol: float) -> List[ql.SwaptionHelper]:
    index = ql.Euribor6M(ts)
    mat_ten = [
        (ql.Period(1, ql.Years), ql.Period(5, ql.Years)),
        (ql.Period(2, ql.Years), ql.Period(5, ql.Years)),
        (ql.Period(5, ql.Years), ql.Period(10, ql.Years)),
        (ql.Period(10, ql.Years), ql.Period(10, ql.Years)),
    ]
    helpers: List[ql.SwaptionHelper] = []
    for maturity, length in mat_ten:
        helper = ql.SwaptionHelper(
            maturity,
            length,
            ql.QuoteHandle(ql.SimpleQuote(flat_atm_vol)),
            index,
            index.tenor(),
            index.dayCounter(),
            index.dayCounter(),
            ts
        )
        helpers.append(helper)
    return helpers


def _df(ts: ql.YieldTermStructureHandle, t: float) -> float:
    return float(ts.discount(t))


# ---- SciPy-based simple calibrators ----

def _targets_from_helpers(helpers: List[ql.SwaptionHelper], atm_vol: float) -> np.ndarray:
    return np.array([h.blackPrice(atm_vol) for h in helpers], dtype=float)


def calibrate_hw_sigma(ts: ql.YieldTermStructureHandle, a: float, atm_vol: float) -> float:
    helpers = _build_helpers(ts, atm_vol)
    targets = _targets_from_helpers(helpers, atm_vol)

    def residuals(theta: np.ndarray) -> np.ndarray:
        sigma = float(theta[0])
        model = ql.HullWhite(ts, a, sigma)
        engine = ql.JamshidianSwaptionEngine(model)
        vals = []
        for h in helpers:
            h.setPricingEngine(engine)
            vals.append(h.modelValue())
        vals = np.array(vals, dtype=float)
        return (vals - targets) / np.maximum(1e-10, targets)

    x0 = np.array([max(1e-4, 0.01 * atm_vol)])
    bounds = (np.array([1e-6]), np.array([2.0]))
    res = opt.least_squares(residuals, x0, bounds=bounds, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)
    return float(res.x[0])


def calibrate_g2_sigmas(ts: ql.YieldTermStructureHandle, a: float, b: float, rho: float, atm_vol: float) -> tuple[float, float]:
    helpers = _build_helpers(ts, atm_vol)
    targets = _targets_from_helpers(helpers, atm_vol)

    def residuals(theta: np.ndarray) -> np.ndarray:
        sigma = float(theta[0])
        eta = float(theta[1])
        model = ql.G2(ts, a, sigma, b, eta, rho)
        engine = ql.G2SwaptionEngine(model, 6.0, 32)
        vals = []
        for h in helpers:
            h.setPricingEngine(engine)
            vals.append(h.modelValue())
        vals = np.array(vals, dtype=float)
        return (vals - targets) / np.maximum(1e-10, targets)

    x0 = np.array([max(1e-4, 0.01 * atm_vol), max(1e-4, 0.005 * atm_vol)])
    bounds = (np.array([1e-6, 1e-6]), np.array([2.0, 2.0]))
    res = opt.least_squares(residuals, x0, bounds=bounds, xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=400)
    return float(res.x[0]), float(res.x[1])


@dataclass
class CalibratedHW:
    ts: ql.YieldTermStructureHandle
    a: float = 0.03
    sigma: float = 0.01

    @classmethod
    def from_atm_level(cls, ts: ql.YieldTermStructureHandle, atm_vol: float, a: float = 0.03) -> 'CalibratedHW':
        sigma = calibrate_hw_sigma(ts, a, atm_vol)
        return cls(ts=ts, a=a, sigma=sigma)

    @property
    def model(self) -> ql.HullWhite:
        return ql.HullWhite(self.ts, self.a, self.sigma)

    def zcb_closed(self, t: float, T: float, x_t: float) -> float:
        h = max(0.0, T - t)
        if h == 0.0:
            return 1.0
        a = self.a
        sigma = self.sigma
        B = (1.0 - math.exp(-a * h)) / a if a > 1e-12 else h
        term1 = h
        term2 = (2.0 / a) * (1.0 - math.exp(-a * h)) if a > 1e-12 else 0.0
        term3 = (1.0 / (2.0 * a)) * (1.0 - math.exp(-2.0 * a * h)) if a > 1e-12 else 0.0
        Var = (sigma * sigma / (a * a)) * (term1 - term2 + term3)
        ratio = _df(self.ts, T) / _df(self.ts, t)
        return ratio * math.exp(-B * x_t + 0.5 * Var)

    def zcb(self, t: float, T: float, x_t: float) -> float:
        return self.zcb_closed(t, T, x_t)

    def with_sigma(self, sigma: float) -> 'CalibratedHW':
        return CalibratedHW(ts=self.ts, a=self.a, sigma=sigma)


@dataclass
class CalibratedG2:
    ts: ql.YieldTermStructureHandle
    a: float = 0.03
    b: float = 0.10
    rho: float = -0.75
    sigma: float = 0.01
    eta: float = 1e-12

    @classmethod
    def from_atm_level(cls, ts: ql.YieldTermStructureHandle, atm_vol: float, a: float = 0.03, b: float = 0.10, rho: float = -0.75) -> 'CalibratedG2':
        sigma, eta = calibrate_g2_sigmas(ts, a, b, rho, atm_vol)
        return cls(ts=ts, a=a, b=b, rho=rho, sigma=sigma, eta=eta)

    @property
    def model(self) -> ql.G2:
        return ql.G2(self.ts, self.a, self.sigma, self.b, self.eta, self.rho)

    def zcb_closed(self, t: float, T: float, x_t: float, y_t: float) -> float:
        h = max(0.0, T - t)
        if h == 0.0:
            return 1.0
        a, b, rho = self.a, self.b, self.rho
        sigma, eta = self.sigma, self.eta
        Bx = (1.0 - math.exp(-a * h)) / a if a > 1e-12 else h
        By = (1.0 - math.exp(-b * h)) / b if b > 1e-12 else h
        def V_single(kappa: float, vol: float) -> float:
            if kappa <= 1e-12:
                return (vol * vol) * (h**3) / 3.0
            term1 = h
            term2 = (2.0 / kappa) * (1.0 - math.exp(-kappa * h))
            term3 = (1.0 / (2.0 * kappa)) * (1.0 - math.exp(-2.0 * kappa * h))
            return (vol * vol / (kappa * kappa)) * (term1 - term2 + term3)
        def Cov_xy() -> float:
            if a <= 1e-12 or b <= 1e-12:
                return 0.0
            term = h - (1.0 - math.exp(-a * h)) / a - (1.0 - math.exp(-b * h)) / b + (1.0 - math.exp(-(a + b) * h)) / (a + b)
            return (rho * sigma * eta / (a * b)) * term
        Var = V_single(a, sigma) + V_single(b, eta) + 2.0 * Cov_xy()
        ratio = _df(self.ts, T) / _df(self.ts, t)
        return ratio * math.exp(-Bx * x_t - By * y_t + 0.5 * Var)

    def zcb(self, t: float, T: float, x_t: float, y_t: float) -> float:
        return self.zcb_closed(t, T, x_t, y_t)

    def with_sigmas(self, sigma: float, eta: float) -> 'CalibratedG2':
        return CalibratedG2(ts=self.ts, a=self.a, b=self.b, rho=self.rho, sigma=sigma, eta=eta)


# ---- Simulation helpers ----

def _build_time_steps(times: list[float]) -> np.ndarray:
    t = np.array(times, dtype=float)
    if np.any(np.diff(t) <= 0):
        raise ValueError("times must be strictly increasing")
    return t


def _sobol_normals(n_paths: int, n_steps: int, seed: int) -> np.ndarray:
    # Scrambled Sobol in [0,1]^d then inverse CDF to N(0,1)
    sampler = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
    u = sampler.random(n=n_paths)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    z = norm.ppf(u, loc=0.0, scale=1.0)
    return z.astype(float)


def simulate_hw(hw: CalibratedHW, times: list[float], n_paths: int, seed: int = 42, antithetic: bool = True, use_sobol: bool = True) -> np.ndarray:
    t = _build_time_steps(times)
    num_steps = t.shape[0]

    a = hw.a
    sigma = hw.sigma

    def step_params(dt: float) -> Tuple[float, float]:
        phi = math.exp(-a * dt)
        sd = sigma * math.sqrt((1.0 - math.exp(-2.0 * a * dt)) / (2.0 * a)) if a > 1e-12 else sigma * math.sqrt(dt)
        return phi, sd

    if use_sobol:
        Z = _sobol_normals(n_paths, num_steps, seed)
        if antithetic:
            Z = np.vstack([Z, -Z])
        n_paths_eff = Z.shape[0]
        x = np.zeros((n_paths_eff, num_steps), dtype=float)
        x_curr = np.zeros(n_paths_eff, dtype=float)
        t_prev = 0.0
        for j in range(num_steps):
            dt = (t[j] - t_prev)
            phi, sd = step_params(dt)
            x_curr = phi * x_curr + sd * Z[:, j]
            x[:, j] = x_curr
            t_prev = t[j]
        return x[:n_paths, :]  # trim to requested paths

    rng = np.random.default_rng(seed)

    def sim_block(n_sim: int) -> np.ndarray:
        x = np.zeros((n_sim, num_steps), dtype=float)
        x_curr = np.zeros(n_sim, dtype=float)
        t_prev = 0.0
        for j in range(num_steps):
            dt = (t[j] - t_prev)
            phi, sd = step_params(dt)
            z = rng.standard_normal(n_sim)
            x_curr = phi * x_curr + sd * z
            x[:, j] = x_curr
            t_prev = t[j]
        return x

    if antithetic:
        n_half = n_paths // 2
        extra = n_paths - 2 * n_half
    else:
        n_half = n_paths
        extra = 0

    blocks = []
    if n_half > 0:
        xb = sim_block(n_half)
        blocks.append(xb)
        if antithetic:
            blocks.append(-xb)
    if extra > 0:
        blocks.append(sim_block(extra))
    x_all = np.vstack(blocks)
    return x_all


def simulate_g2(g2: CalibratedG2, times: list[float], n_paths: int, seed: int = 42, antithetic: bool = True, use_sobol: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    t = _build_time_steps(times)
    num_steps = t.shape[0]

    a, b, rho = g2.a, g2.b, g2.rho
    sigma, eta = g2.sigma, g2.eta

    def step_params(kappa: float, vol: float, dt: float) -> Tuple[float, float]:
        phi = math.exp(-kappa * dt)
        sd = vol * math.sqrt((1.0 - math.exp(-2.0 * kappa * dt)) / (2.0 * kappa)) if kappa > 1e-12 else vol * math.sqrt(dt)
        return phi, sd

    if use_sobol:
        Z1 = _sobol_normals(n_paths, num_steps, seed)
        Z2 = _sobol_normals(n_paths, num_steps, seed + 17)
        if antithetic:
            Z1 = np.vstack([Z1, -Z1])
            Z2 = np.vstack([Z2, -Z2])
        n_paths_eff = Z1.shape[0]
        x = np.zeros((n_paths_eff, num_steps), dtype=float)
        y = np.zeros((n_paths_eff, num_steps), dtype=float)
        x_curr = np.zeros(n_paths_eff, dtype=float)
        y_curr = np.zeros(n_paths_eff, dtype=float)
        t_prev = 0.0
        for j in range(num_steps):
            dt = (t[j] - t_prev)
            phix, sdx = step_params(a, sigma, dt)
            phiy, sdy = step_params(b, eta, dt)
            u1 = Z1[:, j]
            u2 = Z2[:, j]
            z1 = u1
            z2 = rho * u1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * u2
            x_curr = phix * x_curr + sdx * z1
            y_curr = phiy * y_curr + sdy * z2
            x[:, j] = x_curr
            y[:, j] = y_curr
            t_prev = t[j]
        return x[:n_paths, :], y[:n_paths, :]

    rng = np.random.default_rng(seed)

    def sim_block(n_sim: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.zeros((n_sim, num_steps), dtype=float)
        y = np.zeros((n_sim, num_steps), dtype=float)
        x_curr = np.zeros(n_sim, dtype=float)
        y_curr = np.zeros(n_sim, dtype=float)
        t_prev = 0.0
        for j in range(num_steps):
            dt = (t[j] - t_prev)
            phix, sdx = step_params(a, sigma, dt)
            phiy, sdy = step_params(b, eta, dt)
            u1 = rng.standard_normal(n_sim)
            u2 = rng.standard_normal(n_sim)
            z1 = u1
            z2 = rho * u1 + math.sqrt(max(0.0, 1.0 - rho * rho)) * u2
            x_curr = phix * x_curr + sdx * z1
            y_curr = phiy * y_curr + sdy * z2
            x[:, j] = x_curr
            y[:, j] = y_curr
            t_prev = t[j]
        return x, y

    if antithetic:
        n_half = n_paths // 2
        extra = n_paths - 2 * n_half
    else:
        n_half = n_paths
        extra = 0

    xs, ys = [], []
    if n_half > 0:
        xb, yb = sim_block(n_half)
        xs.append(xb)
        ys.append(yb)
        if antithetic:
            xs.append(-xb)
            ys.append(-yb)
    if extra > 0:
        xb, yb = sim_block(extra)
        xs.append(xb)
        ys.append(yb)

    x_all = np.vstack(xs)
    y_all = np.vstack(ys)
    return x_all, y_all


# ---- Calibration fits (price-based) ----

def calibration_fit_rows_hw(hw: CalibratedHW, atm_vol: float) -> List[Dict[str, float]]:
    helpers = _build_helpers(hw.ts, atm_vol)
    rows: List[Dict[str, float]] = []
    ref = hw.ts.referenceDate()
    dc = hw.ts.dayCounter()
    for h in helpers:
        target_price = h.blackPrice(atm_vol)
        h.setPricingEngine(ql.JamshidianSwaptionEngine(hw.model))
        model_price = h.modelValue()
        swpt = h.swaption()
        mat_date = swpt.exercise().dates()[0]
        swap = swpt.underlying()
        end_date = swap.fixedSchedule().endDate()
        maturityY = dc.yearFraction(ref, mat_date)
        tenorY = dc.yearFraction(mat_date, end_date)
        rel_err = (model_price - target_price) / max(1e-12, target_price)
        rows.append({
            "maturityY": float(maturityY),
            "tenorY": float(tenorY),
            "target_vol": atm_vol,
            "target_price": float(target_price),
            "model_price": float(model_price),
            "rel_price_err_%": float(100.0 * rel_err)
        })
    return rows


def calibration_fit_rows_g2(g2: CalibratedG2, atm_vol: float) -> List[Dict[str, float]]:
    helpers = _build_helpers(g2.ts, atm_vol)
    rows: List[Dict[str, float]] = []
    ref = g2.ts.referenceDate()
    dc = g2.ts.dayCounter()
    for h in helpers:
        target_price = h.blackPrice(atm_vol)
        h.setPricingEngine(ql.G2SwaptionEngine(g2.model, 6.0, 32))
        model_price = h.modelValue()
        swpt = h.swaption()
        mat_date = swpt.exercise().dates()[0]
        swap = swpt.underlying()
        end_date = swap.fixedSchedule().endDate()
        maturityY = dc.yearFraction(ref, mat_date)
        tenorY = dc.yearFraction(mat_date, end_date)
        rel_err = (model_price - target_price) / max(1e-12, target_price)
        rows.append({
            "maturityY": float(maturityY),
            "tenorY": float(tenorY),
            "target_vol": atm_vol,
            "target_price": float(target_price),
            "model_price": float(model_price),
            "rel_price_err_%": float(100.0 * rel_err)
        })
    return rows
