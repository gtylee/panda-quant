#!/usr/bin/env python3
import os
import sys
import json
from typing import List, Tuple, Dict

import numpy as np

# Ensure project root on path
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from product_definitions import reconstruct_product_static
from pricers import create_pricer
from scenario_generator import SimpleRandomScenarioGenerator
from tff_approximator import TensorFunctionalFormCalibrate
from rbfi_approximator import RBFICalibrate


def load_portfolio(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    statics = [reconstruct_product_static(d) for d in data]
    return statics


def build_pricers(statics) -> List:
    pricers = []
    for st in statics:
        cls = st.__class__.__name__
        if cls == 'CallableBondStaticBase':
            cfg = {'pricer_module_name': 'quantlib_bond_pricer', 'pricer_class_name': 'QuantLibBondPricer', 'pricer_params': {'method': 'g2', 'grid_steps': 64}}
        elif cls == 'ConvertibleBondStaticBase':
            cfg = {'pricer_module_name': 'quantlib_bond_pricer', 'pricer_class_name': 'QuantLibBondPricer', 'pricer_params': {'method': 'discount', 'convertible_engine_steps': 64}}
        else:
            cfg = {'pricer_module_name': 'quantlib_bond_pricer', 'pricer_class_name': 'QuantLibBondPricer', 'pricer_params': {'method': 'discount'}}
        pricers.append(create_pricer(st, cfg))
    return pricers


def build_rate_factors(pillars: np.ndarray, currency: str = 'USD', index_stub: str = 'GENERIC_IR') -> List[str]:
    return [f"{currency}_{index_stub}_{t:.2f}Y" for t in pillars]


def generate_portfolio_scenarios(
    rate_pillars: np.ndarray,
    base_rates: np.ndarray,
    include_equity: bool = True,
    num_scenarios: int = 500,
    seed: int = 999
) -> Tuple[np.ndarray, List[str]]:
    rate_names = build_rate_factors(rate_pillars)
    factors = list(rate_names)
    base_rates_map = {n: float(v) for n, v in zip(rate_names, base_rates)}

    kwargs = dict(
        base_rates_map=base_rates_map,
        rate_factor_shock_std_dev_map={n: 0.002 for n in rate_names},
        random_seed=seed,
    )

    if include_equity:
        # For the convertible in demo portfolio (AAPL as placeholder)
        kwargs.update(
            base_s0_map={'USD_AAPL_S0': 100.0, 'USD_AAPL_DIVYIELD': 0.01, 'USD_AAPL_CS': 0.002},
            s0_shock_config_map={'USD_AAPL_S0': ('normal_relative', 0.02), 'USD_AAPL_DIVYIELD': ('normal_absolute', 0.002), 'USD_AAPL_CS': ('normal_absolute', 0.0005)},
            base_vol_map={'USD_AAPL_VOL': 0.25},
            vol_shock_config_map={'USD_AAPL_VOL': ('normal_relative', 0.10)},
        )
        factors += ['USD_AAPL_S0', 'USD_AAPL_DIVYIELD', 'USD_AAPL_VOL', 'USD_AAPL_CS']

    sg = SimpleRandomScenarioGenerator(**kwargs)
    scen, names = sg.generate_scenarios(num_scenarios, target_factor_names=factors)
    return scen, names


def price_portfolio(
    statics,
    pricers,
    scenarios: np.ndarray,
    factor_names: List[str],
    rate_pillars: np.ndarray,
) -> np.ndarray:
    g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    conv_kwargs = {'s0_val': 100.0, 'dividend_yield': 0.01, 'equity_volatility': 0.25, 'credit_spread': 0.002}

    prices = []
    for st, pr in zip(statics, pricers):
        cls = st.__class__.__name__
        if cls == 'ConvertibleBondStaticBase':
            vals = pr.price_scenarios(scenarios, factor_names, rate_pillars=rate_pillars, **conv_kwargs)
        elif cls == 'CallableBondStaticBase':
            vals = pr.price_scenarios(scenarios, factor_names, rate_pillars=rate_pillars, g2_params=g2_params)
        else:
            vals = pr.price_scenarios(scenarios, factor_names, rate_pillars=rate_pillars)
        prices.append(vals)
    port = np.sum(np.vstack(prices), axis=0)
    return port


def calibrate_surrogates_for_vanilla(
    statics,
    pricers,
    rate_pillars: np.ndarray,
    base_rates: np.ndarray,
    n_train: int = 48,
    n_test: int = 8,
) -> Tuple[Dict[int, object], Dict[int, object]]:
    tff_models: Dict[int, object] = {}
    rbfi_models: Dict[int, object] = {}
    names_v = build_rate_factors(rate_pillars)

    for i, (st, pr) in enumerate(zip(statics, pricers)):
        if st.__class__.__name__ != 'QuantLibBondStaticBase':
            continue
        base_v = base_rates.copy()
        # Create domain ensuring per-dimension bounds are valid
        halfwidth = 0.002
        extremes = np.vstack([base_v + halfwidth, base_v - halfwidth])
        rng = np.random.default_rng(111 + i)
        rand = np.vstack([base_v + rng.normal(0.0, halfwidth / 2.0, size=base_v.shape) for _ in range(max(n_train, 64) - extremes.shape[0])])
        domain_v = np.vstack([extremes, rand])

        tff_cal = TensorFunctionalFormCalibrate(
            pricer_template=pr,
            tff_input_raw_factor_names=names_v,
            tff_input_raw_base_values=base_v,
            product_static_params_for_worker=st.to_dict(),
            pricer_config_for_worker={'bond_pricer_config': {'method': 'discount'}},
            actual_rate_pillars=rate_pillars,
        )
        tff_model, *_ = tff_cal.sample_and_fit(
            full_market_scenarios_for_tff_factors=domain_v,
            n_train=n_train, n_test=n_test, random_seed=5, order=2
        )
        tff_models[i] = tff_model

        rbfi_cal = RBFICalibrate(
            pricer_template=pr,
            rbfi_input_raw_factor_names=names_v,
            rbfi_input_raw_base_values=base_v,
            product_static_params_for_worker=st.to_dict(),
            pricer_config_for_worker={'bond_pricer_config': {'method': 'discount'}},
            actual_rate_pillars=rate_pillars,
        )
        rbfi_model, *_ = rbfi_cal.sample_and_fit(
            full_market_scenarios_for_rbfi_factors=domain_v,
            n_train=n_train, n_test=n_test, random_seed=5
        )
        rbfi_models[i] = rbfi_model

    return tff_models, rbfi_models


def price_portfolio_surrogate(
    statics,
    pricers,
    scenarios: np.ndarray,
    factor_names: List[str],
    rate_pillars: np.ndarray,
    models: Dict[int, object],
    method_name: str,
) -> np.ndarray:
    g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    conv_kwargs = {'s0_val': 100.0, 'dividend_yield': 0.01, 'equity_volatility': 0.25, 'credit_spread': 0.002}

    port_prices = np.zeros(scenarios.shape[0])
    # Indices for rate factors used by surrogates
    rate_factor_names = build_rate_factors(rate_pillars)
    rate_indices = [factor_names.index(n) for n in rate_factor_names if n in factor_names]
    for idx, (st, pr) in enumerate(zip(statics, pricers)):
        if idx in models:
            model = models[idx]
            x = scenarios[:, rate_indices]
            port_prices += np.atleast_1d(model(x))
        else:
            cls = st.__class__.__name__
            if cls == 'ConvertibleBondStaticBase':
                vals = pr.price_scenarios(scenarios, factor_names, rate_pillars=rate_pillars, **conv_kwargs)
            elif cls == 'CallableBondStaticBase':
                vals = pr.price_scenarios(scenarios, factor_names, rate_pillars=rate_pillars, g2_params=g2_params)
            else:
                vals = pr.price_scenarios(scenarios, factor_names, rate_pillars=rate_pillars)
            port_prices += vals
    return port_prices


def compute_var(prices: np.ndarray, alpha: float = 0.95) -> float:
    base = prices[0]
    pnl = prices - base
    q = 100.0 * (1.0 - alpha)
    return float(np.percentile(pnl, q))


def main():
    # Configuration
    json_path = os.path.join(PROJ_ROOT, 'notebooks', 'data', 'portfolio_small.json')
    # Base curve
    pillars = np.array([0.50, 1.00, 2.00, 5.00, 10.00], dtype=float)
    rates = np.array([0.02, 0.022, 0.024, 0.026, 0.028], dtype=float)

    statics = load_portfolio(json_path)
    pricers = build_pricers(statics)

    # Scenarios include equity inputs for convertibles
    scen, names = generate_portfolio_scenarios(pillars, rates, include_equity=True, num_scenarios=500, seed=999)

    # Baseline pricing
    baseline_prices = price_portfolio(statics, pricers, scen, names, pillars)
    baseline_var = compute_var(baseline_prices, alpha=0.95)

    # Calibrate surrogates for vanilla
    tff_models, rbfi_models = calibrate_surrogates_for_vanilla(statics, pricers, pillars, rates, n_train=48, n_test=8)

    # TFF portfolio (vanilla via TFF, others full)
    tff_prices = price_portfolio_surrogate(statics, pricers, scen, names, pillars, tff_models, 'tff')
    tff_var = compute_var(tff_prices, alpha=0.95)

    # RBFI portfolio (vanilla via RBFI, others full)
    rbfi_prices = price_portfolio_surrogate(statics, pricers, scen, names, pillars, rbfi_models, 'rbfi')
    rbfi_var = compute_var(rbfi_prices, alpha=0.95)

    # Hybrid can be same as TFF portfolio in this setup (vanilla via surrogates, others full)
    hybrid_var = tff_var

    print("Portfolio VaR 95% (baseline vs TFF vs RBFI vs Hybrid):")
    print({
        'baseline_var95': baseline_var,
        'tff_var95': tff_var,
        'rbfi_var95': rbfi_var,
        'hybrid_var95': hybrid_var,
    })


if __name__ == '__main__':
    main()


