import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np
from datetime import date
import os
import QuantLib as ql

from scenario_generator import SimpleRandomScenarioGenerator
from product_definitions_pydantic import ProductStaticRegistry
from product_handlers import ProductHandlerFactory
from approximator_handlers import ApproximatorHandlerFactory
# Import from original workflow manager for PortfolioBuilder and generate_price_strips
from workflow_manager import PortfolioBuilder, generate_price_strips, generate_portfolio_specs_for_serialization
# Import from refactored workflow manager for RefactoredInstrumentProcessor
from workflow import InstrumentProcessor as RefactoredInstrumentProcessor, PortfolioBuilder as RefactoredPortfolioBuilder


def generate_portfolio(num_instruments, val_date):
    """
    Generate a list of instrument specs using Pydantic statics and the registry.
    Alternates between VanillaBond, EuropeanOption, and CallableBond.
    """
    portfolio = []
    for i in range(num_instruments):
        if i % 3 == 0:
            static = ProductStaticRegistry.create_static(
                "VanillaBond",
                valuation_date=val_date,
                maturity_date=val_date.replace(year=val_date.year + 5),
                coupon_rate=0.03,
                face_value=100.0,
                currency="USD",
                index_stub="IR"
            )
            pricer_params = {}
        elif i % 3 == 1:
            static = ProductStaticRegistry.create_static(
                "EuropeanOption",
                valuation_date=val_date,
                expiry_date=val_date.replace(year=val_date.year + 2),
                strike_price=100.0,
                option_type="call",
                underlying_symbol="AAPL",
                currency="USD",
                index_stub="EQ"
            )
            pricer_params = {
                "bs_risk_free_rate": 0.025,
                "bs_dividend_yield": 0.01
            }
        else:
            # CallableBond: 5y maturity, 2 call dates at 2y and 3y, call prices 101, 100.5
            call_dates = [val_date.replace(year=val_date.year + 2).isoformat(), val_date.replace(year=val_date.year + 3).isoformat()]
            call_prices = [101.0, 100.5]
            static = ProductStaticRegistry.create_static(
                "CallableBond",
                valuation_date=val_date,
                maturity_date=val_date.replace(year=val_date.year + 5),
                coupon_rate=0.04,
                face_value=100.0,
                freq=2,
                currency="USD",
                call_dates=call_dates,
                call_prices=call_prices,
                index_stub="IR"
            )
            pricer_params = {"g2_params": (0.01, 0.003, 0.015, 0.006, -0.75), "g2_grid_steps": 32}
        portfolio.append({
            "instrument_id": f"Instr_{i}",
            "product_type": static.product_type,
            "params": static.to_dict(),
            "pricing_preference": "FULL",  # Will be set to TFF/RBFI in the workflow
            "pricer_params": pricer_params,
            "tff_config": {"n_train": 40, "n_test": 8, "seed": i, "order": 2}
        })
    return portfolio


def run_hybrid_var_refactored(
    num_instruments=6,
    num_var_scenarios=1000,
    n_domain_scenarios=2000,
    n_fitting_samples=50,
    hybrid_critical_percentile=0.02,
    approximators=("TFF", "RBFI"),
    random_seed=42
):
    print(f"--- Refactored Hybrid VaR Demo (TFF & RBFI) ---")
    print(f"Approximators: {approximators}")
    np.random.seed(random_seed)
    val_date = date(2025, 5, 18)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    default_g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    default_bs_rfr = 0.025
    default_bs_div = 0.01

    # --- Scenario Generators ---
    base_rates_map = {f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
    base_s0_map = {"USD_AAPL_S0": 100.0}
    base_vol_map = {"USD_AAPL_VOL": 0.25}
    scen_gen_var = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=base_s0_map,
        base_vol_map=base_vol_map, random_seed=random_seed
    )
    var_scenarios, var_factor_names = scen_gen_var.generate_scenarios(num_var_scenarios)

    scen_gen_domain = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=base_s0_map,
        base_vol_map=base_vol_map, random_seed=random_seed + 1
    )
    domain_scenarios, domain_factor_names = scen_gen_domain.generate_scenarios(n_domain_scenarios)

    # --- Portfolio ---
    instrument_definitions = generate_portfolio(num_instruments, val_date)
    holdings_data = [{"client_id": "HybridClient", "instrument_id": d["instrument_id"], "num_holdings": 1000}
                     for d in instrument_definitions]

    # --- Full Revaluation Path ---
    print("\n--- Full Revaluation ---")
    start_time = time.time()
    
    iproc_full = RefactoredInstrumentProcessor(
        scen_gen_var, val_date, tenors, 
        default_g2_params, default_bs_rfr, default_bs_div
    )
    
    full_model_registry = iproc_full.process_instruments(
        instrument_definitions, domain_scenarios, domain_factor_names, batch_size=2
    )
    
    # Build portfolio specs
    portfolio_specs_full = generate_portfolio_specs_for_serialization(
        holdings_data, full_model_registry, instrument_definitions
    )
    
    # Build portfolios
    builder_full = RefactoredPortfolioBuilder(full_model_registry)
    portfolios_full = builder_full.build_portfolios_from_specs(
        portfolio_specs_full, val_date, default_g2_params, default_bs_rfr, default_bs_div
    )
    portfolio_obj_full = portfolios_full["HybridClient"]
    portfolio_values_full = portfolio_obj_full.price_portfolio(var_scenarios, var_factor_names, tenors)
    base_value_full = portfolio_values_full[0]
    losses_full = base_value_full - portfolio_values_full
    sorted_losses_full = np.sort(losses_full)
    N = len(sorted_losses_full)
    idx1 = max(0, int(np.ceil(0.01 * N)) - 1)
    var_1pct_full = -sorted_losses_full[idx1]
    print(f"Full 1% VaR: {var_1pct_full:,.2f}. Base Value: {base_value_full:,.2f}")

    # --- Approximator Paths (TFF, RBFI) ---
    results = {}
    for approx in approximators:
        print(f"\n--- {approx} Approximator Path ---")
        approx_instrument_defs = [dict(d, pricing_preference=approx) for d in instrument_definitions]
        iproc_approx = RefactoredInstrumentProcessor(
            scen_gen_domain, val_date, tenors, default_g2_params, default_bs_rfr, default_bs_div, None, n_fitting_samples
        )
        model_registry = iproc_approx.process_instruments(approx_instrument_defs, domain_scenarios, domain_factor_names, 1)
        portfolio_specs = generate_portfolio_specs_for_serialization(holdings_data, model_registry, approx_instrument_defs)
        builder = RefactoredPortfolioBuilder(model_registry)
        portfolios = builder.build_portfolios_from_specs(
            portfolio_specs, val_date, default_g2_params, default_bs_rfr, default_bs_div
        )
        portfolio_obj = portfolios["HybridClient"]
        portfolio_values = portfolio_obj.price_portfolio(var_scenarios, var_factor_names, tenors)
        base_value = portfolio_values[0]
        losses = base_value - portfolio_values
        sorted_losses = np.sort(losses)
        idx1 = max(0, int(np.ceil(0.01 * N)) - 1)
        var_1pct = -sorted_losses[idx1]
        results[approx] = {"base_value": base_value, "var_1pct": var_1pct, "losses": losses}
        print(f"{approx} 1% VaR: {var_1pct:,.2f}. Base Value: {base_value:,.2f}")

    # --- Hybrid Path for Each Approximator ---
    for approx in approximators:
        print(f"\n--- Hybrid ({approx} + Full) Path ---")
        losses_approx = results[approx]["losses"]
        sorted_losses_approx = np.sort(losses_approx)
        sorted_indices_approx = np.argsort(losses_approx)
        idx_threshold = max(0, int(np.ceil(hybrid_critical_percentile * N)) - 1)
        critical_idx = sorted_indices_approx[:idx_threshold]
        critical_scenarios = var_scenarios[critical_idx]
        # Full reval for critical scenarios
        portfolio_values_full_critical = portfolio_obj_full.price_portfolio(critical_scenarios, var_factor_names, tenors)
        losses_full_critical = base_value_full - portfolio_values_full_critical
        sorted_losses_full_critical = np.sort(losses_full_critical)
        num_critical_scenarios = len(sorted_losses_full_critical)
        if num_critical_scenarios == 0:
            print("No critical scenarios identified. Skipping Hybrid VaR calculation.")
            var_1pct_hybrid = None
        else:
            m_1 = int(np.ceil(0.01 * N)) - 1
            var_1pct_hybrid = -sorted_losses_full_critical[m_1]
            print(f"Hybrid ({approx}) 1% VaR: {var_1pct_hybrid:,.2f}. Base Value: {base_value_full:,.2f}")
        results[approx]["var_1pct_hybrid"] = var_1pct_hybrid

    # --- Summary Table ---
    print("\n--- VaR Comparison Summary ---")
    print(f"{'Method':<15} | {'Base Value':<12} | {'1% VaR':<12} | {'Hybrid 1% VaR':<15}")
    print("-" * 60)
    print(f"{'Full':<15} | {base_value_full:<12.2f} | {var_1pct_full:<12.2f} | {'-':<15}")
    for approx in approximators:
        base_value = results[approx]["base_value"]
        var_1pct = results[approx]["var_1pct"]
        var_1pct_hybrid = results[approx]["var_1pct_hybrid"]
        print(f"{approx:<15} | {base_value:<12.2f} | {var_1pct:<12.2f} | {str(var_1pct_hybrid) if var_1pct_hybrid is not None else '-':<2}")
    
if __name__ == "__main__":
    run_hybrid_var_refactored(
        num_instruments=8,
        num_var_scenarios=1000,
        n_domain_scenarios=2000,
        n_fitting_samples=50,
        hybrid_critical_percentile=0.02,
        approximators=("TFF", "RBFI"),
        random_seed=42
    )

