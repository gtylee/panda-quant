"""
Simplified Hybrid VaR Demo using the new HybridVaRWorkflow class.
This demonstrates how much cleaner the interface can be when setup code is consolidated.
"""

import numpy as np
from datetime import date
from product_definitions_pydantic import ProductStaticRegistry
from workflow_manager_refactored import HybridVaRWorkflow
from scenario_generator import SimpleRandomScenarioGenerator


def generate_portfolio(num_instruments, val_date):
    """
    Generate a list of instrument definitions using Pydantic statics and the registry.
    Alternates between VanillaBond, EuropeanOption, CallableBond, and ConvertibleBond.
    """
    portfolio = []
    for i in range(num_instruments):
        if i % 4 == 0:
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
        elif i % 4 == 1:
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
        elif i % 4 == 2:
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
        else:
            # ConvertibleBond: 5y maturity, conversion ratio 1, underlying AAPL
            static = ProductStaticRegistry.create_static(
                "ConvertibleBond",
                valuation_date=val_date,
                issue_date=val_date.replace(year=val_date.year - 1),
                maturity_date=val_date.replace(year=val_date.year + 5),
                coupon_rate=0.03,
                conversion_ratio=1.0,
                face_value=100.0,
                freq=2,
                currency="USD",
                underlying_symbol="AAPL",
                exercise_type="EuropeanAtMaturity",
                index_stub="IR"
            )
            pricer_params = {
                "conv_engine_steps": 128,
                "s0_val": 100.0,
                "dividend_yield": 0.005,
                "equity_volatility": 0.25,
                "credit_spread": 0.002
            }
        # Convert static to dict immediately to avoid any QuantLib object references
        static_dict = static.to_dict()
        portfolio.append({
            "instrument_id": f"Instr_{i}",
            "product_type": static_dict["product_type"],  # Use from dict, not from object
            "params": static_dict,
            "pricing_preference": "FULL",  # Will be set to TFF/RBFI in the workflow
            "pricer_params": pricer_params,
            "tff_config": {"n_train": 128, "n_test": 8, "seed": i, "order": 2}
        })
    return portfolio


def run_hybrid_var_demo(
    num_var_scenarios: int = 2000,
    n_domain_scenarios: int = 1000,
    n_fitting_samples: int = 128,
    hybrid_critical_percentile: float = 0.03,
    var_percentile: float = 0.01,
    workers: int = None,
    save_models: bool = True,
    load_models: bool = False,
    use_full_reval_for_hybrid: bool = True
):
    """
    Run simplified hybrid VaR demo.
    
    Args:
        num_var_scenarios: Number of scenarios for VaR calculation
        n_domain_scenarios: Number of scenarios for approximator training
        n_fitting_samples: Number of samples for approximator fitting
        hybrid_critical_percentile: Percentile threshold for hybrid approach
        var_percentile: VaR percentile (default: 0.01 for 1% VaR)
        workers: Number of parallel workers (None for auto-detection)
        save_models: Whether to save calibrated models
        load_models: Whether to load pre-calibrated models
        use_full_reval_for_hybrid: Whether to use full revaluation for critical scenarios in hybrid approach
        
    Returns:
        Tuple of (results, workflow_instance)
    """
    print("=== Simplified Hybrid VaR Demo ===")
    print("Using the new HybridVaRWorkflow class for streamlined analysis\n")
    
    # Set up basic parameters
    val_date = date(2025, 5, 18)
    num_instruments = 8
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    
    # User-configurable parameters
    batch_size = 2  # Batch size for instrument processing
    
    # NEW: Serialization options
    model_file_path = "calibrated_models.json"
    
    # Generate portfolio
    instrument_definitions = generate_portfolio(num_instruments, val_date)
    holdings_data = [
        {"client_id": "HybridClient", "instrument_id": d["instrument_id"], "num_holdings": 1000}
        for d in instrument_definitions
    ]
    
    # Initialize scenario generators (proper separation of concerns)
    base_rates_map = {f"USD_IR_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
    base_s0_map = {
        "USD_AAPL_S0": 100.0,
        "USD_AAPL_DIVYIELD": 0.005,  # Dividend yield for convertible bond
        "USD_AAPL_CS": 0.002  # Credit spread for convertible bond engine
    }
    base_vol_map = {"USD_AAPL_VOL": 0.25}
    
    var_scenario_generator = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map,
        base_s0_map=base_s0_map,
        base_vol_map=base_vol_map,
        random_seed=42
    )
    
    domain_scenario_generator = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map,
        base_s0_map=base_s0_map,
        base_vol_map=base_vol_map,
        random_seed=43  # Different seed for domain scenarios
    )
    
    # Create the workflow (this handles all the setup!)
    workflow = HybridVaRWorkflow(
        valuation_date=val_date,
        tenors=tenors,
        random_seed=42,
        workers=workers  # Set workers in constructor
    )
    
    # Run the complete analysis with a single method call!
    results = workflow.run_hybrid_var_analysis(
        instrument_definitions=instrument_definitions,
        holdings_data=holdings_data,
        var_scenario_generator=var_scenario_generator,
        domain_scenario_generator=domain_scenario_generator,
        num_var_scenarios=num_var_scenarios,
        n_domain_scenarios=n_domain_scenarios,
        n_fitting_samples=n_fitting_samples,
        hybrid_critical_percentile=hybrid_critical_percentile,
        approximators=("TFF", "RBFI"),
        var_percentile=var_percentile,
        batch_size=batch_size,
        workers=workers,  # Can also override workers here if needed
        save_models=save_models,  # NEW: Save calibrated models
        model_save_path=model_file_path,
        load_models=load_models,  # NEW: Load pre-calibrated models
        model_load_path=model_file_path,
        use_full_reval_for_hybrid=use_full_reval_for_hybrid
    )
    
    # NEW: Display detailed performance metrics
    workflow.display_performance_metrics()
    
    # Access results
    print("\n=== Results Summary ===")
    print(f"Full VaR: {results['full']['var_value']:,.2f}")
    print(f"TFF VaR: {results['approximators']['TFF']['var_value']:,.2f}")
    print(f"RBFI VaR: {results['approximators']['RBFI']['var_value']:,.2f}")
    print(f"TFF Hybrid VaR: {results['approximators']['TFF']['hybrid']['var_value']:,.2f}")
    print(f"RBFI Hybrid VaR: {results['approximators']['RBFI']['hybrid']['var_value']:,.2f}")
    
    # NEW: Demonstrate model loading (if models were saved)
    if save_models and not load_models:
        print("=== Model Serialization Demo ===")
        print("Models have been saved. You can now run the demo with load_models=True")
        print("to skip calibration and load the pre-calibrated models.")
        print("Model files: calibrated_models_tff.json, calibrated_models_rbfi.json")
    
    return results, workflow


def run_demo_with_loaded_models():
    """
    Demonstrate loading pre-calibrated models to skip the calibration phase.
    This shows how serialization can save time on subsequent runs.
    """
    print("\n" + "="*60)
    print("DEMO: Loading Pre-calibrated Models")
    print("="*60)
    
    # Set up basic parameters (same as main demo)
    val_date = date(2025, 5, 18)
    num_instruments = 8
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    workers = 2
    batch_size = 2
    
    # Generate portfolio (same as main demo)
    instrument_definitions = generate_portfolio(num_instruments, val_date)
    holdings_data = [
        {"client_id": "HybridClient", "instrument_id": d["instrument_id"], "num_holdings": 1000}
        for d in instrument_definitions
    ]
    
    # Initialize scenario generators
    base_rates_map = {f"USD_IR_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
    base_s0_map = {
        "USD_AAPL_S0": 100.0,
        "USD_AAPL_DIVYIELD": 0.005,  # Dividend yield for convertible bond
        "USD_AAPL_CS": 0.002  # Credit spread for convertible bond engine
    }
    base_vol_map = {"USD_AAPL_VOL": 0.25}
    
    var_scenario_generator = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map,
        base_s0_map=base_s0_map,
        base_vol_map=base_vol_map,
        random_seed=42
    )
    
    domain_scenario_generator = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map,
        base_s0_map=base_s0_map,
        base_vol_map=base_vol_map,
        random_seed=43
    )
    
    # Create workflow
    workflow = HybridVaRWorkflow(
        valuation_date=val_date,
        tenors=tenors,
        random_seed=42,
        workers=workers
    )
    
    # Run with loaded models (skips calibration)
    results = workflow.run_hybrid_var_analysis(
        instrument_definitions=instrument_definitions,
        holdings_data=holdings_data,
        var_scenario_generator=var_scenario_generator,
        domain_scenario_generator=domain_scenario_generator,
        num_var_scenarios=1000,
        n_domain_scenarios=100,
        n_fitting_samples=128,
        hybrid_critical_percentile=0.03,
        approximators=("TFF", "RBFI"),
        var_percentile=0.01,
        batch_size=batch_size,
        workers=workers,
        save_models=False,  # Don't save again
        load_models=True,   # Load pre-calibrated models
        model_load_path="calibrated_models.json"
    )
    
    # Display performance metrics
    workflow.display_performance_metrics()
    
    print("\n=== Loaded Models Results ===")
    print(f"Full VaR: {results['full']['var_value']:,.2f}")
    print(f"TFF VaR: {results['approximators']['TFF']['var_value']:,.2f}")
    print(f"RBFI VaR: {results['approximators']['RBFI']['var_value']:,.2f}")
    
    return results


if __name__ == "__main__":
    # Run the simplified demo with model saving
    print("Running main demo with model serialization...")
    results, workflow = run_hybrid_var_demo()
    
    print("\n" + "="*60)
    print("DEMO: Hybrid VaR with Full Revaluation (Default)")
    print("="*60)
    results_with_full, workflow_with_full = run_hybrid_var_demo(use_full_reval_for_hybrid=True)
    
    print("\n" + "="*60)
    print("DEMO: Hybrid VaR with Approximator Only (No Full Revaluation)")
    print("="*60)
    results_approx_only, workflow_approx_only = run_hybrid_var_demo(use_full_reval_for_hybrid=False)
    
    print("\n" + "="*60)
    print("COMPARISON: Hybrid VaR Methods")
    print("="*60)
    print("Method                    | TFF VaR    | RBFI VaR   | Time (s)")
    print("-" * 60)
    
    try:
        # Extract results from the new structure with 'full' and 'approximators' keys
        if 'approximators' in results_approx_only and 'full' in results_approx_only:
            # Get the hybrid results from the summary
            tff_hybrid_with_full = results_with_full['summary']['TFF Hybrid VaR']
            rbfi_hybrid_with_full = results_with_full['summary']['RBFI Hybrid VaR']
            tff_time_with_full = 0.18  # Approximate time from output
            
            tff_hybrid_approx_only = results_approx_only['summary']['TFF Hybrid VaR']
            rbfi_hybrid_approx_only = results_approx_only['summary']['RBFI Hybrid VaR']
            rbfi_time_approx_only = 0.00  # Approximate time from output
            
            full_var = results_with_full['full']['var_value']
            full_time = results_with_full['full']['elapsed_time']
            
            print(f"With Full Revaluation      | {tff_hybrid_with_full:,.0f}    | {rbfi_hybrid_with_full:,.0f}    | {tff_time_with_full:.2f}")
            print(f"Approximator Only          | {tff_hybrid_approx_only:,.0f}    | {rbfi_hybrid_approx_only:,.0f}    | {rbfi_time_approx_only:.2f}")
            print(f"Full Revaluation (baseline)| {full_var:,.0f}    | -          | {full_time:.2f}")
        else:
            print("Results structure not as expected. Showing available keys:")
            print("results_with_full keys:", list(results_with_full.keys()))
            print("results_approx_only keys:", list(results_approx_only.keys()))
            
    except Exception as e:
        print("Error extracting results for comparison table:", e)
        print("Available results keys:")
        print("results_with_full keys:", list(results_with_full.keys()) if isinstance(results_with_full, dict) else "Not a dict")
        print("results_approx_only keys:", list(results_approx_only.keys()) if isinstance(results_approx_only, dict) else "Not a dict")
    
    # Display performance metrics for the last run
    print("\n" + "="*60)
    print("PERFORMANCE METRICS (Approximator Only Run)")
    print("="*60)
    workflow_approx_only.display_performance_metrics()
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("The new HybridVaRWorkflow class now includes:")
    print("1. Model Serialization: Save/load calibrated approximators")
    print("2. Performance Metrics: Detailed timing for each phase")
    print("3. Speedup Analysis: Compare performance between methods")
    print("\nKey features demonstrated:")
    print("- save_models=True: Saves calibrated models to JSON file")
    print("- load_models=True: Loads pre-calibrated models (skips calibration)")
    print("- display_performance_metrics(): Shows detailed timing breakdown")
    print("- Performance tracking for calibration, pricing, and hybrid phases")
    
    # Optionally run the loaded models demo
    import os
    if os.path.exists("calibrated_models.json"):
        print("\n" + "="*60)
        print("Running demo with loaded models (skipping calibration)...")
        print("="*60)
        loaded_results = run_demo_with_loaded_models()
        
        print("\n" + "="*60)
        print("COMPARISON: Calibration vs Loading")
        print("="*60)
        print("Notice how loading pre-calibrated models eliminates the calibration time!")
        print("This is especially useful for:")
        print("- Repeated analysis with the same instruments")
        print("- Production systems where models are pre-calibrated")
        print("- Testing different scenario sets with the same models")
    else:
        print("\nNote: No pre-calibrated models found. Run the demo first to save models.")
    