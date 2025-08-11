"""
Minimal demo comparing RBFI, TFF, and Full revaluation for callable bonds.
Demonstrates the new RBFI approximator alongside existing methods.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import os
from datetime import date, timedelta
import time

# Core imports
from workflow_manager import Portfolio, InstrumentProcessor, PortfolioBuilder, TFFConfigurationFactory, generate_portfolio_specs_for_serialization, generate_price_strips
from registry.product_registry import create_product_static_from_dict
from scenario_generator import SimpleRandomScenarioGenerator

# Approximators
from tff_approximator import TensorFunctionalFormCalibrate
from rbfi_approximator import RBFICalibrate


def run_callable_bond_rbfi_demo(
    num_callable_bonds: int = 5,
    num_var_scenarios: int = 10,
    n_domain_scenarios: int = 100,
    n_fitting_samples: int = 40,
    random_seed: int = 42
):
    """
    Compare RBFI, TFF, and Full pricing for callable bonds.
    """
    print("=== Callable Bond RBFI vs TFF vs Full Demo ===")
    
    # --- 1. Global Setup ---
    np.random.seed(random_seed)
    val_date = date(2025, 5, 18)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    default_g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    default_bs_rfr = 0.025
    default_bs_div = 0.01

    base_rates_map = {f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
    all_underlying_symbols = list(set([f"STOCK_{i%10}" for i in range(1)]))
    base_s0_map_gen, base_vol_map_gen, base_other_map_gen = {}, {}, {}
    for sym in all_underlying_symbols:
        base_s0_map_gen[f"{DEMO_CURRENCY}_{sym}_S0"] = round(90 + np.random.rand() * 20)
        base_vol_map_gen[f"{DEMO_CURRENCY}_{sym}_VOL"] = round(0.20 + np.random.rand() * 0.1, 2)
        base_vol_map_gen[f"{DEMO_CURRENCY}_{sym}_EQVOL"] = round(0.20 + np.random.rand() * 0.1, 2)
        base_other_map_gen[f"{DEMO_CURRENCY}_{sym}_DIVYIELD"] = round(0.01 + np.random.rand() * 0.01, 2)
        base_other_map_gen[f"{DEMO_CURRENCY}_{sym}_CS"] = round(0.01 + np.random.rand() * 0.01, 2)
    merged_s0_map_gen = {**base_s0_map_gen, **base_other_map_gen}

    scen_gen = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=merged_s0_map_gen,
        base_vol_map=base_vol_map_gen, random_seed=random_seed
    )
    var_scenarios, var_factor_names = scen_gen.generate_scenarios(num_var_scenarios)

    # Only generate TFF domain scenarios if TFF or Hybrid methods are enabled
    scen_gen_tff_domain = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=merged_s0_map_gen,
        base_vol_map=base_vol_map_gen, random_seed=random_seed + 1
    )
    tff_domain_scenarios, tff_domain_factor_names = scen_gen_tff_domain.generate_scenarios(n_domain_scenarios)

    var_scenarios, var_factor_names = scen_gen.generate_scenarios(num_var_scenarios)
    domain_scenarios, domain_factor_names = scen_gen.generate_scenarios(n_domain_scenarios)
    
    print(f"Generated scenarios with factors: {var_factor_names[:3]}...{var_factor_names[-2:]}")
    
    # --- Generate Callable Bonds ---
    print(f"\nGenerating {num_callable_bonds} callable bonds...")

    callable_bonds = []
    
    for i in range(num_callable_bonds):
        # Generate call dates (quarterly calls starting 2 years after issue)
        call_start = val_date + timedelta(days=730)  # 2 years
        call_dates = []
        call_prices = []

        for q in range(2):  # 2 quarterly call dates
            call_date = call_start + timedelta(days=90*q)
            call_dates.append(call_date.isoformat())
            call_prices.append(102.0 - q * 0.25)  # Declining call prices
        
        bond_def = {
            'instrument_id': f'CALLABLE_BOND_{i+1:03d}',
            'product_type': 'CallableBond',
            'params': {
                'valuation_date': val_date.isoformat(),
                'maturity_date': (val_date + timedelta(days=3650)).isoformat(),  # 10 years
                'coupon_rate': 0.04 + np.random.uniform(-0.01, 0.01),
                'face_value': 100.0,
                'freq': 2,  # Semi-annual
                'call_dates': call_dates,
                'call_prices': call_prices,
                'currency': 'USD',
                'index_stub': DEMO_RATE_INDEX_STUB
            },
            'pricing_preference': 'FULL',
            'pricer_params': {"g2_params": (0.01, 0.003, 0.015, 0.006, -0.75), "g2_grid_steps": 32},
            'tff_config': {"n_train":n_fitting_samples, "n_test": 8, "seed": i, "order": 2}
        }
        callable_bonds.append(bond_def)
    
    print(f"Created {len(callable_bonds)} callable bonds with call features")
    
    # --- Process with Different Methods ---
    results = {}
    
    # 1. Full Revaluation
    print(f"\n--- Full Revaluation ---")
    start_time = time.time()
    
    iproc_full = InstrumentProcessor(
        scen_gen, val_date, tenors, 
        default_g2_params, default_bs_rfr, default_bs_div
    )
    
    full_model_registry = iproc_full.process_instruments(
        callable_bonds, domain_scenarios, domain_factor_names, batch_size=2
    )
    
    # Create holdings data for the portfolio
    holdings_data = [
        {"client_id": "RBFIClient", "instrument_id": bond["instrument_id"], "num_holdings": 1}
        for bond in callable_bonds
    ]

    # Generate portfolio specs
    full_portfolio_specs = generate_portfolio_specs_for_serialization(
        holdings_data, full_model_registry, callable_bonds
    )

    # Build portfolio using PortfolioBuilder
    builder_full = PortfolioBuilder(full_model_registry)
    portfolio_full_dict = builder_full.build_portfolios_from_specs(
        full_portfolio_specs, val_date, default_g2_params, default_bs_rfr, default_bs_div
    )
    
    # Get the portfolio object and price it
    batch_size = max(1, num_callable_bonds // 100)  # Use half the CPU cores for parallel processing
    portfolio_full = portfolio_full_dict["RBFIClient"]
    strips = generate_price_strips(
        instrument_specs=callable_bonds,
        global_market_scenarios=var_scenarios,
        global_factor_names=var_factor_names,
        iproc=iproc_full,
        num_workers=os.cpu_count(),
        batch_size=batch_size
    )

    full_prices = portfolio_full.price_portfolio_from_strips(strips)
    full_base_value = full_prices[0]
    
    full_time = time.time() - start_time
    print(f"Full revaluation completed in {full_time:.2f}s")
    print(f"Full base value: {full_base_value:,.2f}")
    
    results['full'] = {
        'prices': full_prices,
        'base_value': full_base_value,
        'time': full_time
    }
    
    # 2. TFF Approximation
    print(f"\n--- TFF Approximation ---")
    start_time = time.time()
    
    tff_bonds = [dict(bond, pricing_preference='TFF') for bond in callable_bonds]
    
    
    iproc_tff = InstrumentProcessor(
        scen_gen, val_date, tenors,
        default_g2_params, default_bs_rfr, default_bs_div
    )
    
    tff_model_registry = iproc_tff.process_instruments(
        tff_bonds, domain_scenarios, domain_factor_names, batch_size=2
    )
    
    # Use PortfolioBuilder for TFF as well
    tff_portfolio_specs = generate_portfolio_specs_for_serialization(
        holdings_data, tff_model_registry, tff_bonds
    )
    
    builder_tff = PortfolioBuilder(tff_model_registry)
    portfolio_tff_dict = builder_tff.build_portfolios_from_specs(
        tff_portfolio_specs, val_date, default_g2_params, default_bs_rfr, default_bs_div
    )
    
    portfolio_tff = portfolio_tff_dict["RBFIClient"]
    tff_prices = portfolio_tff.price_portfolio(var_scenarios, var_factor_names, tenors)
    tff_base_value = tff_prices[0]
    
    tff_time = time.time() - start_time
    print(f"TFF approximation completed in {tff_time:.2f}s")
    print(f"TFF base value: {tff_base_value:,.2f}")
    
    results['tff'] = {
        'prices': tff_prices,
        'base_value': tff_base_value,
        'time': tff_time
    }
    
    # 3. RBFI Approximation
    print(f"\n--- RBFI Approximation ---")
    start_time = time.time()
    
    rbfi_results = {}
    
    for bond_id, tff_model in tff_model_registry.items():
        if 'error_tff_calibration' in tff_model:
            print(f"Skipping RBFI for {bond_id} due to TFF calibration error")
            continue
            
        # Get the TFF instance from the model registry
        tff_instance = tff_model
        
        product_static = create_product_static_from_dict(tff_instance['product_static_dict'])
        # find instrument_spec from tff_bonds
        instrument_spec = next((b for b in callable_bonds if b['instrument_id'] == bond_id), None)
        if instrument_spec is None:
            print(f"Error: Could not find instrument spec for {bond_id}")
            continue
        
        pricer_template = iproc_tff._create_pricer_template(product_static=product_static, 
                                          instrument_spec=instrument_spec)

        tff_config_factory = TFFConfigurationFactory(scen_gen, tenors)
        tff_config = tff_config_factory.create_config(product_static)

        # Create RBFI calibrator using the correct attributes from TFF instance
        rbfi_calibrator = RBFICalibrate(
            pricer_template=pricer_template,
            rbfi_input_raw_factor_names=tff_config['tff_input_raw_factor_names'],
            rbfi_input_raw_base_values=tff_config['tff_input_raw_base_values'],
            product_static_params_for_worker=instrument_spec['params'],
            pricer_config_for_worker=tff_config['pricer_config_for_worker'],
            actual_rate_pillars=tenors
        )
        
        # Extract domain scenarios for this instrument's factors, getting slice from raw factors names
        var_factor_names = tff_config['tff_input_raw_factor_names']
        
        # Create domain scenarios for RBFI fitting from scenario generator
        rbfi_domain_scenarios, _ = scen_gen.generate_scenarios(n_domain_scenarios, target_factor_names=var_factor_names)

        # Fit RBFI model with G2 parameters for callable bonds
        try:
            fitted_rbfi, test_inputs, test_prices, rmse, norm_params, base_val, base_rbfi_val = rbfi_calibrator.sample_and_fit(
                full_market_scenarios_for_rbfi_factors=rbfi_domain_scenarios,
                n_train=n_fitting_samples,
                n_test=8,
                random_seed=random_seed,
                length_scale_method='auto',
                regularization=1e-6,
                g2_params=default_g2_params,# Add G2 params for callable bonds
            )
            
            rbfi_results[bond_id] = {
                'rbfi_instance': fitted_rbfi,
                'rmse': rmse,
                'base_value': base_val,
                'base_rbfi_value': base_rbfi_val,
                'normalization_params': norm_params,
                'factor_names': tff_instance['tff_raw_input_names']
            }
            
        except Exception as e:
            print(f"  {bond_id}: RBFI fitting failed: {e}")
    
    # Price portfolio with RBFI - Optimized approach
    rbfi_portfolio_values = np.zeros(num_var_scenarios)
    
    for bond_id, rbfi_model in rbfi_results.items():
        try:
            rbfi_instance = rbfi_model['rbfi_instance']
            factor_names = rbfi_model['factor_names']
            norm_params = rbfi_model['normalization_params']
            
            scenario_data_matrix, _ = scen_gen_tff_domain.generate_scenarios(num_var_scenarios, target_factor_names=factor_names)

            # Apply normalization if needed
            if norm_params.get('is_engineered', False):
                pass
            
            # Price ALL scenarios at once with RBFI (vectorized call)
            instrument_prices = rbfi_instance(scenario_data_matrix)
            
            # Add this instrument's contribution to portfolio values
            rbfi_portfolio_values += instrument_prices
            
        except Exception as e:
            print(f"    Error pricing {bond_id} with RBFI: {e}")
            continue
    
    rbfi_base_value = rbfi_portfolio_values[0]
    
    rbfi_time = time.time() - start_time
    print(f"RBFI approximation completed in {rbfi_time:.2f}s")
    print(f"RBFI base value: {rbfi_base_value:,.2f}")
    
    results['rbfi'] = {
        'prices': rbfi_portfolio_values,  # Now we have full price vector
        'base_value': rbfi_base_value,
        'time': rbfi_time,
        'num_fitted': len(rbfi_results)
    }
    
    # --- Comparison Summary ---
    print(f"\n=== Callable Bond Pricing Comparison ===")
    print(f"{'Method':<15} | {'Base Value':<15} | {'Time (s)':<10} | {'vs Full (%)':<12} | {'Speedup':<10}")
    print("-" * 75)
    
    full_base = results['full']['base_value']
    full_time = results['full']['time']
    
    for method_name, method_results in results.items():
        base_val = method_results['base_value']
        method_time = method_results['time']
        
        diff_pct = 100 * (base_val - full_base) / full_base if full_base != 0 else 0
        speedup = full_time / method_time if method_time > 0 else 0
        
        print(f"{method_name.upper():<15} | {base_val:>13,.2f} | {method_time:>8.2f} | {diff_pct:>+10.3f} | {speedup:>8.1f}x")
    
    # --- Individual Bond Analysis ---
    print(f"\n=== Per-Bond TFF vs RBFI Comparison ===")
    print(f"{'Bond ID':<20} | {'TFF RMSE':<12} | {'RBFI RMSE':<12} | {'Method Comparison':<20}")
    print("-" * 70)
    
    for bond_id in tff_model_registry.keys():
        tff_model = tff_model_registry.get(bond_id, {})
        rbfi_model = rbfi_results.get(bond_id, {})
        
        tff_rmse = tff_model.get('tff_rmse', 'ERROR')
        rbfi_rmse = rbfi_model.get('rmse', 'ERROR')
        
        comparison = "Both fitted"
        if tff_rmse == 'ERROR' and rbfi_rmse == 'ERROR':
            comparison = "Both failed"
        elif tff_rmse == 'ERROR':
            comparison = "RBFI only"
        elif rbfi_rmse == 'ERROR':
            comparison = "TFF only"
        elif isinstance(tff_rmse, (int, float)) and isinstance(rbfi_rmse, (int, float)):
            if rbfi_rmse < tff_rmse:
                comparison = "RBFI better"
            elif tff_rmse < rbfi_rmse:
                comparison = "TFF better"
            else:
                comparison = "Similar"
        
        tff_str = f"{tff_rmse:.6f}" if isinstance(tff_rmse, (int, float)) else str(tff_rmse)
        rbfi_str = f"{rbfi_rmse:.6f}" if isinstance(rbfi_rmse, (int, float)) else str(rbfi_rmse)
        
        print(f"{bond_id:<20} | {tff_str:>10} | {rbfi_str:>10} | {comparison:<20}")
    
    print(f"\n=== Demo Complete ===")
    print(f"Successfully fitted RBFI for {len(rbfi_results)}/{len(callable_bonds)} callable bonds")
    
    return results


if __name__ == "__main__":
    # Run the demo
    results = run_callable_bond_rbfi_demo(
        num_callable_bonds=5,
        num_var_scenarios=100,
        n_domain_scenarios=200,
        n_fitting_samples=128,
        random_seed=42
    )
    
    print(f"\nDemo completed successfully!")
    print(f"Results keys: {list(results.keys())}")

