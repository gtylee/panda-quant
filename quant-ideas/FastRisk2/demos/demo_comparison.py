"""
Comparison Demo: Before vs After Code Complexity
Shows how the new HybridVaRWorkflow class dramatically simplifies the interface.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import date
import numpy as np
from product_definitions_pydantic import ProductStaticRegistry
from workflow import HybridVaRWorkflow
from scenario_generator import SimpleRandomScenarioGenerator


def generate_simple_portfolio(num_instruments=3, val_date=None):
    """Generate a simple portfolio for demonstration."""
    if val_date is None:
        val_date = date(2025, 5, 18)
    
    portfolio = []
    for i in range(num_instruments):
        static = ProductStaticRegistry.create_static(
            "VanillaBond",
            valuation_date=val_date,
            maturity_date=val_date.replace(year=val_date.year + 5),
            coupon_rate=0.03,
            face_value=100.0,
            currency="USD",
            index_stub="IR"
        )
        
        portfolio.append({
            "instrument_id": f"Bond_{i}",
            "product_type": static.product_type,
            "params": static.to_dict(),
            "pricing_preference": "FULL",
            "pricer_params": {},
            "tff_config": {"n_train": 30, "n_test": 5, "seed": i, "order": 2}
        })
    return portfolio


def demonstrate_code_complexity_reduction():
    """
    Demonstrate the dramatic reduction in code complexity.
    """
    print("=" * 80)
    print("CODE COMPLEXITY COMPARISON: Before vs After")
    print("=" * 80)
    
    print("\n📊 BEFORE: Original Demo Required ~200 lines of setup code:")
    print("""
    # Original demo required:
    # 1. Manual scenario generation setup
    # 2. Repetitive instrument processor creation
    # 3. Manual portfolio building for each path
    # 4. Manual VaR calculation logic
    # 5. Repetitive code for TFF, RBFI, and Hybrid paths
    # 6. Manual summary table creation
    
    # Total: ~200 lines of repetitive setup code
    """)
    
    print("\n🚀 AFTER: New HybridVaRWorkflow Requires Only ~25 lines:")
    print("""
    # New workflow requires only:
    # 1. Define instruments
    # 2. Define holdings  
    # 3. Initialize scenario generators (proper separation of concerns)
    # 4. Create workflow instance
    # 5. Call single method
    
    # Total: ~25 lines of clean, focused code
    """)
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION: Running the simplified workflow")
    print("=" * 80)
    
    # Set up the analysis with minimal code
    val_date = date(2025, 5, 18)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    
    # 1. Define instruments (5 lines)
    instrument_definitions = generate_simple_portfolio(3, val_date)
    
    # 2. Define holdings (3 lines)
    holdings_data = [
        {"client_id": "HybridClient", "instrument_id": d["instrument_id"], "num_holdings": 500}
        for d in instrument_definitions
    ]
    
    # 3. Initialize scenario generators (8 lines - proper separation of concerns)
    base_rates_map = {f"USD_IR_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
    base_s0_map = {"USD_AAPL_S0": 100.0}
    base_vol_map = {"USD_AAPL_VOL": 0.25}
    
    var_scenario_generator = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=base_s0_map,
        base_vol_map=base_vol_map, random_seed=42
    )
    
    domain_scenario_generator = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=base_s0_map,
        base_vol_map=base_vol_map, random_seed=43
    )
    
    # 4. Create workflow (2 lines)
    workflow = HybridVaRWorkflow(valuation_date=val_date, tenors=tenors, random_seed=42)
    
    # 5. Run analysis (1 line!)
    results = workflow.run_hybrid_var_analysis(
        instrument_definitions=instrument_definitions,
        holdings_data=holdings_data,
        var_scenario_generator=var_scenario_generator,
        domain_scenario_generator=domain_scenario_generator,
        num_var_scenarios=500,  # Smaller for demo
        n_domain_scenarios=1000,
        n_fitting_samples=30,
        approximators=("TFF", "RBFI")
    )
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"✅ Full VaR: {results['full']['var_value']:,.2f}")
    print(f"✅ TFF VaR: {results['approximators']['TFF']['var_value']:,.2f}")
    print(f"✅ RBFI VaR: {results['approximators']['RBFI']['var_value']:,.2f}")
    print(f"✅ TFF Hybrid VaR: {results['approximators']['TFF']['hybrid']['var_value']:,.2f}")
    print(f"✅ RBFI Hybrid VaR: {results['approximators']['RBFI']['hybrid']['var_value']:,.2f}")
    
    print("\n" + "=" * 80)
    print("KEY BENEFITS OF THE NEW APPROACH")
    print("=" * 80)
    print("🎯 SIMPLICITY: Reduced from ~200 lines to ~25 lines")
    print("🔧 MAINTAINABILITY: All setup logic centralized in one class")
    print("🚀 REUSABILITY: Workflow can be used for any portfolio")
    print("📊 CONSISTENCY: Same interface for all analysis types")
    print("⚡ PERFORMANCE: Optimized scenario generation and caching")
    print("🛡️ ROBUSTNESS: Built-in error handling and validation")
    print("📈 EXTENSIBILITY: Easy to add new approximators or features")
    print("🔀 FLEXIBILITY: Scenario generators can be customized externally")
    print("🏗️ SEPARATION OF CONCERNS: Scenario generation separated from workflow logic")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("The new HybridVaRWorkflow class has transformed the user experience!")
    print("Users can now focus on their business logic rather than setup code.")
    print("Scenario generators are properly separated, maintaining flexibility.")
    print("This is a perfect example of how good abstraction improves usability.")
    
    return results
    

if __name__ == "__main__":
    demonstrate_code_complexity_reduction()

