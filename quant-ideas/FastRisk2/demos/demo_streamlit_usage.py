#!/usr/bin/env python3
"""
Demo script showing how to use the Streamlit Risk Management App programmatically.
This script demonstrates the core functionality without needing the web interface.
"""

import sys
import os
from datetime import date, datetime
import numpy as np
import pandas as pd

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streamlit_risk_app import RealTimeRiskManager, OllamaRiskAssistant

def demo_portfolio_building():
    """Demonstrate portfolio building functionality."""
    print("🏗️  Demo: Portfolio Building")
    print("=" * 50)
    
    # Initialize risk manager
    risk_manager = RealTimeRiskManager()
    success = risk_manager.initialize_workflow(date.today())
    
    if not success:
        print("❌ Failed to initialize risk manager")
        return
    
    print("✅ Risk manager initialized successfully")
    
    # Create sample portfolio
    portfolio_definitions = []
    holdings = []
    
    # Add a vanilla bond
    bond_def = risk_manager.create_instrument_definition(
        "VanillaBond",
        maturity_years=5.0,
        coupon_rate=0.03,
        face_value=100.0,
        quantity=1000
    )
    portfolio_definitions.append(bond_def)
    holdings.append({'quantity': 1000})
    print("📈 Added Vanilla Bond: 5Y, 3% coupon, $100 face, 1000 units")
    
    # Add a European call option
    option_def = risk_manager.create_instrument_definition(
        "EuropeanOption",
        maturity_years=1.0,
        strike=100.0,
        option_type="Call",
        quantity=500
    )
    portfolio_definitions.append(option_def)
    holdings.append({'quantity': 500})
    print("📊 Added European Call: 1Y, $100 strike, 500 units")
    
    # Add a convertible bond
    conv_def = risk_manager.create_instrument_definition(
        "ConvertibleBond",
        maturity_years=3.0,
        coupon_rate=0.025,
        conversion_ratio=1.0,
        quantity=200
    )
    portfolio_definitions.append(conv_def)
    holdings.append({'quantity': 200})
    print("🔄 Added Convertible Bond: 3Y, 2.5% coupon, 1:1 conversion, 200 units")
    
    return risk_manager, portfolio_definitions, holdings

def demo_var_calculation(risk_manager, portfolio_definitions, holdings):
    """Demonstrate VaR calculation."""
    print("\n📊 Demo: VaR Calculation")
    print("=" * 50)
    
    print("🔄 Calculating VaR...")
    var_value, base_value, risk_contributions = risk_manager.calculate_var(
        portfolio_definitions, holdings, num_scenarios=1000
    )
    
    if var_value is not None:
        print(f"✅ VaR Calculation Complete")
        print(f"   Base Portfolio Value: ${base_value:,.2f}")
        print(f"   1% VaR: ${var_value:,.2f}")
        print(f"   VaR as % of Portfolio: {abs(var_value/base_value)*100:.2f}%")
        
        print("\n📋 Risk Contributions:")
        for inst_name, contrib in risk_contributions.items():
            print(f"   {inst_name}: ${contrib['var_contribution']:,.2f} ({contrib['percentage']:.1f}%)")
        
        return var_value, base_value, risk_contributions
    else:
        print("❌ VaR calculation failed")
        return None, None, {}

def demo_market_shocks(risk_manager, portfolio_definitions, holdings):
    """Demonstrate market shock application."""
    print("\n⚡ Demo: Market Shocks")
    print("=" * 50)
    
    # Calculate baseline VaR
    baseline_var, baseline_value, _ = risk_manager.calculate_var(
        portfolio_definitions, holdings, num_scenarios=1000
    )
    
    if baseline_var is None:
        print("❌ Baseline VaR calculation failed")
        return
    
    print(f"📊 Baseline VaR: ${baseline_var:,.2f}")
    
    # Apply interest rate shock
    print("\n📈 Applying Interest Rate Shock (+50 bps across curve)...")
    rate_shock = {f"USD_IR_{t:.2f}Y": 0.005 for t in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]}
    risk_manager.update_market_data(rate_shock)
    
    shocked_var, shocked_value, _ = risk_manager.calculate_var(
        portfolio_definitions, holdings, num_scenarios=1000
    )
    
    if shocked_var is not None:
        var_change = shocked_var - baseline_var
        print(f"📊 New VaR: ${shocked_var:,.2f}")
        print(f"📈 VaR Change: ${var_change:,.2f} ({var_change/baseline_var*100:+.1f}%)")
    
    # Apply equity shock
    print("\n📈 Applying Equity Shock (+10% AAPL price increase)...")
    equity_shock = {"USD_AAPL_S0": 10.0}  # $10 increase
    risk_manager.update_market_data(equity_shock)
    
    equity_shocked_var, equity_shocked_value, _ = risk_manager.calculate_var(
        portfolio_definitions, holdings, num_scenarios=1000
    )
    
    if equity_shocked_var is not None:
        var_change = equity_shocked_var - baseline_var
        print(f"📊 New VaR: ${equity_shocked_var:,.2f}")
        print(f"📈 VaR Change: ${var_change:,.2f} ({var_change/baseline_var*100:+.1f}%)")

def demo_ai_risk_advice(portfolio_definitions, holdings, var_value, base_value):
    """Demonstrate AI risk advice functionality."""
    print("\n🤖 Demo: AI Risk Advice")
    print("=" * 50)
    
    # Create portfolio summary
    portfolio_summary = f"""
    Portfolio Composition:
    - {len(portfolio_definitions)} instruments
    - Total notional: ${sum(h['quantity'] * 100 for h in holdings):,.0f}
    - Instrument types: {', '.join(set(d['product_type'] for d in portfolio_definitions))}
    """
    
    var_result = f"${var_value:,.2f} ({(var_value/base_value)*100:.2f}% of portfolio value)"
    market_conditions = "Current market conditions with moderate volatility"
    
    print("🔄 Getting AI risk advice...")
    
    # Note: This would require Ollama to be running
    # For demo purposes, we'll show what the prompt would look like
    print("📝 AI Prompt:")
    print("-" * 30)
    print(f"Portfolio: {portfolio_summary.strip()}")
    print(f"Current VaR: {var_result}")
    print(f"Market: {market_conditions}")
    print("-" * 30)
    
    print("\n💡 Sample AI Response (if Ollama were running):")
    print("""
    Risk Assessment:
    - Portfolio shows moderate risk with 1.97% VaR
    - Convertible bond provides some downside protection
    - Interest rate sensitivity is manageable
    
    Recommendations:
    1. Consider adding more equity hedges
    2. Monitor interest rate exposure
    3. Review convertible bond conversion terms
    
    Market Outlook:
    - Current volatility suggests maintaining hedges
    - Consider reducing duration exposure
    """)

def demo_real_time_monitoring():
    """Demonstrate real-time monitoring capabilities."""
    print("\n⏱️  Demo: Real-Time Monitoring")
    print("=" * 50)
    
    print("🔄 Simulating real-time VaR updates...")
    
    # Simulate VaR history
    timestamps = pd.date_range(start='2024-01-01 09:00:00', periods=10, freq='H')
    var_values = [28493, 28500, 28450, 28550, 28600, 28580, 28520, 28480, 28530, 28510]
    
    print("📈 VaR History (simulated):")
    for ts, var in zip(timestamps, var_values):
        print(f"   {ts.strftime('%H:%M')}: ${var:,.0f}")
    
    print("\n📊 Key Metrics:")
    print(f"   Average VaR: ${np.mean(var_values):,.0f}")
    print(f"   VaR Volatility: ${np.std(var_values):,.0f}")
    print(f"   Max VaR: ${max(var_values):,.0f}")
    print(f"   Min VaR: ${min(var_values):,.0f}")

def main():
    """Run the complete demo."""
    print("🚀 Streamlit Risk Management App - Programmatic Demo")
    print("=" * 60)
    print("This demo shows the core functionality without the web interface.")
    print("For the full interactive experience, run: streamlit run streamlit_risk_app.py")
    print("=" * 60)
    
    try:
        # Demo 1: Portfolio Building
        result = demo_portfolio_building()
        if result is None:
            return
        risk_manager, portfolio_definitions, holdings = result
        
        # Demo 2: VaR Calculation
        var_value, base_value, risk_contributions = demo_var_calculation(
            risk_manager, portfolio_definitions, holdings
        )
        
        if var_value is None:
            return
        
        # Demo 3: Market Shocks
        demo_market_shocks(risk_manager, portfolio_definitions, holdings)
        
        # Demo 4: AI Risk Advice
        demo_ai_risk_advice(portfolio_definitions, holdings, var_value, base_value)
        
        # Demo 5: Real-Time Monitoring
        demo_real_time_monitoring()
        
        print("\n✅ Demo completed successfully!")
        print("\n🎯 Next Steps:")
        print("   1. Run the full Streamlit app: streamlit run streamlit_risk_app.py")
        print("   2. Install Ollama for AI risk advice: https://ollama.ai")
        print("   3. Experiment with different portfolios and market shocks")
        print("   4. Extend the app with your own instruments and risk metrics")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

