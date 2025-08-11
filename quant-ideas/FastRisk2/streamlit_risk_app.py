import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import json
import threading
import queue
from datetime import datetime, date, timedelta
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our custom modules
from workflow import (
    HybridVaRWorkflow,
    InstrumentProcessor,  # if needed later
    Portfolio,
    PortfolioBuilder as RefactoredPortfolioBuilder,
)
from scenario_generator import SimpleRandomScenarioGenerator
from registry import ProductHandlerFactory, ApproximatorHandlerFactory

class RealTimeRiskManager:
    """Real-time risk management system using TFF calibrated instruments."""
    
    def __init__(self):
        self.workflow = None
        self.portfolio = None
        self.model_registry = None
        self.scenario_generator = None
        self.current_var = None
        self.var_history = []
        self.risk_contributions = {}
        self.last_update = None
        
    def initialize_workflow(self, valuation_date: date):
        """Initialize the workflow and load calibrated models."""
        try:
            self.workflow = HybridVaRWorkflow(
                valuation_date=valuation_date,
                workers=2
            )
            
            # Load pre-calibrated TFF models
            self.model_registry = self.workflow.load_model_registry("calibrated_models_tff.json")
            
            # Initialize scenario generator
            tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
            base_rates_map = {f"USD_IR_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
            base_s0_map = {
                "USD_AAPL_S0": 100.0,
                "USD_AAPL_DIVYIELD": 0.005,
                "USD_AAPL_CS": 0.002
            }
            base_vol_map = {"USD_AAPL_VOL": 0.25}
            
            self.scenario_generator = SimpleRandomScenarioGenerator(
                base_rates_map=base_rates_map,
                base_s0_map=base_s0_map,
                base_vol_map=base_vol_map,
                random_seed=42
            )
            
            return True
        except Exception as e:
            st.error(f"Failed to initialize workflow: {e}")
            return False
    
    def create_instrument_definition(self, instrument_type: str, **params) -> dict:
        """Create instrument definition with given parameters."""
        val_date = date.today()
        
        if instrument_type == "VanillaBond":
            return {
                "product_type": "VanillaBond",
                "valuation_date": val_date.isoformat(),
                "maturity_date": (val_date + timedelta(days=params.get('maturity_years', 5) * 365)).isoformat(),
                "coupon_rate": params.get('coupon_rate', 0.03),
                "face_value": params.get('face_value', 100.0),
                "freq": params.get('freq', 2),
                "currency": params.get('currency', 'USD'),
                "index_stub": params.get('index_stub', 'IR'),
                "pricing_preference": "TFF",
                "pricer_params": {"g2_params": (0.01, 0.003, 0.015, 0.006, -0.75), "g2_grid_steps": 32}
            }
        
        elif instrument_type == "EuropeanOption":
            return {
                "product_type": "EuropeanOption",
                "valuation_date": val_date.isoformat(),
                "maturity_date": (val_date + timedelta(days=params.get('maturity_years', 1) * 365)).isoformat(),
                "strike": params.get('strike', 100.0),
                "option_type": params.get('option_type', 'Call'),
                "underlying_symbol": params.get('underlying_symbol', 'AAPL'),
                "currency": params.get('currency', 'USD'),
                "pricing_preference": "TFF",
                "pricer_params": {"risk_free_rate": 0.025, "dividend_yield": 0.01}
            }
        
        elif instrument_type == "ConvertibleBond":
            return {
                "product_type": "ConvertibleBond",
                "valuation_date": val_date.isoformat(),
                "maturity_date": (val_date + timedelta(days=params.get('maturity_years', 5) * 365)).isoformat(),
                "coupon_rate": params.get('coupon_rate', 0.03),
                "conversion_ratio": params.get('conversion_ratio', 1.0),
                "face_value": params.get('face_value', 100.0),
                "freq": params.get('freq', 2),
                "currency": params.get('currency', 'USD'),
                "underlying_symbol": params.get('underlying_symbol', 'AAPL'),
                "exercise_type": params.get('exercise_type', 'EuropeanAtMaturity'),
                "index_stub": params.get('index_stub', 'IR'),
                "pricing_preference": "TFF",
                "pricer_params": {
                    "conv_engine_steps": 128,
                    "s0_val": 100.0,
                    "dividend_yield": 0.005,
                    "equity_volatility": 0.25,
                    "credit_spread": 0.002
                }
            }
        
        return None
    
    def calculate_var(self, portfolio_definitions: List[dict], holdings: List[dict], 
                     num_scenarios: int = 1000) -> Tuple[float, Dict]:
        """Calculate VaR for the current portfolio."""
        try:
            # Generate scenarios
            var_scenarios = self.scenario_generator.generate_scenarios(num_scenarios)
            var_factor_names = list(self.scenario_generator.base_rates_map.keys()) + \
                              list(self.scenario_generator.base_s0_map.keys()) + \
                              list(self.scenario_generator.base_vol_map.keys())
            
            # Run TFF analysis
            results = self.workflow.run_hybrid_var_analysis(
                instrument_definitions=portfolio_definitions,
                holdings_data=holdings,
                var_scenario_generator=self.scenario_generator,
                domain_scenario_generator=self.scenario_generator,
                num_var_scenarios=num_scenarios,
                n_domain_scenarios=100,  # Small for real-time
                n_fitting_samples=50,
                hybrid_critical_percentile=0.03,
                approximators=("TFF",),
                var_percentile=0.01,
                load_models=True,
                model_load_path="calibrated_models_tff.json",
                use_full_reval_for_hybrid=False
            )
            
            var_value = results['TFF']['var_value']
            base_value = results['TFF']['base_value']
            
            # Calculate risk contributions
            risk_contributions = self.calculate_risk_contributions(
                portfolio_definitions, holdings, var_scenarios, var_factor_names
            )
            
            return var_value, base_value, risk_contributions
            
        except Exception as e:
            st.error(f"Error calculating VaR: {e}")
            return None, None, {}
    
    def calculate_risk_contributions(self, portfolio_definitions: List[dict], 
                                   holdings: List[dict], scenarios: np.ndarray,
                                   factor_names: List[str]) -> Dict:
        """Calculate risk contributions for each instrument."""
        try:
            # Build portfolio
            portfolio_specs = []
            for i, (inst_def, holding) in enumerate(zip(portfolio_definitions, holdings)):
                portfolio_specs.append({
                    "instrument_id": f"inst_{i}",
                    "product_static": inst_def,
                    "num_holdings": holding['quantity'],
                    "pricing_engine_type": "tff",
                    "model_registry": self.model_registry
                })
            
            builder = RefactoredPortfolioBuilder()
            portfolio = builder.build_portfolio_from_specs(
                portfolio_specs, date.today()
            )
            
            # Calculate base portfolio value
            base_scenario = scenarios[0:1]
            base_value = portfolio.price_portfolio(base_scenario, factor_names)[0]
            
            # Calculate individual instrument contributions
            contributions = {}
            for i, (inst_def, holding) in enumerate(zip(portfolio_definitions, holdings)):
                # Create single instrument portfolio
                single_inst_specs = [portfolio_specs[i]]
                single_portfolio = builder.build_portfolio_from_specs(
                    single_inst_specs, date.today()
                )
                
                # Calculate single instrument VaR
                single_values = single_portfolio.price_portfolio(scenarios, factor_names)
                single_losses = base_value - single_values
                single_var = np.percentile(single_losses, 1)  # 1% VaR
                
                contributions[f"Instrument_{i+1}"] = {
                    "type": inst_def["product_type"],
                    "quantity": holding['quantity'],
                    "notional": holding['quantity'] * inst_def.get('face_value', 100),
                    "var_contribution": abs(single_var),
                    "percentage": abs(single_var) / abs(self.current_var) * 100 if self.current_var else 0
                }
            
            return contributions
            
        except Exception as e:
            st.error(f"Error calculating risk contributions: {e}")
            return {}
    
    def update_market_data(self, market_shock: Dict[str, float]):
        """Update market data with shocks."""
        try:
            # Update scenario generator with new market data
            for factor, shock in market_shock.items():
                if factor in self.scenario_generator.base_rates_map:
                    self.scenario_generator.base_rates_map[factor] += shock
                elif factor in self.scenario_generator.base_s0_map:
                    self.scenario_generator.base_s0_map[factor] += shock
                elif factor in self.scenario_generator.base_vol_map:
                    self.scenario_generator.base_vol_map[factor] += shock
            
            self.last_update = datetime.now()
            
        except Exception as e:
            st.error(f"Error updating market data: {e}")

class OllamaRiskAssistant:
    """Ollama-based risk management assistant."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = None
    
    async def get_risk_advice(self, portfolio_summary: str, var_result: str, 
                            market_conditions: str) -> str:
        """Get risk management advice from Ollama."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            prompt = f"""
            As a risk management expert, analyze the following portfolio and provide actionable advice:
            
            Portfolio Summary: {portfolio_summary}
            Current VaR: {var_result}
            Market Conditions: {market_conditions}
            
            Please provide:
            1. Risk assessment
            2. Potential actions to reduce risk
            3. Market outlook and recommendations
            4. Specific instrument-level suggestions
            
            Keep the response concise and actionable.
            """
            
            async with self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": "llama2",
                    "prompt": prompt,
                    "stream": False
                }
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('response', 'Unable to get advice from Ollama')
                else:
                    return f"Error connecting to Ollama: {response.status}"
                    
        except Exception as e:
            return f"Error getting Ollama advice: {e}"

def main():
    st.set_page_config(
        page_title="Real-Time Risk Management",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🚀 Real-Time Risk Management Dashboard")
    st.markdown("---")
    
    # Initialize session state
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = RealTimeRiskManager()
    
    if 'ollama_assistant' not in st.session_state:
        st.session_state.ollama_assistant = OllamaRiskAssistant()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Initialize workflow
        if st.button("Initialize Risk Manager"):
            with st.spinner("Initializing..."):
                success = st.session_state.risk_manager.initialize_workflow(date.today())
                if success:
                    st.success("Risk Manager initialized successfully!")
                else:
                    st.error("Failed to initialize Risk Manager")
        
        # Market data controls
        st.subheader("📈 Market Data")
        
        # Rate shocks
        st.write("**Interest Rate Shocks (bps)**")
        rate_shocks = {}
        for tenor in [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
            shock = st.number_input(
                f"{tenor}Y", 
                value=0.0, 
                step=1.0, 
                key=f"rate_shock_{tenor}"
            )
            if shock != 0:
                rate_shocks[f"USD_IR_{tenor:.2f}Y"] = shock / 10000  # Convert bps to decimal
        
        # Equity shocks
        st.write("**Equity Shocks**")
        equity_shock = st.number_input("AAPL Price Change", value=0.0, step=1.0)
        vol_shock = st.number_input("Volatility Change", value=0.0, step=0.01)
        
        if st.button("Apply Market Shocks"):
            market_shock = {**rate_shocks}
            if equity_shock != 0:
                market_shock["USD_AAPL_S0"] = equity_shock
            if vol_shock != 0:
                market_shock["USD_AAPL_VOL"] = vol_shock
            
            st.session_state.risk_manager.update_market_data(market_shock)
            st.success("Market shocks applied!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Portfolio Builder")
        
        # Portfolio definition
        portfolio_definitions = []
        holdings = []
        
        # Add instruments dynamically
        st.subheader("Add Instruments")
        
        instrument_type = st.selectbox(
            "Instrument Type",
            ["VanillaBond", "EuropeanOption", "ConvertibleBond"]
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            quantity = st.number_input("Quantity", min_value=1, value=1000)
            maturity_years = st.number_input("Maturity (Years)", min_value=0.25, value=5.0, step=0.25)
        
        with col_b:
            if instrument_type == "VanillaBond":
                coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, value=3.0, step=0.1) / 100
                face_value = st.number_input("Face Value", min_value=1.0, value=100.0)
            elif instrument_type == "EuropeanOption":
                strike = st.number_input("Strike Price", min_value=1.0, value=100.0)
                option_type = st.selectbox("Option Type", ["Call", "Put"])
            elif instrument_type == "ConvertibleBond":
                coupon_rate = st.number_input("Coupon Rate (%)", min_value=0.0, value=3.0, step=0.1) / 100
                conversion_ratio = st.number_input("Conversion Ratio", min_value=0.1, value=1.0, step=0.1)
        
        if st.button("Add Instrument"):
            # Create instrument definition
            params = {
                'maturity_years': maturity_years,
                'quantity': quantity
            }
            
            if instrument_type == "VanillaBond":
                params.update({'coupon_rate': coupon_rate, 'face_value': face_value})
            elif instrument_type == "EuropeanOption":
                params.update({'strike': strike, 'option_type': option_type})
            elif instrument_type == "ConvertibleBond":
                params.update({'coupon_rate': coupon_rate, 'conversion_ratio': conversion_ratio})
            
            inst_def = st.session_state.risk_manager.create_instrument_definition(
                instrument_type, **params
            )
            
            if inst_def:
                portfolio_definitions.append(inst_def)
                holdings.append({'quantity': quantity})
                st.success(f"Added {instrument_type} with quantity {quantity}")
        
        # Display current portfolio
        if portfolio_definitions:
            st.subheader("Current Portfolio")
            portfolio_df = pd.DataFrame([
                {
                    "Instrument": f"Inst_{i+1}",
                    "Type": inst["product_type"],
                    "Quantity": holding["quantity"],
                    "Maturity": inst.get("maturity_date", "N/A")
                }
                for i, (inst, holding) in enumerate(zip(portfolio_definitions, holdings))
            ])
            st.dataframe(portfolio_df, use_container_width=True)
            
            # Calculate VaR button
            if st.button("Calculate VaR", type="primary"):
                with st.spinner("Calculating VaR..."):
                    var_value, base_value, risk_contributions = st.session_state.risk_manager.calculate_var(
                        portfolio_definitions, holdings
                    )
                    
                    if var_value is not None:
                        st.session_state.risk_manager.current_var = var_value
                        st.session_state.risk_manager.risk_contributions = risk_contributions
                        
                        # Add to history
                        st.session_state.risk_manager.var_history.append({
                            'timestamp': datetime.now(),
                            'var': var_value,
                            'base_value': base_value
                        })
                        
                        st.success(f"VaR calculated: ${var_value:,.2f}")
    
    with col2:
        st.header("📊 Risk Metrics")
        
        if st.session_state.risk_manager.current_var is not None:
            # Current VaR
            st.metric(
                "Current VaR (1%)",
                f"${st.session_state.risk_manager.current_var:,.2f}",
                delta=None
            )
            
            # Risk contributions
            if st.session_state.risk_manager.risk_contributions:
                st.subheader("Risk Contributions")
                
                contrib_data = []
                for inst_name, contrib in st.session_state.risk_manager.risk_contributions.items():
                    contrib_data.append({
                        "Instrument": inst_name,
                        "Type": contrib["type"],
                        "VaR Contrib": f"${contrib['var_contribution']:,.2f}",
                        "Percentage": f"{contrib['percentage']:.1f}%"
                    })
                
                contrib_df = pd.DataFrame(contrib_data)
                st.dataframe(contrib_df, use_container_width=True)
        
        # Ollama integration
        st.header("🤖 AI Risk Assistant")
        
        if st.button("Get AI Risk Advice"):
            if st.session_state.risk_manager.current_var is not None:
                with st.spinner("Getting AI advice..."):
                    portfolio_summary = f"Portfolio with {len(portfolio_definitions)} instruments"
                    var_result = f"${st.session_state.risk_manager.current_var:,.2f}"
                    market_conditions = "Current market conditions"
                    
                    # Run Ollama query
                    advice = asyncio.run(
                        st.session_state.ollama_assistant.get_risk_advice(
                            portfolio_summary, var_result, market_conditions
                        )
                    )
                    
                    st.text_area("AI Risk Advice", advice, height=200)
            else:
                st.warning("Calculate VaR first to get AI advice")
    
    # Charts and visualizations
    if st.session_state.risk_manager.var_history:
        st.header("📈 VaR History")
        
        # Create VaR history chart
        history_df = pd.DataFrame(st.session_state.risk_manager.var_history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_df['timestamp'],
            y=history_df['var'],
            mode='lines+markers',
            name='VaR',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="VaR Over Time",
            xaxis_title="Time",
            yaxis_title="VaR ($)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Risk contribution pie chart
        if st.session_state.risk_manager.risk_contributions:
            st.subheader("Risk Contribution Breakdown")
            
            labels = list(st.session_state.risk_manager.risk_contributions.keys())
            values = [contrib['percentage'] for contrib in st.session_state.risk_manager.risk_contributions.values()]
            
            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values)])
            fig_pie.update_layout(height=400)
            
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Auto-refresh for real-time updates
    if st.checkbox("Enable Auto-refresh"):
        st.write("Auto-refresh enabled - VaR will be recalculated every 30 seconds")
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main() 