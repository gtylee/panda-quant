# Real-Time Risk Management Dashboard

A sophisticated Streamlit application for real-time risk management using TFF (Tensor Functional Form) calibrated instruments. This app demonstrates live VaR calculation, portfolio management, and risk contribution analysis.

## Features

### 🚀 Core Functionality
- **Real-time Portfolio Building**: Define portfolios on the fly with multiple instrument types
- **Live VaR Calculation**: Calculate VaR using pre-calibrated TFF models
- **Market Data Streaming**: Apply market shocks and see real-time VaR updates
- **Risk Contribution Analysis**: Understand which instruments contribute most to risk
- **AI-Powered Risk Advice**: Get intelligent risk management recommendations via Ollama

### 📊 Instrument Types Supported
- **Vanilla Bonds**: Fixed income instruments with customizable parameters
- **European Options**: Equity options with configurable strikes and types
- **Convertible Bonds**: Hybrid instruments with equity conversion features

### 📈 Real-Time Features
- Live VaR calculation with TFF approximators
- Market shock application (interest rates, equity prices, volatility)
- Risk contribution breakdown by instrument
- Historical VaR tracking and visualization
- Auto-refresh capabilities for continuous monitoring

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Set up Ollama (Optional)
For AI risk advice functionality:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (e.g., llama2)
ollama pull llama2
```

### 3. Ensure Calibrated Models Exist
Make sure you have run the hybrid VaR demo first to generate calibrated models:
```bash
python -m demos.demo_hybrid_var_simplified
```

This will create:
- `calibrated_models_tff.json`
- `calibrated_models_rbfi.json`

## Usage

### Starting the App
```bash
streamlit run streamlit_risk_app.py
```

The app will open in your browser at `http://localhost:8501`

### Step-by-Step Guide

#### 1. Initialize the Risk Manager
- Click "Initialize Risk Manager" in the sidebar
- This loads the pre-calibrated TFF models and sets up the scenario generator

#### 2. Build Your Portfolio
- Select an instrument type (VanillaBond, EuropeanOption, ConvertibleBond)
- Configure the instrument parameters:
  - **Quantity**: Number of instruments to hold
  - **Maturity**: Time to maturity in years
  - **Instrument-specific parameters**: Coupon rates, strikes, conversion ratios, etc.
- Click "Add Instrument" to add it to your portfolio
- Repeat for all desired instruments

#### 3. Calculate VaR
- Once your portfolio is built, click "Calculate VaR"
- The app will use the TFF models to calculate 1% VaR
- Results will show current VaR and risk contributions

#### 4. Apply Market Shocks
- Use the sidebar to apply market shocks:
  - **Interest Rate Shocks**: Modify rates for different tenors (in basis points)
  - **Equity Shocks**: Change AAPL stock price
  - **Volatility Shocks**: Adjust implied volatility
- Click "Apply Market Shocks" to see how VaR changes

#### 5. Get AI Risk Advice
- Click "Get AI Risk Advice" to receive intelligent recommendations
- Requires Ollama to be running with a language model

#### 6. Monitor Real-Time
- Enable auto-refresh for continuous monitoring
- View VaR history charts and risk contribution breakdowns

## Architecture

### Components

#### RealTimeRiskManager
- Core risk management engine
- Handles portfolio building and VaR calculation
- Manages market data updates and scenario generation

#### OllamaRiskAssistant
- AI-powered risk advisory system
- Integrates with Ollama for intelligent recommendations
- Provides actionable risk management advice

#### Streamlit Interface
- User-friendly web interface
- Real-time data visualization
- Interactive portfolio management

### Data Flow
1. **Portfolio Definition** → User defines instruments and quantities
2. **Model Loading** → TFF models loaded from calibrated files
3. **Scenario Generation** → Market scenarios generated with current parameters
4. **VaR Calculation** → TFF approximators calculate portfolio VaR
5. **Risk Analysis** → Risk contributions calculated for each instrument
6. **Visualization** → Results displayed in charts and tables
7. **AI Integration** → Ollama provides risk management advice

## Configuration

### Market Parameters
- **Base Interest Rates**: Configurable for different tenors
- **Equity Parameters**: Stock price, dividend yield, volatility
- **Credit Spreads**: For convertible bonds and credit-sensitive instruments

### Risk Parameters
- **VaR Percentile**: Default 1% (99% confidence level)
- **Scenario Count**: Configurable for accuracy vs. speed trade-off
- **Critical Percentile**: For hybrid VaR calculations

## Advanced Features

### Real-Time Market Data Integration
The app can be extended to integrate with:
- Bloomberg Terminal
- Reuters Eikon
- Custom market data feeds
- WebSocket connections for live data

### Portfolio Optimization
Future enhancements could include:
- Portfolio optimization algorithms
- Risk budgeting tools
- Stress testing scenarios
- Regulatory reporting

### Machine Learning Integration
- Advanced risk models
- Predictive analytics
- Anomaly detection
- Automated risk alerts

## Troubleshooting

### Common Issues

#### "Failed to initialize Risk Manager"
- Ensure calibrated model files exist
- Check that all dependencies are installed
- Verify file paths are correct

#### "Error calculating VaR"
- Check portfolio definition validity
- Ensure sufficient scenarios are generated
- Verify model registry integrity

#### "Ollama connection failed"
- Ensure Ollama is running: `ollama serve`
- Check if model is available: `ollama list`
- Verify network connectivity

#### Performance Issues
- Reduce scenario count for faster calculation
- Use fewer instruments in portfolio
- Disable auto-refresh for better performance

## Development

### Adding New Instrument Types
1. Extend `create_instrument_definition()` method
2. Add instrument-specific parameters
3. Update portfolio builder logic
4. Test with sample data

### Customizing Risk Metrics
1. Modify `calculate_risk_contributions()` method
2. Add new risk measures (ES, CVaR, etc.)
3. Update visualization components
4. Extend AI prompts for new metrics

### Performance Optimization
- Implement caching for repeated calculations
- Use parallel processing for large portfolios
- Optimize scenario generation algorithms
- Add database storage for historical data

## License

This project is part of the FastRisk framework and follows the same licensing terms.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the code documentation
3. Open an issue on GitHub
4. Contact the development team 