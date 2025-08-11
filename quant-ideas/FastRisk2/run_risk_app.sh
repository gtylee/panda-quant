#!/bin/bash

# Real-Time Risk Management Dashboard Launcher
echo "🚀 Starting Real-Time Risk Management Dashboard..."

# Check if calibrated models exist
if [ ! -f "calibrated_models_tff.json" ]; then
    echo "⚠️  Warning: calibrated_models_tff.json not found!"
    echo "   Please run the hybrid VaR demo first:"
    echo "   python -m demos.demo_hybrid_var_simplified"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if Streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements_streamlit.txt
fi

# Check if Ollama is running (optional)
if command -v ollama &> /dev/null; then
    if ! pgrep -x "ollama" > /dev/null; then
        echo "🤖 Ollama not running. Starting Ollama in background..."
        ollama serve &
        sleep 3
    fi
else
    echo "⚠️  Ollama not found. AI risk advice will not be available."
    echo "   Install Ollama from: https://ollama.ai"
fi

# Launch the Streamlit app
echo "📊 Launching Streamlit app..."
echo "   The app will open in your browser at: http://localhost:8501"
echo "   Press Ctrl+C to stop the app"
echo ""

streamlit run streamlit_risk_app.py 