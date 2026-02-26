#!/bin/bash

# Orbital Agronomy - Quick Start Guide
# =====================================

echo "🚀 Orbital Agronomy Setup & Execution Guide"
echo "============================================"
echo ""

# Check current directory
if [ ! -d "orbital_agronomy" ]; then
    echo "❌ Error: Must run from ENIGMA directory"
    exit 1
fi

cd orbital_agronomy

echo "📋 Step 1: Verify Directory Structure"
echo "--------------------------------------"
echo "✅ Models directory:"
ls -la models/
echo ""
echo "✅ Source code:"
ls -la src/
echo ""

echo "📋 Step 2: Install Python Dependencies"
echo "--------------------------------------"
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt
echo "✅ Python dependencies installed"
echo ""

echo "📋 Step 3: Start Backend Server"
echo "--------------------------------------"
echo "Starting FastAPI backend on port 8000..."
echo ""
echo "Command: python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "To start backend, run:"
echo "  cd orbital_agronomy"
echo "  source venv/bin/activate"
echo "  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""

echo "📋 Step 4: Start Frontend (in new terminal)"
echo "--------------------------------------"
echo "Commands:"
echo "  cd frontend"
echo "  npm install"
echo "  npm run dev"
echo ""

echo "📋 Step 5: Access Application"
echo "--------------------------------------"
echo "Frontend: http://localhost:5173"
echo "Backend API: http://localhost:8000"
echo "API Docs: http://localhost:8000/docs"
echo ""

echo "📋 Optional: Train Models"
echo "--------------------------------------"
echo ""
echo "Train Random Forest:"
echo "  python src/Scikitlearn/train_model.py"
echo ""
echo "Train Deep Learning (MLP):"
echo "  python src/Scikitlearn/train_deep_model.py"
echo ""
echo "Train 2D CNN:"
echo "  python src/CNN_2D/train_cnn_2d.py"
echo ""

echo "📋 Model Locations"
echo "--------------------------------------"
echo "U-Net (CNN 2D):        models/CNN_2D/stress_vision_weights.pth"
echo "Deep Learning (MLP):   models/CNN+scikit/deep_stress_model.pkl"
echo "Random Forest:         models/CNN+scikit/stress_vision_model.pkl"
echo ""

echo "✅ Setup guide complete!"
echo "========================"
