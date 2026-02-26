#!/bin/bash
# Quick test script for Orbital Agronomy backend

echo "🧪 Testing FastAPI Backend..."
echo ""

cd /home/smitp/unstop/ENIGMA
source orbital_agronomy/venv/bin/activate

# Test 1: Check imports
echo "1️⃣  Testing imports..."
python -c "from orbital_agronomy.app.main import app; print('✅ Imports successful')" || exit 1

# Test 2: Check if server can start (dry run)
echo "2️⃣  Checking server configuration..."
python -c "
from orbital_agronomy.app.main import app, load_model
load_model()
print('✅ Server configuration valid')
" || exit 1

echo ""
echo "✅ All tests passed!"
echo ""
echo "🚀 Ready to launch:"
echo "   python orbital_agronomy/app/main.py"
echo ""
echo "📊 Frontend:"
echo "   cd frontend && npm run dev"
