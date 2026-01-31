#!/bin/bash

echo "🎯 Starting Fake Account Detection System..."
echo ""

# Check if model exists
if [ ! -f "models/detector.pkl" ]; then
    echo "⚠️  Model not found. Training now..."
    python backend/model_training.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Model training failed. Please check errors above."
        exit 1
    fi
fi

# Start backend API in background
echo ""
echo "🔧 Starting Backend API..."
python backend/app.py &
BACKEND_PID=$!

# Wait for backend to start
echo "   Waiting for API to initialize..."
sleep 5

# Check if backend is running
if ps -p $BACKEND_PID > /dev/null; then
    echo "   ✓ Backend API started (PID: $BACKEND_PID)"
else
    echo "   ❌ Backend failed to start"
    exit 1
fi

# Start dashboard
echo ""
echo "🎨 Starting Dashboard..."
python frontend/dashboard.py &
DASHBOARD_PID=$!

sleep 3

if ps -p $DASHBOARD_PID > /dev/null; then
    echo "   ✓ Dashboard started (PID: $DASHBOARD_PID)"
else
    echo "   ❌ Dashboard failed to start"
    kill $BACKEND_PID
    exit 1
fi

echo ""
echo "="*50
echo "✅ System is running!"
echo "="*50
echo ""
echo "📊 Dashboard: http://localhost:8050"
echo "🔌 API: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
trap "echo ''; echo 'Stopping services...'; kill $BACKEND_PID $DASHBOARD_PID; echo '✓ All services stopped'; exit" INT
wait
