#!/bin/bash

# Combined startup script for HHElrangen + CarePilot

echo "🏥 Starting Smart Clinical Copilot (HHElrangen + CarePilot)..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first."
    exit 1
fi

# Activate virtual environment
source venv/bin/activate

# Check if carepilot-embed is built
if [ ! -d "carepilot-embed/dist" ] || [ ! -f "carepilot-embed/dist/widget.iife.js" ]; then
    echo "📦 Building CarePilot widget..."
    cd carepilot-embed
    npm run build
    cd ..
    echo "✅ Widget built successfully!"
    echo ""
fi

# Start backend
echo "🚀 Starting backend server..."
cd backend
uvicorn main:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

echo "✅ Backend started on http://127.0.0.1:8000"
echo "   - API docs: http://127.0.0.1:8000/docs"
echo "   - Demo page: http://127.0.0.1:8000/demo"
echo ""

# Wait a moment for backend to start
sleep 2

# Start Streamlit frontend (optional)
echo "🎨 Starting Streamlit frontend..."
cd frontend
streamlit run app.py --server.port 8501 &
STREAMLIT_PID=$!
cd ..

echo "✅ Streamlit started on http://127.0.0.1:8501"
echo ""

echo "🎉 All services started!"
echo ""
echo "📍 Access points:"
echo "   - Backend API: http://127.0.0.1:8000"
echo "   - API Docs: http://127.0.0.1:8000/docs"
echo "   - CarePilot Demo: http://127.0.0.1:8000/demo"
echo "   - Streamlit App: http://127.0.0.1:8501"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for user interrupt
trap "kill $BACKEND_PID $STREAMLIT_PID 2>/dev/null; echo ''; echo '🛑 Stopped all services.'; exit" INT
wait

