@echo off
REM Combined startup script for HHElrangen + CarePilot (Windows)

echo 🏥 Starting Smart Clinical Copilot (HHElrangen + CarePilot)...
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo ❌ Virtual environment not found. Please run setup first.
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Check if carepilot-embed is built
if not exist "carepilot-embed\dist\widget.iife.js" (
    echo 📦 Building CarePilot widget...
    cd carepilot-embed
    call npm run build
    cd ..
    echo ✅ Widget built successfully!
    echo.
)

REM Start backend
echo 🚀 Starting backend server...
cd backend
start "Backend Server" cmd /k "uvicorn main:app --reload --port 8000"
cd ..
timeout /t 2 /nobreak >nul

echo ✅ Backend started on http://127.0.0.1:8000
echo    - API docs: http://127.0.0.1:8000/docs
echo    - Demo page: http://127.0.0.1:8000/demo
echo.

REM Start Streamlit frontend
echo 🎨 Starting Streamlit frontend...
cd frontend
start "Streamlit App" cmd /k "streamlit run app.py --server.port 8501"
cd ..

echo ✅ Streamlit started on http://127.0.0.1:8501
echo.

echo 🎉 All services started!
echo.
echo 📍 Access points:
echo    - Backend API: http://127.0.0.1:8000
echo    - API Docs: http://127.0.0.1:8000/docs
echo    - CarePilot Demo: http://127.0.0.1:8000/demo
echo    - Streamlit App: http://127.0.0.1:8501
echo.
echo Press any key to exit...
pause >nul

