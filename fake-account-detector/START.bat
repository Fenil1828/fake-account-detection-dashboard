@echo off
echo ================================================
echo    FAKE ACCOUNT DETECTOR - SECURE ML SYSTEM
echo ================================================
echo.

cd /d "%~dp0"

echo [1/3] Training ML Model...
python backend/model_training.py
if errorlevel 1 (
    echo ERROR: Model training failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Starting Backend API on port 5000...
start "Backend API" cmd /k "python backend/app.py"

echo.
echo [3/3] Starting React Frontend on port 5173...
cd frontend/react-dashboard
start "Frontend" cmd /k "npm run dev -- --host"

echo.
echo ================================================
echo    SYSTEM STARTED SUCCESSFULLY!
echo ================================================
echo.
echo    Backend API:  http://localhost:5000
echo    Frontend:     http://localhost:5173
echo.
echo    Opening browser...
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo Press any key to close this window...
pause >nul
