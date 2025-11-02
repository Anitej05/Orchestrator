@echo off
echo ============================================================
echo Restarting Backend and Frontend Services
echo ============================================================
echo.

echo Step 1: Installing/Updating Backend Dependencies...
cd backend
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo Step 2: Testing WebSocket Connection...
python test_websocket.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: WebSocket test failed - backend might not be running
    echo Please start the backend manually with: python main.py
)
echo.

echo ============================================================
echo Services Ready!
echo ============================================================
echo.
echo To start the backend:
echo   cd backend
echo   python main.py
echo.
echo To start the frontend:
echo   cd frontend
echo   npm run dev
echo.
pause
