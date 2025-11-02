@echo off
echo ============================================================
echo Installing Missing Dependencies
echo ============================================================
echo.

cd backend

echo Installing Python packages...
pip install -r requirements.txt

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo SUCCESS: All dependencies installed!
    echo ============================================================
    echo.
    echo You can now start the backend with:
    echo   cd backend
    echo   python main.py
    echo.
) else (
    echo.
    echo ============================================================
    echo ERROR: Failed to install dependencies
    echo ============================================================
    echo.
    echo Please check the error messages above and try again.
    echo.
)

pause
