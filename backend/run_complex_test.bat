@echo off
echo ========================================
echo Complex Task Test Runner
echo ========================================
echo.

REM Check if virtual environment exists
if exist venv\Scripts\activate.bat (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo No virtual environment found, using system Python
)

echo.
echo Running complex task test...
echo.

python test_complex_task.py

echo.
echo ========================================
echo Test execution complete
echo ========================================
echo.
pause
