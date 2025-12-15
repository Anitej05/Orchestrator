@echo off
REM Start all required microservice agents for Orbimesh

echo ========================================
echo Starting Orbimesh Microservice Agents
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python and try again.
    pause
    exit /b 1
)

echo Starting Document Analysis Agent on port 8070...
start "Document Agent [Port 8070]" cmd /k "cd /d %~dp0 && python agents\document_analysis_agent.py"

timeout /t 2 /nobreak >nul

echo Starting Spreadsheet Agent on port 8041...
start "Spreadsheet Agent [Port 8041]" cmd /k "cd /d %~dp0 && python agents\spreadsheet_agent.py"

echo.
echo ========================================
echo All agents started successfully!
echo ========================================
echo.
echo Running agents:
echo - Document Analysis Agent: http://localhost:8070
echo - Spreadsheet Agent: http://localhost:8041
echo.
echo Press Ctrl+C in each terminal to stop agents
echo ========================================
