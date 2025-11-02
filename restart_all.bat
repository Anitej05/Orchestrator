@echo off
echo Stopping all Python processes...
taskkill /F /IM python.exe /T 2>nul

echo Waiting for processes to stop...
timeout /t 2 /nobreak >nul

echo Starting main server (which will start all agents)...
cd backend
start "Orbimesh Server" python main.py

echo.
echo Server starting...
echo Check the new window for logs.
echo.
pause
