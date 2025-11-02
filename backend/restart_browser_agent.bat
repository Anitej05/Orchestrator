@echo off
echo Stopping browser automation agent on port 8070...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8070" ^| find "LISTENING"') do taskkill /F /PID %%a 2>nul
timeout /t 2 /nobreak >nul
echo Starting browser automation agent...
cd agents
start "Browser Agent" python browser_automation_agent.py
cd ..
echo Browser agent restarted!
