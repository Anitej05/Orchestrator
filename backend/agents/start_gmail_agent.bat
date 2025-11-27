@echo off
REM Start Gmail MCP Agent (Windows)

echo Starting Gmail MCP Agent...

REM Check if .env exists
if not exist "..\\.env" (
    echo Error: backend\.env not found
    echo Please create .env file with:
    echo   COMPOSIO_API_KEY=your_key
    echo   GMAIL_MCP_URL=your_mcp_url
    exit /b 1
)

REM Get port from env or use default
if not defined GMAIL_AGENT_PORT set GMAIL_AGENT_PORT=8095

echo Configuration found
echo Starting agent on port %GMAIL_AGENT_PORT%...

REM Start the agent
cd ..
python agents\gmail_mcp_agent.py
