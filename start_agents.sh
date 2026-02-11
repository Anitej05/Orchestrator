#!/bin/bash
export PYTHONPATH=/home/clawuser/.openclaw/workspace/Orchestrator:/home/clawuser/.openclaw/workspace/Orchestrator/backend
cd /home/clawuser/.openclaw/workspace/Orchestrator

# Kill any existing agent processes
ps aux | grep -v grep | grep -E "spreadsheet_agent|mail_agent|browser_agent|document_agent|zoho_books|uvicorn" | awk '{print $2}' | xargs -r kill -9

# Start Spreadsheet Agent (9000)
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.spreadsheet_agent:app --host 127.0.0.1 --port 9000 > spreadsheet.log 2>&1 &
echo "Spreadsheet Agent starting on 9000..."

# Start Mail Agent (8040)
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.mail_agent.agent:app --host 127.0.0.1 --port 8040 > mail.log 2>&1 &
echo "Mail Agent starting on 8040..."

# Start Browser Agent (8090)
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.browser_agent:app --host 127.0.0.1 --port 8090 > browser.log 2>&1 &
echo "Browser Agent starting on 8090..."

# Start Document Agent (8050)
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.document_agent_lib:app --host 127.0.0.1 --port 8050 > document.log 2>&1 &
echo "Document Agent starting on 8050..."

# Start Zoho Books Agent (8060)
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.zoho_books.zoho_books_agent:app --host 127.0.0.1 --port 8060 > zoho.log 2>&1 &
echo "Zoho Books Agent starting on 8060..."

sleep 15
ps aux | grep uvicorn
