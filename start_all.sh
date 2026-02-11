#!/bin/bash
export PYTHONPATH=/home/clawuser/.openclaw/workspace/Orchestrator:/home/clawuser/.openclaw/workspace/Orchestrator/backend
cd /home/clawuser/.openclaw/workspace/Orchestrator

# Create logs directory
mkdir -p logs_all

# Kill everything
ps aux | grep -v grep | grep -E "spreadsheet_agent|mail_agent|browser_agent|document_agent|zoho_books|uvicorn|main:app|npm|next" | awk '{print $2}' | xargs -r kill -9

# Start Agents
echo "Starting agents..."
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.spreadsheet_agent:app --host 127.0.0.1 --port 9000 > logs_all/spreadsheet.log 2>&1 &
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.mail_agent.agent:app --host 127.0.0.1 --port 8040 > logs_all/mail.log 2>&1 &
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.browser_agent:app --host 127.0.0.1 --port 8090 > logs_all/browser.log 2>&1 &
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.document_agent_lib:app --host 127.0.0.1 --port 8050 > logs_all/document.log 2>&1 &
nohup ./backend/venv/bin/python3 -m uvicorn backend.agents.zoho_books.zoho_books_agent:app --host 127.0.0.1 --port 8060 > logs_all/zoho.log 2>&1 &

# Start Main Backend
echo "Starting main backend..."
cd backend
nohup ../backend/venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > ../logs_all/backend.log 2>&1 &
cd ..

# Start Frontend
echo "Starting frontend..."
cd frontend
# Run without Turbopack
nohup npx next dev -p 3000 > ../logs_all/frontend.log 2>&1 &
cd ..

echo "Waiting for initialization..."
sleep 25
ps aux | grep -E "uvicorn|node|next"

echo "Checking health..."
curl -s http://127.0.0.1:8000/docs | head -n 5
curl -s http://127.0.0.1:9000/health
curl -s http://127.0.0.1:8040/health
curl -s http://127.0.0.1:8090/health
curl -s http://127.0.0.1:8050/health
curl -s http://127.0.0.1:8060/health
