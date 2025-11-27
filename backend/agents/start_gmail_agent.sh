#!/bin/bash
# Start Gmail MCP Agent

echo "🚀 Starting Gmail MCP Agent..."

# Check if .env exists
if [ ! -f "../.env" ]; then
    echo "❌ Error: backend/.env not found"
    echo "Please create .env file with:"
    echo "  COMPOSIO_API_KEY=your_key"
    echo "  GMAIL_MCP_URL=your_mcp_url"
    exit 1
fi

# Check for required env vars
source ../.env

if [ -z "$COMPOSIO_API_KEY" ]; then
    echo "❌ Error: COMPOSIO_API_KEY not set in .env"
    exit 1
fi

if [ -z "$GMAIL_MCP_URL" ]; then
    echo "❌ Error: GMAIL_MCP_URL not set in .env"
    exit 1
fi

# Get port from env or use default
PORT=${GMAIL_AGENT_PORT:-8095}

echo "✅ Configuration found"
echo "📡 Starting agent on port $PORT..."

# Start the agent
cd ..
python agents/gmail_mcp_agent.py
