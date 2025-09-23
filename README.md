# Orbimesh Agent Orchestration Platform

**Orbimesh** is a comprehensive AI agent orchestration platform that enables users to discover, manage, and orchestrate AI agents to complete complex multi-step workflows through natural language prompts.

## 🌟 Features

- **🎯 Natural Language Workflow Creation**: Describe complex tasks in plain English and let the system break them down and find suitable agents
- **🤖 Agent Discovery & Registration**: Browse and register AI agents with various capabilities
- **⚡ Multi-Agent Orchestration**: Execute complex workflows using LangGraph with intelligent agent selection and fallback mechanisms  
- **🔄 Real-time Streaming**: Watch workflows execute with live progress updates via WebSocket connections
- **📊 Vector Search**: Semantic agent discovery using pgvector and sentence transformers
- **🛡️ Error Handling**: Robust retry mechanisms and fallback agent selection
- **🎨 Modern UI**: Clean, responsive interface built with Next.js 15 and shadcn/ui

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│     Next.js     │      │     FastAPI      │      │   PostgreSQL    │
│     Frontend    │◄───► │     Backend      │◄───► │   + pgvector    │
│                 │      │                  │      │                 │
│ • Task Builder  │      │ • LangGraph      │      │ • Agent Storage │
│ • Agent Grid    │      │ • Vector Search  │      │ • Capabilities  │
│ • Registration  │      │ • GROQ LLM       │      │ • Endpoints     │
│ • WebSocket UI  │      │ • WebSocket API  │      │ • Embeddings    │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

### Core Components

**Frontend (Next.js 15 + TypeScript)**
- Task builder for natural language workflow creation
- Agent directory with search and filtering
- Real-time orchestration visualization
- Agent registration forms

**Backend (FastAPI + Python)**
- LangGraph orchestration engine
- Vector-based agent matching
- WebSocket streaming for real-time updates
- RESTful API for agent management
- Three sample agents: Finance, News, and Wikipedia

**Database (PostgreSQL + pgvector)**
- Agent metadata storage
- Vector embeddings for semantic search
- Capability and endpoint management

## 🚀 Quick Start

### Prerequisites

- **Node.js** v18+ and pnpm (recommended package manager)
- **Python** v3.11+
- **PostgreSQL** v14+ with the `pgvector` extension enabled
- **GROQ API Key** (for LLM operations)

### 1. Clone the Repository

```bash
git clone https://github.com/Orbimesh/Orbimesh-App.git
cd Orbimesh-App
```

### 2. Backend Setup

#### Create and Activate Virtual Environment

```bash
# Navigate to the backend directory
cd backend

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# On Windows (Command Prompt)
.venv\Scripts\activate.bat
# On macOS/Linux
source .venv/bin/activate
```

#### Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt
```

#### Environment Configuration

Create a `.env` file in the backend directory:

```bash
# Create .env file (Windows PowerShell)
@"
GROQ_API_KEY=your_groq_api_key_here
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
DB_NAME=agentdb
NEWS_AGENT_API_KEY=your_news_api_key_if_needed
"@ | Out-File -FilePath .env -Encoding utf8

# Or create manually with your preferred text editor
```

#### Database Setup

1. Ensure PostgreSQL is running
2. Create a database named `agentdb`
3. Enable the pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

#### Start Backend Services

```bash
# Create database tables
python create_tables.py

# Start the main FastAPI server
uvicorn main:app --reload
```

The backend will be available at `http://127.0.0.1:8000`.

### 3. Frontend Setup

```bash
# Navigate to frontend directory (from project root)
cd frontend

# Install dependencies using pnpm (recommended)
pnpm install

# Create environment file
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" | Out-File -FilePath .env.local -Encoding utf8

# Start development server
pnpm dev
```

The frontend will be available at `http://localhost:3000`.

## 📊 Project Structure

```
Orbimesh-App/
├── backend/                    # FastAPI backend
│   ├── main.py                # Main FastAPI application
│   ├── database.py            # Database configuration
│   ├── models.py              # SQLAlchemy models
│   ├── schemas.py             # Pydantic schemas
│   ├── requirements.txt       # Python dependencies
│   ├── create_tables.py       # Database initialization
│   ├── .env                   # Environment variables
│   │
│   ├── agents/                # Sample AI agents
│   │   ├── finance_agent.py   # Yahoo Finance integration
│   │   ├── news_agent.py      # News API integration
│   │   └── wiki_agent.py      # Wikipedia API integration
│   │
│   └── orchestrator/          # LangGraph orchestration
│       ├── graph.py           # Main orchestration logic
│       └── state.py           # State management
│
└── frontend/                  # Next.js frontend
    ├── app/                   # Next.js App Router
    │   ├── page.tsx          # Home page with task builder
    │   ├── agents/           # Agent directory pages
    │   └── register-agent/   # Agent registration
    │
    ├── components/            # React components
    │   ├── task-builder.tsx           # Natural language task input
    │   ├── workflow-orchestration.tsx # Real-time execution UI
    │   ├── agent-grid.tsx            # Agent discovery interface
    │   ├── agent-registration-form.tsx # Agent registration
    │   └── ui/                       # shadcn/ui components
    │
    ├── lib/                   # Utilities and API clients
    │   ├── api-client.ts     # Backend API integration
    │   └── types.ts          # TypeScript type definitions
    │
    ├── package.json          # Frontend dependencies
    └── .env.local            # Frontend environment variables
```

## 🛠️ Technology Stack

### Frontend
- **Framework**: Next.js 15 (with App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **Icons**: Lucide React
- **Package Manager**: pnpm
- **Real-time Updates**: WebSocket

### Backend
- **Framework**: FastAPI
- **Orchestration**: LangGraph
- **Language**: Python 3.11+
- **LLM Integration**: GROQ
- **ORM**: SQLAlchemy
- **Data Validation**: Pydantic
- **Vector Processing**: Sentence Transformers
- **HTTP Client**: httpx

### Database
- **Primary Store**: PostgreSQL
- **Vector Search**: pgvector extension
- **Embedding Model**: all-MiniLM-L6-v2

## 🎯 Usage

### 1. Create a Workflow

1. Open the application at `http://localhost:3000`
2. Enter a natural language description of your goal (e.g., "Find the top 3 restaurants in Paris and draft an email to my friends to invite them")
3. Click **Parse & Find Agents** to break down the tasks and find suitable agents
4. Review and modify the suggested agents for each step
5. Execute the workflow and watch real-time progress

### 2. Discover Agents

1. Navigate to the **Agent Directory** from the sidebar
2. Browse available agents or use the search and filter options
3. Filter agents by capabilities, price, or rating
4. Click on an agent to view details and test it with custom prompts

### 3. Register a New Agent

1. Click **Register Agent** from the sidebar
2. Fill in the agent's details, capabilities, and API endpoints
3. Preview how your agent's card will appear in the directory
4. Test endpoint connectivity and save the agent

## 🔌 API Endpoints

### Core Endpoints

- `POST /api/chat`: Process a natural language prompt to create and execute a workflow
- `GET /agents/search`: Search for agents by capabilities, price, or rating
- `GET /agents/{agent_id}`: Get details for a specific agent
- `POST /agents/`: Register a new agent
- `GET /api/health`: Health check for the API
- `WS /ws/{thread_id}`: WebSocket endpoint for real-time updates

### Example API Usage

```bash
# Search for agents capable of email drafting
curl "http://127.0.0.1:8000/agents/search?capability=email_drafting"

# Process a workflow via chat endpoint
curl -X POST "http://127.0.0.1:8000/api/chat" \
 -H "Content-Type: application/json" \
 -d '{"prompt": "Help me find a travel agent and then draft an email to them."}'

# Register a new agent
curl -X POST "http://127.0.0.1:8000/agents/" \
 -H "Content-Type: application/json" \
 -d '{
   "id": "my_agent",
   "name": "My Custom Agent",
   "description": "Does custom tasks",
   "capabilities": ["custom_task"],
   "price_per_call_usd": 0.01,
   "endpoints": [{"endpoint": "http://localhost:8080/api", "http_method": "POST"}]
 }'
```

## 🔧 Configuration

### Backend Environment Variables (`.env`)

```env
# Required
GROQ_API_KEY=your_groq_api_key_here
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
DB_NAME=agentdb

# Optional
NEWS_AGENT_API_KEY=your_news_api_key_if_needed
OPENAI_API_KEY=your_openai_key_if_needed
```

### Frontend Environment Variables (`.env.local`)

```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## 🧪 Testing

### Backend Testing

Run the backend tests from the backend directory:

```bash
cd backend
pytest
```

### Integration Testing

Test the full application stack:

1. **Terminal 1** - Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. **Terminal 2** - Run integration tests:
   ```bash
   python testnew.py
   ```

### API Documentation

FastAPI automatically generates interactive API documentation:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 🚀 Deployment

### Production Considerations

- Use a robust WSGI server like **Gunicorn** for the FastAPI backend
- Configure a reverse proxy like **Nginx** to manage traffic
- Use environment-specific `.env` files and manage secrets securely
- Set up automated database backups
- Implement process managers (e.g., PM2, systemd) to ensure services are always running
- Integrate monitoring and logging solutions (e.g., Grafana, Prometheus, Sentry)

## 🆘 Troubleshooting

### Common Issues

- **Backend won't start:**
  - Ensure PostgreSQL server is running
  - Verify database credentials in `backend/.env` are correct
  - Confirm the `pgvector` extension is installed and enabled in your database

- **Frontend can't connect to backend:**
  - Verify `NEXT_PUBLIC_API_URL` in `.env.local` points to your backend URL
  - Ensure the backend server is running on the specified port (default: 8000)
  - Check for CORS errors in your browser's developer console

- **GROQ API errors:**
  - Ensure `GROQ_API_KEY` in `backend/.env` is set correctly
  - Check that your GROQ API key has sufficient credits
  - Verify your server has internet connectivity

- **Virtual Environment Issues:**
  - Ensure you've activated the virtual environment before installing packages
  - Use the correct activation script for your OS and shell
  - If pip fails, try upgrading pip: `python -m pip install --upgrade pip`

### Getting Help

- Check the [Issues](https://github.com/Orbimesh/Orbimesh-App/issues) page for existing bug reports and feature requests
- Review the API documentation at `http://127.0.0.1:8000/docs`
- Run the backend tests via `pytest` to diagnose potential issues

## 🤝 Contributing

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built by the Orbimesh team**