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

## 🚀 Quick Start & Setup Guide

### ⚠️ Important: Read This First!
This guide covers **fresh installation** and **updating existing installations**. Follow the appropriate section for your scenario.

### Prerequisites

- **Node.js** v18+ (for frontend)
- **Python** v3.11+ (for backend)
- **PostgreSQL** v14+ (with pgvector extension)
- **Git** (for version control)
- **GROQ API Key** (get free key at https://console.groq.com)
- **Cerebras API Key** (optional, for document analysis)

---

## 📋 Setup Instructions

### 1️⃣ Clone or Update Repository

#### Fresh Installation
```bash
git clone https://github.com/Orbimesh/Orbimesh-App.git
cd Orbimesh-App
```

#### Updating Existing Installation
```bash
cd Orbimesh-App
git pull origin main
```

---

### 2️⃣ Backend Setup

#### Step 1: Navigate to Backend Directory
```bash
cd backend
```

#### Step 2: Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# If you get an execution policy error, run:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**On Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**On macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install/Update All Python Dependencies
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# For existing installations, use:
pip install -r requirements.txt --upgrade
```

#### Step 4: Configure Environment Variables

1. **Copy the example environment file:**
   ```bash
   # Windows (PowerShell)
   Copy-Item .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```

2. **Edit `.env` file** with your actual values:
   ```env
   # REQUIRED: API Keys
   GROQ_API_KEY=your_groq_api_key_here
   CEREBRAS_API_KEY=your_cerebras_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   
   # Agent-specific API keys (optional)
   SCHOLARAI_API_KEY=your_scholarai_api_key_here
   NEWS_AGENT_API_KEY=your_news_agent_api_key_here
   OLLAMA_API_KEY=your_ollama_api_key_here
   BROWSER_AGENT_PORT=8070
   
   # REQUIRED: Database Configuration
   PG_USER=postgres
   PG_PASSWORD=your_postgres_password_here
   PG_HOST=localhost
   PG_PORT=5432
   DB_NAME=agentdb
   ```

   **✓ Checklist - Verify all variables from `.env.example` are present in your `.env`**

#### Step 5: Setup PostgreSQL Database

**Prerequisite: PostgreSQL is installed and running**

1. **Create the database and enable pgvector:**
   ```sql
   -- Connect to PostgreSQL with a client (psql, pgAdmin, or DBeaver)
   
   -- Create database
   CREATE DATABASE agentdb;
   
   -- Connect to the database
   \c agentdb
   
   -- Enable pgvector extension
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Alternative: Using psql command line:**
   ```bash
   # Create database
   createdb -U postgres agentdb
   
   # Enable pgvector
   psql -U postgres -d agentdb -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

#### Step 6: Initialize/Update Database Schema

**For Fresh Installation:**
```bash
# Create all tables from scratch
python db_init.py
```

**For Existing Installation (Database Updates):**
```bash
# Run migrations to update schema
alembic upgrade head

# If there are no migrations, create them:
alembic revision --autogenerate -m "Update schema"
alembic upgrade head
```

**Verify Database Setup:**
```bash
# This command should complete without errors
python -c "from database import SessionLocal; db = SessionLocal(); print('✓ Database connection successful')"
```

#### Step 7: Start Backend Server

```bash
# Development mode with auto-reload
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Or simply:
python main.py
```

✓ **Backend is ready when you see:**
```
INFO:     Application startup complete
```

Visit API docs: `http://127.0.0.1:8000/docs`

---

### 3️⃣ Frontend Setup

#### Step 1: Navigate to Frontend Directory
```bash
# From the project root
cd frontend
```

#### Step 2: Install/Update Node Dependencies

**Using pnpm (Recommended - Faster & Better):**
```bash
# Install pnpm globally (if not already installed)
npm install -g pnpm

# Install dependencies
pnpm install

# For existing installation, update dependencies:
pnpm install --latest
```

**Using npm:**
```bash
npm install

# For existing installation:
npm install --legacy-peer-deps
```

**Using yarn:**
```bash
yarn install
```

#### Step 3: Configure Frontend Environment

Create `.env.local` file in the frontend directory:
```bash
# Windows (PowerShell)
@"
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
"@ | Out-File -FilePath .env.local -Encoding utf8

# macOS/Linux
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" > .env.local
```

#### Step 4: Start Frontend Development Server

```bash
pnpm dev
# or: npm run dev
```

✓ **Frontend is ready when you see:**
```
▲ Next.js 15.2.4
  - Local:        http://localhost:3000
```

Visit frontend: `http://localhost:3000`

---

## �️ Database Management & Updates

### Keeping Your Database Up to Date

**Important:** Whenever you pull new code changes, the database schema might have changed. Follow these steps:

#### Check for Pending Migrations
```bash
cd backend

# Activate virtual environment first
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # macOS/Linux

# View current database version
alembic current

# View all migration history
alembic history

# Check for pending/unapplied migrations
alembic status
```

#### Apply Migrations (Required After Code Updates)
```bash
cd backend

# Upgrade to latest migration
alembic upgrade head

# View detailed migration output
alembic upgrade head -v  # verbose mode
```

#### If You Modified Models (Developers Only)

When you make changes to `models.py`, create and apply migrations:

```bash
cd backend

# 1. Create migration script (auto-generates based on model changes)
alembic revision --autogenerate -m "Describe your changes"

# 2. Review the generated migration file in alembic/versions/
# 3. Apply the migration
alembic upgrade head

# 4. Commit the migration file to git
git add alembic/versions/
git commit -m "Add migration for [your changes]"
```

#### Rollback Migrations (If Something Goes Wrong)

```bash
cd backend

# Downgrade to previous version
alembic downgrade -1

# Downgrade multiple versions
alembic downgrade -2

# View current version after rollback
alembic current
```

#### Database Backup (Before Major Updates)

**On Windows (PowerSQL):**
```powershell
# Backup database
pg_dump -U postgres -h localhost agentdb > backup_$(Get-Date -Format 'yyyy-MM-dd_HHmmss').sql

# Restore from backup
psql -U postgres -h localhost agentdb < backup_file.sql
```

**On macOS/Linux:**
```bash
# Backup database
pg_dump -U postgres -h localhost agentdb > backup_$(date +'%Y-%m-%d_%H%M%S').sql

# Restore from backup
psql -U postgres -h localhost agentdb < backup_file.sql
```

#### Database Cleanup (Optional - Advanced)

```bash
# Connect to database
psql -U postgres -d agentdb

# List all tables
\dt

# View table sizes
\d+ table_name

# Reset sequence (if needed after deletes)
SELECT setval('table_id_seq', (SELECT MAX(id) + 1 FROM table_name));
```

#### Common Database Issues

| Issue | Solution |
|-------|----------|
| **"relation does not exist"** | Run `alembic upgrade head` to apply pending migrations |
| **"column does not exist"** | Pull latest code and run `alembic upgrade head` |
| **"connection refused"** | Verify PostgreSQL is running and credentials in `.env` are correct |
| **"permission denied"** | Verify PG_USER has proper permissions on the database |
| **Migration conflicts** | Check migration files in `alembic/versions/` for conflicts |

---

## 🔄 Complete Development Startup

After initial setup, here's how to start everything:

**Terminal 1 - Backend:**
```bash
cd backend
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS/Linux

# ⚠️ IMPORTANT: Apply pending database migrations first
alembic upgrade head

# Then start the server
python -m uvicorn main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
pnpm dev
```

**Terminal 3 - Database (if needed):**
PostgreSQL should already be running in the background.

---

## 📊 Verifying Installation

### Backend Verification
```bash
# Check API is running
curl http://127.0.0.1:8000/health

# Check Swagger docs
# Open http://127.0.0.1:8000/docs in your browser
```

### Frontend Verification
```bash
# Frontend should be accessible at
# http://localhost:3000

# Check browser console for errors (F12)
```

### Database Verification
```bash
# Connect to database
psql -U postgres -d agentdb

# List tables (should see many tables)
\dt

# Check pgvector extension
\dx
```

---

## 🆘 Common Setup Issues & Solutions

### **Issue 1: "ModuleNotFoundError: No module named 'xxx'"**
**Solution:**
```bash
# Activate virtual environment first
cd backend
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # macOS/Linux

# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### **Issue 2: PostgreSQL Connection Error**
**Solution:**
```bash
# Verify PostgreSQL is running
# Windows: Check Services (services.msc) for "PostgreSQL"
# macOS: brew services list
# Linux: sudo systemctl status postgresql

# Check credentials in .env file match your PostgreSQL setup
# Default: PG_USER=postgres, PG_PASSWORD=postgres

# Test connection
psql -U postgres -h localhost -d agentdb
```

### **Issue 3: "pgvector extension not found"**
**Solution:**
```bash
# Install pgvector extension in PostgreSQL
psql -U postgres -d agentdb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Or via pgAdmin:
# 1. Connect to agentdb database
# 2. Right-click Extensions
# 3. Create new Extension
# 4. Search for "vector" and install
```

### **Issue 4: Alembic migration fails**
**Solution:**
```bash
# Check if migrations folder exists
# If not, initialize alembic:
alembic init migrations

# View pending migrations
alembic history

# Downgrade if needed, then upgrade
alembic downgrade -1
alembic upgrade head

# Or reset migrations (⚠️ deletes data):
# alembic downgrade base
# alembic upgrade head
```

### **Issue 5: Frontend dependencies conflict**
**Solution:**
```bash
# Clear cache and reinstall
cd frontend
rm -rf node_modules pnpm-lock.yaml  # or package-lock.json for npm
pnpm install --no-frozen-lockfile

# Or with npm:
npm install --legacy-peer-deps
```

### **Issue 6: GROQ API Key Invalid**
**Solution:**
- Get free GROQ API key: https://console.groq.com
- Copy your API key
- Update `GROQ_API_KEY` in `backend/.env`
- Restart backend server

### **Issue 8: "Table/Column does not exist" after code update**
**Solution:**
```bash
# This means database migrations haven't been applied
cd backend

# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # macOS/Linux

# View pending migrations
alembic status

# Apply all pending migrations
alembic upgrade head

# Verify they applied
alembic current
```

### **Issue 9: Migration conflicts or failed migrations**
**Solution:**
```bash
# First, backup your database (see backup section above)
pg_dump -U postgres agentdb > backup.sql

# Check migration history
cd backend
alembic history

# Downgrade to before the issue
alembic downgrade -1

# Try upgrading again
alembic upgrade head

# If still failing, consult the migration file in alembic/versions/
```

### **Issue 10: Database structure out of sync with code**
**Solution:**
```bash
cd backend

# Check current database state
alembic current

# View all applied migrations
alembic history

# View pending migrations
alembic status

# If status shows pending, apply them:
alembic upgrade head

# If completely broken, reset (⚠️ DELETES ALL DATA):
# alembic downgrade base
# alembic upgrade head
```

---

## 🔧 Updating to Latest Version

**For Team Members with Existing Installation:**

```bash
# 1. Pull latest changes
git pull origin main

# 2. Update backend dependencies and database
cd backend

# Activate virtual environment
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # macOS/Linux

# Update Python packages
pip install -r requirements.txt --upgrade

# ⚠️ CRITICAL: Apply pending database migrations
alembic upgrade head

# Verify migrations applied successfully
alembic current

# 3. Update frontend dependencies
cd ../frontend
pnpm install --latest
# or: npm install --legacy-peer-deps

# 4. Restart both servers
# Terminal 1 - Backend:
# cd backend && python -m uvicorn main:app --reload

# Terminal 2 - Frontend:
# cd frontend && pnpm dev
```

### ✓ Verification After Update
```bash
# In backend directory
alembic current  # Should show latest revision

# In another terminal, test API
curl http://127.0.0.1:8000/health
```

**If migrations fail, see [Database Backup](#database-backup-before-major-updates) section to restore from backup.**

---

## 📝 Verification Checklist

Before starting development, verify:

- [ ] `.env` file exists in backend with all variables from `.env.example`
- [ ] `.env.local` exists in frontend with `NEXT_PUBLIC_API_URL`
- [ ] PostgreSQL is running and `agentdb` database exists
- [ ] pgvector extension is enabled: `psql -U postgres -d agentdb -c "CREATE EXTENSION IF NOT EXISTS vector;"`
- [ ] Virtual environment is activated: `(.venv)` should appear in terminal
- [ ] All dependencies installed: `pip list | grep fastapi` (should show version)
- [ ] **Database migrations are current:** `alembic current` (should show latest revision, not empty)
- [ ] **No pending migrations:** `alembic status` (should show "Target database is up to date")
- [ ] Backend API starts without errors on port 8000: `python -m uvicorn main:app --reload`
- [ ] Frontend builds without errors on port 3000: `pnpm dev`
- [ ] Browser can access http://localhost:3000 without CORS errors
- [ ] API health check passes: `curl http://127.0.0.1:8000/health`

---

## 📚 Additional Resources

- **API Documentation:** `http://127.0.0.1:8000/docs`
- **Database Migration Guide:** See [Database Management & Updates](#-database-management--updates) section
- **Database Backup Guide:** See [Database Backup](#database-backup-before-major-updates) section
- **Update Instructions:** See [Updating to Latest Version](#-updating-to-latest-version) section
- **Troubleshooting:** See [Common Setup Issues & Solutions](#-common-setup-issues--solutions) section
- **Environment Variables:** See [Configuration](#-configuration) section

### Quick Command Reference

**After pulling new code:**
```bash
git pull origin main
cd backend && alembic upgrade head && pip install -r requirements.txt --upgrade
cd ../frontend && pnpm install --latest
```

**Before each development session:**
```bash
# Terminal 1
cd backend && alembic upgrade head && python -m uvicorn main:app --reload

# Terminal 2
cd frontend && pnpm dev
```

**Check database is up to date:**
```bash
cd backend && alembic current && alembic status
```

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

**Required (Without these, the app won't start):**
```env
# LLM & AI Services
GROQ_API_KEY=your_groq_api_key_here              # Required: https://console.groq.com
CEREBRAS_API_KEY=your_cerebras_api_key_here      # Required for document analysis

# Database Connection (Must match your PostgreSQL setup)
PG_USER=postgres                                  # PostgreSQL username
PG_PASSWORD=your_postgres_password_here           # PostgreSQL password
PG_HOST=localhost                                 # PostgreSQL server address
PG_PORT=5432                                      # PostgreSQL port
DB_NAME=agentdb                                   # Database name to create/use
```

**Optional (For specific agents):**
```env
# External API Keys
GOOGLE_API_KEY=your_google_api_key_here
NEWS_AGENT_API_KEY=your_news_api_key_here
SCHOLARAI_API_KEY=your_scholarai_api_key_here
OLLAMA_API_KEY=your_ollama_api_key_here

# Browser Agent Configuration
BROWSER_AGENT_PORT=8070
```

**⚠️ IMPORTANT: All variables in `.env.example` should be defined in your `.env` file before starting the backend!**

### Frontend Environment Variables (`.env.local`)

```env
# Backend API URL (must match where backend is running)
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

### Getting API Keys

1. **GROQ API Key** (Free):
   - Visit: https://console.groq.com
   - Sign up for free account
   - Create new API key
   - Copy and paste into `GROQ_API_KEY` in `.env`

2. **Cerebras API Key** (Optional, for document analysis):
   - Visit: https://console.cerebras.ai
   - Create account and API key
   - Copy and paste into `CEREBRAS_API_KEY` in `.env`

3. **Google API Key** (Optional):
   - Visit: https://console.cloud.google.com
   - Create project and API key
   - Copy and paste into `GOOGLE_API_KEY` in `.env`

4. **News API Key** (Optional):
   - Visit: https://newsapi.org
   - Sign up and get API key
   - Copy and paste into `NEWS_AGENT_API_KEY` in `.env`

---

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