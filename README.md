# Orbimesh Agent Orchestration Platform

**Orbimesh** is a comprehensive AI agent orchestration platform that enables users to discover, manage, and orchestrate AI agents to complete complex multi-step workflows through natural language prompts.

## 🌟 Features

- **🎯 Natural Language Workflow Creation**: Describe complex tasks in plain English
- **🤖 Agent Discovery & Registration**: Browse and register AI agents with various capabilities
- **⚡ Multi-Agent Orchestration**: Execute complex workflows using LangGraph
- **🔄 Real-time Streaming**: Live progress updates via WebSocket
- **📊 Vector Search**: Semantic agent discovery using pgvector
- **🎨 Modern UI**: Built with Next.js 15 and shadcn/ui

## 🚀 Quick Start

### New Developer Setup (Windows)

**1. Run automated setup:**
```powershell
# Creates virtual env, database, and config files
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

**2. Install backend dependencies:**
```powershell
cd backend
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**3. Install frontend dependencies:**
```powershell
cd frontend
pnpm install  # or: npm install --legacy-peer-deps
```

**4. Edit configuration:**
- Edit `backend/.env` with your API keys (GROQ_API_KEY, PG_PASSWORD, etc.)
- Edit `frontend/.env.local` with your Clerk keys if needed

**5. Start development servers:**
```powershell
# Terminal 1 - Backend
cd backend
.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend
pnpm dev
```

Then open http://localhost:3000

### Existing Developer (After git pull)

```powershell
# Update everything (deps, DB, agents)
powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1
```

---

## 📋 Setup Commands Reference

| Task | Command |
|------|---------|
| **Fresh install** | `.\scripts\setup.ps1` |
| **Update after git pull** | `.\scripts\update.ps1` |
| **Start dev** | `.\scripts\dev.ps1` |
| **DB setup** | `python backend/setup.py` |
| **Sync agents** | `python backend/manage.py sync` |

**What gets updated by `update.ps1`:**
- ✅ Python packages (pip install --upgrade)
- ✅ Database migrations (Alembic)
- ✅ Agent definitions (from Agent_entries/*.json)
- ✅ Frontend packages (npm/pnpm)

---

### Prerequisites
- Python 3.11+, Node.js 18+, PostgreSQL 14+
- GROQ API Key (free at https://console.groq.com)

### Fresh Installation

**1. Clone and setup backend:**
```bash
git clone https://github.com/Orbimesh/Orbimesh-App.git
cd Orbimesh-App/backend

# Create virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
Copy-Item .env.example .env  # Windows
# cp .env.example .env       # macOS/Linux

# Edit .env with your GROQ_API_KEY and PostgreSQL password
```

**2. One-command database setup:**
```bash
# This creates database, enables pgvector, and runs migrations automatically
python setup.py
```

**3. Start backend:**
```bash
# This automatically starts all agents as well
python -m uvicorn main:app --reload
```

**4. Setup frontend (new terminal):**
```bash
cd ../frontend
pnpm install
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" > .env.local
pnpm dev
```

✅ **Done!** Visit http://localhost:3000

---

### Updating Existing Installation

**One-liner (Windows):**
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\update.ps1
```

**Manual:**
```bash
git pull origin main

# Backend
cd backend
.venv\Scripts\Activate.ps1  # Windows
pip install -r requirements.txt --upgrade
python setup.py  # Updates database automatically (creates DB, runs Alembic)
python manage.py sync  # Syncs agent definitions

# Frontend (new terminal)
cd ../frontend
pnpm install --latest
pnpm dev
```

---

## 🔧 Configuration

**Required in `backend/.env`:**
```env
GROQ_API_KEY=your_key_here          # Get at https://console.groq.com
PG_PASSWORD=your_postgres_password  # Your PostgreSQL password
DB_NAME=agentdb
```

Check [.env.example](backend/.env.example) for all available options.

---

## 🆘 Troubleshooting

### Setup script failed?

**PostgreSQL not running:**
```bash
# Windows: Check Services (services.msc) for "PostgreSQL"
# macOS: brew services start postgresql
# Linux: sudo systemctl start postgresql
```

**Missing pgvector:**
```bash
# macOS: brew install pgvector
# Linux: apt-get install postgresql-14-pgvector
# Windows: https://github.com/pgvector/pgvector/releases
```

**Manual database setup if needed:**
```bash
createdb -U postgres agentdb
psql -U postgres -d agentdb -c "CREATE EXTENSION vector;"
cd backend && alembic upgrade head
```

### Common Issues

| Problem | Solution |
|---------|----------|
| **ModuleNotFoundError** | Activate venv: `.venv\Scripts\Activate.ps1` then `pip install -r requirements.txt` |
| **Database connection error** | Check PostgreSQL is running & credentials in `.env` |
| **Table doesn't exist** | Run `python setup.py` or `alembic upgrade head` |
| **Frontend can't connect** | Verify backend is running on port 8000 |
| **GROQ API errors** | Get free key at https://console.groq.com |

---

<details>
<summary><b>📖 Detailed Documentation (Click to expand)</b></summary>

## Backend Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Database Management

**Check migrations:**
```bash
cd backend
alembic current  # Show current version
alembic status   # Check for pending migrations
```

**Apply migrations:**
```bash
alembic upgrade head
```

**Create migration (after modifying models.py):**
```bash
alembic revision --autogenerate -m "Description"
alembic upgrade head
git add alembic/versions/ && git commit -m "Add migration"
```

**Rollback:**
```bash
alembic downgrade -1  # Go back one version
```

**Backup database:**
```bash
# Windows
pg_dump -U postgres agentdb > backup_$(Get-Date -Format 'yyyy-MM-dd_HHmmss').sql

# macOS/Linux
pg_dump -U postgres agentdb > backup_$(date +'%Y-%m-%d_%H%M%S').sql
```

## Frontend Package Managers

**pnpm (recommended):**
```bash
npm install -g pnpm
pnpm install
```

**npm:**
```bash
npm install --legacy-peer-deps
```

**yarn:**
```bash
yarn install
```

## Environment Variables

**Backend `.env`:**
```env
# Required
GROQ_API_KEY=your_groq_key
CEREBRAS_API_KEY=your_cerebras_key
PG_USER=postgres
PG_PASSWORD=your_password
PG_HOST=localhost
PG_PORT=5432
DB_NAME=agentdb

# Optional
GOOGLE_API_KEY=your_google_key
NEWS_AGENT_API_KEY=your_news_key
SCHOLARAI_API_KEY=your_scholar_key
BROWSER_AGENT_PORT=8070
```

**Frontend `.env.local`:**
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## Verification Checklist

- [ ] `.env` configured with API keys
- [ ] PostgreSQL running
- [ ] Virtual environment activated
- [ ] Dependencies installed: `pip list | grep fastapi`
- [ ] Database migrations current: `alembic current`
- [ ] Backend starts: `python -m uvicorn main:app --reload`
- [ ] Frontend starts: `pnpm dev`
- [ ] API accessible: http://127.0.0.1:8000/docs

## Development Workflow

**Daily startup:**
```bash
# Terminal 1 - Backend (starts agents automatically)
cd backend
.venv\Scripts\Activate.ps1
python -m uvicorn main:app --reload

# Terminal 2 - Frontend
cd frontend
pnpm dev
```

**After pulling code:**
```bash
git pull
cd backend && pip install -r requirements.txt --upgrade && python setup.py
cd ../frontend && pnpm install --latest
```

</details>

---

## 📚 Additional Resources

- **API Docs:** http://127.0.0.1:8000/docs
- **Get GROQ API Key:** https://console.groq.com (free)
- **Alembic Docs:** https://alembic.sqlalchemy.org
- **pgvector:** https://github.com/pgvector/pgvector

---

## 📊 Project Structure

```
Orbimesh-App/
├── backend/
│   ├── main.py                # FastAPI application
│   ├── setup.py               # One-command setup script
│   ├── database.py            # Database configuration
│   ├── models.py              # SQLAlchemy models
│   ├── requirements.txt       # Python dependencies
│   ├── agents/                # AI agents
│   └── orchestrator/          # LangGraph orchestration
│
└── frontend/
    ├── app/                   # Next.js App Router
    ├── components/            # React components
    └── lib/                   # Utilities
```

## 🛠️ Technology Stack

- **Frontend**: Next.js 15, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: FastAPI, LangGraph, GROQ LLM, SQLAlchemy
- **Database**: PostgreSQL with pgvector extension

## 🎯 Usage

1. Open http://localhost:3000
2. Enter a natural language description (e.g., "Find restaurants in Paris and draft an email")
3. Click **Parse & Find Agents** to break down tasks
4. Review suggested agents and execute the workflow
5. Watch real-time progress

## 🔌 API Endpoints

- `POST /api/chat`: Process workflow from natural language
- `GET /agents/search`: Search for agents
- `GET /agents/{agent_id}`: Get agent details
- `POST /agents/`: Register new agent
- `WS /ws/{thread_id}`: WebSocket for real-time updates
- API Docs: http://127.0.0.1:8000/docs

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Built by the Orbimesh team**
