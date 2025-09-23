# Orbimesh 🌌

A comprehensive full-stack application for discovering, managing, and orchestrating AI agents. This platform serves as a centralized marketplace where users can find AI agents, create complex workflows, and execute tasks through an intelligent orchestration system.

-----

## 🌟 Features

### Frontend (Next.js)

  - **Workflow Builder**: Dynamically parse natural language tasks and select appropriate agents.
  - **Agent Directory**: A searchable and filterable directory to browse all available AI agents.
  - **Real-time Orchestration**: View live workflow execution with dynamic animations.
  - **Agent Registration**: An intuitive form to register new agents with multiple endpoints.
  - **Responsive Design**: Modern and clean UI built with **Tailwind CSS** and **shadcn/ui**.

### Backend (FastAPI + LangGraph)

  - **Intelligent Task Parsing**: Utilizes the **GROQ LLM** to break down complex user prompts into actionable steps.
  - **Vector Search**: Implements semantic agent matching using **pgvector** for high-relevance results.
  - **Agent Registry**: A robust PostgreSQL database with full CRUD operations for agents.
  - **LangGraph Orchestration**: Manages multi-step and multi-agent workflow execution.
  - **RESTful API**: A complete API for agent management and task processing.

-----

## 🏗️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│     Next.js     │      │     FastAPI      │      │   PostgreSQL    │
│     Frontend    │◄───► │     Backend      │◄───► │   + pgvector    │
│                 │      │                  │      │                 │
│ • Task Builder  │      │ • LangGraph      │      │ • Agent Storage │
│ • Agent Grid    │      │ • Vector Search  │      │ • Capabilities  │
│ • Registration  │      │ • GROQ LLM       │      │ • Endpoints     │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

-----

## 🚀 Quick Start

### Prerequisites

  - **Node.js** v18+ and npm/yarn
  - **Python** v3.11+
  - **PostgreSQL** v14+ with the `pgvector` extension enabled
  - **GROQ API Key** (for LLM operations)

### 1\. Clone the Repository

```bash
git clone <repository-url>
cd orbimesh
```

### 2\. Backend Setup

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create and populate the .env file
cat > .env << EOF
GROQ_API_KEY="your_groq_api_key_here"
PG_USER="postgres"
PG_PASSWORD="your_password"
PG_HOST="localhost"
PG_PORT="5432"
DB_NAME="agentdb"
EOF

# Start the backend server
uvicorn main:app --reload
```

The backend will be available at `http://127.0.0.1:8000`.

### 3\. Frontend Setup

```bash
# Navigate to the frontend directory (from the project root)
cd ..

# Install dependencies
npm install

# Create a local environment file
echo "NEXT_PUBLIC_API_URL=http://127.0.0.1:8000" > .env.local

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

-----

## 📊 Database Setup

The application automatically creates the required database and tables on the first run. Please ensure your **PostgreSQL** server is running and the credentials in the backend's `.env` file are correct.

**Required Extension:**

  * `pgvector` - Must be installed and enabled in your PostgreSQL database for semantic search capabilities.

-----

## 🔧 Configuration

### Backend Environment Variables (`backend/.env`)

```env
# Required
GROQ_API_KEY="your_groq_api_key_here"
PG_USER="postgres"
PG_PASSWORD="your_password"
PG_HOST="localhost"
PG_PORT="5432"
DB_NAME="agentdb"

# Optional
OPENAI_API_KEY="your_openai_key_if_needed"
```

### Frontend Environment Variables (`.env.local`)

```env
NEXT_PUBLIC_API_URL="http://127.0.0.1:8000"
```

-----

## 🎯 Usage

### 1\. Create a Workflow

1.  Open the application at `http://localhost:3000`.
2.  Enter a natural language description of your goal (e.g., "Find the top 3 restaurants in Paris and draft an email to my friends to invite them").
3.  Click **Parse & Find Agents** to break down the tasks and find suitable agents.
4.  Review and modify the suggested agents for each step.
5.  Execute the workflow.

### 2\. Discover Agents

1.  Navigate to the **Agent Directory** from the sidebar.
2.  Browse available agents or use the search and filter options.
3.  Filter agents by capabilities, price, or rating.
4.  Click on an agent to view details and test it with custom prompts.

### 3\. Register a New Agent

1.  Click **Register Agent** from the sidebar.
2.  Fill in the agent's details, capabilities, and API endpoints.
3.  Preview how your agent's card will appear in the directory.
4.  Test endpoint connectivity and save the agent.

-----

## 🧪 Testing

### Backend Testing

To run the backend tests, navigate to the `backend` directory and execute:

```bash
cd backend
pytest
```

### API Documentation

With the backend running, visit `http://127.0.0.1:8000/docs` for interactive Swagger/OpenAPI documentation.

-----

## 📁 Project Structure

```
orbimesh/
├── README.md
├── package.json
├── next.config.js
├── tailwind.config.ts
├── .env.local
│
├── app/                  # Next.js App Router
│   ├── layout.tsx
│   ├── page.tsx          # Main workflow builder
│   ├── agents/
│   │   └── page.tsx      # Agent directory
│   └── register-agent/
│       └── page.tsx      # Agent registration
│
├── components/           # React components
│   ├── ui/               # shadcn/ui components
│   ├── task-builder.tsx
│   ├── agent-grid.tsx
│   ├── agent-card.tsx
│   ├── workflow-orchestration.tsx
│   └── ...
│
├── lib/                  # Utilities and types
│   ├── api.ts            # API client
│   ├── types.ts          # TypeScript types
│   └── utils.ts
│
└── backend/              # FastAPI backend
    ├── main.py           # FastAPI app
    ├── database.py       # Database configuration
    ├── models.py         # SQLAlchemy models
    ├── schemas.py        # Pydantic schemas
    ├── requirements.txt
    ├── .env              # Environment variables
    │
    └── orchestrator/     # LangGraph orchestration
        ├── __init__.py
        ├── graph.py      # Main orchestration logic
        └── state.py      # State management
```

-----

## 🔌 API Endpoints

### Core Endpoints

  - `POST /api/chat`: Process a natural language prompt to create and execute a workflow.
  - `GET /agents/search`: Search for agents by capabilities.
  - `GET /agents/{agent_id}`: Get details for a specific agent.
  - `POST /agents/`: Register a new agent.
  - `GET /api/health`: Health check for the API.

### Example API Usage

```bash
# Search for agents capable of email drafting
curl "http://127.0.0.1:8000/agents/search?capability=email_drafting"

# Process a workflow via chat endpoint
curl -X POST "http://127.0.0.1:8000/api/chat" \
 -H "Content-Type: application/json" \
 -d '{"prompt": "Help me find a travel agent and then draft an email to them."}'
```

-----

## 🛠️ Technology Stack

### Frontend

  - **Framework**: Next.js 15 (with App Router)
  - **Language**: TypeScript
  - **Styling**: Tailwind CSS
  - **UI Components**: shadcn/ui
  - **Icons**: Lucide React

### Backend

  - **Framework**: FastAPI
  - **Orchestration**: LangGraph
  - **Language**: Python
  - **LLM Integration**: GROQ
  - **ORM**: SQLAlchemy
  - **Data Validation**: Pydantic

### Database

  - **Primary Store**: PostgreSQL
  - **Vector Search**: pgvector

-----

## 🤝 Contributing

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes (`git commit -m 'Add some amazing feature'`).
4.  Push to the branch (`git push origin feature/amazing-feature`).
5.  Open a Pull Request.

-----

## 📝 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

-----

## 🆘 Troubleshooting

### Common Issues

  - **Backend won't start:**

      - Ensure the PostgreSQL server is running.
      - Verify that the database credentials in `backend/.env` are correct.
      - Confirm the `pgvector` extension is installed and enabled in your database.

  - **Frontend can't connect to backend:**

      - Verify the `NEXT_PUBLIC_API_URL` in `.env.local` points to your backend URL.
      - Ensure the backend server is running on the specified port (default: 8000).
      - Check for any CORS errors in your browser's developer console.

  - **GROQ API errors:**

      - Ensure the `GROQ_API_KEY` in `backend/.env` is set correctly.
      - Check that your GROQ API key has sufficient credits.
      - Verify your server has internet connectivity.

### Getting Help

  - Check the [Issues](https://www.google.com/search?q=issues) page for existing bug reports and feature requests.
  - Review the API documentation at `http://127.0.0.1:8000/docs`.
  - Run the backend tests via `pytest` to diagnose potential https://www.google.com/search?q=issues.

-----

## 🚀 Deployment

### Production Considerations

  - Use a robust WSGI server like **Gunicorn** for the FastAPI backend.
  - Configure a reverse proxy like **Nginx** to manage traffic.
  - Use environment-specific `.env` files and manage secrets securely.
  - Set up automated database backups.
  - Implement process managers (e.g., PM2, systemd) to ensure services are always running.
  - Integrate monitoring and logging solutions (e.g., Grafana, Prometheus, Sentry).
