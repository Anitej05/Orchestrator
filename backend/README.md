# Agent Directory Service

A high-performance, scalable directory service for registering and discovering AI agents. This project serves as the central "Yellow Pages" for an AI agent ecosystem, enabling dynamic task delegation and orchestration.

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI-blue)
![Database](https://img.shields.io/badge/Database-PostgreSQL-blue)
![Testing](https://img.shields.io/badge/Testing-API_Integration-green)
![Status](https://img.shields.io/badge/Status-Development-orange)

---

## Overview

The Agent Directory is a core component in a distributed AI agent system. Its primary responsibility is to maintain a registry of all available agents, their capabilities, and their operational metadata. An **Orchestrator** service can query this directory to find the best agent for a specific task based on criteria like function, price, and quality rating.

This implementation is built using FastAPI for its high performance and developer-friendly features, with PostgreSQL as a robust and scalable data backend.

## Key Features

*   **Agent Registration:** Agents can be registered or updated via a secure API endpoint using an "upsert" logic.
*   **Status Management:** Agents can be marked as `active`, `inactive`, or `deprecated` to manage their availability for tasks.
*   **Rich Metadata:** Stores essential agent information, including capabilities, endpoint, price, rating, and a public key for security.
*   **Multi-Capability Search:** Provides a powerful search endpoint to find `active` agents that possess **all** of the specified capabilities.

## Architecture & Tech Stack

This service is designed as a standalone microservice.

**Data Flow:**
`[Client/Orchestrator] <--- (HTTP API Request) ---> [Agent Directory Service] <--- (SQLAlchemy ORM) ---> [PostgreSQL Database]`

**Technology Stack:**
*   **Backend Framework:** **FastAPI**
*   **Data Validation:** **Pydantic**
*   **Database:** **PostgreSQL**
*   **ORM / DB Toolkit:** **SQLAlchemy**
*   **Web Server:** **Uvicorn**
*   **HTTP Client (for testing):** **Requests**

## Getting Started

Follow these instructions to get the project up and running on your local machine.

### Prerequisites

*   **Python 3.9+**
*   **PostgreSQL** server installed and running.
*   **Git**

### 1. Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/Orbimesh/agent-directory.git
cd agent-directory

# 2. Create and activate a Python virtual environment
# On Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 3. Install the required dependencies
pip install -r requirements.txt
```

### 2. Database Configuration

1.  **Create a PostgreSQL database** for this project. The application is configured to use a database named `agentdb`.
2.  The `database.py` file contains the connection settings. Ensure the credentials (`PG_USER`, `PG_PASSWORD`, etc.) in this file match your local PostgreSQL setup.
3.  **Run database migrations** to ensure your schema is up to date:
    ```bash
    # Run migrations to update database schema
    python manage.py migrate
    
    # Or run the full sync (which includes migrations)
    python manage.py sync
    ```

### 3. Running the Application

With the setup complete, you can run the FastAPI server.

```bash
# This command starts the server with auto-reload on code changes
uvicorn main:app --reload
```
The server will be running at `http://127.0.0.1:8000`.

### 4. Running Agents in Separate Terminals

Each agent (e.g., `news_agent`, `finance_agent`, `wiki_agent`) can be run in a separate terminal. Follow these steps:

1. **Activate the virtual environment in each terminal:**
   ```bash
   # On Windows
   .\.venv\Scripts\Activate.ps1

   # On macOS/Linux
   source .venv/bin/activate
   ```

2. **Run the desired agent script:**
   ```bash
   # Example: Running the News Agent
   python agents/news_agent.py

   # Example: Running the Finance Agent
   python agents/finance_agent.py

   # Example: Running the Wiki Agent
   python agents/wiki_agent.py
   ```

### 5. Interactive API Documentation

Once the server is running, navigate to **`http://127.0.0.1:8000/docs`** in your browser. FastAPI automatically generates interactive Swagger UI documentation where you can explore and test all the API endpoints directly.

## Testing the Full Application Stack

This project uses the `test_main.py` script for end-to-end **integration testing**. This script simulates a real client by making live HTTP requests to the running FastAPI application.

### How it Works

The test script performs three main actions in sequence:
1.  **Clean:** It directly connects to the PostgreSQL database to find and delete any agents with IDs starting with `dummy_agent_`, ensuring a clean state for each run.
2.  **Seed:** It inserts a fresh set of 10 new, randomized dummy agents into the database.
3.  **Validate:** It makes a series of `GET` and `POST` requests to the API endpoints (`/agents/search`, `/agents/{id}`, `/agents/register`) and asserts that the responses are correct.

### How to Run the Integration Test

> **Important:** This test requires a **two-terminal setup**.

1.  **In Terminal 1**, start the FastAPI server:
    ```bash
    uvicorn main:app --reload
    ```
2.  **In Terminal 2**, run the test script:
    ```bash
    python test_main.py
    ```

You will see a step-by-step log of the seeding process followed by the API validation checks.

## API Endpoints

### 1. Register or Update an Agent

*   **Endpoint:** `POST /agents/register`
*   **Description:** Creates a new agent record or updates an existing one if an agent with the same `id` already exists.
*   **Success Response:** `201 Created` (for new agents) or `200 OK` (for updates). Returns the created/updated agent object.

### 2. Search for Agents

*   **Endpoint:** `GET /agents/search`
*   **Description:** Finds all **active** agents that possess **all** of the specified capabilities.
*   **Query Parameters:**
    *   `capabilities` (string, **required**, can be provided multiple times): A list of tasks the agent must be able to perform.
    *   `max_price` (float, optional): The maximum price per call.
    *   `min_rating` (float, optional): The minimum quality rating.
*   **Example Request (finds an agent that can do both):**
    ```bash
    curl "http://127.0.0.1:8000/agents/search?capabilities=translate&capabilities=summarize"
    ```

### 3. Get a Specific Agent

*   **Endpoint:** `GET /agents/{agent_id}`
*   **Description:** Retrieves the full details for a single agent by its unique ID.

### 4. Get All Agents

*   **Endpoint:** `GET /agents/all`
*   **Description:** Returns all agents in the database as a JSON list.
*   **Example Request:**
    ```bash
    curl "http://127.0.0.1:8000/agents/all"
    ```

### 5. Rate an Agent

*   **Endpoint:** `POST /agents/{agent_id}/rate` or `POST /agents/by-name/{agent_name}/rate`
*   **Description:** Submit a rating for an agent (0-5 scale). Updates the agent's average rating and rating count.
*   **Request Body:** `{"rating": 4.5}`

## Database Management

The project includes a comprehensive database management system via `manage.py`:

### Available Commands

```bash
# Run database migrations (recommended for new setups)
python manage.py migrate

# Create database tables from scratch
python manage.py create-tables

# Sync agent entries from JSON files (default action)
python manage.py sync
# or simply
python manage.py
```

### Adding New Migrations

When you modify the database schema (e.g., adding new columns), create a new migration in `migrations.py`:

```python
class YourNewMigration(Migration):
    def __init__(self):
        super().__init__(
            name="your_migration_name",
            description="Description of what this migration does"
        )
    
    def should_run(self, inspector, conn) -> bool:
        # Logic to check if migration should run
        pass
    
    def execute(self, conn):
        # SQL commands to execute
        pass

# Add to MIGRATIONS list
MIGRATIONS = [
    AddRatingCountMigration(),
    YourNewMigration(),  # Add here
]
```

## Roadmap (Future Work)

-   [ ] **Implement Alembic:** Introduce a proper database migration system to manage schema changes over time.
-   [ ] **Automated Unit Testing:** Implement a `pytest` suite with a dedicated test database to run unit tests in a CI/CD pipeline.
-   [ ] **Heartbeat Monitoring:** Add a `/heartbeat` endpoint for agents to periodically ping, allowing for real-time health checks.
-   [ ] **Authentication & Authorization:** Implement robust authentication (e.g., API keys, OAuth2) to ensure only an agent's owner can modify their registered agents.
-   [ ] **Paginated Results:** Add pagination to the `/search` endpoint to handle a large number of agents efficiently.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.