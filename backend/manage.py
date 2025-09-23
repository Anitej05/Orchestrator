import sys
import requests
import json
import os
import argparse
from sqlalchemy import text, inspect, Column, Integer
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.exc import ProgrammingError
from database import SessionLocal, engine, Base
from models import Agent, AgentEndpoint, AgentCapability
from schemas import AgentCard, EndpointDetail
from migrations import run_all_migrations

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:8000"
AGENT_ENTRIES_DIR = "Agent_entries"

def run_migrations():
    """
    Run database migrations to ensure the schema is up to date.
    This function handles schema changes like adding new columns.
    """
    print("--- Running Database Migrations ---")
    
    # Create tables first if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Run all pending migrations
    run_all_migrations(engine)
    
    print("✅ Database migrations completed")

def create_tables():
    """
    Create all database tables if they don't exist.
    """
    print("--- Creating Database Tables ---")
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created successfully")

def is_agent_changed(db_agent: Agent, file_agent: AgentCard) -> bool:
    """
    Compares a database Agent object with an AgentCard from a file to see if an update is needed.
    This version now correctly compares endpoint parameters.
    """
    # Compare simple attributes
    if (getattr(db_agent, 'name') != file_agent.name or
        getattr(db_agent, 'description') != file_agent.description or
        getattr(db_agent, 'owner_id') != file_agent.owner_id or
        getattr(db_agent, 'status').value != file_agent.status or
        getattr(db_agent, 'price_per_call_usd') != file_agent.price_per_call_usd or
        getattr(db_agent, 'public_key_pem') != file_agent.public_key_pem):
        return True

    # Compare capabilities
    db_capabilities = {cap.capability_text for cap in db_agent.capability_vectors}
    file_capabilities = set(file_agent.capabilities)
    if db_capabilities != file_capabilities:
        return True

    # *** START: CORRECTED ENDPOINT AND PARAMETER COMPARISON LOGIC ***
    # Create a comparable representation of endpoints and their parameters from the file
    file_endpoints_set = set()
    for ep in file_agent.endpoints:
        # Create a sorted tuple of parameters for consistent ordering
        params_tuple = tuple(
            sorted(
                (p.name, p.param_type, p.required, p.default_value) for p in ep.parameters
            )
        )
        file_endpoints_set.add((str(ep.endpoint), ep.http_method, ep.description, params_tuple))

    # Create a comparable representation from the database
    db_endpoints_set = set()
    for ep in db_agent.endpoints:
        params_tuple = tuple(
            sorted(
                (p.name, p.param_type, p.required, p.default_value) for p in ep.parameters
            )
        )
        db_endpoints_set.add((ep.endpoint, ep.http_method, ep.description, params_tuple))

    # Now compare the comprehensive sets
    if db_endpoints_set != file_endpoints_set:
        return True
    # *** END: CORRECTED LOGIC ***

    return False

def sync_agent_entries():
    """
    A non-destructive script that synchronizes agent entries from JSON files to the database.
    - Adds new agents if they don't exist.
    - Updates existing agents if their definitions have changed.
    """
    
    # --- Step 1: Run migrations first ---
    run_migrations()
    
    # --- Step 2: Dynamically Find and Sync Agents ---
    print(f"\n--- Step 2: Syncing agents from the '{AGENT_ENTRIES_DIR}' directory ---")
    
    if not os.path.isdir(AGENT_ENTRIES_DIR):
        print(f"❌ ERROR: The directory '{AGENT_ENTRIES_DIR}' was not found.")
        sys.exit(1)
        
    agent_json_files = [f for f in os.listdir(AGENT_ENTRIES_DIR) if f.endswith(".json")]

    if not agent_json_files:
        print(f"⚠️ WARNING: No agent .json files found in '{AGENT_ENTRIES_DIR}'.")
        return

    updated_count = 0
    added_count = 0
    unchanged_count = 0

    with SessionLocal() as db:
        for json_file in agent_json_files:
            file_path = os.path.join(AGENT_ENTRIES_DIR, json_file)
            
            with open(file_path, 'r') as f:
                agent_data = json.load(f)
                file_agent = AgentCard(**agent_data)
                
                # Check if the agent exists in the database, making sure to load relations
                db_agent = db.query(Agent).options(
                    joinedload(Agent.capability_vectors),
                    joinedload(Agent.endpoints).joinedload(AgentEndpoint.parameters)
                ).get(file_agent.id)

                if db_agent:
                    # Agent exists, check if it needs an update
                    if is_agent_changed(db_agent, file_agent):
                        print(f"🔄 Updating agent: {file_agent.name}...")
                        response = requests.post(f"{API_BASE_URL}/agents/register", json=agent_data)
                        if response.status_code == 200:
                            updated_count += 1
                        else:
                            print(f"❌ ERROR updating agent {file_agent.id}: {response.text}")
                    else:
                        print(f"✅ Agent is up-to-date: {file_agent.name}")
                        unchanged_count += 1
                else:
                    # Agent does not exist, create it
                    print(f"➕ Adding new agent: {file_agent.name}...")
                    response = requests.post(f"{API_BASE_URL}/agents/register", json=agent_data)
                    if response.status_code == 201:
                        added_count += 1
                    else:
                        print(f"❌ ERROR adding agent {file_agent.id}: {response.text}")

    print("\n--- Sync Summary ---")
    print(f"New agents added: {added_count}")
    print(f"Existing agents updated: {updated_count}")
    print(f"Agents already up-to-date: {unchanged_count}")

def main():
    """
    Main function to handle command-line arguments and run appropriate actions.
    """
    parser = argparse.ArgumentParser(description="Manage database and agents")
    parser.add_argument('action', choices=['sync', 'migrate', 'create-tables'], 
                       help='Action to perform')
    
    # If no arguments provided, default to sync
    if len(sys.argv) == 1:
        args = parser.parse_args(['sync'])
    else:
        args = parser.parse_args()
    
        args = parser.parse_args()

        if args.create_tables:
            print("Creating tables...")
            Base.metadata.create_all(bind=engine)
            print("Tables created.")
        else:
            parser.print_help()

if __name__ == "__main__":
    main()