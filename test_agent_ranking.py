#!/usr/bin/env python3
"""
Test script to verify agent ranking with endpoint information
"""
import json
import sys
sys.path.insert(0, '/root/backend')

from database import SessionLocal
from models import Agent, AgentEndpoint

def test_agent_endpoints():
    """Verify that agents have the correct endpoints in the database"""
    db = SessionLocal()
    
    try:
        agents = db.query(Agent).all()
        print(f"\n📊 Total Agents in Database: {len(agents)}\n")
        
        for agent in agents:
            print(f"🤖 Agent: {agent.name} (ID: {agent.id})")
            print(f"   Port/URL: {agent.connection_config.get('port') if agent.connection_config else 'N/A'}")
            
            endpoints = db.query(AgentEndpoint).filter(AgentEndpoint.agent_id == agent.id).all()
            print(f"   Endpoints ({len(endpoints)}):")
            for ep in endpoints:
                print(f"     - {ep.http_method:6} {ep.endpoint:20} | {ep.description}")
            print()
        
        # Check for agents with /edit endpoint
        print("\n🔍 Agents with '/edit' endpoint:")
        edit_endpoints = db.query(AgentEndpoint).filter(AgentEndpoint.endpoint == '/edit').all()
        for ep in edit_endpoints:
            agent = db.query(Agent).filter(Agent.id == ep.agent_id).first()
            if agent:
                print(f"   ✅ {agent.name} (port {agent.connection_config.get('port') if agent.connection_config else 'N/A'})")
        
        if not edit_endpoints:
            print("   ❌ No agents found with /edit endpoint!")
            
    finally:
        db.close()

if __name__ == '__main__':
    test_agent_endpoints()
