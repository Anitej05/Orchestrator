#!/usr/bin/env python3
import requests
import json

def check_agent_capabilities():
    """Check the capabilities of image and document analysis agents"""
    try:
        response = requests.get('http://127.0.0.1:8000/api/agents/all')
        if response.status_code != 200:
            print(f"❌ Failed to get agents: {response.status_code}")
            return

        agents = response.json()

        print("=== Agent Capabilities Check ===")

        for agent in agents:
            if 'Image' in agent['name'] or 'Document' in agent['name']:
                print(f"\n📋 {agent['name']}:")
                print(f"   ID: {agent.get('id', 'N/A')}")
                print(f"   Status: {agent.get('status', 'N/A')}")
                print(f"   Capabilities: {agent.get('capabilities', [])}")
                print(f"   Endpoints: {len(agent.get('endpoints', []))}")

                if agent.get('endpoints'):
                    for endpoint in agent['endpoints']:
                        print(f"     - {endpoint.get('http_method', 'N/A')} {endpoint.get('endpoint', 'N/A')}")
                        if endpoint.get('parameters'):
                            print(f"       Parameters: {len(endpoint['parameters'])}")
                            for param in endpoint['parameters']:
                                print(f"         - {param.get('name', 'N/A')}: {param.get('param_type', 'N/A')} ({'required' if param.get('required') else 'optional'})")

    except Exception as e:
        print(f"❌ Error checking agent capabilities: {e}")

if __name__ == "__main__":
    check_agent_capabilities()
