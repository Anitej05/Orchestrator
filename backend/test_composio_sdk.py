#!/usr/bin/env python3
"""Test Composio SDK methods"""

import os
from dotenv import load_dotenv

load_dotenv()

from composio import Composio

api_key = os.getenv("COMPOSIO_API_KEY")
connection_id = os.getenv("GMAIL_CONNECTION_ID")

print(f"API Key: {api_key[:15]}...")
print(f"Connection ID: {connection_id}")
print()

client = Composio(api_key=api_key)

print("Composio client methods:")
methods = [m for m in dir(client) if not m.startswith('_') and callable(getattr(client, m))]
for method in methods:
    print(f"  - {method}")

print()
print("Composio client attributes:")
attrs = [a for a in dir(client) if not a.startswith('_') and not callable(getattr(client, a))]
for attr in attrs:
    print(f"  - {attr}: {type(getattr(client, attr))}")

print()
print("Actions methods:")
actions_methods = [m for m in dir(client.actions) if not m.startswith('_') and callable(getattr(client.actions, m))]
for method in actions_methods:
    print(f"  - {method}")
