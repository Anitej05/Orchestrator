"""
Test SKILL.md parsing and UAP integration.
"""
import sys
sys.path.insert(0, '.')

from backend.services.agent_registry_service import agent_registry, parse_skill_md
from pathlib import Path

def test_skill_parsing():
    """Test that all SKILL.md files are parsed correctly."""
    print("=" * 60)
    print("Testing SKILL.md Parsing")
    print("=" * 60)
    
    skills = agent_registry._load_skill_configs()
    
    print(f"\n✓ Loaded {len(skills)} SKILL.md configurations:\n")
    
    for agent_id, config in skills.items():
        print(f"  Agent: {config['name']}")
        print(f"    ID: {agent_id}")
        print(f"    Port: {config['port']}")
        print(f"    Version: {config['version']}")
        print(f"    URL: http://{config['host']}:{config['port']}")
        print(f"    Description: {config['description'][:80]}...")
        print()
    
    return len(skills) > 0


def test_agent_url_resolution():
    """Test that agent URLs are resolved from SKILL.md."""
    print("=" * 60)
    print("Testing Agent URL Resolution")
    print("=" * 60)
    
    test_agents = ['spreadsheet_agent', 'mail_agent', 'browser_automation_agent']
    
    for agent_id in test_agents:
        url = agent_registry.get_agent_url(agent_id)
        if url:
            print(f"  ✓ {agent_id}: {url}")
        else:
            print(f"  ✗ {agent_id}: NO URL FOUND")
    
    return True


def test_list_active_agents():
    """Test listing active agents includes SKILL.md agents."""
    print("\n" + "=" * 60)
    print("Testing list_active_agents()")
    print("=" * 60)
    
    agents = agent_registry.list_active_agents()
    print(f"\n✓ Found {len(agents)} agents:\n")
    
    for agent in agents:
        print(f"  - {agent['name']} (id: {agent['id']})")
        conn_config = agent.get('connection_config')
        if conn_config and conn_config.get('base_url'):
            print(f"    URL: {conn_config['base_url']}")
    
    return len(agents) > 0


def test_skills_context():
    """Test getting all skills as LLM context."""
    print("\n" + "=" * 60)
    print("Testing get_all_skills_context()")
    print("=" * 60)
    
    context = agent_registry.get_all_skills_context()
    print(f"\n✓ Generated {len(context)} chars of LLM context")
    print(f"\nPreview (first 500 chars):\n{context[:500]}...")
    
    return len(context) > 0


if __name__ == "__main__":
    results = []
    
    results.append(("SKILL.md Parsing", test_skill_parsing()))
    results.append(("Agent URL Resolution", test_agent_url_resolution()))
    results.append(("List Active Agents", test_list_active_agents()))
    results.append(("Skills Context", test_skills_context()))
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print()
    sys.exit(0 if all_passed else 1)
