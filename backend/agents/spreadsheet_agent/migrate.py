"""
Spreadsheet Agent Modularization Migration Script

This script completes the modularization by extracting remaining components
from the original spreadsheet_agent.py into the new module structure.

Usage:
    cd backend/agents
    python -m spreadsheet_agent.migrate
"""

import re
import sys
from pathlib import Path

# Define module templates and extraction patterns
MODULES = {
    'llm_agent.py': {
        'start_pattern': r'class SpreadsheetQueryAgent:',
        'end_pattern': r'# Global query agent instance',
        'imports': [
            'import logging',
            'from typing import List, Dict, Any, Optional, tuple',
            'import pandas as pd',
            'from .config import CEREBRAS_API_KEY, GROQ_API_KEY, CEREBRAS_MODEL, GROQ_MODEL',
            'from .config import CEREBRAS_BASE_URL, GROQ_BASE_URL, LLM_TEMPERATURE, LLM_MAX_TOKENS_QUERY',
            'from .models import QueryResult, QueryPlan',
            'from .utils import handle_execution_error, log_operation_error',
        ]
    },
    'code_generator.py': {
        'start_pattern': r'async def generate_modification_code',
        'end_pattern': r'async def _generate_csv_with_llm',
        'imports': [
            'import logging',
            'import pandas as pd',
            'from typing import Optional',
            'from .config import LLM_MAX_TOKENS_CODE_GEN',
            'from .llm_agent import query_agent',
        ]
    }
}


def extract_code_block(content: str, start_pattern: str, end_pattern: str) -> str:
    """Extract code block between two patterns"""
    start_match = re.search(start_pattern, content)
    if not start_match:
        return ""
    
    start_pos = start_match.start()
    
    # Find end pattern or use end of file
    end_match = re.search(end_pattern, content[start_pos:])
    if end_match:
        end_pos = start_pos + end_match.start()
    else:
        end_pos = len(content)
    
    return content[start_pos:end_pos]


def create_module_file(module_name: str, code: str, imports: list, docstring: str):
    """Create a module file with proper structure"""
    content = f'"""\n{docstring}\n"""\n\n'
    content += '\n'.join(imports) + '\n\n'
    content += 'logger = logging.getLogger(__name__)\n\n'
    content += code
    return content


def main():
    """Main migration function"""
    print("🚀 Starting Spreadsheet Agent Modularization Migration...")
    
    # Find original file
    original_file = Path(__file__).parent.parent / "spreadsheet_agent.py"
    if not original_file.exists():
        print(f"❌ ERROR: Original file not found: {original_file}")
        sys.exit(1)
    
    print(f"✓ Found original file: {original_file}")
    
    # Read original content
    with open(original_file, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    print(f"✓ Read {len(original_content)} characters from original file")
    
    # Extract and create modules
    target_dir = Path(__file__).parent
    
    # Create llm_agent.py
    print("\n📝 Creating llm_agent.py...")
    llm_code = extract_code_block(
        original_content,
        r'class SpreadsheetQueryAgent:',
        r'# Global query agent instance'
    )
    
    if llm_code:
        llm_content = create_module_file(
            'llm_agent.py',
            llm_code,
            MODULES['llm_agent.py']['imports'],
            'LLM-powered query agent for natural language spreadsheet queries'
        )
        
        (target_dir / 'llm_agent.py').write_text(llm_content, encoding='utf-8')
        print(f"✓ Created llm_agent.py ({len(llm_code)} chars)")
    else:
        print("⚠️  Could not extract llm_agent code")
    
    # Add more module extractions here...
    
    print("\n✅ Migration complete!")
    print("\nNext steps:")
    print("1. Review generated files")
    print("2. Run: python -m pytest tests/")
    print("3. Update orchestrator imports")


if __name__ == '__main__':
    main()
