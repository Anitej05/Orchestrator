"""
LLM-powered code generation for spreadsheet modifications.

Generates pandas code for data transformations using natural language instructions.
"""

import logging
from typing import Optional

import pandas as pd

from .llm_agent import query_agent

logger = logging.getLogger(__name__)


async def generate_modification_code(df: pd.DataFrame, instruction: str) -> Optional[str]:
    """
    Generate pandas code for data modifications based on natural language instruction.
    
    Args:
        df: DataFrame to modify
        instruction: Natural language instruction (e.g., "add a serial number column")
    
    Returns:
        Pandas code string or None if generation fails
    """
    # Get dataframe info for context
    columns = df.columns.tolist()
    dtypes = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    sample_row = df.head(1).to_dict('records')[0] if len(df) > 0 else {}
    
    prompt = f"""You are an expert pandas code generator. Your job is to generate ONLY executable Python code to modify a DataFrame.

=== DATAFRAME INFORMATION ===
- Columns: {columns}
- Data types: {dtypes}
- Sample row: {sample_row}
- Total rows: {len(df)}

=== USER'S INSTRUCTION ===
{instruction}

=== STRICT RULES ===
1. Output ONLY valid Python code - NO explanations, NO comments, NO markdown
2. The DataFrame is named 'df'
3. Your code MUST return/result in the modified DataFrame
4. Use proper pandas methods

=== COMMON OPERATIONS (Use these patterns) ===

**Adding Serial Number (Sl.No.) Column:**
df.insert(0, 'Sl.No.', range(1, len(df) + 1))
df

**Adding Total Column (sum of all numeric columns per row):**
df['Total'] = df.select_dtypes(include='number').sum(axis=1)
df

**Adding Total of SPECIFIC columns:**
df['Total'] = df['Col1'] + df['Col2'] + df['Col3']
df

**Renaming Columns:**
df.rename(columns={{'OldName1': 'NewName1', 'OldName2': 'NewName2'}}, inplace=True)
df

**Adding a Calculated Column:**
df['NewCol'] = df['ExistingCol'] * 2
df

**Filtering Rows:**
df = df[df['Col'] > 50]
df

**Sorting:**
df = df.sort_values('Col', ascending=False)
df

**Dropping a Column:**
df = df.drop(columns=['ColName'])
df

=== OUTPUT YOUR CODE NOW ==="""

    # Use query_agent's providers
    providers = query_agent.providers
    if not providers:
        logger.error("No LLM providers available for code generation")
        return None
    
    for provider in providers:
        try:
            provider_name = provider['name']
            client = provider['client']
            model = provider['model']
            
            logger.info(f"🤖 Using {provider_name.upper()} for code generation")
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            
            code = response.choices[0].message.content.strip()
            
            # Clean up code - remove markdown if present
            if code.startswith("```python"):
                code = code[9:]
            if code.startswith("```"):
                code = code[3:]
            if code.endswith("```"):
                code = code[:-3]
            code = code.strip()
            
            logger.info(f"Generated code: {code}")
            return code
            
        except Exception as e:
            logger.warning(f"⚠️ {provider_name.upper()} failed: {e}")
            continue
    
    return None


async def generate_csv_from_instruction(instruction: str, reference_content: Optional[str] = None) -> Optional[str]:
    """
    Generate CSV content from natural language instruction.
    
    Args:
        instruction: Natural language instruction for creating data
        reference_content: Optional reference content to transform
    
    Returns:
        CSV string or None if generation fails
    """
    context = f"""
=== REFERENCE CONTENT ===
{reference_content if reference_content else "No reference content provided."}

=== USER'S INSTRUCTION ===
{instruction}
""" if reference_content else f"""
=== USER'S INSTRUCTION ===
{instruction}
"""

    prompt = f"""You are an expert data generator. Your job is to generate CSV data based on the user's instruction.

{context}

=== STRICT RULES ===
1. Output ONLY valid CSV format - NO markdown, NO code blocks, NO explanations
2. First line MUST be comma-separated headers
3. Following lines are data rows
4. Use proper CSV escaping for special characters
5. Generate realistic, relevant data

=== OUTPUT YOUR CSV NOW ==="""

    # Use query_agent's providers
    providers = query_agent.providers
    if not providers:
        logger.error("No LLM providers available for CSV generation")
        return None
    
    for provider in providers:
        try:
            provider_name = provider['name']
            client = provider['client']
            model = provider['model']
            
            logger.info(f"🤖 Using {provider_name.upper()} for CSV generation")
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=3000
            )
            
            csv_content = response.choices[0].message.content.strip()
            
            # Clean up - remove markdown if present
            if csv_content.startswith("```csv"):
                csv_content = csv_content[6:]
            if csv_content.startswith("```"):
                csv_content = csv_content[3:]
            if csv_content.endswith("```"):
                csv_content = csv_content[:-3]
            csv_content = csv_content.strip()
            
            logger.info(f"Generated CSV ({len(csv_content)} chars)")
            return csv_content
            
        except Exception as e:
            logger.warning(f"⚠️ {provider_name.upper()} failed: {e}")
            continue
    
    return None
