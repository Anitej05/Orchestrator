"""
Helper functions to generate synthetic test datasets for edge case testing.

This module creates DataFrames with various edge cases to test the robustness
of the spreadsheet agent.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any


def create_empty_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with 0 rows but defined schema.
    
    Tests: Count, sum, groupby on empty data
    """
    return pd.DataFrame(columns=['ID', 'Name', 'Value', 'Category', 'Price'])


def create_single_row_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with exactly 1 row.
    
    Tests: Statistical operations (std, median) on single value
    """
    return pd.DataFrame({
        'ID': [1],
        'Name': ['Single Item'],
        'Value': [100],
        'Category': ['A']
    })


def create_unicode_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with Unicode edge cases.
    
    Tests: Emoji, Chinese chars, accents, special quotes
    """
    return pd.DataFrame({
        'ProductID': [1, 2, 3, 4, 5, 6],
        'Name': [
            '🔥 Hot Item',           # Emoji
            '北京店',                 # Chinese
            'Café Racer',            # Accent
            'Item "Special"',        # Quotes
            'Line\nBreak',           # Newline
            "Tab\there"              # Tab
        ],
        'Price': [99.99, 199.99, 149.99, 79.99, 129.99, 89.99],
        'Category': ['Electronics', 'Clothing', 'Accessories', 'Clothing', 'Electronics', 'Home'],
        'Stock': [10, 25, 15, 30, 5, 20]
    })


def create_mixed_types_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with mixed types in same column.
    
    Tests: Type coercion, operations on mixed-type columns
    """
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'MixedColumn': [1, 'two', 3.0, None, True],  # int, str, float, None, bool
        'Value': [100, 200, 300, 400, 500],
        'Category': ['A', 'B', 'A', 'B', 'A']
    })


def create_edge_numerics_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with numeric edge cases.
    
    Tests: Infinity, NaN, division by zero, very large numbers
    """
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6, 7, 8],
        'Value': [100, 0, np.inf, -np.inf, np.nan, 1e15, -999, 0.0000001],
        'Denominator': [10, 0, 5, 0, 10, 1, 0, 2],  # For division tests
        'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
        'Price': [50.0, 0.0, 100.0, -50.0, np.nan, 1000000.0, 25.0, 0.01]
    })


def create_datetime_edge_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with datetime edge cases.
    
    Tests: Mixed date formats, invalid dates, date arithmetic
    """
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6],
        'DateString': [
            '2024-01-01',       # ISO format
            '01/01/2024',       # US format
            'Jan 1, 2024',      # Text format
            '2024-02-30',       # Invalid date
            '2024-12-31',       # Year end
            '1900-01-01'        # Year 1900 (Excel bug)
        ],
        'Value': [100, 200, 300, 400, 500, 600],
        'Category': ['A', 'B', 'A', 'B', 'A', 'B']
    })


def create_duplicate_columns_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with duplicate column names.
    
    Tests: Column access with duplicate names
    """
    df = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    df.columns = ['A', 'A', 'B', 'C']  # Duplicate 'A'
    return df


def create_reserved_keywords_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with reserved Python keywords as columns.
    
    Tests: Bracket notation vs dot notation for reserved words
    """
    return pd.DataFrame({
        'id': [1, 2, 3, 4],
        'lambda': [10, 20, 30, 40],
        'class': ['A', 'B', 'C', 'D'],
        'for': [100, 200, 300, 400],
        'if': [True, False, True, False]
    })


def create_string_edge_cases_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with string edge cases.
    
    Tests: Empty strings, None, whitespace, special characters
    """
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5, 6, 7],
        'Name': [
            'Normal',
            '',                    # Empty string
            None,                  # None
            '   ',                 # Whitespace only
            ' LeadingSpace',       # Leading space
            'TrailingSpace ',      # Trailing space
            "Quote's\"Complex"     # Mixed quotes
        ],
        'Value': [100, 200, 300, 400, 500, 600, 700],
        'Category': ['A', 'B', '', None, 'A', 'B', 'C']
    })


def create_multiindex_result_dataframe() -> pd.DataFrame:
    """
    Create DataFrame that will produce MultiIndex when grouped.
    
    Tests: MultiIndex handling (the bug we fixed)
    """
    return pd.DataFrame({
        'Region': ['North', 'North', 'South', 'South', 'East', 'East'] * 3,
        'Product': ['A', 'B', 'A', 'B', 'A', 'B'] * 3,
        'Sales': np.random.randint(100, 1000, 18),
        'Quantity': np.random.randint(10, 100, 18),
        'Price': np.random.uniform(10, 100, 18)
    })


def create_large_cardinality_dataframe(n_rows: int = 1000, n_unique: int = 500) -> pd.DataFrame:
    """
    Create DataFrame with high cardinality column.
    
    Tests: Performance with many unique values in groupby
    
    Args:
        n_rows: Number of rows to generate
        n_unique: Number of unique categories
    """
    np.random.seed(42)
    return pd.DataFrame({
        'ID': range(n_rows),
        'Category': [f'Cat_{i % n_unique}' for i in range(n_rows)],
        'Value': np.random.randint(1, 1000, n_rows),
        'Price': np.random.uniform(10, 500, n_rows),
        'Quantity': np.random.randint(1, 100, n_rows)
    })


def create_special_chars_column_names_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with special characters in column names.
    
    Tests: Column name handling with spaces, symbols
    """
    return pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Total Amount': [100, 200, 300, 400],           # Space
        'Price (USD)': [10, 20, 30, 40],                # Parentheses
        'Sales-2024': [50, 60, 70, 80],                 # Hyphen
        'Profit%': [10, 20, 30, 40],                    # Percent
        'Cost/Unit': [5, 10, 15, 20]                    # Slash
    })


def create_all_nan_column_dataframe() -> pd.DataFrame:
    """
    Create DataFrame with column containing only NaN values.
    
    Tests: Aggregations on all-NaN column
    """
    return pd.DataFrame({
        'ID': [1, 2, 3, 4],
        'Name': ['A', 'B', 'C', 'D'],
        'Value': [100, 200, 300, 400],
        'AllNaN': [np.nan, np.nan, np.nan, np.nan],
        'SomeNaN': [1.0, np.nan, 3.0, np.nan]
    })


def create_test_datasets_dict() -> Dict[str, pd.DataFrame]:
    """
    Create dictionary of all test datasets.
    
    Returns:
        Dict mapping dataset name to DataFrame
    """
    return {
        'empty': create_empty_dataframe(),
        'single_row': create_single_row_dataframe(),
        'unicode': create_unicode_dataframe(),
        'mixed_types': create_mixed_types_dataframe(),
        'edge_numerics': create_edge_numerics_dataframe(),
        'datetime_edge': create_datetime_edge_dataframe(),
        'duplicate_columns': create_duplicate_columns_dataframe(),
        'reserved_keywords': create_reserved_keywords_dataframe(),
        'string_edge_cases': create_string_edge_cases_dataframe(),
        'multiindex_result': create_multiindex_result_dataframe(),
        'large_cardinality': create_large_cardinality_dataframe(),
        'special_chars_columns': create_special_chars_column_names_dataframe(),
        'all_nan_column': create_all_nan_column_dataframe()
    }


def generate_edge_case_datasets(output_dir: Path = None) -> Dict[str, Path]:
    """
    Generate all synthetic test datasets and save to disk.
    
    Args:
        output_dir: Directory to save datasets (default: edge_case_datasets/)
    
    Returns:
        Dict mapping dataset name to file path
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'edge_case_datasets'
    
    output_dir.mkdir(exist_ok=True)
    
    datasets = create_test_datasets_dict()
    file_paths = {}
    
    for name, df in datasets.items():
        file_path = output_dir / f'{name}.xlsx'
        try:
            df.to_excel(file_path, index=False)
            file_paths[name] = file_path
            print(f"✅ Generated: {file_path}")
        except Exception as e:
            print(f"❌ Failed to generate {name}: {e}")
    
    print(f"\n📊 Generated {len(file_paths)}/{len(datasets)} test datasets")
    return file_paths


if __name__ == "__main__":
    # Generate datasets when run directly
    print("Generating edge case test datasets...")
    file_paths = generate_edge_case_datasets()
    print("\n✨ All datasets generated successfully!")
