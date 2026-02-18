"""
Test Image Analysis Tool
Analyze the temperature_comparison.png file
"""

import sys
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path('backend/.env'))

sys.path.insert(0, 'backend')

from backend.tools.image_tools import analyze_image
import os

print("="*70)
print("IMAGE ANALYSIS TEST")
print("="*70)
print("Analyzing: temperature_comparison.png")
print("="*70)

# Check if file exists
image_path = Path('temperature_comparison.png')
if not image_path.exists():
    print(f"\n[ERROR] File not found: {image_path}")
    print("Looking for the file...")
    # Search for it
    for png in Path('.').glob('**/*.png'):
        print(f"  Found: {png}")
    sys.exit(1)

print(f"\nFile found: {image_path.absolute()}")
print(f"File size: {image_path.stat().st_size} bytes")

# Analyze the image
print("\nAnalyzing image with Groq vision model...")
print("Query: 'What does this chart show? Describe the data and comparison.'")
print("-"*70)

try:
    # Use invoke method since analyze_image is a StructuredTool
    result = analyze_image.invoke({
        "image_path": str(image_path.absolute()),
        "query": "What does this chart show? Describe the data and comparison."
    })
    
    print("\nANALYSIS RESULT:")
    print("="*70)
    
    if 'error' in result:
        print(f"[ERROR] {result['error']}")
    else:
        print(f"Query: {result.get('query', 'N/A')}")
        print(f"\nAnswer:")
        print(result.get('answer', 'No answer generated'))
        
    print("="*70)
    print("\n[TEST COMPLETE]")
    
except Exception as e:
    print(f"\n[ERROR] Analysis failed: {e}")
    import traceback
    traceback.print_exc()
