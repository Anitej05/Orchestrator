"""
Test runner for Document Agent unit tests.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run unit tests only
    python run_tests.py --integration      # Run integration tests only
    python run_tests.py --coverage         # Run with coverage report
"""

import sys
import subprocess
from pathlib import Path

def run_tests(args=None):
    """Run pytest with specified arguments."""
    test_dir = Path(__file__).parent
    
    cmd = ['pytest', str(test_dir), '-v']
    
    if args:
        if '--unit' in args:
            cmd.extend(['-m', 'unit'])
        elif '--integration' in args:
            cmd.extend(['-m', 'integration'])
        elif '--slow' in args:
            cmd.extend(['-m', 'slow'])
        elif '--fast' in args:
            cmd.extend(['-m', 'not slow'])
        
        if '--coverage' in args:
            cmd.extend([
                '--cov=agents.document_agent',
                '--cov-report=html',
                '--cov-report=term'
            ])
        
        if '--verbose' in args or '-vv' in args:
            cmd.append('-vv')
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    args = sys.argv[1:]
    
    if '--help' in args or '-h' in args:
        print(__doc__)
        return 0
    
    return run_tests(args)


if __name__ == '__main__':
    sys.exit(main())
