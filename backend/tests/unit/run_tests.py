#!/usr/bin/env python3
"""
Unit Test Runner for Browser Automation Agent
Runs all unit tests with detailed reporting and coverage
"""

import sys
import os
import subprocess
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(backend_dir))

def run_tests(args=None):
    """Run unit tests with pytest"""
    
    test_dir = Path(__file__).parent
    
    # Default pytest arguments
    pytest_args = [
        'pytest',
        str(test_dir),
        '-v',  # Verbose
        '--tb=short',  # Short traceback
        '--color=yes',  # Colored output
    ]
    
    # Add custom arguments if provided
    if args:
        pytest_args.extend(args)
    
    print("=" * 80)
    print("Running Browser Agent Unit Tests")
    print("=" * 80)
    print(f"Test Directory: {test_dir}")
    print(f"Command: {' '.join(pytest_args)}")
    print("=" * 80)
    print()
    
    # Run pytest
    result = subprocess.run(pytest_args)
    
    return result.returncode

def run_with_coverage():
    """Run tests with coverage report"""
    
    test_dir = Path(__file__).parent
    
    pytest_args = [
        'pytest',
        str(test_dir),
        '-v',
        '--cov=agents.browser_automation_agent',
        '--cov-report=term-missing',
        '--cov-report=html:htmlcov',
        '--cov-report=xml',
    ]
    
    print("=" * 80)
    print("Running Browser Agent Unit Tests with Coverage")
    print("=" * 80)
    print()
    
    result = subprocess.run(pytest_args)
    
    if result.returncode == 0:
        print()
        print("=" * 80)
        print("Coverage report generated:")
        print("  - Terminal: See above")
        print("  - HTML: htmlcov/index.html")
        print("  - XML: coverage.xml")
        print("=" * 80)
    
    return result.returncode

def run_specific_file(filename):
    """Run tests from a specific file"""
    
    test_dir = Path(__file__).parent
    test_file = test_dir / filename
    
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        return 1
    
    pytest_args = [
        'pytest',
        str(test_file),
        '-v',
        '--tb=short',
    ]
    
    print(f"Running tests from: {filename}")
    print("=" * 80)
    
    result = subprocess.run(pytest_args)
    return result.returncode

def run_failed_only():
    """Run only previously failed tests"""
    
    test_dir = Path(__file__).parent
    
    pytest_args = [
        'pytest',
        str(test_dir),
        '-v',
        '--lf',  # Last failed
        '--tb=short',
    ]
    
    print("Running previously failed tests only")
    print("=" * 80)
    
    result = subprocess.run(pytest_args)
    return result.returncode

def list_tests():
    """List all available tests"""
    
    test_dir = Path(__file__).parent
    
    pytest_args = [
        'pytest',
        str(test_dir),
        '--collect-only',
        '-q',
    ]
    
    print("Available Tests:")
    print("=" * 80)
    
    result = subprocess.run(pytest_args)
    return result.returncode

def main():
    """Main entry point"""
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'coverage':
            return run_with_coverage()
        
        elif command == 'failed':
            return run_failed_only()
        
        elif command == 'list':
            return list_tests()
        
        elif command == 'file':
            if len(sys.argv) < 3:
                print("Usage: python run_tests.py file <filename>")
                return 1
            return run_specific_file(sys.argv[2])
        
        elif command == 'help':
            print("Browser Agent Unit Test Runner")
            print()
            print("Usage:")
            print("  python run_tests.py              # Run all tests")
            print("  python run_tests.py coverage     # Run with coverage report")
            print("  python run_tests.py failed       # Run only failed tests")
            print("  python run_tests.py list         # List all tests")
            print("  python run_tests.py file <name>  # Run specific test file")
            print("  python run_tests.py help         # Show this help")
            print()
            print("Examples:")
            print("  python run_tests.py")
            print("  python run_tests.py coverage")
            print("  python run_tests.py file test_vision_manager.py")
            return 0
        
        else:
            # Pass unknown arguments to pytest
            return run_tests(sys.argv[1:])
    
    else:
        # Run all tests by default
        return run_tests()

if __name__ == '__main__':
    sys.exit(main())
