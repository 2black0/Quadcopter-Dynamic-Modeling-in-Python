#!/usr/bin/env python3
"""
Comprehensive CLI test script.
"""

import subprocess
import sys
import os

def run_cli_test(command, description):
    """Run a CLI test and report results."""
    print(f"Testing: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✓ PASSED")
            if result.stdout.strip():
                print(f"  Output: {result.stdout.strip()[:100]}{'...' if len(result.stdout.strip()) > 100 else ''}")
        else:
            print("✗ FAILED")
            print(f"  Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ TIMEOUT")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False
        
    print()
    return True

def main():
    """Run comprehensive CLI tests."""
    print("Running Comprehensive CLI Tests")
    print("=" * 40)
    
    tests = [
        # Basic functionality tests
        ("python -m quadcopter --duration 1 --quiet", "Basic simulation"),
        ("python -m quadcopter --help", "Help command"),
        
        # Controller tests
        ("python -m quadcopter --controller pid --target-pos 1 0 1 --duration 1 --quiet", "PID controller"),
        ("python -m quadcopter --controller lqr --duration 1 --quiet", "LQR controller"),
        
        # Output tests
        ("python -m quadcopter --controller pid --target-pos 0 0 1 --duration 1 --csv test.csv --quiet && rm test.csv", "CSV export"),
        ("python -m quadcopter --controller pid --target-pos 0 0 1 --duration 1 --json test.json --quiet && rm test.json", "JSON export"),
        
        # Academic logging tests
        ("python -m quadcopter --controller pid --target-pos 1 1 1 --duration 2 --academic-log results --quiet && rm -rf results", "Academic logging"),
        
        # Advanced options tests
        ("python -m quadcopter --controller pid --target-pos 0 0 2 --duration 1 --init-pos 0 0 1 --quiet", "Custom initial position"),
        ("python -m quadcopter --controller pid --target-pos 1 0 1 --duration 1 --pid-kp 3 3 5 --pid-ki 0.2 0.2 0.3 --pid-kd 0.6 0.6 1.2 --quiet", "Custom PID gains"),
    ]
    
    passed = 0
    total = len(tests)
    
    for command, description in tests:
        if run_cli_test(command, description):
            passed += 1
    
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All CLI tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())