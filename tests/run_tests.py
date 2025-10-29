#!/usr/bin/env python
"""
Test runner for audio components.

Runs all unit tests and generates a coverage report.

Usage:
    python tests/run_tests.py                    # Run all tests
    python tests/run_tests.py --verbose          # Verbose output
    python tests/run_tests.py --coverage         # With coverage report
    python tests/run_tests.py --unit             # Only unit tests
    python tests/run_tests.py --integration      # Only integration tests
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_tests(args):
    """Run tests based on arguments."""
    import pytest

    pytest_args = []

    # Select test directory
    if args.unit:
        pytest_args.append("tests/unit")
    elif args.integration:
        pytest_args.append("tests/integration")
    else:
        pytest_args.append("tests")

    # Verbosity
    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    # Traceback style
    pytest_args.extend(["--tb=short"])

    # Coverage
    if args.coverage:
        pytest_args.extend([
            "--cov=src",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Specific test file or pattern
    if args.test:
        pytest_args.append(args.test)

    # Stop on first failure
    if args.fail_fast:
        pytest_args.append("-x")

    # Show local variables on failure
    if args.show_locals:
        pytest_args.append("-l")

    print("Running tests with arguments:", " ".join(pytest_args))
    print("=" * 70)

    exit_code = pytest.main(pytest_args)

    if args.coverage:
        print("\nCoverage report generated in htmlcov/index.html")

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="Run audio component tests")

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate coverage report (requires pytest-cov)"
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests"
    )

    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests"
    )

    parser.add_argument(
        "--test", "-t",
        type=str,
        help="Run specific test file or pattern (e.g., test_audio_tokenizer.py)"
    )

    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true",
        help="Stop on first test failure"
    )

    parser.add_argument(
        "--show-locals", "-l",
        action="store_true",
        help="Show local variables on failure"
    )

    args = parser.parse_args()

    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()
