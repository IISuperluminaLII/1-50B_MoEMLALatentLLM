#!/bin/bash
# Run DeepSeek-V3 test suite

set -e

echo "========================================="
echo "DeepSeek-V3 Test Suite"
echo "========================================="
echo ""

# Default options
COVERAGE=true
MARKERS=""
VERBOSE=false
PARALLEL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --gpu)
            MARKERS="-m gpu"
            shift
            ;;
        --unit)
            MARKERS="-m unit"
            shift
            ;;
        --integration)
            MARKERS="-m integration"
            shift
            ;;
        --fast)
            MARKERS="-m 'not slow'"
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -n|--parallel)
            PARALLEL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--no-coverage] [--gpu] [--unit] [--integration] [--fast] [-v] [-n]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv"
fi

if [ "$PARALLEL" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=html --cov-report=term-missing"
fi

if [ -n "$MARKERS" ]; then
    PYTEST_CMD="$PYTEST_CMD $MARKERS"
fi

# Run tests
echo "Running: $PYTEST_CMD"
echo ""

$PYTEST_CMD

# Show coverage report location
if [ "$COVERAGE" = true ]; then
    echo ""
    echo "========================================="
    echo "Coverage report generated:"
    echo "  HTML: htmlcov/index.html"
    echo "========================================="
fi

echo ""
echo "Tests completed!"
