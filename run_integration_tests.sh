#!/bin/bash
# Script to run main integration tests
# Smart Resume Screening System - Main Execution Integration Tests

echo "=========================================="
echo "Main Integration Tests"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[WARN]  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "[PASS] Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo ""
echo "Checking dependencies..."
if ! python -c "import spacy" 2>/dev/null; then
    echo "[WARN]  Dependencies not installed. Installing..."
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    echo "[PASS] Dependencies installed"
else
    echo "[PASS] Dependencies already installed"
fi

echo ""
echo "=========================================="
echo "Running Integration Tests"
echo "=========================================="
echo ""

# Run the integration tests
pytest tests/test_main_integration.py -v --tb=short

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "[PASS] All integration tests passed!"
    echo ""
    echo "Test Coverage:"
    echo "  - CSV Processing Pipeline: 4 tests"
    echo "  - PDF Processing Pipeline: 2 tests"
    echo "  - Cross-Source Validation: 2 tests"
    echo "  - CLI Argument Parsing: 6 tests"
    echo "  - Output File Generation: 5 tests"
    echo "  - Configuration Loading: 3 tests"
    echo "  - Error Handling: 3 tests"
    echo "  - Utility Functions: 1 test"
    echo "  --------------------------------"
    echo "  Total: 26 integration tests"
else
    echo "[FAIL] Some tests failed. Exit code: $EXIT_CODE"
    echo ""
    echo "To see detailed output, run:"
    echo "  pytest tests/test_main_integration.py -v"
fi

echo ""
echo "=========================================="

exit $EXIT_CODE
