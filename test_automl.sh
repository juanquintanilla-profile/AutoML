#!/bin/bash
# Test script for AutoML Agent

echo "====================================="
echo "AutoML Agent - Test Run"
echo "====================================="
echo ""

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated"
    echo "Run: source venv/bin/activate"
    exit 1
fi

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found"
    echo "Run: cp .env.example .env"
    echo "Then edit .env with your credentials"
    exit 1
fi

# Check if dataset exists
if [ ! -f data/customer_churn.csv ]; then
    echo "📊 Generating example dataset..."
    python generate_example_data.py
    echo ""
fi

echo "🚀 Starting AutoML Agent..."
echo "Dataset: data/customer_churn.csv"
echo "Target: churn"
echo ""
echo "This will run a complete AutoML workflow:"
echo "  1. Data analysis and preprocessing"
echo "  2. Model selection and pipeline creation"
echo "  3. Hyperparameter optimization (FLAML)"
echo "  4. Model evaluation and comparison"
echo ""
echo "Results will be saved in: automl_agent/output/"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."
echo ""

# Run AutoML (using the package module)
python -m automl_agent \
  --data data/customer_churn.csv \
  --target churn \
  --config automl_agent/config.yaml \
  --output automl_agent/output

echo ""
echo "====================================="
echo "✓ AutoML run completed!"
echo "====================================="
echo ""
echo "Check results in:"
echo "  - automl_agent/output/model.joblib"
echo "  - automl_agent/output/metrics.json"
echo "  - automl_agent/output/run_summary.json"
