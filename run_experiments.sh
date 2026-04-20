#!/bin/bash
# DELPHIC-LLM: One-command experiment runner
# Run from the directory containing delphic_llm/

set -e

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: Set OPENAI_API_KEY first: export OPENAI_API_KEY=sk-..."
    exit 1
fi

# Check data
if [ ! -f "data/nasa93.csv" ]; then
    echo "ERROR: Place NASA93 dataset at data/nasa93.csv"
    echo "Download from: https://promise.site.uottawa.ca/SERepository"
    exit 1
fi

echo "Installing dependencies..."
pip install openai pydantic sentence-transformers numpy pandas scipy scikit-learn

echo ""
echo "Running experiments (50 projects, 3 seeds, all conditions)..."
echo "Estimated cost: ~\$35-60, time: ~2-4 hours"
echo ""

python -m delphic_llm.evaluation.run_experiment \
    --data data/nasa93.csv \
    --n 50 \
    --seeds 42 43 44 \
    --conditions delphic_full b1 b2 b3 abl1 abl2 \
    --abl3_n 20 \
    --output results/

echo ""
echo "Generating paper tables..."
python -m delphic_llm.evaluation.generate_tables --results results/
