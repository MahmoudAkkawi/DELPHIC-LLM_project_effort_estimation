#!/bin/bash
# Quick test: 3 projects, 1 seed, 2 conditions (~$2-5)
set -e
pip install openai pydantic sentence-transformers numpy pandas scipy scikit-learn -q
python -m delphic_llm.evaluation.run_experiment \
    --data data/nasa93.csv \
    --n 3 \
    --seeds 42 \
    --conditions delphic_full b1 \
    --output results/test/
python -m delphic_llm.evaluation.generate_tables --results results/test/
