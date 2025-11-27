#!/usr/bin/env bash
set -e

# Activate virtual environment (GitHub Actions will handle dependencies)
if [ -d ".venv" ]; then
    echo "Activating local virtual environment..."
    source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
fi

echo "Running pipeline..."
python src/train.py
