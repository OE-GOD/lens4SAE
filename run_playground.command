#!/bin/zsh
cd /Users/oe/featurescope
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== FeatureScope Playground — leave this window open ==="
PYTHONPATH=src:. /Users/oe/sae_project/.venv/bin/python playground.py
