#!/bin/zsh
cd /Users/oe/featurescope
if pgrep -f "phase_d.py" >/dev/null; then echo "phase D already running"; exit 0; fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== Phase D: confirmatory backfire (mixed precision) ==="
PYTHONPATH=src:. /Users/oe/sae_project/.venv/bin/python phase_d.py 2>&1 | tee -a phase_d_terminal.log
echo "=== Phase D finished — you can close this window ==="
