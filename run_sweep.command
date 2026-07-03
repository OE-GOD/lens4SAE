#!/bin/zsh
cd /Users/oe/featurescope
if pgrep -f "phase_bc.py --resume" >/dev/null; then
  echo "sweep already running elsewhere — not starting a second copy"; exit 0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "=== resuming sweep OFFLINE (all models/SAEs served from local cache) ==="
PYTHONPATH=src:. /Users/oe/sae_project/.venv/bin/python phase_bc.py --resume --mixed 2>&1 | tee -a sweep_terminal.log
echo "=== sweep finished — you can close this window ==="
