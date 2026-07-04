#!/bin/zsh
cd /Users/oe/featurescope
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "=== FeatureScope screen: sentiment, top 12, with necessity ==="
PYTHONPATH=src:. /Users/oe/sae_project/.venv/bin/python -c "
from featurescope import FeatureScope
from featurescope import data
import json
pos, neg = data.examples_for('sentiment')
fs = FeatureScope(layer=12).fit(pos, neg).calibrate()
fs.screen(top_k=12)
fs.report()
json.dump(fs.to_dict(), open('screen_sentiment.json','w'), indent=2)
print('wrote screen_sentiment.json')
" 2>&1 | tee screen_run.log
echo "=== done — you can close this window ==="
