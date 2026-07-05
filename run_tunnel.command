#!/bin/zsh
echo "=== Public tunnel to your live playground — leave this window open ==="
/opt/homebrew/bin/cloudflared tunnel --url http://localhost:8765 2>&1 | tee /Users/oe/featurescope/tunnel.log
