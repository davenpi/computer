#!/bin/bash
# Run all 7 shopping items sequentially overnight.
# Usage: nohup ./run_shopping.sh &

set -e

echo "=== Shopping batch started at $(date) ==="

for item in 1 2 3 4 5 6 7; do
    echo ""
    echo "--- Item $item starting at $(date) ---"
    python -m shopping.agent --item "$item" --max-iterations 40
    echo "--- Item $item finished at $(date) ---"

    # Brief pause between runs to let Safari settle
    sleep 5
done

echo ""
echo "=== Shopping batch finished at $(date) ==="
echo "Results:"
ls -la results/
