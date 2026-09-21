#!/usr/bin/env bash
# Run all three POPE splits for Qwen2-VL: random, popular, adversarial
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "Starting Qwen2-VL POPE (random)"
echo "=========================================="
bash "$SCRIPT_DIR/run_qwen2vl_pope_coco.sh" --split random

echo "=========================================="
echo "Starting Qwen2-VL POPE (popular)"
echo "=========================================="
bash "$SCRIPT_DIR/run_qwen2vl_pope_coco.sh" --split popular

echo "=========================================="
echo "Starting Qwen2-VL POPE (adversarial)"
echo "=========================================="
bash "$SCRIPT_DIR/run_qwen2vl_pope_coco.sh" --split adversarial

echo "All Qwen2-VL POPE splits completed successfully!"
