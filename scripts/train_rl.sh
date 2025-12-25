#!/bin/bash
# Setup and run RL training with prime-rl
#
# Usage: ./scripts/train_rl.sh [SFT_CHECKPOINT] [DATASET_PATH]
#
# Environment variables:
#   OPENAI_API_KEY - For LLM judge
#   JUDGE_MODEL - Judge model (default: gpt-4o-mini)
#
# Example:
#   OPENAI_API_KEY=sk-... ./scripts/train_rl.sh ./outputs/checkpoints/sft-final ./data/synthetic/rl_prompts.jsonl

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_CHECKPOINT="${1:-$PROJECT_ROOT/outputs/checkpoints/sft-final}"
DATASET_PATH="${2:-$PROJECT_ROOT/data/synthetic/rl_prompts.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

echo "Setting up RL Training"
echo "  SFT Checkpoint: $SFT_CHECKPOINT"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Judge Model: $JUDGE_MODEL"

# Generate prime-rl config
python3 -m src.training.train rl \
    "$SFT_CHECKPOINT" \
    "$DATASET_PATH" \
    --output "$OUTPUT_DIR" \
    --judge "$JUDGE_MODEL"

CONFIG_PATH="$OUTPUT_DIR/prime_rl_config.yaml"

echo ""
echo "Prime-RL configuration generated: $CONFIG_PATH"
echo ""
echo "To run RL training:"
echo "  python -m prime_rl.train --config $CONFIG_PATH"
echo ""

# Optionally run automatically
if [ "${RUN_PRIME_RL:-false}" = "true" ]; then
    echo "Running prime-rl training..."
    python -m prime_rl.train --config "$CONFIG_PATH"
fi
