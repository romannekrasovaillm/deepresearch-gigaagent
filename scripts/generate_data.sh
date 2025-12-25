#!/bin/bash
# Generate synthetic training data
#
# Usage: ./scripts/generate_data.sh [TOTAL_SAMPLES] [MODEL]
#
# Environment variables:
#   OPENAI_API_KEY - OpenAI API key
#   DEEPSEEK_API_KEY - DeepSeek API key (alternative)
#   LIBRARY_PATH - Path to library (default: /mnt/library)
#
# Example:
#   OPENAI_API_KEY=sk-... ./scripts/generate_data.sh 10000 gpt-4o-mini

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOTAL_SAMPLES="${1:-1000}"
MODEL="${2:-gpt-4o-mini}"
LIBRARY_PATH="${LIBRARY_PATH:-/mnt/library}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/data/synthetic}"

echo "Generating $TOTAL_SAMPLES samples using $MODEL"
echo "Library: $LIBRARY_PATH"
echo "Output: $OUTPUT_DIR"

python3 -m src.generation.synthetic generate \
    --library "$LIBRARY_PATH" \
    --output "$OUTPUT_DIR" \
    --total "$TOTAL_SAMPLES" \
    --model "$MODEL"

echo "Done! Validating dataset..."
python3 -m src.generation.synthetic validate "$OUTPUT_DIR/sft_train.jsonl"
python3 -m src.generation.synthetic validate "$OUTPUT_DIR/rl_prompts.jsonl"
