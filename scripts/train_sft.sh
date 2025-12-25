#!/bin/bash
# Run SFT training
#
# Usage: ./scripts/train_sft.sh [MODEL_PATH] [DATASET_PATH]
#
# For distributed training on multiple GPUs:
#   NUM_GPUS=4 ./scripts/train_sft.sh model_name data.jsonl
#
# Example:
#   ./scripts/train_sft.sh GigaChat-Lightning-Instruct ./data/synthetic/sft_train.jsonl

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-GigaChat-Lightning-Instruct}"
DATASET_PATH="${2:-$PROJECT_ROOT/data/synthetic/sft_train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

# Training parameters
EPOCHS="${SFT_EPOCHS:-3}"
BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
LR="${SFT_LR:-2e-5}"

NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"

echo "Starting SFT Training"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: $NUM_GPUS"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running distributed training..."
    torchrun --nproc_per_node="$NUM_GPUS" \
        -m src.training.train sft \
        "$MODEL_PATH" \
        "$DATASET_PATH" \
        --output "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR"
else
    python3 -m src.training.train sft \
        "$MODEL_PATH" \
        "$DATASET_PATH" \
        --output "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR"
fi

echo "SFT training complete!"
echo "Checkpoint saved to: $OUTPUT_DIR/checkpoints/sft-final"
