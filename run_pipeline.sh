#!/bin/bash
#===============================================================================
# Deep Research Agent Training Pipeline
#
# Full pipeline for training a research agent with Code Reasoning
# on a local library of scientific papers.
#
# Requirements:
#   - 4x H200 GPUs (or adjust NUM_GPUS)
#   - ~500 DOCX papers in DOCX_DIR
#   - pandoc installed (for DOCX conversion)
#   - OpenAI/Anthropic API key (for synthetic data generation)
#
# Usage:
#   ./run_pipeline.sh [stage]
#
# Stages:
#   all        - Run entire pipeline (default)
#   setup      - Install dependencies only
#   convert    - Convert DOCX papers to library
#   generate   - Generate synthetic training data
#   sft        - Run SFT training
#   rl         - Run RL training
#   eval       - Evaluate final model
#===============================================================================

set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration (modify as needed)
#-------------------------------------------------------------------------------

# Paths
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCX_DIR="${DOCX_DIR:-./data/raw_docx}"
LIBRARY_DIR="${LIBRARY_DIR:-./data/library}"
DATASET_DIR="${DATASET_DIR:-./data/datasets}"
WORKSPACE_DIR="${WORKSPACE_DIR:-./data/workspace}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# Model
MODEL_NAME="${MODEL_NAME:-GigaChat-Lightning-Instruct}"
MODEL_PATH="${MODEL_PATH:-}"  # Optional local path

# Training
NUM_GPUS="${NUM_GPUS:-4}"
NUM_SAMPLES="${NUM_SAMPLES:-10000}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
RL_STEPS="${RL_STEPS:-10000}"
RL_ALGORITHM="${RL_ALGORITHM:-grpo}"

# API Keys (for data generation)
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"
GENERATOR_MODEL="${GENERATOR_MODEL:-gpt-4o}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v3}"

#-------------------------------------------------------------------------------
# Utility Functions
#-------------------------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is required but not installed"
    fi
}

create_dirs() {
    mkdir -p "$LIBRARY_DIR" "$DATASET_DIR" "$WORKSPACE_DIR" "$CHECKPOINT_DIR"
    mkdir -p "$CHECKPOINT_DIR/sft" "$CHECKPOINT_DIR/rl"
}

#-------------------------------------------------------------------------------
# Stage: Setup
#-------------------------------------------------------------------------------

stage_setup() {
    log_info "=== Stage: Setup ==="

    # Check required tools
    check_command python3
    check_command pip

    # Check for pandoc (optional but recommended)
    if ! command -v pandoc &> /dev/null; then
        log_warning "pandoc not found. Installing..."
        sudo apt-get update && sudo apt-get install -y pandoc
    fi

    # Check for ripgrep (optional but faster)
    if ! command -v rg &> /dev/null; then
        log_warning "ripgrep not found. Installing for faster search..."
        sudo apt-get install -y ripgrep || true
    fi

    # Create virtual environment
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv venv
    fi

    # Activate and install
    source venv/bin/activate

    log_info "Installing project dependencies..."
    pip install --upgrade pip
    pip install -e ".[training,dev]"

    # Install prime-rl and verifiers
    log_info "Installing prime-rl and verifiers..."
    pip install -e "git+https://github.com/PrimeIntellect-ai/prime-rl.git#egg=prime-rl" || log_warning "prime-rl install failed (optional)"
    pip install -e "git+https://github.com/PrimeIntellect-ai/verifiers.git#egg=verifiers" || log_warning "verifiers install failed (optional)"

    # Create directories
    create_dirs

    log_success "Setup complete!"
}

#-------------------------------------------------------------------------------
# Stage: Convert Library
#-------------------------------------------------------------------------------

stage_convert() {
    log_info "=== Stage: Convert Library ==="
    source venv/bin/activate

    if [ ! -d "$DOCX_DIR" ]; then
        log_error "DOCX directory not found: $DOCX_DIR"
    fi

    docx_count=$(find "$DOCX_DIR" -name "*.docx" | wc -l)
    log_info "Found $docx_count DOCX files in $DOCX_DIR"

    if [ "$docx_count" -eq 0 ]; then
        log_error "No DOCX files found. Please add papers to $DOCX_DIR"
    fi

    log_info "Converting DOCX papers to structured library..."
    python scripts/convert_library.py build "$DOCX_DIR" "$LIBRARY_DIR"

    # Show library info
    python scripts/convert_library.py info "$LIBRARY_DIR"

    log_success "Library conversion complete!"
}

#-------------------------------------------------------------------------------
# Stage: Generate Synthetic Data
#-------------------------------------------------------------------------------

stage_generate() {
    log_info "=== Stage: Generate Synthetic Data ==="
    source venv/bin/activate

    if [ ! -f "$LIBRARY_DIR/index.json" ]; then
        log_error "Library not found. Run 'convert' stage first."
    fi

    # Check for API key
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$ANTHROPIC_API_KEY" ]; then
        log_error "No API key found. Set OPENAI_API_KEY or ANTHROPIC_API_KEY"
    fi

    log_info "Generating $NUM_SAMPLES synthetic training samples..."
    log_info "Generator model: $GENERATOR_MODEL"

    python scripts/generate_dataset.py generate \
        "$LIBRARY_DIR" \
        "$DATASET_DIR" \
        --samples "$NUM_SAMPLES" \
        --model "$GENERATOR_MODEL" \
        --split

    # Show stats
    if [ -f "$DATASET_DIR/sft_train.jsonl" ]; then
        python scripts/generate_dataset.py stats "$DATASET_DIR/sft_train.jsonl"
    fi

    if [ -f "$DATASET_DIR/rl_prompts.jsonl" ]; then
        python scripts/generate_dataset.py stats "$DATASET_DIR/rl_prompts.jsonl"
    fi

    log_success "Data generation complete!"
}

#-------------------------------------------------------------------------------
# Stage: SFT Training
#-------------------------------------------------------------------------------

stage_sft() {
    log_info "=== Stage: SFT Training ==="
    source venv/bin/activate

    SFT_DATA="$DATASET_DIR/sft_train.jsonl"
    if [ ! -f "$SFT_DATA" ]; then
        log_error "SFT data not found: $SFT_DATA. Run 'generate' stage first."
    fi

    sft_samples=$(wc -l < "$SFT_DATA")
    log_info "SFT training on $sft_samples samples"
    log_info "Epochs: $SFT_EPOCHS"

    # Set model path
    model_arg="$MODEL_NAME"
    if [ -n "$MODEL_PATH" ]; then
        model_arg="$MODEL_PATH"
    fi

    log_info "Starting SFT training..."
    python scripts/train_sft.py train \
        "$SFT_DATA" \
        --model "$model_arg" \
        --output "$CHECKPOINT_DIR/sft" \
        --epochs "$SFT_EPOCHS" \
        --batch-size 4 \
        --no-lora

    log_success "SFT training complete! Checkpoint: $CHECKPOINT_DIR/sft"
}

#-------------------------------------------------------------------------------
# Stage: RL Training
#-------------------------------------------------------------------------------

stage_rl() {
    log_info "=== Stage: RL Training ==="
    source venv/bin/activate

    RL_DATA="$DATASET_DIR/rl_prompts.jsonl"
    if [ ! -f "$RL_DATA" ]; then
        log_error "RL data not found: $RL_DATA. Run 'generate' stage first."
    fi

    rl_samples=$(wc -l < "$RL_DATA")
    log_info "RL training on $rl_samples prompts"
    log_info "Algorithm: $RL_ALGORITHM"
    log_info "Total steps: $RL_STEPS"
    log_info "GPUs: $NUM_GPUS"

    # Check for SFT checkpoint
    SFT_CKPT=""
    if [ -d "$CHECKPOINT_DIR/sft/final" ]; then
        SFT_CKPT="--sft $CHECKPOINT_DIR/sft/final"
        log_info "Using SFT checkpoint: $CHECKPOINT_DIR/sft/final"
    else
        log_warning "No SFT checkpoint found, training from base model"
    fi

    # Generate config for prime-rl
    log_info "Generating prime-rl config..."
    python scripts/train_rl.py config \
        "$CHECKPOINT_DIR/rl/prime_rl_config.yaml" \
        --algo "$RL_ALGORITHM" \
        --gpus "$NUM_GPUS"

    log_info "Starting RL training..."

    # For full distributed training with prime-rl:
    # torchrun --nproc_per_node=$NUM_GPUS \
    #     -m prime_rl.train \
    #     --config "$CHECKPOINT_DIR/rl/prime_rl_config.yaml"

    # Fallback to our trainer (for testing/single GPU)
    python scripts/train_rl.py train \
        "$RL_DATA" \
        $SFT_CKPT \
        --output "$CHECKPOINT_DIR/rl" \
        --algo "$RL_ALGORITHM" \
        --steps "$RL_STEPS" \
        --gpus "$NUM_GPUS" \
        --library "$LIBRARY_DIR" \
        --judge "$JUDGE_MODEL"

    log_success "RL training complete! Checkpoint: $CHECKPOINT_DIR/rl"
}

#-------------------------------------------------------------------------------
# Stage: Evaluation
#-------------------------------------------------------------------------------

stage_eval() {
    log_info "=== Stage: Evaluation ==="
    source venv/bin/activate

    # Find latest checkpoint
    CKPT="$CHECKPOINT_DIR/rl"
    if [ ! -d "$CKPT" ]; then
        CKPT="$CHECKPOINT_DIR/sft/final"
    fi

    if [ ! -d "$CKPT" ]; then
        log_error "No checkpoint found for evaluation"
    fi

    log_info "Evaluating model: $CKPT"

    # Use RL data for evaluation (could also use held-out set)
    EVAL_DATA="$DATASET_DIR/rl_prompts.jsonl"

    python scripts/train_rl.py eval \
        "$CKPT" \
        "$EVAL_DATA" \
        --library "$LIBRARY_DIR" \
        --samples 100

    log_success "Evaluation complete!"
}

#-------------------------------------------------------------------------------
# Stage: Full Pipeline
#-------------------------------------------------------------------------------

stage_all() {
    log_info "=== Running Full Pipeline ==="

    stage_setup
    stage_convert
    stage_generate
    stage_sft
    stage_rl
    stage_eval

    log_success "=== Pipeline Complete! ==="
    echo ""
    echo "Trained model: $CHECKPOINT_DIR/rl"
    echo ""
    echo "Next steps:"
    echo "  1. Review evaluation results in $CHECKPOINT_DIR/rl/eval_results.json"
    echo "  2. For production, merge LoRA weights if used"
    echo "  3. Deploy with your inference server"
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

main() {
    echo "=============================================="
    echo "  Deep Research Agent Training Pipeline"
    echo "=============================================="
    echo ""

    cd "$PROJECT_DIR"

    STAGE="${1:-all}"

    case "$STAGE" in
        setup)
            stage_setup
            ;;
        convert)
            stage_convert
            ;;
        generate)
            stage_generate
            ;;
        sft)
            stage_sft
            ;;
        rl)
            stage_rl
            ;;
        eval)
            stage_eval
            ;;
        all)
            stage_all
            ;;
        *)
            echo "Usage: $0 [stage]"
            echo ""
            echo "Stages:"
            echo "  all       - Run entire pipeline (default)"
            echo "  setup     - Install dependencies"
            echo "  convert   - Convert DOCX papers to library"
            echo "  generate  - Generate synthetic training data"
            echo "  sft       - Run SFT training"
            echo "  rl        - Run RL training"
            echo "  eval      - Evaluate final model"
            exit 1
            ;;
    esac
}

main "$@"
