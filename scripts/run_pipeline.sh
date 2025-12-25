#!/bin/bash
# =============================================================================
# Deep Research Agent - Full Training Pipeline
# =============================================================================
# Usage: ./scripts/run_pipeline.sh [OPTIONS]
#
# This script runs the complete training pipeline:
# 1. Environment setup
# 2. Library conversion (DOCX → Markdown)
# 3. Synthetic data generation
# 4. SFT training
# 5. RL training
# 6. Evaluation
# =============================================================================

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================
# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCX_INPUT_DIR="${DOCX_INPUT_DIR:-$PROJECT_ROOT/data/raw}"
LIBRARY_PATH="${LIBRARY_PATH:-/mnt/library}"
WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs}"

# Model
MODEL_NAME="${MODEL_NAME:-GigaChat-Lightning-Instruct}"
MODEL_PATH="${MODEL_PATH:-}"

# Training parameters
TOTAL_SAMPLES="${TOTAL_SAMPLES:-10000}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_BATCH_SIZE="${SFT_BATCH_SIZE:-4}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-64}"

# API Keys (для генерации данных и судьи)
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}"

# Generation model
GEN_MODEL="${GEN_MODEL:-gpt-4o-mini}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"

# Hardware
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"

# Flags
SKIP_CONVERT="${SKIP_CONVERT:-false}"
SKIP_GENERATE="${SKIP_GENERATE:-false}"
SKIP_SFT="${SKIP_SFT:-false}"
SKIP_RL="${SKIP_RL:-false}"
DRY_RUN="${DRY_RUN:-false}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Helper Functions
# =============================================================================
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
}

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "$1 is not installed"
        return 1
    fi
}

run_cmd() {
    if [ "$DRY_RUN" = "true" ]; then
        log_info "[DRY RUN] $*"
    else
        log_info "Running: $*"
        eval "$@"
    fi
}

# =============================================================================
# Step 0: Environment Setup
# =============================================================================
setup_environment() {
    log_info "=========================================="
    log_info "Step 0: Environment Setup"
    log_info "=========================================="

    # Check Python
    check_command python3 || exit 1

    # Check CUDA
    if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        log_warning "CUDA not available. Training will be slow."
    else
        log_info "CUDA available: $NUM_GPUS GPU(s)"
    fi

    # Create directories
    mkdir -p "$LIBRARY_PATH" "$WORKSPACE_PATH" "$DATA_DIR"/{raw,processed,synthetic} "$OUTPUT_DIR"

    # Install dependencies
    log_info "Installing dependencies..."
    run_cmd "pip install -e '$PROJECT_ROOT' -q"

    # Install pandoc if not present
    if ! command -v pandoc &> /dev/null; then
        log_info "Installing pandoc..."
        if [ "$(uname)" = "Linux" ]; then
            run_cmd "sudo apt-get update && sudo apt-get install -y pandoc"
        elif [ "$(uname)" = "Darwin" ]; then
            run_cmd "brew install pandoc"
        fi
    fi

    # Install prime-rl and verifiers if not present
    if ! python3 -c "import prime_rl" 2>/dev/null; then
        log_info "Installing prime-rl..."
        run_cmd "pip install git+https://github.com/PrimeIntellect-ai/prime-rl.git -q"
    fi

    if ! python3 -c "import verifiers" 2>/dev/null; then
        log_info "Installing verifiers..."
        run_cmd "pip install git+https://github.com/PrimeIntellect-ai/verifiers.git -q"
    fi

    log_success "Environment setup complete!"
}

# =============================================================================
# Step 1: Library Conversion
# =============================================================================
convert_library() {
    log_info "=========================================="
    log_info "Step 1: Converting DOCX to Markdown"
    log_info "=========================================="

    if [ "$SKIP_CONVERT" = "true" ]; then
        log_warning "Skipping library conversion (SKIP_CONVERT=true)"
        return 0
    fi

    if [ ! -d "$DOCX_INPUT_DIR" ] || [ -z "$(ls -A "$DOCX_INPUT_DIR"/*.docx 2>/dev/null)" ]; then
        log_warning "No DOCX files found in $DOCX_INPUT_DIR"
        log_info "Please add DOCX files and run again, or set SKIP_CONVERT=true"
        return 0
    fi

    run_cmd "python3 -m src.library.convert convert \
        '$DOCX_INPUT_DIR' \
        --output '$LIBRARY_PATH' \
        --pattern '*.docx'"

    log_success "Library conversion complete!"
    log_info "Library saved to: $LIBRARY_PATH"
}

# =============================================================================
# Step 2: Synthetic Data Generation
# =============================================================================
generate_data() {
    log_info "=========================================="
    log_info "Step 2: Generating Synthetic Training Data"
    log_info "=========================================="

    if [ "$SKIP_GENERATE" = "true" ]; then
        log_warning "Skipping data generation (SKIP_GENERATE=true)"
        return 0
    fi

    # Check for API key
    if [ -z "$OPENAI_API_KEY" ] && [ -z "$DEEPSEEK_API_KEY" ]; then
        log_error "No API key set. Please set OPENAI_API_KEY or DEEPSEEK_API_KEY"
        exit 1
    fi

    # Check library exists
    if [ ! -f "$LIBRARY_PATH/index.json" ]; then
        log_error "Library not found at $LIBRARY_PATH. Run library conversion first."
        exit 1
    fi

    run_cmd "python3 -m src.generation.synthetic generate \
        --library '$LIBRARY_PATH' \
        --output '$DATA_DIR/synthetic' \
        --total $TOTAL_SAMPLES \
        --model '$GEN_MODEL'"

    log_success "Data generation complete!"
    log_info "SFT data: $DATA_DIR/synthetic/sft_train.jsonl"
    log_info "RL data: $DATA_DIR/synthetic/rl_prompts.jsonl"
}

# =============================================================================
# Step 3: SFT Training
# =============================================================================
train_sft() {
    log_info "=========================================="
    log_info "Step 3: Supervised Fine-Tuning (SFT)"
    log_info "=========================================="

    if [ "$SKIP_SFT" = "true" ]; then
        log_warning "Skipping SFT training (SKIP_SFT=true)"
        return 0
    fi

    SFT_DATASET="$DATA_DIR/synthetic/sft_train.jsonl"
    if [ ! -f "$SFT_DATASET" ]; then
        log_error "SFT dataset not found: $SFT_DATASET"
        exit 1
    fi

    # Determine model path
    local model_arg="${MODEL_PATH:-$MODEL_NAME}"

    if [ "$NUM_GPUS" -gt 1 ]; then
        log_info "Running distributed SFT on $NUM_GPUS GPUs"
        run_cmd "torchrun --nproc_per_node=$NUM_GPUS \
            -m src.training.train sft \
            '$model_arg' \
            '$SFT_DATASET' \
            --output '$OUTPUT_DIR' \
            --epochs $SFT_EPOCHS \
            --batch-size $SFT_BATCH_SIZE"
    else
        run_cmd "python3 -m src.training.train sft \
            '$model_arg' \
            '$SFT_DATASET' \
            --output '$OUTPUT_DIR' \
            --epochs $SFT_EPOCHS \
            --batch-size $SFT_BATCH_SIZE"
    fi

    log_success "SFT training complete!"
    log_info "Checkpoint: $OUTPUT_DIR/checkpoints/sft-final"
}

# =============================================================================
# Step 4: RL Training
# =============================================================================
train_rl() {
    log_info "=========================================="
    log_info "Step 4: Reinforcement Learning (RL)"
    log_info "=========================================="

    if [ "$SKIP_RL" = "true" ]; then
        log_warning "Skipping RL training (SKIP_RL=true)"
        return 0
    fi

    RL_DATASET="$DATA_DIR/synthetic/rl_prompts.jsonl"
    SFT_CHECKPOINT="$OUTPUT_DIR/checkpoints/sft-final"

    if [ ! -f "$RL_DATASET" ]; then
        log_error "RL dataset not found: $RL_DATASET"
        exit 1
    fi

    if [ ! -d "$SFT_CHECKPOINT" ]; then
        log_warning "SFT checkpoint not found, using base model"
        SFT_CHECKPOINT="${MODEL_PATH:-$MODEL_NAME}"
    fi

    # Generate prime-rl config
    run_cmd "python3 -m src.training.train rl \
        '$SFT_CHECKPOINT' \
        '$RL_DATASET' \
        --output '$OUTPUT_DIR' \
        --judge '$JUDGE_MODEL'"

    PRIME_CONFIG="$OUTPUT_DIR/prime_rl_config.yaml"

    if [ -f "$PRIME_CONFIG" ]; then
        log_info "Prime-RL config generated: $PRIME_CONFIG"
        log_info ""
        log_info "To run RL training with prime-rl:"
        log_info "  python -m prime_rl.train --config $PRIME_CONFIG"
        log_info ""

        # Optionally run prime-rl
        if [ "${RUN_PRIME_RL:-false}" = "true" ]; then
            run_cmd "python -m prime_rl.train --config '$PRIME_CONFIG'"
        fi
    fi

    log_success "RL setup complete!"
}

# =============================================================================
# Step 5: Evaluation
# =============================================================================
evaluate() {
    log_info "=========================================="
    log_info "Step 5: Evaluation"
    log_info "=========================================="

    # TODO: Add evaluation script
    log_info "Evaluation not yet implemented"
}

# =============================================================================
# Main Pipeline
# =============================================================================
print_banner() {
    echo ""
    echo "=============================================="
    echo "  Deep Research Agent Training Pipeline"
    echo "=============================================="
    echo ""
    echo "Configuration:"
    echo "  Project Root:  $PROJECT_ROOT"
    echo "  Library Path:  $LIBRARY_PATH"
    echo "  Output Dir:    $OUTPUT_DIR"
    echo "  Model:         ${MODEL_PATH:-$MODEL_NAME}"
    echo "  GPUs:          $NUM_GPUS"
    echo "  Total Samples: $TOTAL_SAMPLES"
    echo ""
}

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help              Show this help message"
    echo "  --dry-run           Print commands without executing"
    echo "  --skip-convert      Skip library conversion"
    echo "  --skip-generate     Skip data generation"
    echo "  --skip-sft          Skip SFT training"
    echo "  --skip-rl           Skip RL training"
    echo ""
    echo "Environment Variables:"
    echo "  DOCX_INPUT_DIR      Directory with DOCX files"
    echo "  LIBRARY_PATH        Output library path"
    echo "  MODEL_NAME          Model name or path"
    echo "  TOTAL_SAMPLES       Number of synthetic samples"
    echo "  OPENAI_API_KEY      OpenAI API key for generation"
    echo "  GEN_MODEL           Model for data generation"
    echo "  JUDGE_MODEL         Model for reward judging"
    echo ""
}

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                print_usage
                exit 0
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-convert)
                SKIP_CONVERT=true
                shift
                ;;
            --skip-generate)
                SKIP_GENERATE=true
                shift
                ;;
            --skip-sft)
                SKIP_SFT=true
                shift
                ;;
            --skip-rl)
                SKIP_RL=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done

    print_banner

    # Run pipeline steps
    local start_time=$(date +%s)

    setup_environment
    convert_library
    generate_data
    train_sft
    train_rl
    evaluate

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    echo ""
    log_success "=============================================="
    log_success "Pipeline completed in ${duration}s"
    log_success "=============================================="
    echo ""
    echo "Next steps:"
    echo "  1. Check outputs in: $OUTPUT_DIR"
    echo "  2. Run RL training with prime-rl"
    echo "  3. Evaluate on test set"
    echo ""
}

main "$@"
