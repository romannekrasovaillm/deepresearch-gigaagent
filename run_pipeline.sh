#!/bin/bash
# =============================================================================
# Deep Research Agent - Full Training Pipeline
# =============================================================================
#
# This script runs the complete training pipeline:
# 1. Install dependencies
# 2. Convert DOCX papers to structured library
# 3. Generate synthetic training dataset
# 4. Run SFT training
# 5. Run RL training
#
# Usage:
#   ./run_pipeline.sh [OPTIONS]
#
# Options:
#   --config PATH       Path to config YAML (default: configs/default.yaml)
#   --stage STAGE       Run specific stage: deps, library, data, sft, rl, all
#   --skip-deps         Skip dependency installation
#   --dry-run           Print commands without executing
#   --help              Show this help message
#
# Example:
#   ./run_pipeline.sh --config configs/default.yaml --stage all
#   ./run_pipeline.sh --stage sft --skip-deps
#
# =============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# =============================================================================
# Configuration
# =============================================================================

# Default paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/configs/default.yaml"

# Default directories
RAW_PAPERS_DIR="${SCRIPT_DIR}/data/raw_docx"
LIBRARY_DIR="${SCRIPT_DIR}/data/library"
DATASETS_DIR="${SCRIPT_DIR}/data/datasets"
WORKSPACE_DIR="${SCRIPT_DIR}/data/workspace"
OUTPUT_DIR="${SCRIPT_DIR}/outputs"

# Training defaults
N_SAMPLES=10000
SFT_EPOCHS=3
RL_STEPS=10000
NUM_GPUS=4

# Stage to run
STAGE="all"
SKIP_DEPS=false
DRY_RUN=false

# =============================================================================
# Parse Arguments
# =============================================================================

print_help() {
    sed -n '2,/^# =====/p' "$0" | grep '^#' | sed 's/^# //'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            print_help
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY-RUN] $*"
    else
        log "Running: $*"
        eval "$@"
    fi
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log "WARNING: nvidia-smi not found. GPU may not be available."
        return 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log "Found $GPU_COUNT GPU(s)"

    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        log "WARNING: Expected $NUM_GPUS GPUs, found $GPU_COUNT"
    fi

    return 0
}

# =============================================================================
# Stage 1: Install Dependencies
# =============================================================================

install_dependencies() {
    log "========================================="
    log "Stage 1: Installing Dependencies"
    log "========================================="

    # System packages
    log "Installing system packages..."
    run_cmd "sudo apt-get update -qq"
    run_cmd "sudo apt-get install -y -qq pandoc ripgrep"

    # Python packages
    log "Installing Python packages..."
    run_cmd "pip install -q -r ${SCRIPT_DIR}/requirements.txt"

    # Clone training libraries
    if [ ! -d "${SCRIPT_DIR}/external/prime-rl" ]; then
        log "Cloning prime-rl..."
        run_cmd "git clone https://github.com/PrimeIntellect-ai/prime-rl.git ${SCRIPT_DIR}/external/prime-rl"
        run_cmd "pip install -q -e ${SCRIPT_DIR}/external/prime-rl"
    fi

    if [ ! -d "${SCRIPT_DIR}/external/verifiers" ]; then
        log "Cloning verifiers..."
        run_cmd "git clone https://github.com/PrimeIntellect-ai/verifiers.git ${SCRIPT_DIR}/external/verifiers"
        run_cmd "pip install -q -e ${SCRIPT_DIR}/external/verifiers"
    fi

    log "Dependencies installed successfully!"
}

# =============================================================================
# Stage 2: Convert Library
# =============================================================================

convert_library() {
    log "========================================="
    log "Stage 2: Converting DOCX to Library"
    log "========================================="

    # Create directories
    run_cmd "mkdir -p ${LIBRARY_DIR}"

    # Check for input files
    if [ ! -d "${RAW_PAPERS_DIR}" ]; then
        log "Creating sample papers directory: ${RAW_PAPERS_DIR}"
        run_cmd "mkdir -p ${RAW_PAPERS_DIR}"
        log "WARNING: No DOCX files found. Place your papers in ${RAW_PAPERS_DIR}"
        return 0
    fi

    DOCX_COUNT=$(find "${RAW_PAPERS_DIR}" -name "*.docx" 2>/dev/null | wc -l)
    log "Found ${DOCX_COUNT} DOCX files"

    if [ "$DOCX_COUNT" -eq 0 ]; then
        log "WARNING: No DOCX files found in ${RAW_PAPERS_DIR}"
        return 0
    fi

    # Run conversion
    run_cmd "python -m src.library.converter ${RAW_PAPERS_DIR} ${LIBRARY_DIR}"

    log "Library conversion complete!"
    log "Output: ${LIBRARY_DIR}"
}

# =============================================================================
# Stage 3: Generate Synthetic Dataset
# =============================================================================

generate_dataset() {
    log "========================================="
    log "Stage 3: Generating Synthetic Dataset"
    log "========================================="

    # Create directories
    run_cmd "mkdir -p ${DATASETS_DIR}"

    # Check library exists
    if [ ! -f "${LIBRARY_DIR}/index.json" ]; then
        log "ERROR: Library index not found. Run 'library' stage first."
        exit 1
    fi

    PAPER_COUNT=$(python -c "import json; print(len(json.load(open('${LIBRARY_DIR}/index.json'))))" 2>/dev/null || echo "0")
    log "Library contains ${PAPER_COUNT} papers"

    if [ "$PAPER_COUNT" -eq 0 ]; then
        log "ERROR: Library is empty. Add papers first."
        exit 1
    fi

    # Run generation
    log "Generating ${N_SAMPLES} training samples..."
    run_cmd "python -m src.data_generation.generator \\
        ${LIBRARY_DIR} \\
        ${DATASETS_DIR} \\
        --n-samples ${N_SAMPLES} \\
        --backend openai \\
        --model gpt-4o"

    log "Dataset generation complete!"
    log "Output: ${DATASETS_DIR}"
}

# =============================================================================
# Stage 4: SFT Training
# =============================================================================

train_sft() {
    log "========================================="
    log "Stage 4: Supervised Fine-Tuning"
    log "========================================="

    # Check data exists
    if [ ! -f "${DATASETS_DIR}/sft_train.jsonl" ]; then
        log "ERROR: SFT training data not found. Run 'data' stage first."
        exit 1
    fi

    SFT_OUTPUT="${OUTPUT_DIR}/sft"
    run_cmd "mkdir -p ${SFT_OUTPUT}"

    SFT_SAMPLES=$(wc -l < "${DATASETS_DIR}/sft_train.jsonl")
    log "Training on ${SFT_SAMPLES} SFT examples"

    # Check GPUs
    check_gpu

    # Run SFT training
    log "Starting SFT training..."
    run_cmd "torchrun --nproc_per_node=${NUM_GPUS} \\
        -m src.training.sft_trainer \\
        --train-data ${DATASETS_DIR}/sft_train.jsonl \\
        --output-dir ${SFT_OUTPUT} \\
        --epochs ${SFT_EPOCHS} \\
        --batch-size 4 \\
        --lr 2e-5"

    log "SFT training complete!"
    log "Model saved to: ${SFT_OUTPUT}"
}

# =============================================================================
# Stage 5: RL Training
# =============================================================================

train_rl() {
    log "========================================="
    log "Stage 5: Reinforcement Learning"
    log "========================================="

    # Check data exists
    if [ ! -f "${DATASETS_DIR}/rl_prompts.jsonl" ]; then
        log "ERROR: RL prompts not found. Run 'data' stage first."
        exit 1
    fi

    # Check SFT model exists
    SFT_MODEL="${OUTPUT_DIR}/sft"
    if [ ! -d "${SFT_MODEL}" ]; then
        log "WARNING: SFT model not found. Using base model."
        SFT_MODEL=""
    fi

    RL_OUTPUT="${OUTPUT_DIR}/rl"
    run_cmd "mkdir -p ${RL_OUTPUT}"
    run_cmd "mkdir -p ${WORKSPACE_DIR}"

    RL_PROMPTS=$(wc -l < "${DATASETS_DIR}/rl_prompts.jsonl")
    log "Training on ${RL_PROMPTS} RL prompts"

    # Check GPUs
    check_gpu

    # Run RL training
    log "Starting RL training for ${RL_STEPS} steps..."

    if [ -n "${SFT_MODEL}" ]; then
        MODEL_ARG="--model-path ${SFT_MODEL}"
    else
        MODEL_ARG=""
    fi

    run_cmd "torchrun --nproc_per_node=${NUM_GPUS} \\
        -m src.training.rl_trainer \\
        ${MODEL_ARG} \\
        --library-path ${LIBRARY_DIR} \\
        --prompts-path ${DATASETS_DIR}/rl_prompts.jsonl \\
        --output-dir ${RL_OUTPUT} \\
        --total-steps ${RL_STEPS} \\
        --num-gpus ${NUM_GPUS}"

    log "RL training complete!"
    log "Model saved to: ${RL_OUTPUT}"
}

# =============================================================================
# Main
# =============================================================================

main() {
    log "========================================="
    log "Deep Research Agent Training Pipeline"
    log "========================================="
    log "Config: ${CONFIG_FILE}"
    log "Stage: ${STAGE}"
    log "Dry run: ${DRY_RUN}"
    log ""

    case $STAGE in
        deps)
            install_dependencies
            ;;
        library)
            convert_library
            ;;
        data)
            generate_dataset
            ;;
        sft)
            train_sft
            ;;
        rl)
            train_rl
            ;;
        all)
            if [ "$SKIP_DEPS" = false ]; then
                install_dependencies
            fi
            convert_library
            generate_dataset
            train_sft
            train_rl
            ;;
        *)
            log "ERROR: Unknown stage: ${STAGE}"
            log "Valid stages: deps, library, data, sft, rl, all"
            exit 1
            ;;
    esac

    log ""
    log "========================================="
    log "Pipeline Complete!"
    log "========================================="
}

# Run main
main
