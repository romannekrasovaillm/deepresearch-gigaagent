#!/bin/bash
#===============================================================================
# Deep Research Agent - Full Training Pipeline
#===============================================================================
#
# This script runs the complete training pipeline:
# 1. Setup: Install dependencies, clone prime-rl & verifiers
# 2. Ingestion: Convert DOCX papers to structured Markdown library
# 3. Synthesis: Generate synthetic training data using LLM
# 4. SFT: Supervised fine-tuning on tool usage demonstrations
# 5. RL: Reinforcement learning with multi-objective rewards
#
# Usage:
#   ./scripts/run_pipeline.sh [options]
#
# Options:
#   --stage STAGE      Start from specific stage (setup|ingest|synthetic|sft|rl|all)
#   --config CONFIG    Path to config file (default: configs/base.yaml)
#   --gpus N           Number of GPUs to use (default: 4)
#   --dry-run          Print commands without executing
#   --help             Show this help message
#
# Example:
#   ./scripts/run_pipeline.sh --stage all --gpus 4
#   ./scripts/run_pipeline.sh --stage rl --config configs/h200_4gpu.yaml
#
#===============================================================================

set -euo pipefail

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

# Defaults
STAGE="all"
CONFIG="configs/base.yaml"
NUM_GPUS=4
DRY_RUN=false
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Paths
DOCX_INPUT="${PROJECT_ROOT}/data/docx_input"
LIBRARY_PATH="/mnt/library"
DATASETS_PATH="${PROJECT_ROOT}/data/datasets"
CHECKPOINTS_PATH="${PROJECT_ROOT}/checkpoints"
WORKSPACE_PATH="/workspace"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

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

run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY-RUN]${NC} $*"
    else
        log_info "Running: $*"
        "$@"
    fi
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_warning "nvidia-smi not found. GPU training may not work."
        return 1
    fi

    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    log_info "Found $GPU_COUNT GPU(s)"

    if [ "$GPU_COUNT" -lt "$NUM_GPUS" ]; then
        log_warning "Requested $NUM_GPUS GPUs but only $GPU_COUNT available"
        NUM_GPUS=$GPU_COUNT
    fi
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --stage)
                STAGE="$2"
                shift 2
                ;;
            --config)
                CONFIG="$2"
                shift 2
                ;;
            --gpus)
                NUM_GPUS="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                head -30 "$0" | tail -25
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                ;;
        esac
    done
}

#-------------------------------------------------------------------------------
# Stage 0: Setup
#-------------------------------------------------------------------------------

stage_setup() {
    log_info "========================================"
    log_info "Stage 0: Setup & Dependencies"
    log_info "========================================"

    cd "$PROJECT_ROOT"

    # Check Python version
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"

    # Install system dependencies
    log_info "Installing system dependencies..."
    if command -v apt-get &> /dev/null; then
        run_cmd sudo apt-get update
        run_cmd sudo apt-get install -y pandoc ripgrep
    elif command -v yum &> /dev/null; then
        run_cmd sudo yum install -y pandoc ripgrep
    fi

    # Create virtual environment if not exists
    if [ ! -d "venv" ]; then
        log_info "Creating virtual environment..."
        run_cmd python3 -m venv venv
    fi

    # Activate venv
    source venv/bin/activate

    # Install Python dependencies
    log_info "Installing Python dependencies..."
    run_cmd pip install --upgrade pip
    run_cmd pip install -e .

    # Clone and install prime-rl
    if [ ! -d "external/prime-rl" ]; then
        log_info "Cloning prime-rl..."
        mkdir -p external
        run_cmd git clone https://github.com/PrimeIntellect-ai/prime-rl.git external/prime-rl
        run_cmd pip install -e external/prime-rl
    fi

    # Clone and install verifiers
    if [ ! -d "external/verifiers" ]; then
        log_info "Cloning verifiers..."
        run_cmd git clone https://github.com/PrimeIntellect-ai/verifiers.git external/verifiers
        run_cmd pip install -e external/verifiers
    fi

    # Create directories
    log_info "Creating directory structure..."
    run_cmd mkdir -p "$DOCX_INPUT"
    run_cmd mkdir -p "$LIBRARY_PATH"
    run_cmd mkdir -p "$DATASETS_PATH"
    run_cmd mkdir -p "$CHECKPOINTS_PATH"/{sft,rl}
    run_cmd mkdir -p "$WORKSPACE_PATH"

    log_success "Setup complete!"
}

#-------------------------------------------------------------------------------
# Stage 1: Library Ingestion
#-------------------------------------------------------------------------------

stage_ingest() {
    log_info "========================================"
    log_info "Stage 1: Library Ingestion (DOCX → MD)"
    log_info "========================================"

    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # Check for input files
    DOCX_COUNT=$(find "$DOCX_INPUT" -name "*.docx" 2>/dev/null | wc -l)

    if [ "$DOCX_COUNT" -eq 0 ]; then
        log_warning "No DOCX files found in $DOCX_INPUT"
        log_info "Please add your paper files to: $DOCX_INPUT"
        log_info "Supported formats: .docx"
        log_info ""
        log_info "Creating sample structure for testing..."

        # Create sample paper for testing
        mkdir -p "$LIBRARY_PATH/by_id/paper_0001/sections"
        echo '{"paper_id": "paper_0001", "title": "Sample Paper", "author": "Test"}' > "$LIBRARY_PATH/by_id/paper_0001/metadata.json"
        echo "# Sample Paper\n\nThis is a sample paper for testing the pipeline." > "$LIBRARY_PATH/by_id/paper_0001/full_text.md"
        echo "This is a sample abstract." > "$LIBRARY_PATH/by_id/paper_0001/abstract.txt"
        echo '[{"paper_id": "paper_0001", "title": "Sample Paper"}]' > "$LIBRARY_PATH/index.json"

        log_success "Created sample library structure"
        return 0
    fi

    log_info "Found $DOCX_COUNT DOCX files to process"

    # Run conversion
    run_cmd python -m src.library.convert convert \
        "$DOCX_INPUT" \
        "$LIBRARY_PATH" \
        --workers 4 \
        --pattern "*.docx"

    # Verify conversion
    run_cmd python -m src.library.convert verify "$LIBRARY_PATH"

    log_success "Library ingestion complete!"
    log_info "Library location: $LIBRARY_PATH"
}

#-------------------------------------------------------------------------------
# Stage 2: Synthetic Data Generation
#-------------------------------------------------------------------------------

stage_synthetic() {
    log_info "========================================"
    log_info "Stage 2: Synthetic Data Generation"
    log_info "========================================"

    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # Check for API key
    if [ -z "${OPENAI_API_KEY:-}" ] && [ -z "${ANTHROPIC_API_KEY:-}" ]; then
        log_warning "No API key found (OPENAI_API_KEY or ANTHROPIC_API_KEY)"
        log_info "Creating placeholder dataset for testing..."

        # Create minimal test dataset
        mkdir -p "$DATASETS_PATH"

        # SFT dataset
        cat > "$DATASETS_PATH/sft_train.jsonl" << 'EOF'
{"task_id": "sft_00001", "task_type": "sft_demo", "question": "What is the main contribution of paper_0001?", "golden_answer": "The paper introduces a novel approach.", "required_papers": ["paper_0001"], "metadata": {"turns": [{"thought": "I need to search for the paper", "action": "grep_search", "action_input": {"pattern": "contribution"}, "observation": "Found: novel approach"}]}}
EOF

        # RL dataset
        cat > "$DATASETS_PATH/rl_prompts.jsonl" << 'EOF'
{"task_id": "retrieval_00001", "task_type": "retrieval", "question": "What method does paper_0001 use?", "golden_answer": "The paper uses a sample method.", "required_papers": ["paper_0001"]}
{"task_id": "multihop_00001", "task_type": "multihop", "question": "Compare the approaches in papers.", "golden_answer": "The papers use different approaches.", "required_papers": ["paper_0001"]}
EOF

        log_success "Created placeholder datasets"
        return 0
    fi

    # Determine provider
    if [ -n "${OPENAI_API_KEY:-}" ]; then
        PROVIDER="openai"
    elif [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        PROVIDER="anthropic"
    fi

    log_info "Using provider: $PROVIDER"

    # Run generation
    run_cmd python -m src.synthetic.generate generate \
        "$LIBRARY_PATH" \
        "$DATASETS_PATH" \
        --provider "$PROVIDER" \
        --total 10000

    log_success "Synthetic data generation complete!"
    log_info "SFT data: $DATASETS_PATH/sft_train.jsonl"
    log_info "RL data: $DATASETS_PATH/rl_prompts.jsonl"
}

#-------------------------------------------------------------------------------
# Stage 3: SFT Training
#-------------------------------------------------------------------------------

stage_sft() {
    log_info "========================================"
    log_info "Stage 3: Supervised Fine-Tuning (SFT)"
    log_info "========================================"

    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # Check GPU
    check_gpu

    # Check for SFT data
    if [ ! -f "$DATASETS_PATH/sft_train.jsonl" ]; then
        log_error "SFT data not found: $DATASETS_PATH/sft_train.jsonl"
    fi

    SFT_SAMPLES=$(wc -l < "$DATASETS_PATH/sft_train.jsonl")
    log_info "SFT samples: $SFT_SAMPLES"

    # Run SFT training
    if [ "$NUM_GPUS" -gt 1 ]; then
        log_info "Running distributed SFT on $NUM_GPUS GPUs..."
        run_cmd torchrun --nproc_per_node="$NUM_GPUS" \
            -m src.training.train sft \
            --data-path "$DATASETS_PATH/sft_train.jsonl" \
            --output-dir "$CHECKPOINTS_PATH/sft" \
            --epochs 3 \
            --batch-size 4
    else
        run_cmd python -m src.training.train sft \
            --data-path "$DATASETS_PATH/sft_train.jsonl" \
            --output-dir "$CHECKPOINTS_PATH/sft" \
            --epochs 3 \
            --batch-size 4
    fi

    log_success "SFT training complete!"
    log_info "Checkpoint: $CHECKPOINTS_PATH/sft"
}

#-------------------------------------------------------------------------------
# Stage 4: RL Training
#-------------------------------------------------------------------------------

stage_rl() {
    log_info "========================================"
    log_info "Stage 4: Reinforcement Learning (RL)"
    log_info "========================================"

    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # Check GPU
    check_gpu

    # Check for RL data
    if [ ! -f "$DATASETS_PATH/rl_prompts.jsonl" ]; then
        log_error "RL data not found: $DATASETS_PATH/rl_prompts.jsonl"
    fi

    RL_SAMPLES=$(wc -l < "$DATASETS_PATH/rl_prompts.jsonl")
    log_info "RL samples: $RL_SAMPLES"

    # Check for SFT checkpoint
    SFT_CHECKPOINT=""
    if [ -d "$CHECKPOINTS_PATH/sft" ]; then
        SFT_CHECKPOINT="--sft-checkpoint $CHECKPOINTS_PATH/sft"
        log_info "Starting from SFT checkpoint"
    else
        log_warning "No SFT checkpoint found, starting from base model"
    fi

    # Run RL training
    if [ "$NUM_GPUS" -gt 1 ]; then
        log_info "Running distributed RL on $NUM_GPUS GPUs..."
        run_cmd torchrun --nproc_per_node="$NUM_GPUS" \
            -m src.training.train rl \
            --data-path "$DATASETS_PATH/rl_prompts.jsonl" \
            --output-dir "$CHECKPOINTS_PATH/rl" \
            --library-path "$LIBRARY_PATH" \
            --epochs 10 \
            --num-envs 32 \
            $SFT_CHECKPOINT
    else
        run_cmd python -m src.training.train rl \
            --data-path "$DATASETS_PATH/rl_prompts.jsonl" \
            --output-dir "$CHECKPOINTS_PATH/rl" \
            --library-path "$LIBRARY_PATH" \
            --epochs 10 \
            --num-envs 8 \
            $SFT_CHECKPOINT
    fi

    log_success "RL training complete!"
    log_info "Final model: $CHECKPOINTS_PATH/rl/checkpoint_final"
}

#-------------------------------------------------------------------------------
# Stage 5: Evaluation (Optional)
#-------------------------------------------------------------------------------

stage_eval() {
    log_info "========================================"
    log_info "Stage 5: Evaluation"
    log_info "========================================"

    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # TODO: Add evaluation script
    log_warning "Evaluation not implemented yet"
    log_info "Manual testing: python -m src.env.run_agent"
}

#-------------------------------------------------------------------------------
# Main
#-------------------------------------------------------------------------------

main() {
    parse_args "$@"

    echo ""
    echo "=============================================="
    echo " Deep Research Agent Training Pipeline"
    echo "=============================================="
    echo " Stage:  $STAGE"
    echo " Config: $CONFIG"
    echo " GPUs:   $NUM_GPUS"
    echo " Dry-run: $DRY_RUN"
    echo "=============================================="
    echo ""

    case $STAGE in
        setup)
            stage_setup
            ;;
        ingest)
            stage_ingest
            ;;
        synthetic)
            stage_synthetic
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
            stage_setup
            stage_ingest
            stage_synthetic
            stage_sft
            stage_rl
            stage_eval
            ;;
        *)
            log_error "Unknown stage: $STAGE"
            ;;
    esac

    echo ""
    log_success "Pipeline complete!"
    echo ""
    echo "Next steps:"
    echo "  1. Test the model: python -m src.env.run_agent"
    echo "  2. Add more papers: python -m src.library.add_paper add <file.docx> $LIBRARY_PATH"
    echo "  3. Fine-tune further: ./scripts/run_pipeline.sh --stage rl"
    echo ""
}

main "$@"
