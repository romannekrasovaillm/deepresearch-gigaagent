# Deep Research GigaAgent - Makefile
# Convenience commands for development and training

.PHONY: help install setup ingest synthetic sft rl all test clean

# Default target
help:
	@echo "Deep Research GigaAgent - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install     Install dependencies"
	@echo "  make setup       Full setup (install + directories)"
	@echo ""
	@echo "Pipeline:"
	@echo "  make ingest      Convert DOCX papers to library"
	@echo "  make synthetic   Generate synthetic training data"
	@echo "  make sft         Run supervised fine-tuning"
	@echo "  make rl          Run reinforcement learning"
	@echo "  make all         Run complete pipeline"
	@echo ""
	@echo "Development:"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linters"
	@echo "  make agent       Run interactive agent"
	@echo "  make clean       Clean generated files"

# Installation
install:
	pip install -e .
	pip install -e ".[dev]"

setup:
	./scripts/run_pipeline.sh --stage setup

# Pipeline stages
ingest:
	./scripts/run_pipeline.sh --stage ingest

synthetic:
	./scripts/run_pipeline.sh --stage synthetic

sft:
	./scripts/run_pipeline.sh --stage sft

rl:
	./scripts/run_pipeline.sh --stage rl

all:
	./scripts/run_pipeline.sh --stage all

# Distributed training (4 GPUs)
sft-distributed:
	torchrun --nproc_per_node=4 -m src.training.train sft

rl-distributed:
	torchrun --nproc_per_node=4 -m src.training.train rl

# Development
test:
	pytest tests/ -v

lint:
	ruff check src/
	black --check src/

format:
	black src/
	ruff check --fix src/

agent:
	python -m src.env.run_agent --interactive

# Utilities
add-paper:
	@echo "Usage: python -m src.library.add_paper add <file.docx> /mnt/library"

verify-library:
	python -m src.library.convert verify /mnt/library

# Clean
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.egg-info
	rm -rf build dist
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-all: clean
	rm -rf venv
	rm -rf external
	rm -rf checkpoints
	rm -rf data/datasets
	rm -rf /workspace/*
