# Deep Research Agent

**Scientific Synthesis with Code Reasoning on Local Paper Library**

Train an agent to perform deep research on a local library of scientific papers using Full RL on H200 GPUs.

## Features

- **Bash + Code Interpreter**: Hybrid tool environment combining deterministic file navigation with Python analytical reasoning
- **MOA-style Rewards**: Multi-objective alignment with independent reward functions (R_fact, R_process, R_citation, R_code)
- **Curriculum Learning**: Progressive training from simple retrieval to full synthesis
- **Full RL Training**: No LoRA - full parameter updates with FSDP on 4×H200

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Deep Research Agent                       │
├─────────────────────────────────────────────────────────────┤
│  Tools                                                       │
│  ├── Bash: grep_search, read_file_chunk, find_files         │
│  ├── Code: execute_python (pandas, numpy, matplotlib)       │
│  └── Notes: add_to_notes, read_notes, save_figure           │
├─────────────────────────────────────────────────────────────┤
│  Rewards (MOA-style)                                         │
│  ├── R_fact (0.4): LLM-as-Judge factual accuracy            │
│  ├── R_process (0.2): Efficiency, no repeats                │
│  ├── R_citation (0.2): Citation verification                │
│  └── R_code (0.2): Successful code execution                │
├─────────────────────────────────────────────────────────────┤
│  Training                                                    │
│  ├── SFT: Tool usage format (2-5k samples)                  │
│  └── RL: PPO/GRPO with prime-rl (10k+ prompts)              │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/your-org/deepresearch-gigaagent.git
cd deepresearch-gigaagent

# Install dependencies
pip install -e .

# Install pandoc for DOCX conversion
sudo apt install pandoc  # Linux
brew install pandoc      # macOS
```

### 2. Prepare Library

Convert your DOCX papers to structured Markdown:

```bash
# Put DOCX files in data/raw/
./scripts/convert_library.sh ./data/raw /mnt/library
```

### 3. Generate Training Data

```bash
# Set API key for data generation
export OPENAI_API_KEY="sk-..."

# Generate synthetic dataset
./scripts/generate_data.sh 10000 gpt-4o-mini
```

### 4. Train

```bash
# Run full pipeline
./scripts/run_pipeline.sh

# Or run steps separately:
./scripts/train_sft.sh GigaChat-Lightning-Instruct ./data/synthetic/sft_train.jsonl
./scripts/train_rl.sh ./outputs/checkpoints/sft-final ./data/synthetic/rl_prompts.jsonl
```

### 5. Run Agent

```bash
# Interactive mode
python -m src.tools.agent run ./outputs/checkpoints/sft-final

# Single question
python -m src.tools.agent run ./outputs/checkpoints/sft-final \
    --question "Compare reward modeling approaches in the library"
```

## Project Structure

```
deepresearch-gigaagent/
├── src/
│   ├── library/          # DOCX → Markdown conversion
│   │   └── convert.py
│   ├── tools/            # Agent tools
│   │   ├── bash_tools.py      # grep, read, find, verify
│   │   ├── code_interpreter.py # Python sandbox
│   │   ├── scratchpad.py      # Notes & figures
│   │   ├── tool_env.py        # ToolEnv for RL
│   │   └── agent.py           # Interactive agent
│   ├── rewards/          # MOA reward system
│   │   ├── reward_functions.py
│   │   └── judge.py           # LLM-as-Judge
│   ├── generation/       # Synthetic data
│   │   ├── prompts.py
│   │   └── synthetic.py
│   └── training/         # SFT & RL
│       ├── train.py
│       └── rollout.py
├── configs/
│   └── training.yaml     # Full training config
├── scripts/
│   ├── run_pipeline.sh   # Full pipeline
│   ├── convert_library.sh
│   ├── generate_data.sh
│   ├── train_sft.sh
│   └── train_rl.sh
└── data/
    ├── raw/              # Input DOCX files
    ├── processed/        # Intermediate data
    └── synthetic/        # Generated datasets
```

## Configuration

Edit `configs/training.yaml` for full control:

```yaml
model:
  name: GigaChat-Lightning-Instruct
  precision: bf16

sft:
  epochs: 3
  batch_size: 4
  learning_rate: 2.0e-5

rl:
  algorithm: ppo
  batch_size: 64
  kl_coef: 0.05

reward:
  weights:
    fact: 0.4
    process: 0.2
    citation: 0.2
    code: 0.2
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | 1× A100 40GB | 4× H200 141GB |
| RAM | 64 GB | 256 GB |
| Storage | 100 GB | 500 GB SSD |

### VRAM Estimation (10B model, BF16)

- Model weights: ~20 GB
- Optimizer states (AdamW): ~40-60 GB
- Gradients: ~20 GB
- **Total**: ~80-100 GB (fits on H200)

With FSDP on 4× H200: ~25-30 GB per card

## Tool Reference

### Bash Tools

| Tool | Description |
|------|-------------|
| `grep_search` | Search pattern in library (uses ripgrep) |
| `read_file_chunk` | Stream file contents |
| `find_files` | Find files by name pattern |
| `verify_quote` | Verify citation exists (anti-hallucination) |
| `list_papers` | List papers by topic |
| `get_paper_metadata` | Get paper metadata |
| `get_paper_abstract` | Get paper abstract |
| `get_paper_section` | Get specific section |

### Code Interpreter

```python
# Available in execute_python:
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
```

### Scratchpad

| Tool | Description |
|------|-------------|
| `add_to_notes` | Save findings |
| `read_notes` | Review notes |
| `save_figure` | Register generated figure |

## Reward System

### R_fact (40%)
LLM-as-Judge compares agent answer to golden answer.

### R_process (20%)
- Penalty for >10 turns
- Penalty for repeated actions
- Penalty for failed tool calls

### R_citation (20%)
- Verifies cited papers were accessed
- Checks quote accuracy

### R_code (20%)
- Rewards successful Python execution
- Bonus for producing outputs (figures, tables)

## Curriculum Levels

1. **Retrieval**: Find specific facts
2. **Extraction**: Extract complex information
3. **Computation**: Calculate metrics with code
4. **Multi-hop**: Compare multiple papers
5. **Visualization**: Generate charts
6. **Synthesis**: Full topic review

## Integration with prime-rl

This project generates configuration for [PrimeIntellect-ai/prime-rl](https://github.com/PrimeIntellect-ai/prime-rl):

```bash
# Generate config
./scripts/train_rl.sh ./outputs/checkpoints/sft-final ./data/synthetic/rl_prompts.jsonl

# Run with prime-rl
python -m prime_rl.train --config ./outputs/prime_rl_config.yaml
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | For data generation & judge | - |
| `DEEPSEEK_API_KEY` | Alternative API | - |
| `LIBRARY_PATH` | Paper library path | `/mnt/library` |
| `WORKSPACE_PATH` | Agent workspace | `/workspace` |
| `GEN_MODEL` | Generation model | `gpt-4o-mini` |
| `JUDGE_MODEL` | Judge model | `gpt-4o-mini` |

## License

MIT

## Citation

```bibtex
@software{deepresearch_agent,
  title = {Deep Research Agent},
  year = {2024},
  description = {Scientific Synthesis with Code Reasoning}
}
```
