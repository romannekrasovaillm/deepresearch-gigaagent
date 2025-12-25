# Deep Research GigaAgent

**Scientific Research Agent with Code Reasoning** - Full RL Training Pipeline for 4×H200

Train a Deep Research agent that can navigate, analyze, and synthesize information from a local library of scientific papers using bash tools and Python code interpreter.

## Features

- **Bash-based Navigation**: Deterministic grep/sed/find tools for reliable paper search
- **Code Reasoning**: Python interpreter (pandas, numpy, scipy, matplotlib) for analysis
- **Multi-objective Rewards**: MOA-style training with factual, process, citation, and code rewards
- **Full RL on H200**: No LoRA needed - full model training with FSDP on 4×H200 (141GB each)
- **Curriculum Learning**: Progressive difficulty from simple retrieval to full synthesis

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Deep Research Agent                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ Bash Tools  │  │   Code      │  │     Scratchpad          │  │
│  │             │  │ Interpreter │  │                         │  │
│  │ grep_search │  │             │  │ add_to_notes()          │  │
│  │ read_file   │  │ pandas      │  │ save_data()             │  │
│  │ find_files  │  │ numpy       │  │ save_figure()           │  │
│  │ verify_quote│  │ scipy       │  │                         │  │
│  │ list_papers │  │ matplotlib  │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│                    Paper Library (/mnt/library)                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ index.json │ by_id/paper_001/{metadata,full_text,sections}│  │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <repo-url>
cd deepresearch-gigaagent

# Run full pipeline
./scripts/run_pipeline.sh --stage all --gpus 4
```

### 2. Add Your Papers

```bash
# Add DOCX files to input directory
cp /path/to/papers/*.docx ./data/docx_input/

# Or add individual papers
python -m src.library.add_paper add paper.docx /mnt/library
```

### 3. Run Training

```bash
# Full pipeline (setup → ingest → synthetic → SFT → RL)
./scripts/run_pipeline.sh --stage all

# Or run individual stages
./scripts/run_pipeline.sh --stage ingest    # Convert DOCX to library
./scripts/run_pipeline.sh --stage synthetic # Generate training data
./scripts/run_pipeline.sh --stage sft       # Supervised fine-tuning
./scripts/run_pipeline.sh --stage rl        # Reinforcement learning
```

### 4. Test the Agent

```bash
python -m src.env.run_agent --interactive
```

## Project Structure

```
deepresearch-gigaagent/
├── src/
│   ├── library/           # DOCX → Markdown conversion
│   │   ├── convert.py     # Main conversion pipeline
│   │   └── add_paper.py   # Incremental paper addition
│   ├── tools/             # Agent tools
│   │   ├── bash_tools.py  # grep_search, read_file, etc.
│   │   ├── code_interpreter.py  # Python sandbox
│   │   ├── scratchpad.py  # Notes and data persistence
│   │   └── registry.py    # Tool registry for verifiers
│   ├── synthetic/         # Training data generation
│   │   ├── prompts.py     # LLM prompts for generation
│   │   └── generate.py    # Dataset generator
│   ├── env/               # RL environment
│   │   ├── tool_env.py    # Research environment
│   │   ├── rewards.py     # Multi-objective rewards
│   │   └── judge.py       # LLM-as-Judge evaluator
│   └── training/          # Training pipelines
│       ├── sft.py         # Supervised fine-tuning
│       ├── rl.py          # Reinforcement learning
│       └── data.py        # Data loading utilities
├── scripts/
│   └── run_pipeline.sh    # Main orchestration script
├── configs/
│   ├── base.yaml          # Base configuration
│   └── h200_4gpu.yaml     # H200-optimized config
├── data/
│   ├── docx_input/        # Input DOCX papers
│   ├── library/           # Converted library
│   └── datasets/          # Generated training data
└── checkpoints/           # Model checkpoints
```

## Configuration

### Base Configuration (`configs/base.yaml`)

```yaml
model:
  name: "ai-sage/GigaChat-Lightning-Instruct"
  precision: "bf16"

rewards:
  weights:
    r_fact: 0.4      # Factual accuracy
    r_process: 0.2   # Search efficiency
    r_citation: 0.2  # Citation accuracy
    r_code: 0.2      # Code reasoning quality

rl:
  algorithm: "ppo"
  epochs: 10
  batch_size: 64
  num_envs: 32
  max_turns: 15
```

### H200 Optimization (`configs/h200_4gpu.yaml`)

Memory calculation for GigaChat Lightning (10B params):
- Model weights (BF16): ~20 GB
- Optimizer states (AdamW): ~40-60 GB
- Gradients: ~20 GB
- **Total: ~80-100 GB** (fits in single H200 with room for activations)
- With FSDP on 4 cards: ~25-30 GB per card

## Tools Reference

### Bash Tools (Deterministic)

| Tool | Description | Example |
|------|-------------|---------|
| `grep_search(pattern, path)` | Search for text in papers | `grep_search("PPO-Clip", "by_id/paper_042")` |
| `read_file_chunk(path, start, num_lines)` | Read file portion | `read_file_chunk("paper_042/methods.txt", 10, 30)` |
| `find_files(pattern)` | Find files by name | `find_files("*reward*")` |
| `verify_quote(path, snippet)` | Verify citation exists | `verify_quote("paper_042/methods.txt", "clipped objective")` |
| `list_papers(topic, keyword)` | List papers from index | `list_papers(keyword="reward")` |
| `read_abstract(paper_id)` | Read paper abstract | `read_abstract("paper_042")` |

### Code Interpreter (Python Sandbox)

```python
# Available libraries: pandas, numpy, scipy.stats, matplotlib, sympy

# Example: Compare metrics
execute_python('''
import pandas as pd
results = pd.DataFrame({
    'paper': ['A', 'B', 'C'],
    'accuracy': [0.89, 0.92, 0.87]
})
print(f"Mean: {results.accuracy.mean():.2%}")
''')

# Example: Statistical test
execute_python('''
from scipy import stats
z, p = stats.proportions_ztest([89, 92], [100, 100])
print(f"Z={z:.2f}, p={p:.4f}")
''')

# Example: Visualization
execute_python('''
import matplotlib.pyplot as plt
years = [2020, 2021, 2022, 2023]
scores = [0.72, 0.81, 0.89, 0.94]
plt.plot(years, scores, marker='o')
plt.savefig('/workspace/progress.png')
''')
```

### Scratchpad (Memory)

| Tool | Description |
|------|-------------|
| `add_to_notes(text, section)` | Save findings |
| `read_notes()` | Recall saved notes |
| `save_data(key, value)` | Store structured data |
| `load_data(key)` | Retrieve stored data |
| `save_figure(path, name)` | Save generated figures |

## Reward System

Multi-objective rewards (MOA-style):

| Reward | Weight | Description |
|--------|--------|-------------|
| R_fact | 0.4 | Factual accuracy vs. golden answer (LLM-Judge) |
| R_process | 0.2 | Search efficiency (fewer steps, no repeats) |
| R_citation | 0.2 | Citation accuracy (verified quotes) |
| R_code | 0.2 | Code reasoning quality (correct execution, relevance) |

## Training Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│   Ingestion  │ => │  Synthetic   │ => │     SFT     │ => │     RL     │
│  DOCX → MD   │    │  Generation  │    │ Tool Format │    │  Rewards   │
└──────────────┘    └──────────────┘    └─────────────┘    └────────────┘
     pandoc           GPT-4/Claude       2-5k demos        10k+ prompts
```

### Curriculum Learning

| Level | Tasks | Skills |
|-------|-------|--------|
| 1 | Simple retrieval | grep, find |
| 2 | Multi-hop reasoning | compare papers |
| 3 | Computation | execute_python |
| 4 | Visualization | matplotlib plots |
| 5 | Full synthesis | all tools |

## API Keys

For synthetic data generation, set one of:

```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
# or
export DEEPSEEK_API_KEY="..."
```

## Requirements

- Python 3.10+
- CUDA 12.0+ (for GPU training)
- 4× H200 GPUs (141GB each) - or adjust batch sizes for other hardware
- `pandoc` for DOCX conversion
- `ripgrep` (rg) for fast search

## License

MIT

## Citation

```bibtex
@software{deepresearch_gigaagent,
  title = {Deep Research GigaAgent: Scientific Research Agent with Code Reasoning},
  year = {2024},
  url = {https://github.com/...}
}
```
