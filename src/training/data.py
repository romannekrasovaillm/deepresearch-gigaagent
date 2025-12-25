"""
Data loading utilities for training.
"""

import json
from pathlib import Path
from typing import Iterator

from datasets import Dataset


def load_sft_dataset(data_path: Path) -> Dataset:
    """
    Load SFT dataset from JSONL file.

    Args:
        data_path: Path to sft_train.jsonl

    Returns:
        HuggingFace Dataset
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"SFT data not found: {data_path}")

    records = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)

            # Convert to conversation format
            turns = record.get("metadata", {}).get("turns", [])

            if turns:
                # Build messages from turns
                messages = [
                    {"role": "user", "content": record["question"]}
                ]

                for turn in turns:
                    # Agent thought + action
                    assistant_content = f"Thought: {turn.get('thought', '')}\n"
                    assistant_content += f"Action: {turn.get('action', '')}\n"
                    assistant_content += f"Action Input: {json.dumps(turn.get('action_input', {}))}"
                    messages.append({"role": "assistant", "content": assistant_content})

                    # Observation
                    messages.append({
                        "role": "user",
                        "content": f"Observation: {turn.get('observation', '')}"
                    })

                # Final answer
                messages.append({
                    "role": "assistant",
                    "content": f"<final_answer>\n{record['golden_answer']}\n</final_answer>"
                })

                records.append({
                    "messages": messages,
                    "task_id": record.get("task_id"),
                    "task_type": record.get("task_type"),
                })

    return Dataset.from_list(records)


def load_rl_prompts(data_path: Path) -> Iterator[dict]:
    """
    Load RL prompts from JSONL file.

    Args:
        data_path: Path to rl_prompts.jsonl

    Yields:
        Task dictionaries with question, golden_answer, required_papers
    """
    data_path = Path(data_path)

    if not data_path.exists():
        raise FileNotFoundError(f"RL prompts not found: {data_path}")

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            yield {
                "task_id": record.get("task_id"),
                "question": record.get("question"),
                "golden_answer": record.get("golden_answer"),
                "required_papers": record.get("required_papers", []),
                "task_type": record.get("task_type"),
                "metadata": record.get("metadata", {}),
            }


def format_tool_call(thought: str, tool: str, tool_input: dict) -> str:
    """Format a tool call for training."""
    return f"""Thought: {thought}
Action: {tool}
Action Input: {json.dumps(tool_input, ensure_ascii=False)}"""


def format_final_answer(answer: str) -> str:
    """Format final answer for training."""
    return f"""<final_answer>
{answer}
</final_answer>"""


def create_curriculum_splits(
    data_path: Path,
    output_dir: Path,
    levels: int = 5,
) -> list[Path]:
    """
    Create curriculum learning splits.

    Args:
        data_path: Path to full dataset
        output_dir: Output directory for splits
        levels: Number of difficulty levels

    Returns:
        List of paths to curriculum splits
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all tasks
    tasks_by_type = {
        "retrieval": [],
        "multihop": [],
        "computation": [],
        "synthesis": [],
    }

    with open(data_path, 'r') as f:
        for line in f:
            record = json.loads(line)
            task_type = record.get("task_type", "retrieval")
            if task_type in tasks_by_type:
                tasks_by_type[task_type].append(record)

    # Create curriculum levels
    curriculum = {
        1: ["retrieval"],  # Level 1: Simple retrieval
        2: ["retrieval", "multihop"],  # Level 2: Add multi-hop
        3: ["retrieval", "multihop", "computation"],  # Level 3: Add computation
        4: ["retrieval", "multihop", "computation"],  # Level 4: Same but harder
        5: ["retrieval", "multihop", "computation", "synthesis"],  # Level 5: All
    }

    split_paths = []

    for level in range(1, levels + 1):
        level_tasks = []
        for task_type in curriculum.get(level, []):
            level_tasks.extend(tasks_by_type.get(task_type, []))

        # Sort by difficulty within each type
        level_tasks.sort(key=lambda x: len(x.get("required_papers", [])))

        # For higher levels, include harder examples
        if level >= 3:
            # Take latter half (harder) for higher levels
            mid = len(level_tasks) // 2
            level_tasks = level_tasks[mid:]

        split_path = output_dir / f"curriculum_level_{level}.jsonl"

        with open(split_path, 'w') as f:
            for task in level_tasks:
                f.write(json.dumps(task, ensure_ascii=False) + '\n')

        split_paths.append(split_path)

    return split_paths
