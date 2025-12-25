#!/usr/bin/env python3
"""CLI script for generating synthetic training data."""

import sys
from pathlib import Path

import typer
from rich.console import Console

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data import generate_dataset, SyntheticGenerator, TaskType

app = typer.Typer(help="Generate synthetic training data")
console = Console()


@app.command("generate")
def generate(
    library_path: Path = typer.Argument(..., help="Path to paper library"),
    output_path: Path = typer.Argument(..., help="Output directory for datasets"),
    n_samples: int = typer.Option(1000, "--samples", "-n", help="Number of samples"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Generator model"),
    split: bool = typer.Option(True, "--split/--no-split", help="Split SFT/RL"),
):
    """Generate synthetic training dataset."""
    console.print(f"[bold]Generating {n_samples} samples[/bold]")
    console.print(f"Library: {library_path}")
    console.print(f"Model: {model}")

    generate_dataset(
        library_path=library_path,
        output_path=output_path,
        n_samples=n_samples,
        generator_model=model,
        split_sft_rl=split,
    )


@app.command("preview")
def preview(
    library_path: Path = typer.Argument(..., help="Path to paper library"),
    task_type: str = typer.Option("retrieval", "--type", "-t", help="Task type"),
    count: int = typer.Option(3, "--count", "-c", help="Number to preview"),
):
    """Preview generated samples without saving."""
    generator = SyntheticGenerator(library_path=library_path)

    task_map = {
        "retrieval": TaskType.RETRIEVAL,
        "multihop": TaskType.MULTIHOP,
        "computation": TaskType.COMPUTATION,
        "synthesis": TaskType.SYNTHESIS,
    }

    if task_type not in task_map:
        console.print(f"[red]Unknown task type: {task_type}[/red]")
        console.print(f"Valid types: {list(task_map.keys())}")
        raise typer.Exit(1)

    console.print(f"[bold]Generating {count} {task_type} samples...[/bold]\n")

    for i, task in enumerate(generator.generate_dataset(count)):
        if task.task_type == task_map[task_type]:
            console.print(f"[cyan]--- Sample {i + 1} ---[/cyan]")
            console.print(f"[bold]Q:[/bold] {task.question}")
            console.print(f"[bold]A:[/bold] {task.golden_answer[:200]}...")
            console.print(f"[dim]Papers: {task.required_papers}[/dim]")
            console.print()


@app.command("stats")
def stats(
    data_path: Path = typer.Argument(..., help="Path to dataset JSONL"),
):
    """Show dataset statistics."""
    import json
    from collections import Counter

    if not data_path.exists():
        console.print(f"[red]File not found: {data_path}[/red]")
        raise typer.Exit(1)

    type_counts = Counter()
    difficulty_counts = Counter()
    total = 0

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            type_counts[data.get("type", "unknown")] += 1
            difficulty_counts[data.get("difficulty", "unknown")] += 1
            total += 1

    console.print(f"[bold]Dataset: {data_path}[/bold]")
    console.print(f"Total samples: {total}")
    console.print("\nBy type:")
    for t, c in type_counts.most_common():
        console.print(f"  {t}: {c} ({c / total * 100:.1f}%)")
    console.print("\nBy difficulty:")
    for d, c in difficulty_counts.most_common():
        console.print(f"  {d}: {c} ({c / total * 100:.1f}%)")


def main():
    app()


if __name__ == "__main__":
    main()
