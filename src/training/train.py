"""
Main training entry point.

Unified CLI for SFT and RL training.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

console = Console()
app = typer.Typer()


@app.command("sft")
def train_sft_cmd(
    model_name: str = typer.Option(
        "ai-sage/GigaChat-Lightning-Instruct",
        help="HuggingFace model name"
    ),
    data_path: Path = typer.Option(
        "./data/datasets/sft_train.jsonl",
        help="Path to SFT training data"
    ),
    output_dir: Path = typer.Option(
        "./checkpoints/sft",
        help="Output directory for checkpoints"
    ),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Batch size per device"),
    learning_rate: float = typer.Option(2e-5, help="Learning rate"),
):
    """Run Supervised Fine-Tuning on tool usage demonstrations."""
    from .sft import SFTConfig, SFTTrainer

    console.print("[bold blue]Starting SFT Training[/bold blue]")
    console.print(f"  Model: {model_name}")
    console.print(f"  Data: {data_path}")
    console.print(f"  Output: {output_dir}")

    config = SFTConfig(
        model_name=model_name,
        data_path=str(data_path),
        output_dir=str(output_dir),
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    trainer = SFTTrainer(config)
    trainer.train()


@app.command("rl")
def train_rl_cmd(
    model_name: str = typer.Option(
        "ai-sage/GigaChat-Lightning-Instruct",
        help="Base model name"
    ),
    sft_checkpoint: Optional[Path] = typer.Option(
        None,
        help="Path to SFT checkpoint (if any)"
    ),
    data_path: Path = typer.Option(
        "./data/datasets/rl_prompts.jsonl",
        help="Path to RL prompts"
    ),
    output_dir: Path = typer.Option(
        "./checkpoints/rl",
        help="Output directory"
    ),
    library_path: Path = typer.Option(
        "/mnt/library",
        help="Path to paper library"
    ),
    epochs: int = typer.Option(10, help="Number of RL epochs"),
    batch_size: int = typer.Option(64, help="Rollout batch size"),
    num_envs: int = typer.Option(32, help="Number of parallel environments"),
):
    """Run Reinforcement Learning with tool-use environment."""
    from .rl import RLConfig, RLTrainer

    console.print("[bold green]Starting RL Training[/bold green]")
    console.print(f"  Model: {model_name}")
    console.print(f"  SFT checkpoint: {sft_checkpoint or 'None (starting from base)'}")
    console.print(f"  Data: {data_path}")
    console.print(f"  Library: {library_path}")

    config = RLConfig(
        model_name=model_name,
        sft_checkpoint=str(sft_checkpoint) if sft_checkpoint else None,
        data_path=str(data_path),
        output_dir=str(output_dir),
        library_path=str(library_path),
        num_epochs=epochs,
        batch_size=batch_size,
        num_envs=num_envs,
    )

    trainer = RLTrainer(config)
    trainer.train()


@app.command("curriculum")
def train_curriculum_cmd(
    model_name: str = typer.Option(
        "ai-sage/GigaChat-Lightning-Instruct",
        help="Base model name"
    ),
    data_path: Path = typer.Option(
        "./data/datasets/rl_prompts.jsonl",
        help="Path to full RL dataset"
    ),
    output_dir: Path = typer.Option(
        "./checkpoints/curriculum",
        help="Output directory"
    ),
    levels: int = typer.Option(5, help="Number of curriculum levels"),
):
    """Run curriculum learning (progressive difficulty)."""
    from .data import create_curriculum_splits
    from .rl import RLConfig, RLTrainer

    console.print("[bold yellow]Starting Curriculum Learning[/bold yellow]")

    # Create curriculum splits
    splits_dir = Path(output_dir) / "splits"
    split_paths = create_curriculum_splits(data_path, splits_dir, levels)

    console.print(f"Created {len(split_paths)} curriculum levels")

    checkpoint = None

    for level, split_path in enumerate(split_paths, 1):
        console.print(f"\n[bold]Level {level}/{levels}[/bold]")

        config = RLConfig(
            model_name=model_name,
            sft_checkpoint=checkpoint,
            data_path=str(split_path),
            output_dir=str(Path(output_dir) / f"level_{level}"),
            num_epochs=2,  # Fewer epochs per level
        )

        trainer = RLTrainer(config)
        trainer.train()

        # Use this checkpoint for next level
        checkpoint = str(Path(output_dir) / f"level_{level}" / "checkpoint_final")

    console.print("\n[bold green]Curriculum training complete![/bold green]")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
