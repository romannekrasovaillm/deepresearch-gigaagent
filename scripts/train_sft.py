#!/usr/bin/env python3
"""CLI script for SFT training."""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training import SFTTrainer, SFTConfig

app = typer.Typer(help="SFT training for tool usage")
console = Console()


@app.command("train")
def train(
    train_data: Path = typer.Argument(..., help="Training data JSONL"),
    model_name: str = typer.Option(
        "GigaChat-Lightning-Instruct", "--model", "-m", help="Model name/path"
    ),
    output_dir: Path = typer.Option(
        "./checkpoints/sft", "--output", "-o", help="Output directory"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of epochs"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Batch size"),
    learning_rate: float = typer.Option(2e-5, "--lr", help="Learning rate"),
    max_length: int = typer.Option(4096, "--max-length", help="Max sequence length"),
    use_lora: bool = typer.Option(False, "--lora/--no-lora", help="Use LoRA"),
    eval_data: Optional[Path] = typer.Option(None, "--eval", help="Eval data JSONL"),
):
    """Run SFT training on tool usage demonstrations."""
    if not train_data.exists():
        console.print(f"[red]Training data not found: {train_data}[/red]")
        raise typer.Exit(1)

    config = SFTConfig(
        model_name=model_name,
        output_dir=str(output_dir),
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        max_length=max_length,
        use_lora=use_lora,
    )

    console.print("[bold]SFT Training Configuration[/bold]")
    console.print(f"Model: {config.model_name}")
    console.print(f"Train data: {train_data}")
    console.print(f"Output: {config.output_dir}")
    console.print(f"Epochs: {config.num_epochs}")
    console.print(f"Batch size: {config.batch_size}")
    console.print(f"Learning rate: {config.learning_rate}")
    console.print(f"LoRA: {config.use_lora}")
    console.print()

    trainer = SFTTrainer(config)
    trainer.train(train_data, eval_data)

    console.print(f"[green]Training complete! Model saved to {output_dir}[/green]")


@app.command("export")
def export(
    checkpoint: Path = typer.Argument(..., help="Checkpoint path"),
    output: Path = typer.Argument(..., help="Output path for merged model"),
):
    """Export LoRA checkpoint to full model."""
    console.print(f"[yellow]Merging LoRA weights from {checkpoint}[/yellow]")

    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Load base model
        base_path = checkpoint.parent.parent / "base"
        if not base_path.exists():
            console.print(f"[red]Base model not found at {base_path}[/red]")
            raise typer.Exit(1)

        model = AutoModelForCausalLM.from_pretrained(base_path)
        model = PeftModel.from_pretrained(model, checkpoint)
        model = model.merge_and_unload()

        model.save_pretrained(output)
        tokenizer = AutoTokenizer.from_pretrained(base_path)
        tokenizer.save_pretrained(output)

        console.print(f"[green]Exported to {output}[/green]")

    except ImportError:
        console.print("[red]peft package required for LoRA export[/red]")
        raise typer.Exit(1)


def main():
    app()


if __name__ == "__main__":
    main()
