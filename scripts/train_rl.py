#!/usr/bin/env python3
"""CLI script for RL training."""

import sys
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training import RLTrainer, RLConfig
from rewards import RewardWeights

app = typer.Typer(help="RL training for research agent")
console = Console()


@app.command("train")
def train(
    data_path: Path = typer.Argument(..., help="RL prompts JSONL"),
    sft_checkpoint: Optional[Path] = typer.Option(
        None, "--sft", "-s", help="SFT checkpoint to start from"
    ),
    model_name: str = typer.Option(
        "GigaChat-Lightning-Instruct", "--model", "-m", help="Model name/path"
    ),
    output_dir: Path = typer.Option(
        "./checkpoints/rl", "--output", "-o", help="Output directory"
    ),
    algorithm: str = typer.Option("grpo", "--algo", "-a", help="RL algorithm (ppo/grpo)"),
    total_steps: int = typer.Option(10000, "--steps", help="Total training steps"),
    num_gpus: int = typer.Option(4, "--gpus", "-g", help="Number of GPUs"),
    library_path: Path = typer.Option(
        "./data/library", "--library", "-l", help="Paper library path"
    ),
    judge_model: str = typer.Option(
        "deepseek-v3", "--judge", "-j", help="Judge model for rewards"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Config YAML file"
    ),
):
    """Run RL training on research tasks."""
    if not data_path.exists():
        console.print(f"[red]Data not found: {data_path}[/red]")
        raise typer.Exit(1)

    # Load config from file or create from args
    if config_file and config_file.exists():
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
        config = RLConfig(**cfg)
    else:
        config = RLConfig(
            model_name=model_name,
            sft_checkpoint=str(sft_checkpoint) if sft_checkpoint else None,
            output_dir=str(output_dir),
            algorithm=algorithm,
            total_steps=total_steps,
            num_gpus=num_gpus,
            library_path=str(library_path),
            judge_model=judge_model,
        )

    console.print("[bold]RL Training Configuration[/bold]")
    console.print(f"Model: {config.model_name}")
    console.print(f"SFT checkpoint: {config.sft_checkpoint or 'None'}")
    console.print(f"Algorithm: {config.algorithm}")
    console.print(f"Total steps: {config.total_steps}")
    console.print(f"GPUs: {config.num_gpus}")
    console.print(f"Judge model: {config.judge_model}")
    console.print()

    trainer = RLTrainer(config)
    trainer.train(data_path)


@app.command("config")
def generate_config(
    output: Path = typer.Argument(..., help="Output config YAML path"),
    algorithm: str = typer.Option("grpo", "--algo", "-a", help="RL algorithm"),
    num_gpus: int = typer.Option(4, "--gpus", "-g", help="Number of GPUs"),
):
    """Generate prime-rl config file."""
    config = RLConfig(
        algorithm=algorithm,
        num_gpus=num_gpus,
    )

    trainer = RLTrainer(config)
    trainer.save_prime_rl_config(output)


@app.command("eval")
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Model checkpoint"),
    eval_data: Path = typer.Argument(..., help="Evaluation data JSONL"),
    library_path: Path = typer.Option(
        "./data/library", "--library", "-l", help="Paper library path"
    ),
    num_samples: int = typer.Option(100, "--samples", "-n", help="Number of samples"),
):
    """Evaluate trained model on held-out data."""
    import json

    config = RLConfig(
        model_path=str(checkpoint),
        library_path=str(library_path),
    )

    trainer = RLTrainer(config)

    console.print(f"[bold]Evaluating {checkpoint}[/bold]")
    console.print(f"Samples: {num_samples}")

    # Load eval data
    samples = []
    with open(eval_data, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line))
            if len(samples) >= num_samples:
                break

    total_reward = 0.0
    results = []

    for sample in samples:
        episode = trainer.run_episode(sample)
        total_reward += episode["reward"]
        results.append({
            "question": sample["question"],
            "reward": episode["reward"],
        })

    avg_reward = total_reward / len(samples)
    console.print(f"\n[green]Average reward: {avg_reward:.4f}[/green]")

    # Save results
    results_path = checkpoint.parent / "eval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {"avg_reward": avg_reward, "results": results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    console.print(f"Results saved to {results_path}")


def main():
    app()


if __name__ == "__main__":
    main()
