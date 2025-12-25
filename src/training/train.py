#!/usr/bin/env python3
"""
Training Pipeline for Deep Research Agent
SFT and RL training with prime-rl/verifiers integration.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()
app = typer.Typer(help="Train Deep Research Agent")


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    model_name: str = "GigaChat-Lightning-Instruct"
    model_path: Optional[str] = None
    precision: str = "bf16"

    # Training strategy
    strategy: str = "fsdp"  # fsdp, ddp, deepspeed
    flash_attention: bool = True
    gradient_checkpointing: bool = True

    # SFT settings
    sft_epochs: int = 3
    sft_batch_size: int = 4
    sft_learning_rate: float = 2e-5
    sft_warmup_ratio: float = 0.1
    sft_max_length: int = 4096

    # RL settings
    rl_epochs: int = 1
    rl_batch_size: int = 64
    rl_learning_rate: float = 1e-6
    rl_kl_coef: float = 0.05
    rl_clip_range: float = 0.2
    rl_value_clip: float = 0.2
    rl_gamma: float = 0.99
    rl_gae_lambda: float = 0.95

    # Environment
    max_turns: int = 15
    library_path: str = "/mnt/library"
    workspace_path: str = "/workspace"

    # Reward weights
    reward_weights: dict = field(default_factory=lambda: {
        "fact": 0.4,
        "process": 0.2,
        "citation": 0.2,
        "code": 0.2,
    })

    # Judge settings
    judge_model: str = "gpt-4o-mini"

    # Logging
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 100
    wandb_project: Optional[str] = "deepresearch-agent"


@dataclass
class SFTSample:
    """SFT training sample."""
    input_text: str
    output_text: str
    metadata: dict = field(default_factory=dict)


class Trainer:
    """
    Main trainer class for Deep Research Agent.
    Integrates with prime-rl and verifiers.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.optimizer = None

    def setup_model(self):
        """Load and setup model for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"[bold]Loading model: {self.config.model_name}[/bold]")

        model_path = self.config.model_path or self.config.model_name

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        torch_dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.flash_attention else "eager",
        )

        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        console.print(f"[green]Model loaded: {self.model.num_parameters():,} parameters[/green]")

    def setup_fsdp(self):
        """Setup Fully Sharded Data Parallel."""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

        # FSDP configuration for H200
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
        )

        console.print("[green]FSDP initialized[/green]")

    def _format_sft_sample(self, sample: dict) -> str:
        """Format sample for SFT training."""
        # Build conversation format
        messages = []

        # System message with tool descriptions
        from ..tools.tool_env import ToolEnv
        env = ToolEnv(
            library_path=self.config.library_path,
            workspace_path=self.config.workspace_path,
        )
        tool_desc = env.get_tool_descriptions()

        system = f"""You are a research assistant with access to a library of scientific papers.
Your task is to answer questions by searching and reading papers, taking notes, and using code for analysis.

{tool_desc}

Use this format:
Thought: your reasoning
Action: tool_name
Action Input: {{"param": "value"}}
Observation: tool output

When done, provide:
Final Answer: your complete answer with citations"""

        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": sample["question"]})

        # Add turns as assistant messages
        assistant_parts = []
        for turn in sample.get("turns", []):
            if turn.get("thought"):
                assistant_parts.append(f"Thought: {turn['thought']}")
            if turn.get("action"):
                assistant_parts.append(f"Action: {turn['action']}")
                assistant_parts.append(f"Action Input: {json.dumps(turn['action_input'])}")
            if turn.get("observation"):
                assistant_parts.append(f"Observation: {turn['observation']}")

        assistant_parts.append(f"Final Answer: {sample.get('golden_answer', '')}")
        messages.append({"role": "assistant", "content": "\n".join(assistant_parts)})

        # Format with tokenizer's chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False)
        else:
            # Fallback formatting
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    def train_sft(self, dataset_path: str):
        """
        Train with Supervised Fine-Tuning.

        Args:
            dataset_path: Path to SFT JSONL file
        """
        from torch.utils.data import Dataset, DataLoader
        from transformers import get_linear_schedule_with_warmup

        console.print("[bold]Starting SFT Training[/bold]")

        # Load dataset
        samples = []
        with open(dataset_path) as f:
            for line in f:
                samples.append(json.loads(line))

        console.print(f"Loaded {len(samples)} SFT samples")

        # Create dataset
        class SFTDataset(Dataset):
            def __init__(self, samples, tokenizer, max_length, formatter):
                self.samples = samples
                self.tokenizer = tokenizer
                self.max_length = max_length
                self.formatter = formatter

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                sample = self.samples[idx]
                text = self.formatter(sample)

                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding="max_length",
                    return_tensors="pt",
                )

                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "labels": encoding["input_ids"].squeeze(),
                }

        dataset = SFTDataset(
            samples,
            self.tokenizer,
            self.config.sft_max_length,
            self._format_sft_sample,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.sft_batch_size,
            shuffle=True,
            num_workers=4,
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.sft_learning_rate,
            weight_decay=0.01,
        )

        num_training_steps = len(dataloader) * self.config.sft_epochs
        num_warmup_steps = int(num_training_steps * self.config.sft_warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training loop
        self.model.train()
        global_step = 0

        for epoch in range(self.config.sft_epochs):
            console.print(f"\n[bold]Epoch {epoch + 1}/{self.config.sft_epochs}[/bold]")

            epoch_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                # Move to device
                batch = {k: v.cuda() for k, v in batch.items()}

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    console.print(f"  Step {global_step}: loss={avg_loss:.4f}")

                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"sft-step-{global_step}")

            avg_epoch_loss = epoch_loss / len(dataloader)
            console.print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

        # Save final model
        self._save_checkpoint("sft-final")
        console.print("[green]SFT training complete![/green]")

    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save config
        config_path = checkpoint_dir / "training_config.json"
        config_path.write_text(json.dumps(self.config.__dict__, indent=2))

        console.print(f"[dim]Saved checkpoint: {checkpoint_dir}[/dim]")


def train_sft(
    model_path: str,
    dataset_path: str,
    output_dir: str = "./outputs",
    **kwargs,
) -> str:
    """
    Train model with SFT.

    Args:
        model_path: Path to base model
        dataset_path: Path to SFT dataset (JSONL)
        output_dir: Output directory
        **kwargs: Additional training config

    Returns:
        Path to trained model checkpoint
    """
    config = TrainingConfig(
        model_path=model_path,
        output_dir=output_dir,
        **kwargs,
    )

    trainer = Trainer(config)
    trainer.setup_model()

    if torch.cuda.device_count() > 1:
        trainer.setup_fsdp()

    trainer.train_sft(dataset_path)

    return str(Path(output_dir) / "checkpoints" / "sft-final")


def train_rl(
    model_path: str,
    dataset_path: str,
    output_dir: str = "./outputs",
    **kwargs,
) -> str:
    """
    Train model with RL (PPO/GRPO).

    This function integrates with prime-rl for distributed training.

    Args:
        model_path: Path to SFT model
        dataset_path: Path to RL prompts (JSONL)
        output_dir: Output directory
        **kwargs: Additional training config

    Returns:
        Path to trained model checkpoint
    """
    # This would integrate with prime-rl
    # For now, create the configuration file for prime-rl
    config = TrainingConfig(
        model_path=model_path,
        output_dir=output_dir,
        **kwargs,
    )

    # Generate prime-rl config
    prime_config = {
        "model": {
            "name": config.model_name,
            "path": config.model_path,
            "precision": config.precision,
        },
        "training": {
            "strategy": config.strategy,
            "flash_attention": config.flash_attention,
            "batch_size": config.rl_batch_size,
            "learning_rate": config.rl_learning_rate,
            "epochs": config.rl_epochs,
        },
        "ppo": {
            "kl_coef": config.rl_kl_coef,
            "clip_range": config.rl_clip_range,
            "value_clip": config.rl_value_clip,
            "gamma": config.rl_gamma,
            "gae_lambda": config.rl_gae_lambda,
        },
        "environment": {
            "type": "ToolEnv",
            "max_turns": config.max_turns,
            "library_path": config.library_path,
            "workspace_path": config.workspace_path,
        },
        "reward": {
            "judge_model": config.judge_model,
            "weights": config.reward_weights,
        },
        "data": {
            "prompts_path": dataset_path,
        },
        "output": {
            "dir": output_dir,
            "logging_steps": config.logging_steps,
            "save_steps": config.save_steps,
        },
    }

    config_path = Path(output_dir) / "prime_rl_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml
    config_path.write_text(yaml.dump(prime_config, default_flow_style=False))

    console.print(f"[green]Generated prime-rl config: {config_path}[/green]")
    console.print("\nTo run RL training with prime-rl:")
    console.print(f"  python -m prime_rl.train --config {config_path}")

    return str(config_path)


# CLI Commands
@app.command()
def sft(
    model_path: str = typer.Argument(..., help="Path to base model"),
    dataset_path: Path = typer.Argument(..., help="Path to SFT dataset"),
    output_dir: Path = typer.Option(
        Path("./outputs"),
        "--output", "-o",
        help="Output directory"
    ),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    batch_size: int = typer.Option(4, "--batch-size", "-b"),
    learning_rate: float = typer.Option(2e-5, "--lr"),
):
    """Run SFT training."""
    train_sft(
        model_path=model_path,
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        sft_epochs=epochs,
        sft_batch_size=batch_size,
        sft_learning_rate=learning_rate,
    )


@app.command()
def rl(
    model_path: str = typer.Argument(..., help="Path to SFT model"),
    dataset_path: Path = typer.Argument(..., help="Path to RL prompts"),
    output_dir: Path = typer.Option(
        Path("./outputs"),
        "--output", "-o",
        help="Output directory"
    ),
    judge_model: str = typer.Option("gpt-4o-mini", "--judge"),
):
    """Generate RL training config for prime-rl."""
    train_rl(
        model_path=model_path,
        dataset_path=str(dataset_path),
        output_dir=str(output_dir),
        judge_model=judge_model,
    )


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
