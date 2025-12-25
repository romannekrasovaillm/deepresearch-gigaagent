"""RL Trainer using prime-rl and verifiers integration."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterator

import torch
import yaml
from rich.console import Console

from ..environment import ResearchToolEnv, ToolEnvConfig
from ..rewards import CombinedReward, RewardWeights

console = Console()


@dataclass
class RLConfig:
    """Configuration for RL training."""

    # Model
    model_name: str = "GigaChat-Lightning-Instruct"
    model_path: Optional[str] = None
    sft_checkpoint: Optional[str] = None

    # Training
    algorithm: str = "grpo"  # "ppo" or "grpo"
    output_dir: str = "./checkpoints/rl"
    total_steps: int = 10000
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 10

    # Rollouts
    num_rollouts: int = 64
    max_turns: int = 15
    temperature: float = 0.7
    top_p: float = 0.95

    # PPO
    ppo_epochs: int = 4
    batch_size: int = 16
    mini_batch_size: int = 4
    learning_rate: float = 1e-6
    kl_coef: float = 0.02
    clip_range: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # GRPO
    grpo_num_generations: int = 8
    grpo_learning_rate: float = 5e-7
    grpo_beta: float = 0.1

    # Distributed
    num_gpus: int = 4
    fsdp_enabled: bool = True
    bf16: bool = True

    # Environment
    library_path: str = "./data/library"
    workspace_path: str = "./data/workspace"

    # Reward
    judge_model: str = "deepseek-v3"
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # Curriculum
    curriculum_enabled: bool = True
    curriculum_stages: list = field(default_factory=list)


class RLDataLoader:
    """Load RL prompts for training."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> list[dict]:
        """Load prompts from JSONL."""
        prompts = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if data.get("type") != "sft_demo":  # Skip SFT demos
                    prompts.append(data)
        return prompts

    def sample_batch(self, batch_size: int) -> list[dict]:
        """Sample a batch of prompts."""
        import random

        return random.sample(self.prompts, min(batch_size, len(self.prompts)))

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.prompts)


class RLTrainer:
    """
    RL Trainer for research agent.

    Integrates with prime-rl for distributed training
    and verifiers for tool environments.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment
        env_config = ToolEnvConfig(
            library_path=config.library_path,
            workspace_path=config.workspace_path,
            max_turns=config.max_turns,
        )
        self.env = ResearchToolEnv(env_config)

        # Initialize reward
        self.reward_fn = CombinedReward(
            weights=config.reward_weights,
            judge_model=config.judge_model,
            library_path=config.library_path,
        )

        # Curriculum state
        self.current_stage = 0
        self.steps_in_stage = 0

    def _get_curriculum_distribution(self) -> dict:
        """Get current task distribution based on curriculum stage."""
        if not self.config.curriculum_enabled or not self.config.curriculum_stages:
            return None

        if self.current_stage >= len(self.config.curriculum_stages):
            return self.config.curriculum_stages[-1].get("task_distribution")

        return self.config.curriculum_stages[self.current_stage].get(
            "task_distribution"
        )

    def _update_curriculum(self) -> None:
        """Update curriculum stage if needed."""
        if not self.config.curriculum_enabled:
            return

        self.steps_in_stage += 1

        if self.current_stage < len(self.config.curriculum_stages):
            stage = self.config.curriculum_stages[self.current_stage]
            if self.steps_in_stage >= stage.get("steps", 1000):
                self.current_stage += 1
                self.steps_in_stage = 0
                console.print(
                    f"[blue]Advancing to curriculum stage {self.current_stage}[/blue]"
                )

    def run_episode(self, prompt: dict) -> dict:
        """Run single episode in environment."""
        question = prompt["question"]
        golden_answer = prompt["golden_answer"]
        required_papers = prompt.get("required_papers", [])

        # Reset environment
        obs = self.env.reset(question)
        trajectory = [{"observation": obs["observation"]}]
        done = False

        while not done:
            # Generate action (placeholder - actual generation via model)
            action = self._generate_action(trajectory)

            # Step environment
            result = self.env.step(action)
            trajectory.append({
                "action": action,
                "observation": result["observation"],
            })
            done = result["done"]

        # Get final answer
        agent_answer = ""
        if "answer" in result:
            agent_answer = result["answer"]

        # Compute reward
        metrics = self.env.get_metrics()
        reward_score = self.reward_fn.compute(
            question=question,
            golden_answer=golden_answer,
            agent_answer=agent_answer,
            env_metrics=metrics,
            required_papers=required_papers,
            timeout=result.get("timeout", False),
        )

        return {
            "question": question,
            "trajectory": trajectory,
            "answer": agent_answer,
            "reward": reward_score.clipped,
            "reward_breakdown": self.reward_fn.get_reward_breakdown(reward_score),
            "metrics": metrics,
        }

    def _generate_action(self, trajectory: list) -> str:
        """
        Generate action from model.

        Note: This is a placeholder. In actual training,
        this would call the model being trained.
        """
        # For now, return a simple search action
        return json.dumps({
            "name": "list_papers",
            "arguments": {"limit": 10},
        })

    def generate_prime_rl_config(self) -> dict:
        """Generate configuration for prime-rl library."""
        config = {
            "model": {
                "name": self.config.model_name,
                "path": self.config.model_path or self.config.sft_checkpoint,
                "precision": "bf16" if self.config.bf16 else "fp32",
            },
            "training": {
                "algorithm": self.config.algorithm,
                "total_steps": self.config.total_steps,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.batch_size
                // self.config.mini_batch_size,
            },
            "distributed": {
                "strategy": "fsdp" if self.config.fsdp_enabled else "ddp",
                "num_gpus": self.config.num_gpus,
            },
            "rollout": {
                "num_rollouts": self.config.num_rollouts,
                "max_turns": self.config.max_turns,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
            "ppo": {
                "epochs": self.config.ppo_epochs,
                "learning_rate": self.config.learning_rate,
                "kl_coef": self.config.kl_coef,
                "clip_range": self.config.clip_range,
                "value_loss_coef": self.config.value_loss_coef,
                "entropy_coef": self.config.entropy_coef,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
            },
            "grpo": {
                "num_generations": self.config.grpo_num_generations,
                "learning_rate": self.config.grpo_learning_rate,
                "beta": self.config.grpo_beta,
            },
            "environment": {
                "type": "ResearchToolEnv",
                "config": {
                    "library_path": self.config.library_path,
                    "workspace_path": self.config.workspace_path,
                    "max_turns": self.config.max_turns,
                },
            },
            "reward": {
                "type": "CombinedReward",
                "judge_model": self.config.judge_model,
                "weights": {
                    "R_fact": self.config.reward_weights.R_fact,
                    "R_process": self.config.reward_weights.R_process,
                    "R_citation": self.config.reward_weights.R_citation,
                    "R_code": self.config.reward_weights.R_code,
                },
            },
            "logging": {
                "output_dir": self.config.output_dir,
                "save_steps": self.config.save_steps,
                "eval_steps": self.config.eval_steps,
                "logging_steps": self.config.logging_steps,
            },
        }

        if self.config.curriculum_enabled:
            config["curriculum"] = {
                "enabled": True,
                "stages": self.config.curriculum_stages,
            }

        return config

    def save_prime_rl_config(self, path: str | Path) -> None:
        """Save prime-rl config to file."""
        config = self.generate_prime_rl_config()
        Path(path).write_text(
            yaml.dump(config, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        console.print(f"[green]Saved prime-rl config to {path}[/green]")

    def train(self, data_path: str | Path) -> None:
        """
        Run RL training.

        Note: This is a simplified training loop.
        For full distributed training, use prime-rl directly
        with the generated config.
        """
        console.print("[bold]Starting RL Training[/bold]")
        console.print(f"Algorithm: {self.config.algorithm}")
        console.print(f"Total steps: {self.config.total_steps}")

        # Load data
        data_loader = RLDataLoader(data_path)
        console.print(f"Loaded {len(data_loader)} prompts")

        # Save prime-rl config
        config_path = Path(self.config.output_dir) / "prime_rl_config.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_prime_rl_config(config_path)

        console.print(
            "\n[yellow]For full distributed training on 4xH200:[/yellow]"
        )
        console.print(f"  torchrun --nproc_per_node=4 \\")
        console.print(f"    -m prime_rl.train \\")
        console.print(f"    --config {config_path}")

        # Simple training loop for testing
        step = 0
        total_reward = 0.0

        while step < min(self.config.total_steps, 100):  # Limit for testing
            batch = data_loader.sample_batch(self.config.batch_size)

            for prompt in batch:
                episode = self.run_episode(prompt)
                total_reward += episode["reward"]
                step += 1

                if step % self.config.logging_steps == 0:
                    avg_reward = total_reward / step
                    console.print(
                        f"Step {step}: avg_reward={avg_reward:.4f}"
                    )

                self._update_curriculum()

                if step >= self.config.total_steps:
                    break

        console.print(f"\n[green]Training complete! Steps: {step}[/green]")
