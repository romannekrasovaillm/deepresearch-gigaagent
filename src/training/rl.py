"""
Reinforcement Learning Training with prime-rl.

Implements Full RL training on 4×H200 with FSDP.
Uses multi-objective rewards (MOA-style) for scientific research agent.
"""

import os
import json
from pathlib import Path
from typing import Optional, Iterator
from dataclasses import dataclass, field
import asyncio

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.env.tool_env import ResearchToolEnv
from src.env.rewards import RewardCalculator
from .data import load_rl_prompts


@dataclass
class RLConfig:
    """Configuration for RL training."""
    # Model
    model_name: str = "ai-sage/GigaChat-Lightning-Instruct"
    sft_checkpoint: Optional[str] = None  # Start from SFT if provided

    # Paths
    output_dir: str = "./checkpoints/rl"
    data_path: str = "./data/datasets/rl_prompts.jsonl"
    library_path: str = "/mnt/library"
    workspace_path: str = "/workspace"

    # RL Algorithm
    algorithm: str = "ppo"  # ppo, grpo
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Training
    num_epochs: int = 10
    batch_size: int = 64
    mini_batch_size: int = 8
    rollout_batch_size: int = 32
    learning_rate: float = 1e-6
    kl_coef: float = 0.1

    # Environment
    max_turns: int = 15
    num_envs: int = 32  # Parallel environments

    # Hardware
    bf16: bool = True
    fsdp: bool = True
    num_gpus: int = 4

    # Rewards
    reward_weights: dict = field(default_factory=lambda: {
        "r_fact": 0.4,
        "r_process": 0.2,
        "r_citation": 0.2,
        "r_code": 0.2,
    })

    # Logging
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50


class RLTrainer:
    """
    Reinforcement Learning trainer using FSDP for distributed training.

    Designed for 4×H200 setup with full model training (no LoRA needed).
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.reward_calculator = None

        # Distributed setup
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.is_main = self.rank == 0

    def setup_distributed(self):
        """Initialize distributed training."""
        if self.world_size > 1:
            dist.init_process_group(backend="nccl")
            torch.cuda.set_device(self.rank)

    def load_model(self):
        """Load model with FSDP wrapping."""
        print(f"[Rank {self.rank}] Loading model...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_path = self.config.sft_checkpoint or self.config.model_name

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float32,
            trust_remote_code=True,
        )

        # Wrap with FSDP
        if self.config.fsdp and self.world_size > 1:
            self.model = FSDP(
                model,
                # FSDP config for optimal memory usage
                sharding_strategy="FULL_SHARD",
                cpu_offload=None,  # H200 has enough memory
                mixed_precision=torch.bfloat16 if self.config.bf16 else None,
            )
        else:
            self.model = model.cuda()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

    def setup_environment(self) -> list[ResearchToolEnv]:
        """Create parallel environments."""
        envs = []
        for i in range(self.config.num_envs):
            env = ResearchToolEnv(
                library_path=self.config.library_path,
                workspace_path=f"{self.config.workspace_path}/env_{i}",
                max_turns=self.config.max_turns,
            )
            envs.append(env)
        return envs

    def setup_rewards(self):
        """Initialize reward calculator."""
        self.reward_calculator = RewardCalculator(
            weights=self.config.reward_weights,
            library_path=self.config.library_path,
        )

    async def collect_rollouts(
        self,
        envs: list[ResearchToolEnv],
        prompts: list[dict],
    ) -> list[dict]:
        """
        Collect rollouts from environments.

        Args:
            envs: List of environments
            prompts: List of task prompts

        Returns:
            List of trajectory dictionaries
        """
        trajectories = []

        for env, prompt in zip(envs, prompts):
            # Reset environment
            obs = env.reset(prompt)

            # Rollout loop
            done = False
            while not done:
                # Generate action from model
                action = await self._generate_action(obs.content, env)

                # Step environment
                obs, step_reward, done, info = env.step(action)

            # Get trajectory
            trajectory = env.get_trajectory()

            # Compute rewards
            rewards = await self.reward_calculator.compute(trajectory)
            trajectory["rewards"] = rewards

            trajectories.append(trajectory)

        return trajectories

    async def _generate_action(
        self,
        observation: str,
        env: ResearchToolEnv,
    ) -> dict:
        """
        Generate action from model.

        Args:
            observation: Current observation
            env: Environment for context

        Returns:
            Action dictionary
        """
        # Build prompt
        system = env.get_system_prompt()
        prompt = f"{system}\n\nObservation: {observation}\n\nAssistant:"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        # Parse action from response
        return self._parse_action(response)

    def _parse_action(self, response: str) -> dict:
        """Parse model response into action."""
        import re

        # Check for final answer
        final_match = re.search(
            r'<final_answer>(.*?)</final_answer>',
            response,
            re.DOTALL
        )
        if final_match:
            return {"final_answer": final_match.group(1).strip()}

        # Parse tool call
        action_match = re.search(r'Action:\s*(\w+)', response)
        input_match = re.search(r'Action Input:\s*(\{.*?\})', response, re.DOTALL)

        if action_match:
            tool = action_match.group(1)
            try:
                tool_input = json.loads(input_match.group(1)) if input_match else {}
            except json.JSONDecodeError:
                tool_input = {}

            return {"tool": tool, "input": tool_input}

        # Default: empty action
        return {"tool": "read_notes", "input": {}}

    def compute_ppo_loss(
        self,
        trajectories: list[dict],
    ) -> torch.Tensor:
        """
        Compute PPO loss from trajectories.

        Args:
            trajectories: List of trajectory dicts with rewards

        Returns:
            Loss tensor
        """
        total_loss = torch.tensor(0.0, device=self.model.device)

        for traj in trajectories:
            rewards = traj.get("rewards", {})
            total_reward = rewards.total if hasattr(rewards, 'total') else 0.0

            # Simplified PPO loss (full implementation would use GAE, value function, etc.)
            # This is a placeholder - real implementation uses prime-rl
            loss = -total_reward

            total_loss += loss

        return total_loss / len(trajectories)

    def train_step(self, trajectories: list[dict]) -> dict:
        """
        Perform one training step.

        Args:
            trajectories: Collected trajectories

        Returns:
            Training metrics
        """
        self.optimizer.zero_grad()

        loss = self.compute_ppo_loss(trajectories)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_reward": sum(
                t["rewards"].total for t in trajectories
                if hasattr(t.get("rewards", {}), 'total')
            ) / len(trajectories),
        }

    def train(self):
        """Run full RL training loop."""
        self.setup_distributed()
        self.load_model()
        self.setup_rewards()

        envs = self.setup_environment()
        prompts = list(load_rl_prompts(Path(self.config.data_path)))

        if self.is_main:
            print(f"Starting RL training with {len(prompts)} prompts")
            print(f"Using {self.config.num_envs} parallel environments")

        for epoch in range(self.config.num_epochs):
            epoch_metrics = []

            # Shuffle prompts each epoch
            import random
            random.shuffle(prompts)

            for batch_idx in range(0, len(prompts), self.config.rollout_batch_size):
                batch_prompts = prompts[batch_idx:batch_idx + self.config.rollout_batch_size]

                # Pad to num_envs
                while len(batch_prompts) < self.config.num_envs:
                    batch_prompts.append(random.choice(prompts))

                # Collect rollouts
                trajectories = asyncio.run(
                    self.collect_rollouts(envs, batch_prompts[:self.config.num_envs])
                )

                # Training step
                metrics = self.train_step(trajectories)
                epoch_metrics.append(metrics)

                if self.is_main and batch_idx % self.config.log_interval == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"loss={metrics['loss']:.4f}, reward={metrics['mean_reward']:.4f}")

            # Save checkpoint
            if self.is_main and epoch % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

        if self.is_main:
            self.save_checkpoint("final")
            print("RL training complete!")

    def save_checkpoint(self, name):
        """Save model checkpoint."""
        save_path = Path(self.config.output_dir) / f"checkpoint_{name}"
        save_path.mkdir(parents=True, exist_ok=True)

        # Save with FSDP state dict
        if self.config.fsdp:
            # Get full state dict
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                save_policy,
            ):
                state_dict = self.model.state_dict()
                if self.is_main:
                    torch.save(state_dict, save_path / "model.pt")
        else:
            if self.is_main:
                self.model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)


def train_rl(
    model_name: str,
    data_path: str,
    output_dir: str,
    sft_checkpoint: Optional[str] = None,
    **kwargs,
):
    """
    Convenience function for RL training.

    Args:
        model_name: Base model name
        data_path: Path to RL prompts
        output_dir: Output directory
        sft_checkpoint: Optional SFT checkpoint to start from
        **kwargs: Additional config options
    """
    config = RLConfig(
        model_name=model_name,
        sft_checkpoint=sft_checkpoint,
        data_path=data_path,
        output_dir=output_dir,
        **kwargs,
    )

    trainer = RLTrainer(config)
    trainer.train()
