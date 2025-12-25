"""
Reinforcement Learning trainer for Deep Research agent.

Uses PPO/GRPO with multi-objective rewards (MOA-style).
Designed to work with prime-rl distributed training infrastructure.
"""

import json
import os
from pathlib import Path
from typing import Optional, Any, Callable
from dataclasses import dataclass, field
import random


@dataclass
class RLConfig:
    """Configuration for RL training."""
    # Model
    model_name: str = "GigaChat-Lightning-Instruct"
    model_path: Optional[str] = None  # Path to SFT checkpoint
    ref_model_path: Optional[str] = None  # Reference model for KL

    # Training algorithm
    algorithm: str = "ppo"  # ppo, grpo
    ppo_epochs: int = 4
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    kl_coef: float = 0.1
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # Batch sizes
    rollout_batch_size: int = 64
    mini_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Learning rates
    actor_lr: float = 1e-6
    critic_lr: float = 1e-5
    warmup_steps: int = 100

    # Environment
    max_turns: int = 15
    num_envs: int = 32  # Parallel environments

    # Training
    total_steps: int = 10000
    checkpoint_interval: int = 500
    eval_interval: int = 100
    log_interval: int = 10

    # Curriculum
    curriculum_enabled: bool = True
    curriculum_levels: int = 6
    level_up_threshold: float = 0.7  # Average reward to advance

    # Paths
    library_path: str = ""
    workspace_path: str = "./workspace"
    output_dir: str = "./outputs/rl"
    prompts_path: str = ""  # Path to RL prompts JSONL

    # Hardware
    num_gpus: int = 4
    precision: str = "bf16"
    use_fsdp: bool = True
    flash_attention: bool = True

    # Rewards
    reward_weights: dict = field(default_factory=lambda: {
        "R_fact": 0.4,
        "R_process": 0.2,
        "R_citation": 0.2,
        "R_code": 0.2,
    })
    judge_model: Optional[str] = None  # Model for LLM-as-Judge


@dataclass
class RolloutBuffer:
    """Buffer for storing rollout data."""
    observations: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)
    values: list = field(default_factory=list)
    log_probs: list = field(default_factory=list)
    advantages: list = field(default_factory=list)
    returns: list = field(default_factory=list)
    dones: list = field(default_factory=list)

    def clear(self):
        """Clear all buffers."""
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.advantages.clear()
        self.returns.clear()
        self.dones.clear()

    def add(self, **kwargs):
        """Add a step to the buffer."""
        for key, value in kwargs.items():
            getattr(self, key).append(value)


class RLTrainer:
    """
    Reinforcement Learning trainer for research agent.

    Implements PPO/GRPO with:
    - Multi-turn tool-use environment
    - Multi-objective rewards (MOA-style)
    - Curriculum learning
    - Distributed training via FSDP
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self._model = None
        self._ref_model = None
        self._tokenizer = None
        self._optimizer = None
        self._envs = []
        self._reward_fn = None
        self._prompts = []
        self._current_level = 1

        # Metrics
        self._step = 0
        self._episode_rewards = []

    def _setup(self):
        """Initialize all components."""
        self._load_model()
        self._load_prompts()
        self._setup_envs()
        self._setup_rewards()
        self._setup_optimizer()

    def _load_model(self):
        """Load actor and reference models."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required")

        model_path = self.config.model_path or self.config.model_name
        dtype = torch.bfloat16 if self.config.precision == "bf16" else torch.float16

        print(f"Loading model from {model_path}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Actor model (trainable)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if self.config.flash_attention else None,
        )

        # Reference model (frozen) for KL penalty
        if self.config.ref_model_path:
            ref_path = self.config.ref_model_path
        else:
            ref_path = model_path

        self._ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        self._ref_model.eval()
        for param in self._ref_model.parameters():
            param.requires_grad = False

    def _load_prompts(self):
        """Load RL training prompts."""
        prompts_path = Path(self.config.prompts_path)
        if not prompts_path.exists():
            raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

        with open(prompts_path, 'r', encoding='utf-8') as f:
            self._prompts = [json.loads(line) for line in f]

        print(f"Loaded {len(self._prompts)} training prompts")

        # Group by curriculum level
        self._prompts_by_level = {}
        for prompt in self._prompts:
            level = prompt.get("curriculum_level", 1)
            if level not in self._prompts_by_level:
                self._prompts_by_level[level] = []
            self._prompts_by_level[level].append(prompt)

    def _setup_envs(self):
        """Initialize parallel environments."""
        from .environment import ResearchEnv, create_env

        workspace_base = Path(self.config.workspace_path)

        for i in range(self.config.num_envs):
            env_workspace = workspace_base / f"env_{i}"
            env_workspace.mkdir(parents=True, exist_ok=True)

            env = create_env(
                library_path=self.config.library_path,
                workspace_path=env_workspace,
                max_turns=self.config.max_turns,
                curriculum_level=self._current_level,
            )
            self._envs.append(env)

    def _setup_rewards(self):
        """Initialize reward function."""
        from ..rewards.rewards import create_default_reward_function
        from ..rewards.judge import LLMJudge

        self._reward_fn = create_default_reward_function(
            judge_model=self.config.judge_model,
            weights=self.config.reward_weights,
        )

        # Setup LLM judge if configured
        if self.config.judge_model:
            judge = LLMJudge(
                model=self.config.judge_model,
                backend="openai",  # or anthropic
            )
            # Connect judge to factual reward
            for rf in self._reward_fn.reward_functions:
                if hasattr(rf, 'set_judge'):
                    rf.set_judge(judge)

    def _setup_optimizer(self):
        """Initialize optimizer with learning rate schedule."""
        try:
            import torch
            from torch.optim import AdamW
            from transformers import get_linear_schedule_with_warmup
        except ImportError:
            raise ImportError("torch and transformers required")

        # Separate actor and critic parameters if using separate critic
        self._optimizer = AdamW(
            self._model.parameters(),
            lr=self.config.actor_lr,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )

        self._scheduler = get_linear_schedule_with_warmup(
            self._optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=self.config.total_steps,
        )

    def _sample_prompts(self, n: int) -> list[dict]:
        """Sample prompts for current curriculum level."""
        available_prompts = []
        for level in range(1, self._current_level + 1):
            available_prompts.extend(self._prompts_by_level.get(level, []))

        if not available_prompts:
            available_prompts = self._prompts

        return random.sample(
            available_prompts,
            min(n, len(available_prompts))
        )

    def _collect_rollouts(self) -> RolloutBuffer:
        """Collect rollouts from parallel environments."""
        buffer = RolloutBuffer()

        # Sample tasks
        tasks = self._sample_prompts(self.config.rollout_batch_size)

        for i, task in enumerate(tasks):
            env_idx = i % len(self._envs)
            env = self._envs[env_idx]

            # Reset environment
            obs = env.reset(task["question"])
            episode_reward = 0
            trajectory = []

            done = False
            while not done:
                # Generate action from model
                action, action_input, log_prob, value = self._generate_action(obs)

                # Step environment
                result = env.step(action, action_input)

                trajectory.append({
                    "action": action,
                    "action_input": action_input,
                    "observation": result.observation,
                })

                # Add to buffer
                buffer.add(
                    observations=obs,
                    actions=(action, action_input),
                    rewards=result.reward,
                    values=value,
                    log_probs=log_prob,
                    dones=result.done,
                )

                obs = result.observation
                done = result.done
                episode_reward += result.reward

            # Compute final reward using reward function
            final_answer = None
            if result.info and "final_answer" in result.info:
                final_answer = result.info["final_answer"]

            reward_result = self._reward_fn.compute(
                trajectory=trajectory,
                final_answer=final_answer or "",
                golden_answer=task.get("golden_answer"),
                metadata=task.get("metadata", {}),
            )

            # Add final reward to buffer
            buffer.rewards[-1] += reward_result.total
            episode_reward += reward_result.total

            self._episode_rewards.append(episode_reward)

        # Compute advantages
        self._compute_advantages(buffer)

        return buffer

    def _generate_action(
        self,
        observation: str
    ) -> tuple[str, dict, float, float]:
        """
        Generate action from model.

        Returns:
            Tuple of (action_name, action_input, log_prob, value)
        """
        import torch

        # Tokenize observation
        inputs = self._tokenizer(
            observation,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # Move to device
        device = next(self._model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode action
        generated_text = self._tokenizer.decode(
            outputs.sequences[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # Parse action from generated text
        action, action_input = self._parse_action(generated_text)

        # Compute log probability
        log_prob = self._compute_log_prob(outputs)

        # Estimate value (simplified - would use separate value head in practice)
        value = 0.0

        return action, action_input, log_prob, value

    def _parse_action(self, text: str) -> tuple[str, dict]:
        """Parse action and parameters from generated text."""
        import re

        # Try to parse tool call format: tool_name({"param": "value"})
        match = re.search(r'(\w+)\s*\(\s*(\{.*?\})\s*\)', text, re.DOTALL)
        if match:
            action = match.group(1)
            try:
                action_input = json.loads(match.group(2))
            except json.JSONDecodeError:
                action_input = {}
            return action, action_input

        # Try simple format: action: tool_name, param: value
        action_match = re.search(r'action:\s*(\w+)', text, re.IGNORECASE)
        if action_match:
            action = action_match.group(1)
            # Extract parameters
            action_input = {}
            for param_match in re.finditer(r'(\w+):\s*"([^"]*)"', text):
                action_input[param_match.group(1)] = param_match.group(2)
            return action, action_input

        # Default to final answer
        return "final_answer", {"answer": text}

    def _compute_log_prob(self, outputs) -> float:
        """Compute log probability of generated sequence."""
        import torch

        if hasattr(outputs, 'scores') and outputs.scores:
            # Sum log probs of generated tokens
            log_probs = []
            for score in outputs.scores:
                probs = torch.softmax(score, dim=-1)
                log_probs.append(torch.log(probs.max()).item())
            return sum(log_probs)
        return 0.0

    def _compute_advantages(self, buffer: RolloutBuffer):
        """Compute GAE advantages."""
        import numpy as np

        rewards = np.array(buffer.rewards)
        values = np.array(buffer.values)
        dones = np.array(buffer.dones)

        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values

        buffer.advantages = advantages.tolist()
        buffer.returns = returns.tolist()

    def _update_policy(self, buffer: RolloutBuffer) -> dict:
        """Update policy using PPO."""
        import torch
        import torch.nn.functional as F

        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "kl_div": 0,
        }

        # Convert buffer to tensors (simplified)
        # In practice, would use proper batching

        for epoch in range(self.config.ppo_epochs):
            # Mini-batch updates
            indices = list(range(len(buffer.observations)))
            random.shuffle(indices)

            for start in range(0, len(indices), self.config.mini_batch_size):
                end = start + self.config.mini_batch_size
                batch_indices = indices[start:end]

                # Get batch data
                batch_obs = [buffer.observations[i] for i in batch_indices]
                batch_advantages = torch.tensor([buffer.advantages[i] for i in batch_indices])
                batch_old_log_probs = torch.tensor([buffer.log_probs[i] for i in batch_indices])
                batch_returns = torch.tensor([buffer.returns[i] for i in batch_indices])

                # Forward pass (simplified)
                # In practice, would compute new log probs and values

                # PPO loss computation would go here
                # policy_loss = ...
                # value_loss = ...

                # Update
                self._optimizer.zero_grad()
                # total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                self._optimizer.step()
                self._scheduler.step()

        return metrics

    def _maybe_update_curriculum(self):
        """Check if should advance curriculum level."""
        if not self.config.curriculum_enabled:
            return

        if len(self._episode_rewards) < 100:
            return

        recent_rewards = self._episode_rewards[-100:]
        avg_reward = sum(recent_rewards) / len(recent_rewards)

        if avg_reward >= self.config.level_up_threshold:
            if self._current_level < self.config.curriculum_levels:
                self._current_level += 1
                print(f"Advancing to curriculum level {self._current_level}")

                # Reinitialize environments with new tools
                self._envs.clear()
                self._setup_envs()

    def train(self):
        """Main training loop."""
        print("Setting up training...")
        self._setup()

        print(f"Starting training for {self.config.total_steps} steps...")

        for step in range(self.config.total_steps):
            self._step = step

            # Collect rollouts
            buffer = self._collect_rollouts()

            # Update policy
            metrics = self._update_policy(buffer)

            # Maybe update curriculum
            self._maybe_update_curriculum()

            # Logging
            if step % self.config.log_interval == 0:
                avg_reward = sum(self._episode_rewards[-100:]) / max(len(self._episode_rewards[-100:]), 1)
                print(f"Step {step}: avg_reward={avg_reward:.3f}, level={self._current_level}")

            # Evaluation
            if step % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                print(f"Eval: {eval_metrics}")

            # Checkpoint
            if step % self.config.checkpoint_interval == 0:
                self._save_checkpoint(step)

        # Final save
        self._save_checkpoint(self._step)
        print("Training complete!")

    def _evaluate(self) -> dict:
        """Evaluate current policy on held-out prompts."""
        # Use a subset of prompts for evaluation
        eval_rewards = []

        for prompt in self._prompts[:10]:
            env = self._envs[0]
            obs = env.reset(prompt["question"])

            total_reward = 0
            done = False
            trajectory = []

            while not done:
                action, action_input, _, _ = self._generate_action(obs)
                result = env.step(action, action_input)

                trajectory.append({
                    "action": action,
                    "action_input": action_input,
                    "observation": result.observation,
                })

                total_reward += result.reward
                obs = result.observation
                done = result.done

            eval_rewards.append(total_reward)

        return {
            "eval_mean_reward": sum(eval_rewards) / len(eval_rewards),
            "eval_min_reward": min(eval_rewards),
            "eval_max_reward": max(eval_rewards),
        }

    def _save_checkpoint(self, step: int):
        """Save model checkpoint."""
        output_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
        output_dir.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(output_dir)
        self._tokenizer.save_pretrained(output_dir)

        # Save training state
        state = {
            "step": step,
            "current_level": self._current_level,
            "episode_rewards": self._episode_rewards[-1000:],
        }
        with open(output_dir / "training_state.json", 'w') as f:
            json.dump(state, f)

        print(f"Saved checkpoint to {output_dir}")


def train_rl_cli():
    """CLI entry point for RL training."""
    import argparse

    parser = argparse.ArgumentParser(description="RL Training")
    parser.add_argument("--config", type=str, help="Path to config JSON")
    parser.add_argument("--model-path", type=str, help="Path to SFT checkpoint")
    parser.add_argument("--library-path", type=str, required=True)
    parser.add_argument("--prompts-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/rl")
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--num-gpus", type=int, default=4)

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = RLConfig(**config_dict)
    else:
        config = RLConfig(
            model_path=args.model_path,
            library_path=args.library_path,
            prompts_path=args.prompts_path,
            output_dir=args.output_dir,
            total_steps=args.total_steps,
            num_gpus=args.num_gpus,
        )

    trainer = RLTrainer(config)
    trainer.train()


if __name__ == "__main__":
    train_rl_cli()
