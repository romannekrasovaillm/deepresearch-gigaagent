#!/usr/bin/env python3
"""
Rollout Generation for RL Training
Generates trajectories from agent interactions with ToolEnv.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
from dataclasses import dataclass, field, asdict
from datetime import datetime

import torch
from rich.console import Console

from ..tools.tool_env import ToolEnv, ToolResult, parse_tool_call, format_tool_result
from ..rewards.reward_functions import compute_rewards, RewardConfig, RewardResult
from ..rewards.judge import LLMJudge, JudgeModel

console = Console()


@dataclass
class RolloutConfig:
    """Configuration for rollout generation."""
    # Environment
    library_path: str = "/mnt/library"
    workspace_path: str = "/workspace"
    max_turns: int = 15

    # Model
    model_path: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 1024

    # Rollout settings
    num_rollouts: int = 1000
    batch_size: int = 32
    num_workers: int = 4

    # Reward
    judge_model: str = "gpt-4o-mini"
    reward_config: RewardConfig = field(default_factory=RewardConfig)


@dataclass
class Trajectory:
    """A single agent trajectory."""
    question: str
    turns: list[dict] = field(default_factory=list)
    final_answer: str = ""
    rewards: RewardResult = None
    total_reward: float = 0.0
    metadata: dict = field(default_factory=dict)


class RolloutWorker:
    """
    Worker for generating agent rollouts.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model=None,
        tokenizer=None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer

        self.env = ToolEnv(
            library_path=config.library_path,
            workspace_path=config.workspace_path,
            max_turns=config.max_turns,
        )

        self.judge = LLMJudge(model=JudgeModel(config.judge_model))

    def _format_prompt(self, question: str, trajectory: list[dict]) -> str:
        """Format prompt with current trajectory."""
        tool_desc = self.env.get_tool_descriptions()

        system = f"""You are a research assistant with access to a library of scientific papers.
Your task is to answer questions by searching and reading papers, taking notes, and using code for analysis.

{tool_desc}

Use this format:
Thought: your reasoning about what to do next
Action: tool_name
Action Input: {{"param": "value"}}

When you have enough information, provide your final answer:
Thought: I now have all the information needed.
Final Answer: your complete answer with citations (paper_XXXX)"""

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": f"Question: {question}"})

        # Add trajectory
        if trajectory:
            assistant_content = []
            for turn in trajectory:
                if turn.get("thought"):
                    assistant_content.append(f"Thought: {turn['thought']}")
                if turn.get("action"):
                    assistant_content.append(f"Action: {turn['action']}")
                    assistant_content.append(f"Action Input: {json.dumps(turn['action_input'])}")
                if turn.get("observation"):
                    assistant_content.append(f"Observation: {turn['observation']}")

            messages.append({"role": "assistant", "content": "\n".join(assistant_content)})

        # Format with chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Generate response from model."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_tokens * 3,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode only new tokens
        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def _parse_response(self, response: str) -> tuple[Optional[str], dict, Optional[str]]:
        """
        Parse model response into thought, action, and final answer.

        Returns:
            Tuple of (thought, action_dict, final_answer)
            action_dict contains 'name' and 'input' if action found
            final_answer is set if model produced final answer
        """
        import re

        thought = ""
        action = {"name": None, "input": {}}
        final_answer = None

        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        # Check for final answer
        final_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL)
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, action, final_answer

        # Extract action
        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action["name"] = action_match.group(1)

            # Extract action input
            input_match = re.search(r'Action Input:\s*(\{.*?\})', response, re.DOTALL)
            if input_match:
                try:
                    action["input"] = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    pass

        return thought, action, final_answer

    async def generate_rollout(
        self,
        question: str,
        golden_answer: str = "",
    ) -> Trajectory:
        """
        Generate a single rollout for a question.

        Args:
            question: The research question
            golden_answer: Ground truth answer for reward computation

        Returns:
            Complete trajectory with rewards
        """
        # Reset environment
        episode = self.env.reset(question)
        trajectory = []
        final_answer = None

        for turn_num in range(self.config.max_turns):
            # Format prompt with current trajectory
            prompt = self._format_prompt(question, trajectory)

            # Generate response
            response = self._generate(prompt)

            # Parse response
            thought, action, answer = self._parse_response(response)

            if answer:
                # Agent produced final answer
                final_answer = answer
                break

            if not action["name"]:
                # No valid action, try again or end
                continue

            # Execute action
            result = await self.env.step(
                action["name"],
                action["input"],
                thought,
            )

            # Record turn
            trajectory.append({
                "turn": turn_num + 1,
                "thought": thought,
                "action": action["name"],
                "action_input": action["input"],
                "observation": result.output,
                "status": result.status.value,
            })

            if self.env.is_done():
                break

        # Finalize episode
        episode = self.env.finish(final_answer or "")

        # Compute rewards
        judge_score = None
        if golden_answer:
            try:
                judge_result = await self.judge.judge(
                    question,
                    golden_answer,
                    final_answer or "",
                )
                judge_score = judge_result.get("total_score", 0.5)
            except Exception as e:
                console.print(f"[yellow]Judge failed: {e}[/yellow]")

        rewards = compute_rewards(
            episode,
            golden_answer=golden_answer,
            judge_score=judge_score,
            library_path=self.config.library_path,
            config=self.config.reward_config,
        )

        return Trajectory(
            question=question,
            turns=trajectory,
            final_answer=final_answer or "",
            rewards=rewards,
            total_reward=rewards.total,
            metadata={
                "num_turns": len(trajectory),
                "judge_score": judge_score,
            },
        )

    async def generate_batch(
        self,
        samples: list[dict],
    ) -> list[Trajectory]:
        """
        Generate rollouts for a batch of samples.

        Args:
            samples: List of dicts with 'question' and optional 'golden_answer'

        Returns:
            List of trajectories
        """
        tasks = [
            self.generate_rollout(
                s["question"],
                s.get("golden_answer", ""),
            )
            for s in samples
        ]

        return await asyncio.gather(*tasks)


async def generate_rollouts(
    model_path: str,
    prompts_path: str,
    output_path: str,
    config: RolloutConfig = None,
) -> list[Trajectory]:
    """
    Generate rollouts for RL training.

    Args:
        model_path: Path to model checkpoint
        prompts_path: Path to prompts JSONL
        output_path: Output path for trajectories

    Returns:
        List of generated trajectories
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if config is None:
        config = RolloutConfig(model_path=model_path)

    # Load model
    console.print(f"[bold]Loading model: {model_path}[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Load prompts
    prompts = []
    with open(prompts_path) as f:
        for line in f:
            prompts.append(json.loads(line))

    console.print(f"Loaded {len(prompts)} prompts")

    # Create worker
    worker = RolloutWorker(config, model, tokenizer)

    # Generate rollouts in batches
    all_trajectories = []
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for i in range(0, len(prompts), config.batch_size):
            batch = prompts[i:i + config.batch_size]
            console.print(f"Processing batch {i // config.batch_size + 1}...")

            trajectories = await worker.generate_batch(batch)

            for traj in trajectories:
                all_trajectories.append(traj)
                # Write immediately for recovery
                f.write(json.dumps(asdict(traj), ensure_ascii=False) + '\n')
                f.flush()

    console.print(f"[green]Generated {len(all_trajectories)} trajectories[/green]")
    console.print(f"[dim]Saved to {output_path}[/dim]")

    # Statistics
    avg_reward = sum(t.total_reward for t in all_trajectories) / len(all_trajectories)
    avg_turns = sum(len(t.turns) for t in all_trajectories) / len(all_trajectories)

    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Average reward: {avg_reward:.3f}")
    console.print(f"  Average turns: {avg_turns:.1f}")

    return all_trajectories


def convert_to_ppo_format(
    trajectories_path: str,
    output_path: str,
):
    """
    Convert trajectories to format expected by prime-rl PPO.

    Args:
        trajectories_path: Path to trajectories JSONL
        output_path: Output path for PPO format
    """
    trajectories = []
    with open(trajectories_path) as f:
        for line in f:
            trajectories.append(json.loads(line))

    ppo_samples = []
    for traj in trajectories:
        # Build prompt (without final answer)
        prompt_parts = [f"Question: {traj['question']}"]

        for turn in traj["turns"]:
            prompt_parts.append(f"Thought: {turn['thought']}")
            prompt_parts.append(f"Action: {turn['action']}")
            prompt_parts.append(f"Action Input: {json.dumps(turn['action_input'])}")
            prompt_parts.append(f"Observation: {turn['observation']}")

        # Response is the final answer
        response = f"Thought: I have gathered enough information.\nFinal Answer: {traj['final_answer']}"

        ppo_samples.append({
            "prompt": "\n".join(prompt_parts),
            "response": response,
            "reward": traj["total_reward"],
            "reward_breakdown": traj.get("rewards", {}),
        })

    with open(output_path, 'w') as f:
        for sample in ppo_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    console.print(f"[green]Converted {len(ppo_samples)} samples to PPO format[/green]")
