#!/usr/bin/env python3
"""
Synthetic Dataset Generator
Generates training data for Deep Research Agent.
"""

import os
import json
import random
import asyncio
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from tqdm import tqdm

from .prompts import (
    RETRIEVAL_PROMPT,
    MULTIHOP_PROMPT,
    COMPUTATION_PROMPT,
    SYNTHESIS_PROMPT,
    SFT_DEMO_PROMPT,
)

console = Console()
app = typer.Typer(help="Generate synthetic training data")


class TaskType(Enum):
    """Types of training tasks."""
    RETRIEVAL = "retrieval"
    MULTIHOP = "multihop"
    COMPUTATION = "computation"
    SYNTHESIS = "synthesis"
    SFT_DEMO = "sft_demo"


@dataclass
class GeneratedSample:
    """A generated training sample."""
    task_type: str
    question: str
    golden_answer: str
    required_papers: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    requires_code: bool = False
    metadata: dict = field(default_factory=dict)

    # For SFT samples
    turns: list[dict] = field(default_factory=list)


@dataclass
class GenerationConfig:
    """Configuration for dataset generation."""
    # Distribution of task types
    distribution: dict = field(default_factory=lambda: {
        TaskType.RETRIEVAL: 0.30,
        TaskType.MULTIHOP: 0.25,
        TaskType.COMPUTATION: 0.20,
        TaskType.SYNTHESIS: 0.15,
        TaskType.SFT_DEMO: 0.10,
    })

    # Generation parameters
    total_samples: int = 10000
    max_concurrent: int = 10
    temperature: float = 0.7

    # Model settings
    model: str = "gpt-4o-mini"  # or deepseek-chat, claude-3-5-sonnet
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class DatasetGenerator:
    """
    Generates synthetic training data using LLMs.
    """

    def __init__(
        self,
        library_path: str = "/mnt/library",
        output_path: str = "./data/synthetic",
        config: GenerationConfig = None,
    ):
        self.library_path = Path(library_path)
        self.output_path = Path(output_path)
        self.config = config or GenerationConfig()

        self._client = None
        self._papers = []
        self._index = []

    def _get_client(self):
        """Get or create API client."""
        if self._client is not None:
            return self._client

        from openai import OpenAI

        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        base_url = self.config.base_url

        # Auto-detect DeepSeek
        if "deepseek" in self.config.model.lower():
            api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            base_url = base_url or "https://api.deepseek.com/v1"

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        return self._client

    def _load_library(self):
        """Load paper index from library."""
        index_path = self.library_path / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Library index not found: {index_path}")

        self._index = json.loads(index_path.read_text())
        self._papers = []

        for paper in self._index:
            paper_id = paper["paper_id"]
            paper_dir = self.library_path / "by_id" / paper_id

            # Load full text
            full_text_path = paper_dir / "full_text.md"
            if full_text_path.exists():
                paper["full_text"] = full_text_path.read_text()[:8000]  # Truncate
            else:
                paper["full_text"] = ""

            # Load abstract
            abstract_path = paper_dir / "abstract.txt"
            if abstract_path.exists():
                paper["abstract"] = abstract_path.read_text()
            else:
                paper["abstract"] = ""

            self._papers.append(paper)

        console.print(f"[green]Loaded {len(self._papers)} papers[/green]")

    async def _call_model(self, prompt: str, system: str = "") -> str:
        """Call the generation model."""
        client = self._get_client()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=2000,
                    response_format={"type": "json_object"},
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            console.print(f"[red]API error: {e}[/red]")
            return "{}"

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from model response."""
        try:
            # Handle markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())
        except json.JSONDecodeError:
            return {}

    async def generate_retrieval(self, paper: dict) -> list[GeneratedSample]:
        """Generate retrieval questions for a paper."""
        prompt = RETRIEVAL_PROMPT.format(
            paper_title=paper["title"],
            paper_id=paper["paper_id"],
            paper_text=paper.get("full_text", "")[:5000],
        )

        response = await self._call_model(prompt)
        data = self._parse_json_response(response)

        samples = []
        for q in data.get("questions", []):
            samples.append(GeneratedSample(
                task_type=TaskType.RETRIEVAL.value,
                question=q.get("question", ""),
                golden_answer=q.get("answer", ""),
                required_papers=[paper["paper_id"]],
                difficulty=q.get("difficulty", "medium"),
                metadata={"section_hint": q.get("section_hint", "")},
            ))

        return samples

    async def generate_multihop(self, paper_a: dict, paper_b: dict) -> GeneratedSample:
        """Generate multi-hop comparison question."""
        prompt = MULTIHOP_PROMPT.format(
            paper_a_title=paper_a["title"],
            paper_a_id=paper_a["paper_id"],
            paper_a_abstract=paper_a.get("abstract", "")[:1000],
            paper_b_title=paper_b["title"],
            paper_b_id=paper_b["paper_id"],
            paper_b_abstract=paper_b.get("abstract", "")[:1000],
        )

        response = await self._call_model(prompt)
        data = self._parse_json_response(response)

        return GeneratedSample(
            task_type=TaskType.MULTIHOP.value,
            question=data.get("question", ""),
            golden_answer=data.get("golden_answer", ""),
            required_papers=data.get("required_papers", [paper_a["paper_id"], paper_b["paper_id"]]),
            difficulty=data.get("difficulty", "hard"),
            metadata={"reasoning_type": data.get("reasoning_type", "comparison")},
        )

    async def generate_computation(self, papers: list[dict]) -> GeneratedSample:
        """Generate computational task."""
        papers_info = "\n".join([
            f"- {p['title']} (ID: {p['paper_id']}): {p.get('abstract', '')[:300]}"
            for p in papers[:5]
        ])

        prompt = COMPUTATION_PROMPT.format(papers_with_metrics=papers_info)

        response = await self._call_model(prompt)
        data = self._parse_json_response(response)

        return GeneratedSample(
            task_type=TaskType.COMPUTATION.value,
            question=data.get("question", ""),
            golden_answer=data.get("golden_answer", ""),
            required_papers=data.get("required_papers", [p["paper_id"] for p in papers[:3]]),
            difficulty=data.get("difficulty", "medium"),
            requires_code=True,
            metadata={
                "expected_code": data.get("expected_code", ""),
                "computation_type": data.get("computation_type", ""),
            },
        )

    async def generate_synthesis(self, topic: str, papers: list[dict]) -> GeneratedSample:
        """Generate synthesis task."""
        papers_info = "\n".join([
            f"- {p['title']} (ID: {p['paper_id']})"
            for p in papers[:7]
        ])

        prompt = SYNTHESIS_PROMPT.format(
            topic_name=topic,
            list_of_papers=papers_info,
        )

        response = await self._call_model(prompt)
        data = self._parse_json_response(response)

        return GeneratedSample(
            task_type=TaskType.SYNTHESIS.value,
            question=data.get("question", ""),
            golden_answer=data.get("golden_answer", ""),
            required_papers=data.get("required_papers", [p["paper_id"] for p in papers[:5]]),
            difficulty="very_hard",
            metadata={
                "evaluation_criteria": data.get("evaluation_criteria", []),
                "expected_length": data.get("expected_length", ""),
            },
        )

    async def generate_sft_demo(
        self,
        question: str,
        answer: str,
        paper_id: str,
    ) -> GeneratedSample:
        """Generate SFT demonstration with tool traces."""
        prompt = SFT_DEMO_PROMPT.format(
            question=question,
            answer=answer,
            paper_id=paper_id,
        )

        response = await self._call_model(prompt)
        data = self._parse_json_response(response)

        return GeneratedSample(
            task_type=TaskType.SFT_DEMO.value,
            question=data.get("question", question),
            golden_answer=data.get("final_answer", answer),
            required_papers=[paper_id],
            turns=data.get("turns", []),
            metadata={
                "tools_used": data.get("tools_used", []),
                "num_turns": data.get("num_turns", 0),
            },
        )

    async def generate_batch(
        self,
        task_type: TaskType,
        count: int,
    ) -> list[GeneratedSample]:
        """Generate a batch of samples of given type."""
        samples = []

        if task_type == TaskType.RETRIEVAL:
            # Sample papers and generate questions
            selected_papers = random.sample(self._papers, min(count, len(self._papers)))
            for paper in selected_papers:
                try:
                    paper_samples = await self.generate_retrieval(paper)
                    samples.extend(paper_samples[:1])  # Take one per paper
                except Exception as e:
                    console.print(f"[yellow]Retrieval generation failed: {e}[/yellow]")

        elif task_type == TaskType.MULTIHOP:
            # Generate paper pairs
            for _ in range(count):
                if len(self._papers) < 2:
                    break
                paper_a, paper_b = random.sample(self._papers, 2)
                try:
                    sample = await self.generate_multihop(paper_a, paper_b)
                    if sample.question:
                        samples.append(sample)
                except Exception as e:
                    console.print(f"[yellow]Multihop generation failed: {e}[/yellow]")

        elif task_type == TaskType.COMPUTATION:
            # Generate computational tasks
            for _ in range(count):
                papers = random.sample(self._papers, min(5, len(self._papers)))
                try:
                    sample = await self.generate_computation(papers)
                    if sample.question:
                        samples.append(sample)
                except Exception as e:
                    console.print(f"[yellow]Computation generation failed: {e}[/yellow]")

        elif task_type == TaskType.SYNTHESIS:
            # Generate synthesis tasks for different topics
            topics = ["reinforcement learning", "alignment", "reward modeling",
                      "evaluation", "scaling", "reasoning", "agents"]
            for topic in random.sample(topics, min(count, len(topics))):
                try:
                    sample = await self.generate_synthesis(topic, self._papers)
                    if sample.question:
                        samples.append(sample)
                except Exception as e:
                    console.print(f"[yellow]Synthesis generation failed: {e}[/yellow]")

        elif task_type == TaskType.SFT_DEMO:
            # Generate SFT demos from existing retrieval samples
            # First generate some retrieval questions
            retrieval_samples = await self.generate_batch(TaskType.RETRIEVAL, count)
            for sample in retrieval_samples:
                try:
                    demo = await self.generate_sft_demo(
                        sample.question,
                        sample.golden_answer,
                        sample.required_papers[0] if sample.required_papers else "",
                    )
                    if demo.turns:
                        samples.append(demo)
                except Exception as e:
                    console.print(f"[yellow]SFT demo generation failed: {e}[/yellow]")

        return samples

    async def generate_dataset(self) -> tuple[list[GeneratedSample], list[GeneratedSample]]:
        """
        Generate complete dataset according to distribution.

        Returns:
            Tuple of (sft_samples, rl_samples)
        """
        self._load_library()
        self.output_path.mkdir(parents=True, exist_ok=True)

        all_samples = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            for task_type, ratio in self.config.distribution.items():
                count = int(self.config.total_samples * ratio)
                task = progress.add_task(f"Generating {task_type.value}...", total=count)

                # Generate in batches
                batch_size = 10
                generated = 0
                while generated < count:
                    batch_count = min(batch_size, count - generated)
                    samples = await self.generate_batch(task_type, batch_count)
                    all_samples.extend(samples)
                    generated += len(samples)
                    progress.update(task, advance=len(samples))

        # Split into SFT and RL datasets
        sft_samples = [s for s in all_samples if s.task_type == TaskType.SFT_DEMO.value]
        rl_samples = [s for s in all_samples if s.task_type != TaskType.SFT_DEMO.value]

        # Save datasets
        self._save_dataset(sft_samples, "sft_train.jsonl")
        self._save_dataset(rl_samples, "rl_prompts.jsonl")

        console.print(f"[green]Generated {len(sft_samples)} SFT samples[/green]")
        console.print(f"[green]Generated {len(rl_samples)} RL samples[/green]")

        return sft_samples, rl_samples

    def _save_dataset(self, samples: list[GeneratedSample], filename: str):
        """Save samples to JSONL file."""
        filepath = self.output_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(asdict(sample), ensure_ascii=False) + '\n')
        console.print(f"[dim]Saved to {filepath}[/dim]")


# CLI Commands
@app.command()
def generate(
    library_path: Path = typer.Option(
        Path("/mnt/library"),
        "--library", "-l",
        help="Path to paper library"
    ),
    output_path: Path = typer.Option(
        Path("./data/synthetic"),
        "--output", "-o",
        help="Output directory"
    ),
    total: int = typer.Option(
        1000,
        "--total", "-n",
        help="Total samples to generate"
    ),
    model: str = typer.Option(
        "gpt-4o-mini",
        "--model", "-m",
        help="Model for generation"
    ),
):
    """Generate synthetic training dataset."""
    config = GenerationConfig(
        total_samples=total,
        model=model,
    )

    generator = DatasetGenerator(
        library_path=str(library_path),
        output_path=str(output_path),
        config=config,
    )

    asyncio.run(generator.generate_dataset())


@app.command()
def validate(
    dataset_path: Path = typer.Argument(..., help="Path to JSONL dataset"),
):
    """Validate generated dataset."""
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        return

    samples = []
    with open(dataset_path) as f:
        for line in f:
            samples.append(json.loads(line))

    console.print(f"[bold]Dataset: {dataset_path}[/bold]")
    console.print(f"Total samples: {len(samples)}")

    # Statistics
    by_type = {}
    by_difficulty = {}
    with_code = 0

    for s in samples:
        t = s.get("task_type", "unknown")
        by_type[t] = by_type.get(t, 0) + 1

        d = s.get("difficulty", "unknown")
        by_difficulty[d] = by_difficulty.get(d, 0) + 1

        if s.get("requires_code"):
            with_code += 1

    console.print("\n[bold]By task type:[/bold]")
    for t, count in sorted(by_type.items()):
        console.print(f"  {t}: {count}")

    console.print("\n[bold]By difficulty:[/bold]")
    for d, count in sorted(by_difficulty.items()):
        console.print(f"  {d}: {count}")

    console.print(f"\nWith code: {with_code}")

    # Check for issues
    issues = []
    for i, s in enumerate(samples):
        if not s.get("question"):
            issues.append(f"Sample {i}: empty question")
        if not s.get("golden_answer"):
            issues.append(f"Sample {i}: empty answer")

    if issues:
        console.print(f"\n[yellow]Issues found: {len(issues)}[/yellow]")
        for issue in issues[:10]:
            console.print(f"  - {issue}")


async def generate_dataset(
    library_path: str = "/mnt/library",
    output_path: str = "./data/synthetic",
    total_samples: int = 1000,
    model: str = "gpt-4o-mini",
) -> tuple[list[GeneratedSample], list[GeneratedSample]]:
    """
    Convenience function for programmatic use.

    Args:
        library_path: Path to paper library
        output_path: Output directory
        total_samples: Total samples to generate
        model: Model for generation

    Returns:
        Tuple of (sft_samples, rl_samples)
    """
    config = GenerationConfig(
        total_samples=total_samples,
        model=model,
    )

    generator = DatasetGenerator(
        library_path=library_path,
        output_path=output_path,
        config=config,
    )

    return await generator.generate_dataset()


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
