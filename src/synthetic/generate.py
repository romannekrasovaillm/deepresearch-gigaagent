"""
Synthetic Dataset Generation for Deep Research Agent Training.

Generates training tasks using a powerful LLM (GPT-4, Claude, DeepSeek-V3)
based on the paper library. Produces both SFT demonstrations and RL prompts.
"""

import json
import random
import asyncio
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field, asdict
from datetime import datetime

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .prompts import GENERATION_PROMPTS

console = Console()
app = typer.Typer()


@dataclass
class GeneratedTask:
    """A generated training task."""
    task_id: str
    task_type: str  # retrieval, multihop, computation, synthesis, sft_demo
    question: str
    golden_answer: str
    required_papers: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class Paper:
    """Paper metadata for generation."""
    paper_id: str
    title: str
    abstract: str
    sections: list[str]
    full_text: str = ""


def load_papers(library_path: Path, max_papers: Optional[int] = None) -> list[Paper]:
    """
    Load papers from library for generation.

    Args:
        library_path: Path to library directory
        max_papers: Maximum papers to load

    Returns:
        List of Paper objects
    """
    index_path = library_path / "index.json"
    if not index_path.exists():
        console.print("[red]Library index not found![/red]")
        return []

    index = json.loads(index_path.read_text())
    papers = []

    for entry in index[:max_papers]:
        paper_id = entry.get("paper_id")
        paper_dir = library_path / "by_id" / paper_id

        if not paper_dir.exists():
            continue

        # Load abstract
        abstract_path = paper_dir / "abstract.txt"
        abstract = abstract_path.read_text() if abstract_path.exists() else ""

        # Load full text (for retrieval tasks)
        full_text_path = paper_dir / "full_text.md"
        full_text = ""
        if full_text_path.exists():
            full_text = full_text_path.read_text()
            # Truncate for API limits
            if len(full_text) > 15000:
                full_text = full_text[:15000] + "\n... [truncated]"

        # Get sections
        sections = entry.get("sections", [])

        papers.append(Paper(
            paper_id=paper_id,
            title=entry.get("title", "Unknown"),
            abstract=abstract,
            sections=sections,
            full_text=full_text,
        ))

    console.print(f"[green]Loaded {len(papers)} papers[/green]")
    return papers


class LLMClient:
    """Unified client for different LLM providers."""

    def __init__(
        self,
        provider: Literal["openai", "anthropic", "deepseek"] = "openai",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = provider
        self.model = model or self._default_model()
        self.api_key = api_key

    def _default_model(self) -> str:
        defaults = {
            "openai": "gpt-4-turbo-preview",
            "anthropic": "claude-3-opus-20240229",
            "deepseek": "deepseek-chat",
        }
        return defaults.get(self.provider, "gpt-4")

    async def generate(self, system: str, user: str) -> str:
        """Generate completion from LLM."""
        if self.provider == "openai":
            return await self._openai_generate(system, user)
        elif self.provider == "anthropic":
            return await self._anthropic_generate(system, user)
        elif self.provider == "deepseek":
            return await self._deepseek_generate(system, user)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    async def _openai_generate(self, system: str, user: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=self.api_key)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        return response.choices[0].message.content

    async def _anthropic_generate(self, system: str, user: str) -> str:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=self.api_key)

        response = await client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    async def _deepseek_generate(self, system: str, user: str) -> str:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1",
        )

        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=4000,
        )
        return response.choices[0].message.content


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response."""
    # Try to find JSON block
    import re

    # Look for ```json blocks
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    # Look for raw JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        text = json_match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


async def generate_retrieval_tasks(
    papers: list[Paper],
    llm: LLMClient,
    num_tasks: int = 100,
) -> list[GeneratedTask]:
    """
    Generate simple fact retrieval tasks.

    Args:
        papers: List of papers
        llm: LLM client
        num_tasks: Number of tasks to generate

    Returns:
        List of generated tasks
    """
    tasks = []
    prompts = GENERATION_PROMPTS["retrieval"]

    # Sample papers for generation
    sample_size = min(len(papers), num_tasks // 3 + 1)
    sampled_papers = random.sample(papers, sample_size)

    for paper in sampled_papers:
        user_prompt = prompts["user"].format(
            paper_title=paper.title,
            paper_id=paper.paper_id,
            paper_text=paper.full_text[:10000],
        )

        try:
            response = await llm.generate(prompts["system"], user_prompt)
            data = parse_json_response(response)

            for q in data.get("questions", []):
                task = GeneratedTask(
                    task_id=f"retrieval_{len(tasks):05d}",
                    task_type="retrieval",
                    question=q.get("question", ""),
                    golden_answer=q.get("answer", ""),
                    required_papers=[paper.paper_id],
                    metadata={
                        "difficulty": q.get("difficulty", "medium"),
                        "evidence_text": q.get("evidence_text", ""),
                    }
                )
                tasks.append(task)

                if len(tasks) >= num_tasks:
                    return tasks

        except Exception as e:
            console.print(f"[yellow]Error generating for {paper.paper_id}: {e}[/yellow]")

    return tasks


async def generate_multihop_tasks(
    papers: list[Paper],
    llm: LLMClient,
    num_tasks: int = 50,
) -> list[GeneratedTask]:
    """
    Generate multi-hop reasoning tasks requiring multiple papers.

    Args:
        papers: List of papers
        llm: LLM client
        num_tasks: Number of tasks to generate

    Returns:
        List of generated tasks
    """
    tasks = []
    prompts = GENERATION_PROMPTS["multihop"]

    # Generate pairs of papers
    pairs = []
    for i, paper_a in enumerate(papers):
        for paper_b in papers[i+1:]:
            pairs.append((paper_a, paper_b))

    random.shuffle(pairs)

    for paper_a, paper_b in pairs[:num_tasks]:
        user_prompt = prompts["user"].format(
            paper_a_title=paper_a.title,
            paper_a_id=paper_a.paper_id,
            paper_a_abstract=paper_a.abstract,
            paper_a_sections=", ".join(paper_a.sections[:5]),
            paper_b_title=paper_b.title,
            paper_b_id=paper_b.paper_id,
            paper_b_abstract=paper_b.abstract,
            paper_b_sections=", ".join(paper_b.sections[:5]),
        )

        try:
            response = await llm.generate(prompts["system"], user_prompt)
            data = parse_json_response(response)

            if data.get("question"):
                task = GeneratedTask(
                    task_id=f"multihop_{len(tasks):05d}",
                    task_type="multihop",
                    question=data.get("question", ""),
                    golden_answer=data.get("golden_answer", ""),
                    required_papers=[paper_a.paper_id, paper_b.paper_id],
                    metadata={
                        "reasoning_type": data.get("reasoning_type", "comparison"),
                        "reasoning_steps": data.get("reasoning_steps", []),
                    }
                )
                tasks.append(task)

        except Exception as e:
            console.print(f"[yellow]Error generating multihop: {e}[/yellow]")

        if len(tasks) >= num_tasks:
            break

    return tasks


async def generate_computation_tasks(
    papers: list[Paper],
    llm: LLMClient,
    num_tasks: int = 50,
) -> list[GeneratedTask]:
    """
    Generate tasks requiring Python computation.

    Args:
        papers: List of papers
        llm: LLM client
        num_tasks: Number of tasks to generate

    Returns:
        List of generated tasks
    """
    tasks = []
    prompts = GENERATION_PROMPTS["computation"]

    # Group papers for computation tasks (3-5 papers per task)
    for _ in range(num_tasks):
        sample_size = random.randint(3, min(5, len(papers)))
        sampled = random.sample(papers, sample_size)

        papers_info = "\n".join([
            f"- {p.paper_id}: {p.title}\n  Abstract: {p.abstract[:300]}..."
            for p in sampled
        ])

        user_prompt = prompts["user"].format(papers_with_metrics=papers_info)

        try:
            response = await llm.generate(prompts["system"], user_prompt)
            data = parse_json_response(response)

            if data.get("question"):
                task = GeneratedTask(
                    task_id=f"computation_{len(tasks):05d}",
                    task_type="computation",
                    question=data.get("question", ""),
                    golden_answer=data.get("golden_answer", ""),
                    required_papers=[p.paper_id for p in sampled],
                    metadata={
                        "requires_code": True,
                        "expected_code": data.get("expected_code", ""),
                        "extracted_values": data.get("extracted_values", {}),
                    }
                )
                tasks.append(task)

        except Exception as e:
            console.print(f"[yellow]Error generating computation: {e}[/yellow]")

    return tasks


async def generate_synthesis_tasks(
    papers: list[Paper],
    llm: LLMClient,
    num_tasks: int = 30,
) -> list[GeneratedTask]:
    """
    Generate literature synthesis tasks.

    Args:
        papers: List of papers
        llm: LLM client
        num_tasks: Number of tasks to generate

    Returns:
        List of generated tasks
    """
    tasks = []
    prompts = GENERATION_PROMPTS["synthesis"]

    # Define topics based on common themes
    topics = [
        "reinforcement learning from human feedback",
        "reward modeling",
        "language model alignment",
        "scaling laws",
        "chain of thought reasoning",
        "multi-agent systems",
        "tool use in LLMs",
        "instruction following",
    ]

    for topic in topics:
        # Find papers related to topic
        related = [p for p in papers if topic.lower() in p.abstract.lower()
                   or topic.lower() in p.title.lower()]

        if len(related) < 3:
            related = random.sample(papers, min(5, len(papers)))

        papers_list = "\n".join([
            f"- {p.paper_id}: {p.title}" for p in related[:5]
        ])

        user_prompt = prompts["user"].format(
            topic_name=topic,
            papers_list=papers_list,
        )

        try:
            response = await llm.generate(prompts["system"], user_prompt)
            data = parse_json_response(response)

            if data.get("question"):
                task = GeneratedTask(
                    task_id=f"synthesis_{len(tasks):05d}",
                    task_type="synthesis",
                    question=data.get("question", ""),
                    golden_answer=data.get("golden_answer", ""),
                    required_papers=[p.paper_id for p in related[:5]],
                    metadata={
                        "topic": topic,
                        "evaluation_criteria": data.get("evaluation_criteria", []),
                    }
                )
                tasks.append(task)

        except Exception as e:
            console.print(f"[yellow]Error generating synthesis: {e}[/yellow]")

        if len(tasks) >= num_tasks:
            break

    return tasks


async def generate_sft_demos(
    papers: list[Paper],
    llm: LLMClient,
    num_demos: int = 200,
) -> list[GeneratedTask]:
    """
    Generate SFT demonstrations with tool usage traces.

    Args:
        papers: List of papers
        llm: LLM client
        num_demos: Number of demos to generate

    Returns:
        List of generated demonstrations
    """
    tasks = []
    prompts = GENERATION_PROMPTS["sft_demo"]

    for paper in random.sample(papers, min(len(papers), num_demos)):
        # Generate a simple question first
        simple_q = f"What is the main contribution of paper {paper.paper_id}?"
        simple_a = f"Based on the abstract, the main contribution is: {paper.abstract[:500]}"

        user_prompt = prompts["user"].format(
            question=simple_q,
            answer=simple_a,
            paper_id=paper.paper_id,
        )

        try:
            response = await llm.generate(prompts["system"], user_prompt)
            data = parse_json_response(response)

            if data.get("turns"):
                task = GeneratedTask(
                    task_id=f"sft_{len(tasks):05d}",
                    task_type="sft_demo",
                    question=simple_q,
                    golden_answer=data.get("final_answer", simple_a),
                    required_papers=[paper.paper_id],
                    metadata={
                        "turns": data.get("turns", []),
                        "num_turns": len(data.get("turns", [])),
                    }
                )
                tasks.append(task)

        except Exception as e:
            console.print(f"[yellow]Error generating SFT demo: {e}[/yellow]")

        if len(tasks) >= num_demos:
            break

    return tasks


async def generate_dataset(
    library_path: Path,
    output_path: Path,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    distribution: Optional[dict] = None,
    total_samples: int = 10000,
):
    """
    Generate complete training dataset.

    Args:
        library_path: Path to paper library
        output_path: Output directory for datasets
        provider: LLM provider (openai, anthropic, deepseek)
        model: Model name
        api_key: API key
        distribution: Task type distribution
        total_samples: Total number of samples
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Default distribution
    if distribution is None:
        distribution = {
            "retrieval": 0.30,
            "multihop": 0.25,
            "computation": 0.20,
            "synthesis": 0.15,
            "sft_demo": 0.10,
        }

    # Load papers
    papers = load_papers(library_path)
    if not papers:
        console.print("[red]No papers loaded![/red]")
        return

    # Initialize LLM client
    llm = LLMClient(provider=provider, model=model, api_key=api_key)

    all_tasks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:

        # Generate each task type
        for task_type, ratio in distribution.items():
            num_samples = int(total_samples * ratio)
            task = progress.add_task(f"Generating {task_type}...", total=num_samples)

            if task_type == "retrieval":
                tasks = await generate_retrieval_tasks(papers, llm, num_samples)
            elif task_type == "multihop":
                tasks = await generate_multihop_tasks(papers, llm, num_samples)
            elif task_type == "computation":
                tasks = await generate_computation_tasks(papers, llm, num_samples)
            elif task_type == "synthesis":
                tasks = await generate_synthesis_tasks(papers, llm, num_samples)
            elif task_type == "sft_demo":
                tasks = await generate_sft_demos(papers, llm, num_samples)
            else:
                tasks = []

            all_tasks.extend(tasks)
            progress.update(task, completed=len(tasks))

    # Split into SFT and RL datasets
    sft_tasks = [t for t in all_tasks if t.task_type == "sft_demo"]
    rl_tasks = [t for t in all_tasks if t.task_type != "sft_demo"]

    # Save datasets
    def save_jsonl(tasks: list[GeneratedTask], path: Path):
        with open(path, 'w', encoding='utf-8') as f:
            for task in tasks:
                f.write(json.dumps(asdict(task), ensure_ascii=False) + '\n')

    save_jsonl(sft_tasks, output_path / "sft_train.jsonl")
    save_jsonl(rl_tasks, output_path / "rl_prompts.jsonl")

    # Save full dataset
    save_jsonl(all_tasks, output_path / "full_dataset.jsonl")

    # Save stats
    stats = {
        "total": len(all_tasks),
        "sft": len(sft_tasks),
        "rl": len(rl_tasks),
        "by_type": {t: len([x for x in all_tasks if x.task_type == t])
                    for t in distribution.keys()},
        "created_at": datetime.now().isoformat(),
    }
    (output_path / "stats.json").write_text(json.dumps(stats, indent=2))

    console.print(f"\n[green]Dataset generated successfully![/green]")
    console.print(f"  Total tasks: {len(all_tasks)}")
    console.print(f"  SFT demos: {len(sft_tasks)} -> {output_path / 'sft_train.jsonl'}")
    console.print(f"  RL prompts: {len(rl_tasks)} -> {output_path / 'rl_prompts.jsonl'}")


@app.command()
def generate(
    library_dir: Path = typer.Argument(..., help="Path to paper library"),
    output_dir: Path = typer.Argument(..., help="Output directory"),
    provider: str = typer.Option("openai", help="LLM provider"),
    model: Optional[str] = typer.Option(None, help="Model name"),
    total: int = typer.Option(10000, help="Total samples to generate"),
):
    """Generate synthetic training dataset from paper library."""
    import os
    api_key = os.getenv(f"{provider.upper()}_API_KEY")

    asyncio.run(generate_dataset(
        library_path=library_dir,
        output_path=output_dir,
        provider=provider,
        model=model,
        api_key=api_key,
        total_samples=total,
    ))


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
