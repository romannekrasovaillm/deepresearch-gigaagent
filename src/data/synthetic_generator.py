"""Synthetic dataset generator for research agent training."""

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .prompts import (
    RETRIEVAL_PROMPT,
    MULTIHOP_PROMPT,
    COMPUTATION_PROMPT,
    SYNTHESIS_PROMPT,
    SFT_DEMO_PROMPT,
    VISUALIZATION_PROMPT,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


console = Console()


class TaskType(Enum):
    """Types of training tasks."""

    RETRIEVAL = "retrieval"
    MULTIHOP = "multihop"
    COMPUTATION = "computation"
    SYNTHESIS = "synthesis"
    VISUALIZATION = "visualization"
    SFT_DEMO = "sft_demo"


@dataclass
class GeneratedTask:
    """A generated training task."""

    task_type: TaskType
    question: str
    golden_answer: str
    required_papers: list[str]
    difficulty: str = "medium"
    requires_code: bool = False
    requires_visualization: bool = False
    expected_code: Optional[str] = None
    sft_trace: Optional[list[dict]] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.task_type.value,
            "question": self.question,
            "golden_answer": self.golden_answer,
            "required_papers": self.required_papers,
            "difficulty": self.difficulty,
            "requires_code": self.requires_code,
            "requires_visualization": self.requires_visualization,
            "expected_code": self.expected_code,
            "sft_trace": self.sft_trace,
            "metadata": self.metadata,
        }


class SyntheticGenerator:
    """Generate synthetic training tasks using LLM."""

    DEFAULT_DISTRIBUTION = {
        TaskType.RETRIEVAL: 0.30,
        TaskType.MULTIHOP: 0.25,
        TaskType.COMPUTATION: 0.20,
        TaskType.SYNTHESIS: 0.15,
        TaskType.VISUALIZATION: 0.05,
        TaskType.SFT_DEMO: 0.05,
    }

    def __init__(
        self,
        library_path: str | Path,
        generator_model: str = "gpt-4o",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.library_path = Path(library_path)
        self.generator_model = generator_model

        # Load paper index
        self.papers = self._load_papers()

        # Initialize LLM client
        if "gpt" in generator_model.lower() or "o1" in generator_model.lower():
            if OpenAI is None:
                raise ImportError("openai package required")
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.client_type = "openai"
        elif "claude" in generator_model.lower():
            if Anthropic is None:
                raise ImportError("anthropic package required")
            self.client = Anthropic(api_key=api_key)
            self.client_type = "anthropic"
        else:
            # Assume OpenAI-compatible
            if OpenAI is None:
                raise ImportError("openai package required")
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.client_type = "openai"

    def _load_papers(self) -> list[dict]:
        """Load paper index."""
        index_path = self.library_path / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Library index not found: {index_path}")
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _get_paper_content(self, paper_id: str, max_chars: int = 5000) -> str:
        """Get paper content for prompt."""
        paper_path = self.library_path / "by_id" / paper_id / "full_text.md"
        if not paper_path.exists():
            return ""
        content = paper_path.read_text(encoding="utf-8")
        return content[:max_chars]

    def _call_llm(self, prompt: str) -> str:
        """Call LLM for generation."""
        if self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.generator_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            return response.choices[0].message.content

        elif self.client_type == "anthropic":
            response = self.client.messages.create(
                model=self.generator_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return ""

    def _parse_json_response(self, response: str) -> dict:
        """Parse JSON from LLM response."""
        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]

        return json.loads(response.strip())

    def generate_retrieval(self, paper: dict) -> list[GeneratedTask]:
        """Generate retrieval tasks from a paper."""
        paper_id = paper["paper_id"]
        content = self._get_paper_content(paper_id)

        if not content:
            return []

        prompt = RETRIEVAL_PROMPT.format(
            paper_id=paper_id,
            paper_title=paper.get("title", "Unknown"),
            paper_content=content[:4000],
        )

        try:
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)

            tasks = []
            for q in data.get("questions", []):
                tasks.append(
                    GeneratedTask(
                        task_type=TaskType.RETRIEVAL,
                        question=q["question"],
                        golden_answer=q["answer"],
                        required_papers=[paper_id],
                        difficulty=q.get("difficulty", "medium"),
                        metadata={"section_hint": q.get("section_hint")},
                    )
                )
            return tasks

        except Exception as e:
            console.print(f"[yellow]Retrieval generation failed: {e}[/yellow]")
            return []

    def generate_multihop(
        self, paper_a: dict, paper_b: dict
    ) -> Optional[GeneratedTask]:
        """Generate multi-hop comparison task."""
        prompt = MULTIHOP_PROMPT.format(
            paper_a_id=paper_a["paper_id"],
            paper_a_title=paper_a.get("title", ""),
            paper_a_abstract=paper_a.get("abstract", "")[:1000],
            paper_b_id=paper_b["paper_id"],
            paper_b_title=paper_b.get("title", ""),
            paper_b_abstract=paper_b.get("abstract", "")[:1000],
        )

        try:
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)

            return GeneratedTask(
                task_type=TaskType.MULTIHOP,
                question=data["question"],
                golden_answer=data["golden_answer"],
                required_papers=data["required_papers"],
                difficulty=data.get("difficulty", "medium"),
                metadata={"reasoning_type": data.get("reasoning_type")},
            )

        except Exception as e:
            console.print(f"[yellow]Multihop generation failed: {e}[/yellow]")
            return None

    def generate_computation(self, papers: list[dict]) -> Optional[GeneratedTask]:
        """Generate computation task requiring Python."""
        papers_info = "\n".join(
            f"- {p['paper_id']}: {p.get('title', '')[:50]}"
            for p in papers[:5]
        )

        prompt = COMPUTATION_PROMPT.format(papers_with_metrics=papers_info)

        try:
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)

            return GeneratedTask(
                task_type=TaskType.COMPUTATION,
                question=data["question"],
                golden_answer=data["golden_answer"],
                required_papers=data["required_papers"],
                requires_code=True,
                expected_code=data.get("expected_code"),
                metadata={"computation_type": data.get("computation_type")},
            )

        except Exception as e:
            console.print(f"[yellow]Computation generation failed: {e}[/yellow]")
            return None

    def generate_synthesis(
        self, topic: str, papers: list[dict]
    ) -> Optional[GeneratedTask]:
        """Generate synthesis task."""
        papers_info = "\n".join(
            f"- {p['paper_id']}: {p.get('title', '')[:60]}"
            for p in papers[:6]
        )

        prompt = SYNTHESIS_PROMPT.format(
            topic_name=topic,
            papers_on_topic=papers_info,
        )

        try:
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)

            return GeneratedTask(
                task_type=TaskType.SYNTHESIS,
                question=data["question"],
                golden_answer=data["golden_answer"],
                required_papers=data["required_papers"],
                difficulty="hard",
                metadata={"evaluation_criteria": data.get("evaluation_criteria", [])},
            )

        except Exception as e:
            console.print(f"[yellow]Synthesis generation failed: {e}[/yellow]")
            return None

    def generate_sft_demo(
        self, question: str, answer: str, paper_id: str
    ) -> Optional[GeneratedTask]:
        """Generate SFT demonstration trace."""
        prompt = SFT_DEMO_PROMPT.format(
            question=question,
            answer=answer,
            paper_id=paper_id,
        )

        try:
            response = self._call_llm(prompt)
            data = self._parse_json_response(response)

            return GeneratedTask(
                task_type=TaskType.SFT_DEMO,
                question=question,
                golden_answer=data.get("final_answer", answer),
                required_papers=[paper_id],
                sft_trace=data.get("turns", []),
            )

        except Exception as e:
            console.print(f"[yellow]SFT demo generation failed: {e}[/yellow]")
            return None

    def generate_dataset(
        self,
        n_samples: int = 1000,
        distribution: Optional[dict[TaskType, float]] = None,
    ) -> Iterator[GeneratedTask]:
        """Generate full dataset with specified distribution."""
        dist = distribution or self.DEFAULT_DISTRIBUTION

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Generating tasks...", total=n_samples)
            generated = 0

            while generated < n_samples:
                # Sample task type
                task_type = random.choices(
                    list(dist.keys()),
                    weights=list(dist.values()),
                )[0]

                progress.update(task, description=f"Generating {task_type.value}...")

                try:
                    if task_type == TaskType.RETRIEVAL:
                        paper = random.choice(self.papers)
                        tasks = self.generate_retrieval(paper)
                        for t in tasks[:1]:  # One per paper
                            yield t
                            generated += 1
                            progress.advance(task)

                    elif task_type == TaskType.MULTIHOP:
                        if len(self.papers) >= 2:
                            pair = random.sample(self.papers, 2)
                            result = self.generate_multihop(pair[0], pair[1])
                            if result:
                                yield result
                                generated += 1
                                progress.advance(task)

                    elif task_type == TaskType.COMPUTATION:
                        if len(self.papers) >= 3:
                            sample = random.sample(
                                self.papers, min(5, len(self.papers))
                            )
                            result = self.generate_computation(sample)
                            if result:
                                yield result
                                generated += 1
                                progress.advance(task)

                    elif task_type == TaskType.SYNTHESIS:
                        # Use random topic from first paper's title
                        paper = random.choice(self.papers)
                        topic = paper.get("title", "research").split()[0]
                        sample = random.sample(
                            self.papers, min(5, len(self.papers))
                        )
                        result = self.generate_synthesis(topic, sample)
                        if result:
                            yield result
                            generated += 1
                            progress.advance(task)

                    elif task_type == TaskType.SFT_DEMO:
                        # First generate a retrieval task, then demo
                        paper = random.choice(self.papers)
                        retrieval_tasks = self.generate_retrieval(paper)
                        if retrieval_tasks:
                            rt = retrieval_tasks[0]
                            result = self.generate_sft_demo(
                                rt.question,
                                rt.golden_answer,
                                paper["paper_id"],
                            )
                            if result:
                                yield result
                                generated += 1
                                progress.advance(task)

                except Exception as e:
                    console.print(f"[red]Generation error: {e}[/red]")
                    continue

        console.print(f"[green]Generated {generated} tasks[/green]")


def generate_dataset(
    library_path: str | Path,
    output_path: str | Path,
    n_samples: int = 1000,
    generator_model: str = "gpt-4o",
    split_sft_rl: bool = True,
) -> None:
    """
    Generate and save synthetic dataset.

    Args:
        library_path: Path to paper library
        output_path: Output directory for datasets
        n_samples: Number of samples to generate
        generator_model: LLM to use for generation
        split_sft_rl: Whether to split into SFT and RL datasets
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    generator = SyntheticGenerator(
        library_path=library_path,
        generator_model=generator_model,
    )

    all_tasks = list(generator.generate_dataset(n_samples))

    if split_sft_rl:
        sft_tasks = [t for t in all_tasks if t.task_type == TaskType.SFT_DEMO]
        rl_tasks = [t for t in all_tasks if t.task_type != TaskType.SFT_DEMO]

        # Save SFT dataset
        sft_path = output_path / "sft_train.jsonl"
        with open(sft_path, "w", encoding="utf-8") as f:
            for task in sft_tasks:
                f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
        console.print(f"[green]Saved {len(sft_tasks)} SFT samples to {sft_path}[/green]")

        # Save RL dataset
        rl_path = output_path / "rl_prompts.jsonl"
        with open(rl_path, "w", encoding="utf-8") as f:
            for task in rl_tasks:
                f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
        console.print(f"[green]Saved {len(rl_tasks)} RL samples to {rl_path}[/green]")

    else:
        # Save all to single file
        all_path = output_path / "dataset.jsonl"
        with open(all_path, "w", encoding="utf-8") as f:
            for task in all_tasks:
                f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
        console.print(f"[green]Saved {len(all_tasks)} samples to {all_path}[/green]")
