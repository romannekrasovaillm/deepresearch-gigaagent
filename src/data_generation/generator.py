"""
Synthetic dataset generator for Deep Research agent training.

Generates diverse training tasks from a library of scientific papers:
- Simple retrieval questions
- Multi-hop reasoning tasks
- Computation-required questions
- Full synthesis tasks
- SFT demonstrations
"""

import json
import random
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib

from .prompts import (
    GENERATION_PROMPTS,
    DEFAULT_DISTRIBUTION,
    CURRICULUM_LEVELS,
)


@dataclass
class GeneratedTask:
    """A single generated training task."""
    task_id: str
    task_type: str
    question: str
    golden_answer: str
    required_papers: list[str]
    requires_code: bool = False
    difficulty: str = "medium"
    curriculum_level: int = 1
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SyntheticDataGenerator:
    """
    Generator for synthetic training data.

    Uses a powerful LLM to generate diverse tasks from paper library.
    """

    def __init__(
        self,
        library_path: str | Path,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.library_path = Path(library_path)
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.base_url = base_url

        self._papers = []
        self._client = None
        self._load_papers()

    def _load_papers(self):
        """Load paper metadata from library index."""
        index_path = self.library_path / "index.json"
        if index_path.exists():
            self._papers = json.loads(index_path.read_text(encoding='utf-8'))
        else:
            print(f"Warning: No index found at {index_path}")

    def _get_client(self):
        """Get or create LLM client."""
        if self._client is not None:
            return self._client

        if self.llm_backend == "openai":
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        elif self.llm_backend == "anthropic":
            from anthropic import Anthropic
            self._client = Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unknown backend: {self.llm_backend}")

        return self._client

    def _call_llm(self, prompt: str, max_tokens: int = 2048) -> str:
        """Call LLM with prompt."""
        client = self._get_client()

        if self.llm_backend == "openai":
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        elif self.llm_backend == "anthropic":
            response = client.messages.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            return response.content[0].text

        return ""

    def _parse_json_response(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        # Try to find JSON block
        json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        return {}

    def _get_paper_text(self, paper_id: str, max_chars: int = 3000) -> str:
        """Get paper text for prompt."""
        paper_dir = self.library_path / "by_id" / paper_id

        # Try full text first
        full_text = paper_dir / "full_text.md"
        if full_text.exists():
            text = full_text.read_text(encoding='utf-8')
            return text[:max_chars]

        # Fallback to abstract + sections
        parts = []
        abstract = paper_dir / "abstract.txt"
        if abstract.exists():
            parts.append(abstract.read_text(encoding='utf-8'))

        sections_dir = paper_dir / "sections"
        if sections_dir.exists():
            for section_file in list(sections_dir.glob("*.txt"))[:3]:
                parts.append(section_file.read_text(encoding='utf-8')[:500])

        return '\n\n'.join(parts)[:max_chars]

    def _generate_task_id(self, task_type: str, content: str) -> str:
        """Generate unique task ID."""
        hash_input = f"{task_type}:{content}:{datetime.now().isoformat()}"
        return f"{task_type}_{hashlib.md5(hash_input.encode()).hexdigest()[:8]}"

    def generate_retrieval_task(self, paper: dict) -> list[GeneratedTask]:
        """Generate simple retrieval tasks from a paper."""
        paper_id = paper.get('paper_id')
        paper_title = paper.get('title', 'Untitled')
        paper_text = self._get_paper_text(paper_id)

        if not paper_text:
            return []

        prompt = GENERATION_PROMPTS["retrieval"].format(
            paper_title=paper_title,
            paper_id=paper_id,
            paper_text=paper_text
        )

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)

            tasks = []
            for q in result.get("questions", []):
                task = GeneratedTask(
                    task_id=self._generate_task_id("retrieval", q.get("question", "")),
                    task_type="retrieval",
                    question=q.get("question", ""),
                    golden_answer=q.get("answer", ""),
                    required_papers=[paper_id],
                    difficulty=q.get("difficulty", "medium"),
                    curriculum_level=1,
                    metadata={
                        "evidence_quote": q.get("evidence_quote", ""),
                        "source_paper_title": paper_title,
                    }
                )
                if task.question and task.golden_answer:
                    tasks.append(task)

            return tasks

        except Exception as e:
            print(f"Error generating retrieval task: {e}")
            return []

    def generate_multihop_task(
        self,
        paper_a: dict,
        paper_b: dict
    ) -> Optional[GeneratedTask]:
        """Generate multi-hop comparison task from two papers."""
        paper_a_id = paper_a.get('paper_id')
        paper_b_id = paper_b.get('paper_id')

        paper_a_text = self._get_paper_text(paper_a_id, max_chars=1500)
        paper_b_text = self._get_paper_text(paper_b_id, max_chars=1500)

        if not paper_a_text or not paper_b_text:
            return None

        prompt = GENERATION_PROMPTS["multihop"].format(
            paper_a_title=paper_a.get('title', ''),
            paper_a_id=paper_a_id,
            paper_a_abstract=paper_a.get('abstract_preview', ''),
            paper_a_excerpt=paper_a_text[:800],
            paper_b_title=paper_b.get('title', ''),
            paper_b_id=paper_b_id,
            paper_b_abstract=paper_b.get('abstract_preview', ''),
            paper_b_excerpt=paper_b_text[:800],
        )

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)

            if not result.get("question"):
                return None

            return GeneratedTask(
                task_id=self._generate_task_id("multihop", result["question"]),
                task_type="multihop",
                question=result["question"],
                golden_answer=result.get("golden_answer", ""),
                required_papers=[paper_a_id, paper_b_id],
                difficulty="hard",
                curriculum_level=4,
                metadata={
                    "reasoning_type": result.get("reasoning_type", "comparison"),
                    "key_points": result.get("key_points", []),
                }
            )

        except Exception as e:
            print(f"Error generating multihop task: {e}")
            return None

    def generate_computation_task(
        self,
        papers: list[dict]
    ) -> Optional[GeneratedTask]:
        """Generate task requiring code computation."""
        # Format papers with metrics
        papers_info = []
        for p in papers[:5]:
            paper_id = p.get('paper_id')
            text = self._get_paper_text(paper_id, max_chars=800)
            papers_info.append(f"Paper: {p.get('title')}\nID: {paper_id}\nExcerpt: {text[:400]}")

        prompt = GENERATION_PROMPTS["computation"].format(
            papers_with_metrics='\n\n---\n\n'.join(papers_info)
        )

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)

            if not result.get("question"):
                return None

            return GeneratedTask(
                task_id=self._generate_task_id("computation", result["question"]),
                task_type="computation",
                question=result["question"],
                golden_answer=result.get("golden_answer", ""),
                required_papers=result.get("required_papers", [p.get('paper_id') for p in papers[:3]]),
                requires_code=True,
                difficulty="hard",
                curriculum_level=3,
                metadata={
                    "expected_code": result.get("expected_code", ""),
                    "extracted_values": result.get("extracted_values", {}),
                }
            )

        except Exception as e:
            print(f"Error generating computation task: {e}")
            return None

    def generate_synthesis_task(
        self,
        topic: str,
        papers: list[dict]
    ) -> Optional[GeneratedTask]:
        """Generate full synthesis task for a topic."""
        papers_list = '\n'.join([
            f"- {p.get('paper_id')}: {p.get('title')}"
            for p in papers[:5]
        ])

        prompt = GENERATION_PROMPTS["synthesis"].format(
            topic_name=topic,
            papers_list=papers_list
        )

        try:
            response = self._call_llm(prompt, max_tokens=3000)
            result = self._parse_json_response(response)

            if not result.get("question"):
                return None

            return GeneratedTask(
                task_id=self._generate_task_id("synthesis", result["question"]),
                task_type="synthesis",
                question=result["question"],
                golden_answer=result.get("golden_answer", ""),
                required_papers=result.get("required_papers", [p.get('paper_id') for p in papers]),
                difficulty="expert",
                curriculum_level=6,
                metadata={
                    "evaluation_criteria": result.get("evaluation_criteria", []),
                    "key_themes": result.get("key_themes", []),
                    "topic": topic,
                }
            )

        except Exception as e:
            print(f"Error generating synthesis task: {e}")
            return None

    def generate_sft_demonstration(
        self,
        task: GeneratedTask
    ) -> Optional[dict]:
        """Generate SFT demonstration for a task."""
        # Get relevant paper info
        paper_id = task.required_papers[0] if task.required_papers else None
        if not paper_id:
            return None

        paper = next((p for p in self._papers if p.get('paper_id') == paper_id), None)
        if not paper:
            return None

        prompt = GENERATION_PROMPTS["sft_demo"].format(
            question=task.question,
            answer=task.golden_answer,
            paper_id=paper_id,
            relevant_section="methods"  # Default
        )

        try:
            response = self._call_llm(prompt, max_tokens=3000)
            result = self._parse_json_response(response)

            if not result.get("turns"):
                return None

            return {
                "task_id": f"sft_{task.task_id}",
                "task": task.question,
                "turns": result["turns"],
                "final_answer": result.get("final_answer", task.golden_answer),
                "source_task_id": task.task_id,
            }

        except Exception as e:
            print(f"Error generating SFT demo: {e}")
            return None

    def generate_dataset(
        self,
        n_samples: int = 1000,
        distribution: Optional[dict] = None,
        output_dir: Optional[Path] = None,
        include_sft: bool = True
    ) -> tuple[list[GeneratedTask], list[dict]]:
        """
        Generate complete training dataset.

        Args:
            n_samples: Total number of RL prompts to generate
            distribution: Task type distribution (default: DEFAULT_DISTRIBUTION)
            output_dir: Directory to save generated data
            include_sft: Whether to generate SFT demonstrations

        Returns:
            Tuple of (RL tasks, SFT demonstrations)
        """
        if distribution is None:
            distribution = DEFAULT_DISTRIBUTION

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        rl_tasks = []
        sft_demos = []

        # Calculate samples per type
        samples_per_type = {
            task_type: int(n_samples * ratio)
            for task_type, ratio in distribution.items()
            if task_type != "sft_demo"
        }

        print(f"Generating {n_samples} samples...")
        print(f"Distribution: {samples_per_type}")

        # Generate retrieval tasks
        n_retrieval = samples_per_type.get("retrieval", 0)
        if n_retrieval > 0:
            print(f"Generating {n_retrieval} retrieval tasks...")
            papers_sample = random.sample(
                self._papers,
                min(n_retrieval // 3 + 1, len(self._papers))
            )
            for paper in papers_sample:
                tasks = self.generate_retrieval_task(paper)
                rl_tasks.extend(tasks)
                if len(rl_tasks) >= n_retrieval:
                    break

        # Generate multihop tasks
        n_multihop = samples_per_type.get("multihop", 0)
        if n_multihop > 0 and len(self._papers) >= 2:
            print(f"Generating {n_multihop} multihop tasks...")
            for _ in range(n_multihop):
                paper_a, paper_b = random.sample(self._papers, 2)
                task = self.generate_multihop_task(paper_a, paper_b)
                if task:
                    rl_tasks.append(task)

        # Generate computation tasks
        n_computation = samples_per_type.get("computation", 0)
        if n_computation > 0 and len(self._papers) >= 3:
            print(f"Generating {n_computation} computation tasks...")
            for _ in range(n_computation):
                papers = random.sample(self._papers, min(5, len(self._papers)))
                task = self.generate_computation_task(papers)
                if task:
                    rl_tasks.append(task)

        # Generate synthesis tasks
        n_synthesis = samples_per_type.get("synthesis", 0)
        if n_synthesis > 0:
            print(f"Generating {n_synthesis} synthesis tasks...")
            # Extract topics from titles
            topics = self._extract_topics()
            for topic in topics[:n_synthesis]:
                related_papers = [
                    p for p in self._papers
                    if topic.lower() in p.get('title', '').lower()
                    or topic.lower() in p.get('abstract_preview', '').lower()
                ][:5]
                if len(related_papers) >= 3:
                    task = self.generate_synthesis_task(topic, related_papers)
                    if task:
                        rl_tasks.append(task)

        # Generate SFT demonstrations
        if include_sft:
            n_sft = int(n_samples * distribution.get("sft_demo", 0.1))
            print(f"Generating {n_sft} SFT demonstrations...")

            # Use subset of RL tasks for SFT demos
            sft_source_tasks = random.sample(
                rl_tasks,
                min(n_sft, len(rl_tasks))
            )
            for task in sft_source_tasks:
                demo = self.generate_sft_demonstration(task)
                if demo:
                    sft_demos.append(demo)

        # Save to files
        if output_dir:
            # Save RL prompts
            rl_path = output_dir / "rl_prompts.jsonl"
            with open(rl_path, 'w', encoding='utf-8') as f:
                for task in rl_tasks:
                    f.write(json.dumps(asdict(task), ensure_ascii=False) + '\n')
            print(f"Saved {len(rl_tasks)} RL prompts to {rl_path}")

            # Save SFT demonstrations
            sft_path = output_dir / "sft_train.jsonl"
            with open(sft_path, 'w', encoding='utf-8') as f:
                for demo in sft_demos:
                    f.write(json.dumps(demo, ensure_ascii=False) + '\n')
            print(f"Saved {len(sft_demos)} SFT demos to {sft_path}")

            # Save metadata
            meta_path = output_dir / "dataset_meta.json"
            meta_path.write_text(json.dumps({
                "generated_at": datetime.now().isoformat(),
                "n_rl_prompts": len(rl_tasks),
                "n_sft_demos": len(sft_demos),
                "distribution": distribution,
                "library_path": str(self.library_path),
                "n_papers": len(self._papers),
            }, indent=2))

        print(f"Done! Generated {len(rl_tasks)} RL tasks, {len(sft_demos)} SFT demos")
        return rl_tasks, sft_demos

    def _extract_topics(self) -> list[str]:
        """Extract common topics from paper titles."""
        # Simple keyword extraction
        keywords = {}
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'for', 'to', 'with',
            'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'using',
            'via', 'through', 'towards', 'toward', 'based', 'learning', 'model',
            'models', 'method', 'methods', 'approach', 'approaches',
        }

        for paper in self._papers:
            title = paper.get('title', '')
            words = re.findall(r'\b[a-zA-Z]{4,}\b', title.lower())
            for word in words:
                if word not in stop_words:
                    keywords[word] = keywords.get(word, 0) + 1

        # Return most common keywords
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return [kw for kw, count in sorted_keywords[:20] if count >= 2]


def generate_dataset_cli():
    """CLI entry point for dataset generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic training data")
    parser.add_argument("library_path", help="Path to paper library")
    parser.add_argument("output_dir", help="Output directory for generated data")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--backend", default="openai", help="LLM backend (openai/anthropic)")
    parser.add_argument("--model", default="gpt-4o", help="LLM model name")
    parser.add_argument("--no-sft", action="store_true", help="Skip SFT demonstration generation")

    args = parser.parse_args()

    generator = SyntheticDataGenerator(
        library_path=args.library_path,
        llm_backend=args.backend,
        llm_model=args.model,
    )

    generator.generate_dataset(
        n_samples=args.n_samples,
        output_dir=Path(args.output_dir),
        include_sft=not args.no_sft,
    )


if __name__ == "__main__":
    generate_dataset_cli()
