"""
LLM-as-Judge for evaluating agent responses.

Uses a powerful LLM to assess factual accuracy, relevance,
and quality of agent's answers compared to golden answers.
"""

import json
import re
from typing import Optional, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    score: float  # 0.0 to 1.0
    explanation: str
    criteria_scores: dict[str, float]


FACTUAL_JUDGE_PROMPT = """You are an expert evaluator assessing the factual accuracy of an AI assistant's response.

## Task
Compare the ASSISTANT'S ANSWER to the GOLDEN ANSWER and evaluate factual accuracy.

## Evaluation Criteria
1. **Key Facts** (40%): Are the main facts correct?
2. **Details** (30%): Are specific details (numbers, names, methods) accurate?
3. **Completeness** (20%): Does the answer cover all key points from golden?
4. **No Hallucinations** (10%): Is there any made-up information?

## Golden Answer
{golden_answer}

## Assistant's Answer
{answer}

## Additional Context
{context}

## Instructions
Provide your evaluation in the following JSON format:
```json
{{
    "key_facts_score": <0.0-1.0>,
    "details_score": <0.0-1.0>,
    "completeness_score": <0.0-1.0>,
    "no_hallucinations_score": <0.0-1.0>,
    "overall_score": <0.0-1.0>,
    "explanation": "<brief explanation>"
}}
```

Respond ONLY with the JSON, no other text."""


PROCESS_JUDGE_PROMPT = """You are evaluating the research process of an AI agent.

## Agent's Research Trajectory
{trajectory}

## Task
{task}

## Evaluation Criteria
1. **Efficiency** (40%): Did the agent find information efficiently?
2. **Strategy** (30%): Was the search strategy logical?
3. **Thoroughness** (20%): Did the agent verify information?
4. **No Redundancy** (10%): Were there unnecessary repeated actions?

## Instructions
Provide your evaluation in JSON format:
```json
{{
    "efficiency_score": <0.0-1.0>,
    "strategy_score": <0.0-1.0>,
    "thoroughness_score": <0.0-1.0>,
    "no_redundancy_score": <0.0-1.0>,
    "overall_score": <0.0-1.0>,
    "explanation": "<brief explanation>"
}}
```"""


class BaseLLMJudge(ABC):
    """Base class for LLM-based judges."""

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with a prompt."""
        pass

    def _parse_json_response(self, response: str) -> dict:
        """Extract JSON from LLM response."""
        # Try to find JSON in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"overall_score": 0.5, "explanation": "Failed to parse judge response"}


class LLMJudge(BaseLLMJudge):
    """
    LLM-as-Judge implementation.

    Supports multiple backends: OpenAI, Anthropic, local models.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        backend: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1
    ):
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self._client = None

    def _get_client(self):
        """Lazy initialization of API client."""
        if self._client is not None:
            return self._client

        if self.backend == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url
                )
            except ImportError:
                raise ImportError("openai package required: pip install openai")

        elif self.backend == "anthropic":
            try:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")

        elif self.backend == "vllm":
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key="EMPTY",
                    base_url=self.base_url or "http://localhost:8000/v1"
                )
            except ImportError:
                raise ImportError("openai package required for vLLM: pip install openai")

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self._client

    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt."""
        client = self._get_client()

        if self.backend in ["openai", "vllm"]:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1024
            )
            return response.choices[0].message.content

        elif self.backend == "anthropic":
            response = client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=1024
            )
            return response.content[0].text

        return ""

    def evaluate_factual(
        self,
        answer: str,
        golden_answer: str,
        metadata: Optional[dict] = None
    ) -> Tuple[float, str]:
        """
        Evaluate factual accuracy of answer.

        Returns:
            Tuple of (score, explanation)
        """
        context = ""
        if metadata:
            if "required_papers" in metadata:
                context += f"Required papers: {metadata['required_papers']}\n"
            if "task_type" in metadata:
                context += f"Task type: {metadata['task_type']}\n"

        prompt = FACTUAL_JUDGE_PROMPT.format(
            golden_answer=golden_answer,
            answer=answer,
            context=context or "None"
        )

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)

            score = result.get("overall_score", 0.5)
            explanation = result.get("explanation", "")

            return float(score), explanation

        except Exception as e:
            return 0.5, f"Judge error: {str(e)}"

    def evaluate_process(
        self,
        trajectory: list[dict],
        task: str
    ) -> Tuple[float, str]:
        """
        Evaluate research process quality.

        Returns:
            Tuple of (score, explanation)
        """
        # Format trajectory for prompt
        trajectory_str = self._format_trajectory(trajectory)

        prompt = PROCESS_JUDGE_PROMPT.format(
            trajectory=trajectory_str,
            task=task
        )

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)

            score = result.get("overall_score", 0.5)
            explanation = result.get("explanation", "")

            return float(score), explanation

        except Exception as e:
            return 0.5, f"Judge error: {str(e)}"

    def _format_trajectory(self, trajectory: list[dict], max_steps: int = 10) -> str:
        """Format trajectory for prompt."""
        lines = []
        for i, step in enumerate(trajectory[:max_steps]):
            action = step.get("action", "unknown")
            action_input = step.get("action_input", {})
            observation = step.get("observation", "")[:200]

            lines.append(f"Step {i+1}: {action}")
            lines.append(f"  Input: {json.dumps(action_input)[:100]}")
            lines.append(f"  Result: {observation}...")
            lines.append("")

        if len(trajectory) > max_steps:
            lines.append(f"... ({len(trajectory) - max_steps} more steps)")

        return '\n'.join(lines)


class RubricJudge(BaseLLMJudge):
    """
    Rubric-based judge with customizable criteria.

    Allows defining specific evaluation rubrics for different task types.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        backend: str = "openai",
        api_key: Optional[str] = None
    ):
        self.model = model
        self.backend = backend
        self.api_key = api_key
        self._llm_judge = LLMJudge(model, backend, api_key)

    def _call_llm(self, prompt: str) -> str:
        return self._llm_judge._call_llm(prompt)

    def evaluate_with_rubric(
        self,
        answer: str,
        rubric: dict[str, dict],
        context: Optional[str] = None
    ) -> JudgeResult:
        """
        Evaluate answer against a custom rubric.

        Args:
            answer: Agent's answer
            rubric: Dict mapping criterion name to {weight, description}
            context: Additional context for evaluation

        Returns:
            JudgeResult with scores per criterion
        """
        criteria_text = "\n".join([
            f"- **{name}** ({spec['weight']*100:.0f}%): {spec['description']}"
            for name, spec in rubric.items()
        ])

        prompt = f"""Evaluate this answer against the following rubric:

## Answer
{answer}

## Context
{context or 'None'}

## Rubric
{criteria_text}

## Instructions
Score each criterion from 0.0 to 1.0. Respond in JSON:
```json
{{
    "scores": {{
        "<criterion_name>": <score>,
        ...
    }},
    "explanation": "<brief explanation>"
}}
```"""

        try:
            response = self._call_llm(prompt)
            result = self._parse_json_response(response)

            scores = result.get("scores", {})

            # Calculate weighted total
            total = 0.0
            for name, spec in rubric.items():
                score = scores.get(name, 0.5)
                total += score * spec['weight']

            return JudgeResult(
                score=total,
                explanation=result.get("explanation", ""),
                criteria_scores=scores
            )

        except Exception as e:
            return JudgeResult(
                score=0.5,
                explanation=f"Evaluation error: {str(e)}",
                criteria_scores={}
            )


class MockJudge(BaseLLMJudge):
    """
    Mock judge for testing without API calls.

    Returns deterministic scores based on simple heuristics.
    """

    def _call_llm(self, prompt: str) -> str:
        return '{"overall_score": 0.7, "explanation": "Mock evaluation"}'

    def evaluate_factual(
        self,
        answer: str,
        golden_answer: str,
        metadata: Optional[dict] = None
    ) -> Tuple[float, str]:
        # Simple word overlap heuristic
        answer_words = set(answer.lower().split())
        golden_words = set(golden_answer.lower().split())

        if not golden_words:
            return 0.5, "Empty golden answer"

        overlap = len(answer_words & golden_words)
        score = min(overlap / len(golden_words), 1.0)

        return score, f"Word overlap: {overlap}/{len(golden_words)}"

    def evaluate_process(
        self,
        trajectory: list[dict],
        task: str
    ) -> Tuple[float, str]:
        # Score based on trajectory length
        n_steps = len(trajectory)
        if n_steps == 0:
            return 0.0, "Empty trajectory"
        if n_steps <= 5:
            return 0.9, "Efficient process"
        if n_steps <= 10:
            return 0.7, "Moderate process"
        return 0.5, "Long process"
