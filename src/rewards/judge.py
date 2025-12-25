#!/usr/bin/env python3
"""
LLM-as-Judge for Answer Evaluation
Uses external LLM to evaluate factual accuracy.
"""

import os
import json
import asyncio
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class JudgeModel(Enum):
    """Supported judge models."""
    GPT4O = "gpt-4o"
    GPT4O_MINI = "gpt-4o-mini"
    CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
    DEEPSEEK_V3 = "deepseek-chat"


@dataclass
class JudgeRubric:
    """Rubric for judging answers."""

    # Evaluation criteria and weights
    accuracy: float = 0.4      # Factual correctness
    completeness: float = 0.3  # Covers all key points
    relevance: float = 0.2     # Stays on topic
    clarity: float = 0.1       # Well-structured answer

    # Prompt template
    system_prompt: str = """You are an expert evaluator for research synthesis tasks.
Your job is to compare an agent's answer against a reference answer and score it.

Evaluate on these criteria:
1. ACCURACY (40%): Are the facts correct? Any hallucinations?
2. COMPLETENESS (30%): Does it cover all key points from the reference?
3. RELEVANCE (20%): Does it stay on topic and answer the question?
4. CLARITY (10%): Is it well-organized and clear?

Output a JSON object with:
{
  "accuracy_score": 0.0-1.0,
  "completeness_score": 0.0-1.0,
  "relevance_score": 0.0-1.0,
  "clarity_score": 0.0-1.0,
  "total_score": weighted average,
  "reasoning": "brief explanation"
}"""

    user_template: str = """Question: {question}

Reference Answer:
{reference}

Agent's Answer:
{answer}

Evaluate the agent's answer against the reference. Output JSON only."""


class LLMJudge:
    """
    LLM-based judge for evaluating agent answers.
    """

    def __init__(
        self,
        model: JudgeModel = JudgeModel.GPT4O_MINI,
        rubric: JudgeRubric = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.rubric = rubric or JudgeRubric()
        self.api_key = api_key
        self.base_url = base_url

        self._client = None

    def _get_client(self):
        """Get or create API client."""
        if self._client is not None:
            return self._client

        if self.model in [JudgeModel.GPT4O, JudgeModel.GPT4O_MINI]:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or os.getenv("OPENAI_API_KEY"),
                base_url=self.base_url,
            )
        elif self.model == JudgeModel.CLAUDE_SONNET:
            from anthropic import Anthropic
            self._client = Anthropic(
                api_key=self.api_key or os.getenv("ANTHROPIC_API_KEY"),
            )
        elif self.model == JudgeModel.DEEPSEEK_V3:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key or os.getenv("DEEPSEEK_API_KEY"),
                base_url=self.base_url or "https://api.deepseek.com/v1",
            )

        return self._client

    async def judge(
        self,
        question: str,
        reference_answer: str,
        agent_answer: str,
        max_retries: int = 3,
    ) -> dict:
        """
        Judge agent's answer against reference.

        Args:
            question: Original question
            reference_answer: Ground truth answer
            agent_answer: Agent's generated answer
            max_retries: Number of retries on failure

        Returns:
            Dict with scores and reasoning
        """
        user_prompt = self.rubric.user_template.format(
            question=question,
            reference=reference_answer,
            answer=agent_answer,
        )

        for attempt in range(max_retries):
            try:
                response = await self._call_model(user_prompt)
                result = self._parse_response(response)
                return result
            except Exception as e:
                if attempt == max_retries - 1:
                    # Return neutral score on failure
                    return {
                        "accuracy_score": 0.5,
                        "completeness_score": 0.5,
                        "relevance_score": 0.5,
                        "clarity_score": 0.5,
                        "total_score": 0.5,
                        "reasoning": f"Judge failed: {e}",
                        "error": str(e),
                    }
                await asyncio.sleep(1)

    async def _call_model(self, user_prompt: str) -> str:
        """Call the judge model."""
        client = self._get_client()

        if self.model in [JudgeModel.GPT4O, JudgeModel.GPT4O_MINI, JudgeModel.DEEPSEEK_V3]:
            # OpenAI-compatible API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=self.model.value,
                    messages=[
                        {"role": "system", "content": self.rubric.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    response_format={"type": "json_object"},
                )
            )
            return response.choices[0].message.content

        elif self.model == JudgeModel.CLAUDE_SONNET:
            # Anthropic API
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.messages.create(
                    model=self.model.value,
                    max_tokens=1000,
                    system=self.rubric.system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
            )
            return response.content[0].text

    def _parse_response(self, response: str) -> dict:
        """Parse judge response into scores."""
        # Try to extract JSON
        try:
            # Handle potential markdown code blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            result = json.loads(response.strip())
        except json.JSONDecodeError:
            # Fallback: try to find JSON object
            import re
            match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if match:
                result = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse judge response: {response[:200]}")

        # Ensure all required fields
        defaults = {
            "accuracy_score": 0.5,
            "completeness_score": 0.5,
            "relevance_score": 0.5,
            "clarity_score": 0.5,
            "total_score": 0.5,
            "reasoning": "",
        }

        for key, default in defaults.items():
            if key not in result:
                result[key] = default

        # Recalculate total if needed
        if "total_score" not in result or result["total_score"] == 0.5:
            result["total_score"] = (
                self.rubric.accuracy * result["accuracy_score"] +
                self.rubric.completeness * result["completeness_score"] +
                self.rubric.relevance * result["relevance_score"] +
                self.rubric.clarity * result["clarity_score"]
            )

        return result

    async def batch_judge(
        self,
        samples: list[dict],
        max_concurrent: int = 5,
    ) -> list[dict]:
        """
        Judge multiple samples concurrently.

        Args:
            samples: List of dicts with 'question', 'reference', 'answer' keys
            max_concurrent: Maximum concurrent API calls

        Returns:
            List of judge results
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def judge_one(sample):
            async with semaphore:
                return await self.judge(
                    sample["question"],
                    sample["reference"],
                    sample["answer"],
                )

        tasks = [judge_one(s) for s in samples]
        return await asyncio.gather(*tasks)


# Convenience function for single-shot judging
async def judge_answer(
    question: str,
    reference_answer: str,
    agent_answer: str,
    model: str = "gpt-4o-mini",
) -> float:
    """
    Quick judge function returning total score.

    Args:
        question: Original question
        reference_answer: Ground truth
        agent_answer: Agent's answer
        model: Judge model name

    Returns:
        Total score 0-1
    """
    judge = LLMJudge(model=JudgeModel(model))
    result = await judge.judge(question, reference_answer, agent_answer)
    return result["total_score"]
