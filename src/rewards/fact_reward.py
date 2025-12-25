"""R_fact: Factual accuracy reward using LLM-as-Judge."""

import json
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None


@dataclass
class FactScore:
    """Factual accuracy scores."""

    accuracy: float
    completeness: float
    relevance: float
    total: float
    reasoning: str = ""


JUDGE_PROMPT = """You are evaluating a research agent's answer for factual accuracy.

Question: {question}

Golden Answer (reference): {golden_answer}

Agent's Answer: {agent_answer}

Evaluate the agent's answer on three criteria:
1. Accuracy (0-1): Is the information factually correct compared to the reference?
2. Completeness (0-1): Does it address all key aspects from the reference?
3. Relevance (0-1): Is it directly relevant to the question asked?

Respond in JSON format:
{{
    "accuracy": <float 0-1>,
    "completeness": <float 0-1>,
    "relevance": <float 0-1>,
    "reasoning": "<brief explanation>"
}}"""


class FactReward:
    """LLM-as-Judge for factual accuracy."""

    def __init__(
        self,
        judge_model: str = "gpt-4o",
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    ):
        self.judge_model = judge_model
        self.weights = weights

        # Initialize client based on model
        if "gpt" in judge_model.lower() or "o1" in judge_model.lower():
            if OpenAI is None:
                raise ImportError("openai package required for GPT judge")
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.client_type = "openai"
        elif "claude" in judge_model.lower():
            if Anthropic is None:
                raise ImportError("anthropic package required for Claude judge")
            self.client = Anthropic(api_key=api_key)
            self.client_type = "anthropic"
        else:
            # Assume OpenAI-compatible API
            if OpenAI is None:
                raise ImportError("openai package required")
            self.client = OpenAI(api_key=api_key, base_url=api_base)
            self.client_type = "openai"

    def _call_judge(self, prompt: str) -> str:
        """Call the judge model."""
        if self.client_type == "openai":
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            return response.choices[0].message.content

        elif self.client_type == "anthropic":
            response = self.client.messages.create(
                model=self.judge_model,
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        return ""

    def compute(
        self,
        question: str,
        golden_answer: str,
        agent_answer: str,
    ) -> FactScore:
        """
        Compute factual accuracy reward.

        Args:
            question: Original research question
            golden_answer: Reference answer
            agent_answer: Agent's answer to evaluate

        Returns:
            FactScore with accuracy, completeness, relevance
        """
        prompt = JUDGE_PROMPT.format(
            question=question,
            golden_answer=golden_answer,
            agent_answer=agent_answer,
        )

        try:
            response = self._call_judge(prompt)

            # Parse JSON from response
            json_match = response
            if "```json" in response:
                json_match = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_match = response.split("```")[1].split("```")[0]

            data = json.loads(json_match)

            accuracy = float(data.get("accuracy", 0))
            completeness = float(data.get("completeness", 0))
            relevance = float(data.get("relevance", 0))
            reasoning = data.get("reasoning", "")

            # Weighted total
            total = (
                self.weights[0] * accuracy
                + self.weights[1] * completeness
                + self.weights[2] * relevance
            )

            return FactScore(
                accuracy=accuracy,
                completeness=completeness,
                relevance=relevance,
                total=total,
                reasoning=reasoning,
            )

        except Exception as e:
            # Return zero scores on error
            return FactScore(
                accuracy=0,
                completeness=0,
                relevance=0,
                total=0,
                reasoning=f"Judge error: {e}",
            )


def compute_fact_reward(
    question: str,
    golden_answer: str,
    agent_answer: str,
    judge_model: str = "gpt-4o",
) -> float:
    """Convenience function for fact reward."""
    reward = FactReward(judge_model=judge_model)
    score = reward.compute(question, golden_answer, agent_answer)
    return score.total
