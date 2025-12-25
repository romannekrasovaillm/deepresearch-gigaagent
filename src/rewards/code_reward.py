"""R_code: Code reasoning quality reward."""

import json
from dataclasses import dataclass
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


@dataclass
class CodeScore:
    """Code reasoning scores."""

    execution_success: int
    execution_failure: int
    relevance_score: float
    result_used: bool
    visualization_bonus: float
    statistical_bonus: float
    total: float
    details: str = ""


RELEVANCE_PROMPT = """Evaluate if this code execution was relevant and useful for answering the research question.

Question: {question}

Code executed:
```python
{code}
```

Code output:
{output}

Agent's final answer (excerpt):
{answer_excerpt}

Evaluate:
1. Was the code relevant to answering the question? (0-1)
2. Was the output actually used in the final answer? (true/false)
3. Did it provide statistical analysis? (true/false)
4. Did it create useful visualization? (true/false)

Respond in JSON:
{{
    "relevance": <float 0-1>,
    "result_used": <bool>,
    "statistical_analysis": <bool>,
    "visualization": <bool>
}}"""


class CodeReward:
    """Reward for code reasoning quality."""

    def __init__(
        self,
        judge_model: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        execution_success_reward: float = 0.2,
        execution_failure_penalty: float = -0.1,
        result_used_reward: float = 0.2,
        result_ignored_penalty: float = -0.05,
        visualization_bonus: float = 0.1,
        statistical_bonus: float = 0.15,
    ):
        self.judge_model = judge_model
        self.execution_success_reward = execution_success_reward
        self.execution_failure_penalty = execution_failure_penalty
        self.result_used_reward = result_used_reward
        self.result_ignored_penalty = result_ignored_penalty
        self.visualization_bonus = visualization_bonus
        self.statistical_bonus = statistical_bonus

        if judge_model and OpenAI:
            self.client = OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.client = None

    def _call_judge(self, prompt: str) -> dict:
        """Call judge model for relevance evaluation."""
        if self.client is None:
            # Default to assuming relevance
            return {
                "relevance": 0.5,
                "result_used": True,
                "statistical_analysis": False,
                "visualization": False,
            }

        try:
            response = self.client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=200,
            )
            content = response.choices[0].message.content

            # Parse JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content)

        except Exception:
            return {
                "relevance": 0.5,
                "result_used": True,
                "statistical_analysis": False,
                "visualization": False,
            }

    def compute(
        self,
        question: str,
        code_executions: list[dict],
        agent_answer: str,
    ) -> CodeScore:
        """
        Compute code reasoning reward.

        Args:
            question: Research question
            code_executions: List of {"code": str, "result": str, "success": bool}
            agent_answer: Agent's final answer

        Returns:
            CodeScore with breakdown
        """
        details = []

        if not code_executions:
            return CodeScore(
                execution_success=0,
                execution_failure=0,
                relevance_score=0,
                result_used=False,
                visualization_bonus=0,
                statistical_bonus=0,
                total=0,
                details="No code executed",
            )

        # Count successes and failures
        execution_success = sum(1 for e in code_executions if e.get("success", True))
        execution_failure = len(code_executions) - execution_success

        if execution_failure > 0:
            details.append(f"{execution_failure} code execution(s) failed")

        # Evaluate relevance with judge
        relevance_scores = []
        result_used = False
        has_visualization = False
        has_statistical = False

        for execution in code_executions:
            code = execution.get("code", "")
            result = execution.get("result", "")

            # Check for visualization patterns
            if any(
                pattern in code.lower()
                for pattern in ["plt.", "matplotlib", ".plot(", ".savefig("]
            ):
                has_visualization = True

            # Check for statistical patterns
            if any(
                pattern in code.lower()
                for pattern in [
                    "scipy.stats",
                    "ttest",
                    "chi2",
                    ".mean()",
                    ".std()",
                    "correlation",
                    "regression",
                ]
            ):
                has_statistical = True

            # Call judge for relevance
            if self.client:
                prompt = RELEVANCE_PROMPT.format(
                    question=question,
                    code=code[:500],
                    output=result[:300],
                    answer_excerpt=agent_answer[:500],
                )
                eval_result = self._call_judge(prompt)
                relevance_scores.append(eval_result.get("relevance", 0.5))
                if eval_result.get("result_used"):
                    result_used = True

        # Calculate scores
        avg_relevance = (
            sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.5
        )

        visualization_bonus = self.visualization_bonus if has_visualization else 0
        statistical_bonus = self.statistical_bonus if has_statistical else 0

        if has_visualization:
            details.append("Created visualization")
        if has_statistical:
            details.append("Performed statistical analysis")

        # Calculate total
        total = (
            execution_success * self.execution_success_reward
            + execution_failure * self.execution_failure_penalty
            + avg_relevance * 0.3  # Relevance weight
            + (self.result_used_reward if result_used else self.result_ignored_penalty)
            + visualization_bonus
            + statistical_bonus
        )

        return CodeScore(
            execution_success=execution_success,
            execution_failure=execution_failure,
            relevance_score=avg_relevance,
            result_used=result_used,
            visualization_bonus=visualization_bonus,
            statistical_bonus=statistical_bonus,
            total=total,
            details="; ".join(details) if details else "Code executed successfully",
        )


def compute_code_reward(
    question: str,
    code_executions: list[dict],
    agent_answer: str,
    judge_model: Optional[str] = None,
) -> float:
    """Convenience function for code reward."""
    reward = CodeReward(judge_model=judge_model)
    score = reward.compute(question, code_executions, agent_answer)
    return score.total
