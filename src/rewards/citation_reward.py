"""R_citation: Citation accuracy reward."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CitationScore:
    """Citation accuracy scores."""

    correct_citations: int
    incorrect_citations: int
    missing_citations: int
    verified_quotes: int
    failed_quotes: int
    total: float
    details: str = ""


class CitationReward:
    """Reward for citation accuracy."""

    def __init__(
        self,
        library_path: Optional[str] = None,
        correct_citation_reward: float = 0.2,
        incorrect_citation_penalty: float = -0.3,
        missing_citation_penalty: float = -0.1,
        quote_match_threshold: float = 0.85,
    ):
        self.library_path = Path(library_path) if library_path else None
        self.correct_citation_reward = correct_citation_reward
        self.incorrect_citation_penalty = incorrect_citation_penalty
        self.missing_citation_penalty = missing_citation_penalty
        self.quote_match_threshold = quote_match_threshold

    def _extract_citations(self, text: str) -> list[str]:
        """Extract paper IDs from text."""
        # Match patterns like paper_0001, [paper_0001], (paper_0001)
        pattern = r'\b(paper_\d{3,4})\b'
        return list(set(re.findall(pattern, text, re.IGNORECASE)))

    def _extract_quotes(self, text: str) -> list[tuple[str, str]]:
        """Extract quotes with their claimed sources."""
        # Match patterns like "quote text" (paper_0001)
        pattern = r'"([^"]+)"\s*\(?(paper_\d{3,4})\)?'
        return re.findall(pattern, text, re.IGNORECASE)

    def _verify_quote(self, quote: str, paper_id: str) -> bool:
        """Verify quote exists in paper."""
        if self.library_path is None:
            return True  # Can't verify without library

        paper_path = self.library_path / "by_id" / paper_id / "full_text.md"
        if not paper_path.exists():
            return False

        content = paper_path.read_text(encoding="utf-8")

        # Exact match
        if quote in content:
            return True

        # Normalized match
        normalized_quote = " ".join(quote.lower().split())
        normalized_content = " ".join(content.lower().split())

        if normalized_quote in normalized_content:
            return True

        # Fuzzy match
        quote_words = set(normalized_quote.split())
        content_words = set(normalized_content.split())

        if not quote_words:
            return False

        overlap = len(quote_words & content_words) / len(quote_words)
        return overlap >= self.quote_match_threshold

    def compute(
        self,
        agent_answer: str,
        opened_files: list[str],
        required_papers: Optional[list[str]] = None,
    ) -> CitationScore:
        """
        Compute citation accuracy reward.

        Args:
            agent_answer: Agent's final answer
            opened_files: Files actually opened during search
            required_papers: Papers that should be cited

        Returns:
            CitationScore with breakdown
        """
        details = []

        # Extract citations from answer
        cited_papers = self._extract_citations(agent_answer)

        # Check against opened files
        opened_paper_ids = set()
        for path in opened_files:
            # Extract paper_id from path like "by_id/paper_0001/..."
            match = re.search(r'(paper_\d{3,4})', path)
            if match:
                opened_paper_ids.add(match.group(1))

        # Count correct/incorrect citations
        correct_citations = 0
        incorrect_citations = 0

        for paper in cited_papers:
            if paper in opened_paper_ids:
                correct_citations += 1
            else:
                incorrect_citations += 1
                details.append(f"Hallucinated citation: {paper}")

        # Check missing citations
        missing_citations = 0
        if required_papers:
            required_set = set(required_papers)
            cited_set = set(cited_papers)
            missing = required_set - cited_set
            missing_citations = len(missing)
            if missing:
                details.append(f"Missing citations: {missing}")

        # Verify quotes
        quotes = self._extract_quotes(agent_answer)
        verified_quotes = 0
        failed_quotes = 0

        for quote, paper_id in quotes:
            if self._verify_quote(quote, paper_id):
                verified_quotes += 1
            else:
                failed_quotes += 1
                details.append(f"Unverified quote from {paper_id}")

        # Calculate total reward
        total = (
            correct_citations * self.correct_citation_reward
            + incorrect_citations * self.incorrect_citation_penalty
            + missing_citations * self.missing_citation_penalty
            + failed_quotes * self.incorrect_citation_penalty * 0.5
        )

        return CitationScore(
            correct_citations=correct_citations,
            incorrect_citations=incorrect_citations,
            missing_citations=missing_citations,
            verified_quotes=verified_quotes,
            failed_quotes=failed_quotes,
            total=total,
            details="; ".join(details) if details else "All citations valid",
        )


def compute_citation_reward(
    agent_answer: str,
    opened_files: list[str],
    required_papers: Optional[list[str]] = None,
    library_path: Optional[str] = None,
) -> float:
    """Convenience function for citation reward."""
    reward = CitationReward(library_path=library_path)
    score = reward.compute(agent_answer, opened_files, required_papers)
    return score.total
