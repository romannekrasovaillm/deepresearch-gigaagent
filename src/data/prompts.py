"""Prompts for synthetic data generation."""

# =============================================================================
# RETRIEVAL: Simple fact-finding tasks
# =============================================================================
RETRIEVAL_PROMPT = """You are a synthetic data generator for training a research agent.
Your task: Create a factual question that can be answered ONLY from this paper.

Paper ID: {paper_id}
Title: {paper_title}
Content: {paper_content}

Generate 3 questions of increasing difficulty. Each question should:
1. Have a specific, verifiable answer from this paper
2. Require reading this specific paper (not general knowledge)
3. Be answerable in 1-3 sentences

Respond in JSON format:
{{
  "questions": [
    {{
      "question": "What batch size was used in the main experiment?",
      "answer": "The batch size was 256 as stated in Section 4.1.",
      "source_paper_id": "{paper_id}",
      "difficulty": "easy",
      "section_hint": "methods"
    }},
    {{
      "question": "...",
      "answer": "...",
      "source_paper_id": "{paper_id}",
      "difficulty": "medium",
      "section_hint": "..."
    }},
    {{
      "question": "...",
      "answer": "...",
      "source_paper_id": "{paper_id}",
      "difficulty": "hard",
      "section_hint": "..."
    }}
  ]
}}"""


# =============================================================================
# MULTIHOP: Comparison/synthesis across papers
# =============================================================================
MULTIHOP_PROMPT = """You are a synthetic data generator for training a research agent.
Create a question that REQUIRES reading and comparing information from BOTH papers.

Paper A:
- ID: {paper_a_id}
- Title: {paper_a_title}
- Abstract: {paper_a_abstract}

Paper B:
- ID: {paper_b_id}
- Title: {paper_b_title}
- Abstract: {paper_b_abstract}

Generate a comparison question that:
1. Cannot be answered from either paper alone
2. Requires extracting specific information from both
3. Asks for synthesis/comparison (not just listing facts)

Respond in JSON format:
{{
  "question": "Compare the optimization approaches used in papers A and B. What are the key differences?",
  "golden_answer": "Paper A uses Adam optimizer with learning rate 1e-4, while Paper B uses...",
  "required_papers": ["{paper_a_id}", "{paper_b_id}"],
  "reasoning_type": "comparison",
  "difficulty": "medium"
}}"""


# =============================================================================
# COMPUTATION: Tasks requiring Python analysis
# =============================================================================
COMPUTATION_PROMPT = """You are a synthetic data generator for training a research agent.
Create a task that REQUIRES Python code to solve (statistical analysis, calculations, etc.).

Papers with numerical data:
{papers_with_metrics}

Generate a computation task that:
1. Requires extracting numbers from multiple papers
2. Needs actual computation (mean, comparison, statistics)
3. Cannot be answered by just reading (needs calculation)

Respond in JSON format:
{{
  "question": "Calculate the average improvement in accuracy across papers A, B, C compared to their baselines.",
  "golden_answer": "The average improvement is 12.3% (A: +8%, B: +15%, C: +14%)",
  "required_papers": ["paper_a", "paper_b", "paper_c"],
  "expected_code": "import pandas as pd\\ndf = pd.DataFrame({{...}})\\ndf['improvement'].mean()",
  "requires_code": true,
  "computation_type": "statistical_comparison"
}}"""


# =============================================================================
# SYNTHESIS: Comprehensive literature review
# =============================================================================
SYNTHESIS_PROMPT = """You are a synthetic data generator for training a research agent.
Create a comprehensive synthesis task requiring review of multiple papers.

Topic: {topic_name}
Relevant papers:
{papers_on_topic}

Generate a synthesis task that:
1. Requires reviewing 3-5 papers on this topic
2. Asks for identification of patterns, evolution, or contradictions
3. Needs structured analysis, not just summaries

Respond in JSON format:
{{
  "question": "Analyze the evolution of {topic_name} approaches across the papers in the library. Identify key methodological shifts and remaining open problems.",
  "golden_answer": "A comprehensive analysis covering: 1) Early approaches (papers X, Y)..., 2) Modern methods (papers Z, W)..., 3) Open problems identified...",
  "required_papers": ["paper_1", "paper_2", "paper_3", "paper_4"],
  "evaluation_criteria": [
    "covers_all_papers",
    "identifies_common_patterns",
    "notes_evolution",
    "identifies_contradictions",
    "accurate_citations"
  ],
  "difficulty": "hard"
}}"""


# =============================================================================
# SFT_DEMO: Tool usage demonstrations
# =============================================================================
SFT_DEMO_PROMPT = """You are generating a demonstration of CORRECT tool usage for a research agent.

Question: {question}
Correct Answer: {answer}
Source Paper: {paper_id}

Generate a complete interaction trace showing the agent:
1. Thinking about the approach
2. Using tools correctly (grep_search, read_file_chunk, execute_python, etc.)
3. Building up to the answer step by step
4. Verifying findings before answering

Tool format: {{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}

Respond in JSON format:
{{
  "turns": [
    {{
      "thought": "I need to find papers about PPO-Clip...",
      "action": {{"name": "grep_search", "arguments": {{"pattern": "PPO-Clip", "path": null}}}},
      "observation": "by_id/paper_042/methods.txt:15: PPO-Clip implementation..."
    }},
    {{
      "thought": "Found it in paper_042. Let me read the methods section...",
      "action": {{"name": "read_file_chunk", "arguments": {{"path": "by_id/paper_042/sections/methods.txt", "start": 10, "num_lines": 30}}}},
      "observation": "The PPO-Clip algorithm uses a clipped surrogate objective..."
    }},
    {{
      "thought": "Now I have the information. Let me save this finding.",
      "action": {{"name": "add_to_notes", "arguments": {{"text": "PPO-Clip uses clipped surrogate objective with epsilon=0.2", "section": "PPO-Clip"}}}},
      "observation": "Added to notes: 67 characters"
    }},
    {{
      "thought": "I should verify this quote before using it.",
      "action": {{"name": "verify_quote", "arguments": {{"path": "by_id/paper_042/sections/methods.txt", "snippet": "clipped surrogate objective"}}}},
      "observation": "VERIFIED: Exact match found."
    }},
    {{
      "thought": "I have verified information. I can now provide the answer.",
      "action": null,
      "observation": null
    }}
  ],
  "final_answer": "[ANSWER] Based on paper_042, PPO-Clip uses..."
}}"""


# =============================================================================
# VISUALIZATION: Tasks requiring plots/graphs
# =============================================================================
VISUALIZATION_PROMPT = """You are a synthetic data generator for training a research agent.
Create a task that requires creating a visualization to answer.

Papers with temporal/comparative data:
{papers_with_data}

Generate a visualization task that:
1. Requires extracting data points from papers
2. Needs a plot/chart to properly answer
3. Asks for trend analysis or comparison

Respond in JSON format:
{{
  "question": "Create a timeline visualization showing the performance improvements on {benchmark} from 2020-2024 based on papers in the library.",
  "golden_answer": "The visualization shows steady improvement from X% (2020) to Y% (2024), with a notable jump in 2023 when method Z was introduced.",
  "required_papers": ["paper_a", "paper_b", "paper_c"],
  "expected_code": "import matplotlib.pyplot as plt\\nyears = [...]\\nscores = [...]\\nplt.plot(years, scores)\\nplt.savefig('progress.png')",
  "requires_visualization": true,
  "visualization_type": "timeline"
}}"""
