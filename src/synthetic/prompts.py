"""
Prompts for Synthetic Data Generation.

These prompts are used with a powerful LLM (GPT-4, Claude, DeepSeek-V3)
to generate training tasks from the paper library.
"""

# System prompts for different task types

RETRIEVAL_SYSTEM = """You are a training data generator for a scientific research agent.
Your task: given a paper's text, create questions that can ONLY be answered by finding
specific facts in THIS paper (unique identifiers, numbers, method names, etc.).

Requirements:
- Questions must have factual, verifiable answers
- Answers should be specific (not general knowledge)
- Include the exact text span that contains the answer
- Vary difficulty: easy (single fact), medium (requires reading), hard (requires inference)"""

MULTIHOP_SYSTEM = """You are a training data generator for multi-step reasoning tasks.
Your task: create questions that require comparing information from MULTIPLE papers.

Requirements:
- Question must need information from BOTH papers
- Focus on comparisons: methods, results, approaches, limitations
- The answer should synthesize information, not just concatenate
- Include reasoning steps in golden answer"""

COMPUTATION_SYSTEM = """You are a training data generator for computational research tasks.
Your task: create questions that require CALCULATIONS based on data from papers.

Requirements:
- Question must require extracting numbers and computing something
- Include: averages, differences, percentages, statistical tests
- Provide expected Python code snippet
- Answer should be a computed value, not just extracted"""

SYNTHESIS_SYSTEM = """You are a training data generator for research synthesis tasks.
Your task: create questions that require reviewing multiple papers on a topic.

Requirements:
- Question should ask for a literature review or synthesis
- Answer must cite multiple papers with specific findings
- Include: common patterns, differences, evolution of ideas
- Evaluation criteria should check coverage and accuracy"""

SFT_DEMO_SYSTEM = """You are a training data generator for tool-use demonstrations.
Your task: create a complete trace of an agent correctly using tools to answer a question.

Available tools:
- grep_search(pattern, path): Search for text in papers
- read_file_chunk(path, start, num_lines): Read file portion
- execute_python(code): Run Python for analysis
- add_to_notes(text, section): Save findings
- verify_quote(path, snippet): Verify citation accuracy

Requirements:
- Show realistic tool usage sequence
- Include thought process before each action
- Show realistic observations (file contents, search results)
- End with final answer that uses gathered information"""


# User prompt templates

RETRIEVAL_USER = """Paper: {paper_title}
ID: {paper_id}
Text:
{paper_text}

Generate 3 questions of varying difficulty (easy, medium, hard).
Each question should have an answer findable ONLY in this paper.

Return JSON:
{{
  "questions": [
    {{
      "question": "What batch size was used in experiment X?",
      "answer": "The batch size was 256.",
      "source_paper_id": "{paper_id}",
      "evidence_text": "We trained with batch size 256 for...",
      "difficulty": "easy"
    }}
  ]
}}"""

MULTIHOP_USER = """Paper A: {paper_a_title} (ID: {paper_a_id})
Abstract A: {paper_a_abstract}
Key sections A: {paper_a_sections}

Paper B: {paper_b_title} (ID: {paper_b_id})
Abstract B: {paper_b_abstract}
Key sections B: {paper_b_sections}

Generate a comparison question that requires reading BOTH papers.

Return JSON:
{{
  "question": "Compare the reward shaping approaches in papers A and B.",
  "golden_answer": "Paper A uses... while Paper B uses... The key difference is...",
  "required_papers": ["{paper_a_id}", "{paper_b_id}"],
  "reasoning_type": "comparison",
  "reasoning_steps": ["Find reward shaping in A", "Find reward shaping in B", "Compare"]
}}"""

COMPUTATION_USER = """Papers with numerical data:
{papers_with_metrics}

Generate a question that requires extracting numbers and computing a result.

Return JSON:
{{
  "question": "Calculate the average accuracy improvement across papers A, B, C.",
  "golden_answer": "The average improvement is 12.3%",
  "required_papers": ["paper_a", "paper_b", "paper_c"],
  "expected_code": "import pandas as pd\\nscores = [0.89, 0.92, 0.87]\\nprint(f'Average: {{sum(scores)/len(scores):.1%}}')",
  "requires_code": true,
  "extracted_values": {{"paper_a": 0.89, "paper_b": 0.92, "paper_c": 0.87}}
}}"""

SYNTHESIS_USER = """Topic: {topic_name}
Relevant papers:
{papers_list}

Generate a synthesis question requiring review of 3-5 papers.

Return JSON:
{{
  "question": "Survey the approaches to {topic_name} based on papers in the library. Identify key methods, their advantages, and limitations.",
  "golden_answer": "Detailed review with citations...",
  "required_papers": ["paper_1", "paper_2", "paper_3", "paper_4"],
  "evaluation_criteria": [
    "covers_all_required_papers",
    "identifies_common_patterns",
    "notes_differences",
    "accurate_citations",
    "logical_structure"
  ]
}}"""

SFT_DEMO_USER = """Question: {question}
Correct answer: {answer}
Source paper: {paper_id}
Paper location: by_id/{paper_id}/

Generate a complete agent trace showing correct tool usage.

Return JSON:
{{
  "turns": [
    {{
      "thought": "I need to find information about X. Let me search for it.",
      "action": "grep_search",
      "action_input": {{"pattern": "X", "path": "by_id/{paper_id}"}},
      "observation": "{paper_id}/sections/methods.txt:15: The X method uses..."
    }},
    {{
      "thought": "Found a match. Let me read more context.",
      "action": "read_file_chunk",
      "action_input": {{"path": "by_id/{paper_id}/sections/methods.txt", "start": 10, "num_lines": 20}},
      "observation": "10: ...\\n11: ...\\n15: The X method uses..."
    }},
    {{
      "thought": "I have enough information to answer. Let me verify the quote.",
      "action": "verify_quote",
      "action_input": {{"path": "by_id/{paper_id}/sections/methods.txt", "snippet": "X method uses"}},
      "observation": "VERIFIED: Quote found at line 15"
    }}
  ],
  "final_answer": "{answer}"
}}"""


# Aggregated prompts dict
GENERATION_PROMPTS = {
    "retrieval": {
        "system": RETRIEVAL_SYSTEM,
        "user": RETRIEVAL_USER,
    },
    "multihop": {
        "system": MULTIHOP_SYSTEM,
        "user": MULTIHOP_USER,
    },
    "computation": {
        "system": COMPUTATION_SYSTEM,
        "user": COMPUTATION_USER,
    },
    "synthesis": {
        "system": SYNTHESIS_SYSTEM,
        "user": SYNTHESIS_USER,
    },
    "sft_demo": {
        "system": SFT_DEMO_SYSTEM,
        "user": SFT_DEMO_USER,
    },
}
