"""
Prompts for synthetic dataset generation.

Contains prompt templates for generating different types
of training tasks: retrieval, multi-hop, computation, synthesis.
"""

# Simple Retrieval - finding a single fact
PROMPT_RETRIEVAL = """You are generating training data for a scientific research agent.

Your task: Create a question that can ONLY be answered by reading this specific paper.
The question should target a UNIQUE fact (number, method name, specific finding).

## Paper Information
Title: {paper_title}
Paper ID: {paper_id}
Text excerpt:
{paper_text}

## Requirements
1. The answer must be found ONLY in this paper
2. Include specific details (numbers, names, dates)
3. Avoid generic questions answerable from common knowledge
4. Create 3 questions of varying difficulty

## Output Format (JSON)
{{
  "questions": [
    {{
      "question": "What batch size was used in the PPO training experiments?",
      "answer": "The batch size was 256 samples.",
      "source_paper_id": "{paper_id}",
      "difficulty": "easy",
      "evidence_quote": "We trained with a batch size of 256..."
    }},
    {{
      "question": "...",
      "answer": "...",
      "source_paper_id": "{paper_id}",
      "difficulty": "medium",
      "evidence_quote": "..."
    }},
    {{
      "question": "...",
      "answer": "...",
      "source_paper_id": "{paper_id}",
      "difficulty": "hard",
      "evidence_quote": "..."
    }}
  ]
}}

Generate questions now:"""


# Multi-hop Reasoning - comparing multiple papers
PROMPT_MULTIHOP = """You are generating training data for a scientific research agent.

Your task: Create a question that REQUIRES reading BOTH papers to answer.
The question should involve comparison, contrast, or synthesis of information.

## Paper A
Title: {paper_a_title}
Paper ID: {paper_a_id}
Abstract: {paper_a_abstract}
Key excerpt: {paper_a_excerpt}

## Paper B
Title: {paper_b_title}
Paper ID: {paper_b_id}
Abstract: {paper_b_abstract}
Key excerpt: {paper_b_excerpt}

## Requirements
1. Answer MUST require information from BOTH papers
2. Focus on: methodology differences, result comparisons, theoretical connections
3. The question should not be answerable from either paper alone

## Output Format (JSON)
{{
  "question": "How does the reward shaping approach in Paper A differ from Paper B's method?",
  "golden_answer": "Paper A uses intrinsic motivation based on..., while Paper B employs...",
  "required_papers": ["{paper_a_id}", "{paper_b_id}"],
  "reasoning_type": "comparison",
  "key_points": [
    "Paper A approach: ...",
    "Paper B approach: ...",
    "Key difference: ..."
  ]
}}

Generate the question now:"""


# Computation Required - needs code execution
PROMPT_COMPUTATION = """You are generating training data for a scientific research agent with code execution capabilities.

Your task: Create a question that requires NUMERICAL COMPUTATION to answer.
The agent must extract numbers from papers and perform calculations.

## Papers with Numerical Data
{papers_with_metrics}

## Computation Types (choose one)
1. Statistical comparison (t-test, effect size)
2. Average/aggregation across papers
3. Percentage improvement calculation
4. Trend analysis over time

## Requirements
1. Question must require extracting specific numbers from papers
2. Answer requires Python code (pandas, numpy, scipy)
3. Provide the expected numerical result

## Output Format (JSON)
{{
  "question": "Calculate the average improvement in GSM8K accuracy across papers A, B, and C compared to the GPT-4 baseline.",
  "golden_answer": "The average improvement is 12.3% (±2.1% std dev)",
  "required_papers": ["paper_001", "paper_002", "paper_003"],
  "expected_code": "import pandas as pd\\ndf = pd.DataFrame({{...}})\\ndf['improvement'].mean()",
  "requires_code": true,
  "extracted_values": {{
    "paper_001": {{"gsm8k": 0.89, "baseline": 0.76}},
    "paper_002": {{"gsm8k": 0.91, "baseline": 0.76}},
    "paper_003": {{"gsm8k": 0.87, "baseline": 0.76}}
  }}
}}

Generate the question now:"""


# Full Synthesis - comprehensive review
PROMPT_SYNTHESIS = """You are generating training data for a scientific research agent.

Your task: Create a SYNTHESIS question requiring review of 3-5 papers on a topic.
The agent must identify patterns, contradictions, and evolution of ideas.

## Topic
{topic_name}

## Available Papers on This Topic
{papers_list}

## Synthesis Requirements
1. Cover ALL listed papers
2. Identify common patterns or methodologies
3. Note any contradictions between papers
4. Trace evolution of the approach over time (if applicable)

## Output Format (JSON)
{{
  "question": "Provide a comprehensive review of reward modeling approaches based on the papers in the library. Identify key methodologies, their advantages, and limitations.",
  "golden_answer": "Detailed review covering:\\n1. Main approaches: ...\\n2. Common patterns: ...\\n3. Key differences: ...\\n4. Evolution: ...\\n5. Open questions: ...",
  "required_papers": ["paper_001", "paper_005", "paper_012", "paper_023"],
  "evaluation_criteria": [
    "covers_all_papers",
    "identifies_common_patterns",
    "notes_contradictions",
    "traces_evolution",
    "accurate_citations"
  ],
  "key_themes": ["theme1", "theme2", "theme3"]
}}

Generate the synthesis question now:"""


# SFT Demonstration - tool usage example
PROMPT_SFT_DEMO = """You are generating a DEMONSTRATION of correct tool usage for training.

Given a question and answer, generate a complete agent trajectory showing
how to properly use tools to find the answer.

## Question
{question}

## Correct Answer
{answer}

## Relevant Paper
Paper ID: {paper_id}
Relevant section: {relevant_section}

## Available Tools
1. grep_search(pattern, path) - Search for pattern in files
2. read_file_chunk(path, start, num_lines) - Read portion of file
3. execute_python(code) - Run Python code for analysis
4. add_to_notes(text, tag) - Save findings to notebook
5. verify_quote(path, snippet) - Verify citation exists

## Output Format (JSON)
Generate a sequence of thought-action-observation steps:
{{
  "task": "{question}",
  "turns": [
    {{
      "thought": "I need to find information about X. Let me search for papers mentioning it.",
      "action": "grep_search",
      "action_input": {{"pattern": "X", "path": "/mnt/library"}},
      "observation": "by_id/paper_042/methods.txt:15: X is described as..."
    }},
    {{
      "thought": "Found a relevant paper. Let me read the methods section.",
      "action": "read_file_chunk",
      "action_input": {{"path": "by_id/paper_042/methods.txt", "start": 10, "num_lines": 30}},
      "observation": "10: The methodology involves...\\n11: ..."
    }},
    {{
      "thought": "I found the relevant information. Let me save this finding.",
      "action": "add_to_notes",
      "action_input": {{"text": "Paper 042 describes X as...", "tag": "finding"}},
      "observation": "Note added (1 total notes)"
    }},
    {{
      "thought": "Before finalizing, let me verify the quote.",
      "action": "verify_quote",
      "action_input": {{"path": "by_id/paper_042/methods.txt", "snippet": "X is described as"}},
      "observation": "VERIFIED (fuzzy match): ...X is described as..."
    }}
  ],
  "final_answer": "{answer}"
}}

Generate the demonstration trajectory now:"""


# Visualization task
PROMPT_VISUALIZATION = """You are generating training data for a scientific research agent with visualization capabilities.

Your task: Create a question that requires CREATING A VISUALIZATION to answer.
The agent must extract data from multiple papers and create an informative plot.

## Papers with Timeline/Comparative Data
{papers_data}

## Visualization Types (choose one)
1. Timeline plot showing evolution of a metric
2. Bar chart comparing methods across papers
3. Scatter plot of performance vs. complexity
4. Heatmap of method capabilities

## Requirements
1. Data must be extracted from actual papers
2. Visualization must be meaningful and interpretable
3. Include axis labels, title, legend

## Output Format (JSON)
{{
  "question": "Create a timeline visualization showing the improvement in MMLU scores from 2020 to 2024 based on the papers in the library.",
  "golden_answer": "The visualization shows a clear upward trend from 65% in 2020 to 89% in 2024, with major jumps in 2022 (GPT-3.5) and 2023 (GPT-4).",
  "required_papers": ["paper_003", "paper_007", "paper_015", "paper_021"],
  "requires_code": true,
  "expected_code": "import matplotlib.pyplot as plt\\nyears = [2020, 2021, 2022, 2023, 2024]\\nscores = [0.65, 0.68, 0.78, 0.86, 0.89]\\nplt.plot(years, scores, marker='o')\\nplt.xlabel('Year')\\nplt.ylabel('MMLU Score')\\nplt.title('MMLU Performance Evolution')\\nplt.savefig('/workspace/mmlu_timeline.png')",
  "extracted_data": {{
    "2020": {{"paper": "paper_003", "score": 0.65}},
    "2022": {{"paper": "paper_007", "score": 0.78}},
    "2024": {{"paper": "paper_021", "score": 0.89}}
  }}
}}

Generate the visualization question now:"""


# All prompts registry
GENERATION_PROMPTS = {
    "retrieval": PROMPT_RETRIEVAL,
    "multihop": PROMPT_MULTIHOP,
    "computation": PROMPT_COMPUTATION,
    "synthesis": PROMPT_SYNTHESIS,
    "sft_demo": PROMPT_SFT_DEMO,
    "visualization": PROMPT_VISUALIZATION,
}


# Task type distribution for balanced dataset
DEFAULT_DISTRIBUTION = {
    "retrieval": 0.30,      # 30% simple fact retrieval
    "multihop": 0.25,       # 25% multi-paper comparison
    "computation": 0.15,    # 15% requires code
    "visualization": 0.05,  # 5% visualization tasks
    "synthesis": 0.15,      # 15% full synthesis
    "sft_demo": 0.10,       # 10% SFT demonstrations
}


# Curriculum levels for progressive training
CURRICULUM_LEVELS = {
    1: {
        "name": "Retrieval",
        "tasks": ["retrieval"],
        "tools": ["grep_search", "read_file_chunk", "list_papers"],
        "max_steps": 5,
    },
    2: {
        "name": "Extraction",
        "tasks": ["retrieval"],
        "tools": ["grep_search", "read_file_chunk", "add_to_notes", "list_papers"],
        "max_steps": 8,
    },
    3: {
        "name": "Computation",
        "tasks": ["retrieval", "computation"],
        "tools": ["grep_search", "read_file_chunk", "execute_python", "add_to_notes"],
        "max_steps": 10,
    },
    4: {
        "name": "Multi-hop",
        "tasks": ["retrieval", "computation", "multihop"],
        "tools": ["grep_search", "read_file_chunk", "execute_python", "add_to_notes", "verify_quote"],
        "max_steps": 12,
    },
    5: {
        "name": "Visualization",
        "tasks": ["computation", "multihop", "visualization"],
        "tools": ["grep_search", "read_file_chunk", "execute_python", "add_to_notes", "verify_quote", "save_figure"],
        "max_steps": 15,
    },
    6: {
        "name": "Full Synthesis",
        "tasks": ["multihop", "synthesis", "visualization"],
        "tools": ["all"],
        "max_steps": 20,
    },
}
