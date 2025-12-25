#!/usr/bin/env python3
"""
Prompts for Synthetic Data Generation
Templates for generating training tasks of different types.
"""

# Prompt 1: Simple Retrieval (fact finding)
RETRIEVAL_PROMPT = """SYSTEM:
You are a training data generator for a scientific research agent.
Your task: given a paper's text, create a question whose answer can ONLY be found in this specific paper (a unique fact).

Requirements:
- The question should require reading the paper to answer
- The answer should be specific and verifiable
- Avoid overly obvious questions (not just the title)
- Include questions about methods, results, specific numbers

USER:
Paper: {paper_title}
ID: {paper_id}
Text: {paper_text}

Generate 3 questions of varying difficulty. Output JSON format:
{{
  "questions": [
    {{
      "question": "What batch size was used in experiment X?",
      "answer": "The batch size was 256.",
      "source_paper_id": "{paper_id}",
      "difficulty": "easy",
      "section_hint": "methods"
    }},
    {{
      "question": "What was the improvement in accuracy compared to baseline?",
      "answer": "The improvement was 4.2% over the PPO baseline.",
      "source_paper_id": "{paper_id}",
      "difficulty": "medium",
      "section_hint": "results"
    }},
    {{
      "question": "Why did the authors choose reward shaping over direct optimization?",
      "answer": "Because direct optimization led to reward hacking in preliminary experiments.",
      "source_paper_id": "{paper_id}",
      "difficulty": "hard",
      "section_hint": "discussion"
    }}
  ]
}}

IMPORTANT: Questions and answers must be in the same language as the paper.
Output valid JSON only."""

# Prompt 2: Multi-hop Reasoning (comparison)
MULTIHOP_PROMPT = """SYSTEM:
You are a generator of complex research questions.
Create a question that REQUIRES reading BOTH papers and comparing information from them.

Requirements:
- The question should be unanswerable with just one paper
- Focus on comparisons: methods, results, approaches, limitations
- The answer should synthesize information from both sources

USER:
Paper A: {paper_a_title} (ID: {paper_a_id})
Abstract A: {paper_a_abstract}

Paper B: {paper_b_title} (ID: {paper_b_id})
Abstract B: {paper_b_abstract}

Generate a comparison question. Output JSON format:
{{
  "question": "Compare the approaches to reward shaping in papers A and B. What are the key differences?",
  "golden_answer": "Paper A uses dense reward shaping with intermediate goals, while Paper B employs sparse terminal rewards with hindsight relabeling. The key difference is that A requires domain knowledge for reward design, whereas B learns from failures...",
  "required_papers": ["{paper_a_id}", "{paper_b_id}"],
  "reasoning_type": "comparison",
  "difficulty": "hard"
}}

Output valid JSON only."""

# Prompt 3: Computation Required (calculations)
COMPUTATION_PROMPT = """SYSTEM:
You are a generator of computational research tasks.
Create a task that requires CALCULATIONS - the agent must extract numbers from papers and perform mathematical operations.

Task types:
- Calculate averages, differences, percentages
- Compare metrics across papers
- Verify calculations from papers
- Statistical significance estimation

USER:
Papers with numerical data:
{papers_with_metrics}

Generate a computational task. Output JSON format:
{{
  "question": "Calculate the average accuracy improvement on GSM8K across papers A, B, and C compared to their respective baselines.",
  "golden_answer": "Average improvement is 12.3%. Paper A: 8.5%, Paper B: 15.2%, Paper C: 13.2%",
  "required_papers": ["paper_a", "paper_b", "paper_c"],
  "expected_code": "import pandas as pd\\ndata = {{'paper': ['A', 'B', 'C'], 'improvement': [8.5, 15.2, 13.2]}}\\ndf = pd.DataFrame(data)\\nprint(f'Average: {{df.improvement.mean():.1f}}%')",
  "requires_code": true,
  "computation_type": "average",
  "difficulty": "medium"
}}

Output valid JSON only."""

# Prompt 4: Full Synthesis (topic review)
SYNTHESIS_PROMPT = """SYSTEM:
You are a generator of synthesis tasks requiring knowledge integration.
Create a task requiring review of 3-5 papers on a topic.
The agent must find common patterns, contradictions, and evolution of ideas.

USER:
Topic: {topic_name}
Relevant papers:
{list_of_papers}

Generate a synthesis task. Output JSON format:
{{
  "question": "Write a review of reward modeling methods based on papers in the library. Identify key approaches, their advantages, and limitations.",
  "golden_answer": "A detailed review with citations... Three main approaches have emerged: 1) Human preference learning (paper_001, paper_005), 2) AI feedback (paper_003), 3) Hybrid methods (paper_007). Key advantages include... Limitations are...",
  "required_papers": ["paper_001", "paper_003", "paper_005", "paper_007"],
  "evaluation_criteria": [
    "covers_all_papers",
    "identifies_common_patterns",
    "notes_contradictions",
    "accurate_citations",
    "coherent_narrative"
  ],
  "difficulty": "very_hard",
  "expected_length": "500-1000 words"
}}

Output valid JSON only."""

# Prompt 5: SFT Demo (tool usage examples)
SFT_DEMO_PROMPT = """SYSTEM:
You are a generator of tool usage demonstrations.
Create an example dialogue showing the agent CORRECTLY using tools:
grep_search, read_file_chunk, execute_python, add_to_notes, verify_quote.

The demonstration should show:
1. Systematic search strategy
2. Efficient file reading
3. Note-taking for complex tasks
4. Quote verification before citing
5. Code for calculations when needed

USER:
Question: {question}
Correct answer: {answer}
Paper with answer: {paper_id}

Generate a COMPLETE agent trace. Output JSON format:
{{
  "question": "{question}",
  "turns": [
    {{
      "thought": "I need to find the paper discussing PPO-Clip...",
      "action": "grep_search",
      "action_input": {{"pattern": "PPO-Clip", "path": "/mnt/library"}},
      "observation": "paper_042/methods.txt:15: PPO-Clip с clipping range..."
    }},
    {{
      "thought": "Found the paper, reading the methods section...",
      "action": "read_file_chunk",
      "action_input": {{"path": "/mnt/library/by_id/paper_042/sections/methods.txt", "start": 10, "num_lines": 30}},
      "observation": "The PPO-Clip algorithm uses..."
    }},
    {{
      "thought": "I found the key information. Let me note it down.",
      "action": "add_to_notes",
      "action_input": {{"text": "PPO-Clip uses epsilon=0.2 for clipping", "source": "paper_042", "category": "findings"}},
      "observation": "[Note added] Category: findings, Source: paper_042"
    }},
    {{
      "thought": "I should verify this quote before using it.",
      "action": "verify_quote",
      "action_input": {{"path": "/mnt/library/by_id/paper_042/sections/methods.txt", "snippet": "epsilon=0.2"}},
      "observation": "{{\\"verified\\": true, \\"exact_match\\": true}}"
    }},
    {{
      "thought": "Quote verified. I can now provide the final answer.",
      "action": null,
      "action_input": null,
      "observation": null
    }}
  ],
  "final_answer": "{answer}",
  "tools_used": ["grep_search", "read_file_chunk", "add_to_notes", "verify_quote"],
  "num_turns": 4
}}

IMPORTANT:
- Thoughts should explain reasoning
- Actions should be efficient (no unnecessary steps)
- Demonstrate proper tool usage patterns
- Include note-taking for complex tasks
- Verify quotes before citing

Output valid JSON only."""

# Prompt for generating diverse questions within a category
DIVERSITY_PROMPT = """SYSTEM:
You are reviewing generated questions for diversity.
Given a set of existing questions, suggest ways to make the next questions more diverse.

Existing questions:
{existing_questions}

Suggest 3 different angles/aspects that haven't been covered:
1.
2.
3."""

# Curriculum levels mapping
CURRICULUM_PROMPTS = {
    1: RETRIEVAL_PROMPT,      # Level 1: Simple retrieval
    2: RETRIEVAL_PROMPT,      # Level 2: Extraction (same prompt, harder examples)
    3: COMPUTATION_PROMPT,    # Level 3: Computation
    4: MULTIHOP_PROMPT,       # Level 4: Multi-hop reasoning
    5: COMPUTATION_PROMPT,    # Level 5: Visualization (computation with figures)
    6: SYNTHESIS_PROMPT,      # Level 6: Full synthesis
}
