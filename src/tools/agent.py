#!/usr/bin/env python3
"""
Deep Research Agent - Interactive CLI
Run the trained agent for interactive research queries.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional

import torch
import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .tool_env import ToolEnv, parse_tool_call, format_tool_result

console = Console()
app = typer.Typer(help="Run Deep Research Agent interactively")


class Agent:
    """
    Interactive Deep Research Agent.
    """

    def __init__(
        self,
        model_path: str,
        library_path: str = "/mnt/library",
        workspace_path: str = "/workspace",
        max_turns: int = 15,
    ):
        self.model_path = model_path
        self.max_turns = max_turns

        # Setup environment
        self.env = ToolEnv(
            library_path=library_path,
            workspace_path=workspace_path,
            max_turns=max_turns,
        )

        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load the model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        console.print(f"[bold]Loading model: {self.model_path}[/bold]")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        console.print("[green]Model loaded![/green]")

    def _format_prompt(self, question: str, trajectory: list[dict]) -> str:
        """Format prompt with trajectory."""
        tool_desc = self.env.get_tool_descriptions()

        system = f"""You are a research assistant with access to a library of scientific papers.
Your task is to answer questions by searching and reading papers, taking notes, and using code for analysis.

{tool_desc}

Use this format:
Thought: your reasoning about what to do next
Action: tool_name
Action Input: {{"param": "value"}}

When you have enough information, provide your final answer:
Thought: I now have all the information needed.
Final Answer: your complete answer with citations (paper_XXXX)"""

        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": f"Question: {question}"})

        if trajectory:
            assistant_content = []
            for turn in trajectory:
                if turn.get("thought"):
                    assistant_content.append(f"Thought: {turn['thought']}")
                if turn.get("action"):
                    assistant_content.append(f"Action: {turn['action']}")
                    assistant_content.append(f"Action Input: {json.dumps(turn['action_input'])}")
                if turn.get("observation"):
                    assistant_content.append(f"Observation: {turn['observation']}")

            messages.append({"role": "assistant", "content": "\n".join(assistant_content)})

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            return "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    @torch.no_grad()
    def _generate(self, prompt: str) -> str:
        """Generate response."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        response = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        return response

    def _parse_response(self, response: str):
        """Parse model response."""
        import re

        thought = ""
        action = {"name": None, "input": {}}
        final_answer = None

        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|Final Answer:|$)', response, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()

        final_match = re.search(r'Final Answer:\s*(.+?)$', response, re.DOTALL)
        if final_match:
            final_answer = final_match.group(1).strip()
            return thought, action, final_answer

        action_match = re.search(r'Action:\s*(\w+)', response)
        if action_match:
            action["name"] = action_match.group(1)

            input_match = re.search(r'Action Input:\s*(\{.*?\})', response, re.DOTALL)
            if input_match:
                try:
                    action["input"] = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    pass

        return thought, action, final_answer

    async def answer(self, question: str, verbose: bool = True) -> str:
        """
        Answer a research question.

        Args:
            question: The research question
            verbose: Print intermediate steps

        Returns:
            Final answer
        """
        if verbose:
            console.print(Panel(question, title="Question", border_style="blue"))

        episode = self.env.reset(question)
        trajectory = []
        final_answer = None

        for turn_num in range(self.max_turns):
            prompt = self._format_prompt(question, trajectory)
            response = self._generate(prompt)

            thought, action, answer = self._parse_response(response)

            if verbose and thought:
                console.print(f"[dim]Thought: {thought}[/dim]")

            if answer:
                final_answer = answer
                break

            if not action["name"]:
                continue

            if verbose:
                console.print(f"[yellow]Action: {action['name']}[/yellow]")
                console.print(f"[dim]Input: {json.dumps(action['input'])}[/dim]")

            result = await self.env.step(action["name"], action["input"], thought)

            if verbose:
                # Truncate long observations
                obs = result.output
                if len(obs) > 500:
                    obs = obs[:500] + "..."
                console.print(f"[green]Observation:[/green] {obs}")

            trajectory.append({
                "turn": turn_num + 1,
                "thought": thought,
                "action": action["name"],
                "action_input": action["input"],
                "observation": result.output,
            })

        if verbose:
            console.print("\n")
            if final_answer:
                console.print(Panel(
                    Markdown(final_answer),
                    title="Answer",
                    border_style="green",
                ))
            else:
                console.print("[red]No answer generated within max turns[/red]")

        return final_answer or ""


@app.command()
def run(
    model_path: str = typer.Argument(..., help="Path to trained model"),
    library_path: str = typer.Option(
        "/mnt/library",
        "--library", "-l",
        help="Path to paper library"
    ),
    question: Optional[str] = typer.Option(
        None,
        "--question", "-q",
        help="Single question to answer (non-interactive)"
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Run in interactive mode"
    ),
):
    """Run the Deep Research Agent."""
    agent = Agent(
        model_path=model_path,
        library_path=library_path,
    )
    agent.load_model()

    if question:
        # Single question mode
        answer = asyncio.run(agent.answer(question))
        return

    if interactive:
        # Interactive mode
        console.print("\n[bold]Deep Research Agent[/bold]")
        console.print("Type your research questions. Type 'quit' to exit.\n")

        while True:
            try:
                question = Prompt.ask("[bold blue]Question[/bold blue]")

                if question.lower() in ['quit', 'exit', 'q']:
                    break

                if not question.strip():
                    continue

                asyncio.run(agent.answer(question))
                console.print("\n")

            except KeyboardInterrupt:
                break

        console.print("\n[dim]Goodbye![/dim]")


@app.command()
def demo(
    library_path: str = typer.Option(
        "/mnt/library",
        "--library", "-l",
    ),
):
    """Run a demo without a model (uses mock responses)."""
    console.print("[bold]Demo Mode[/bold] (no model loaded)\n")

    env = ToolEnv(library_path=library_path)
    console.print("Available tools:")
    console.print(env.get_tool_descriptions())


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
