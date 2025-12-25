#!/usr/bin/env python3
"""
Interactive test of agent tools.

Usage:
    python scripts/test_tools.py --library ./data/library
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.bash_tools import create_tools
from src.tools.code_interpreter import CodeInterpreter
from src.tools.scratchpad import Scratchpad


def main():
    parser = argparse.ArgumentParser(description="Test agent tools")
    parser.add_argument(
        "--library",
        default="./data/library",
        help="Path to library directory"
    )
    parser.add_argument(
        "--workspace",
        default="./data/workspace/test",
        help="Path to workspace directory"
    )

    args = parser.parse_args()

    library_path = Path(args.library)
    workspace_path = Path(args.workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 50)
    print("  Agent Tools Test")
    print("=" * 50)

    # Create tools
    bash_tools = create_tools(library_path)
    code_interpreter = CodeInterpreter(workspace_path)
    scratchpad = Scratchpad(workspace_path)

    print(f"\nAvailable bash tools: {list(bash_tools.keys())}")

    # Test grep_search
    print("\n--- Testing grep_search ---")
    result = bash_tools["grep_search"].execute(pattern="method", path="")
    print(f"Success: {result.success}")
    print(f"Output (first 200 chars): {result.output[:200]}...")

    # Test list_papers
    print("\n--- Testing list_papers ---")
    result = bash_tools["list_papers"].execute(limit=5)
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output}")

    # Test code_interpreter
    print("\n--- Testing code_interpreter ---")
    code = """
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'paper': ['A', 'B', 'C'],
    'score': [0.85, 0.92, 0.88]
})

print("Sample DataFrame:")
print(data)
print(f"\\nMean score: {data['score'].mean():.3f}")
"""
    result = code_interpreter.execute(code=code)
    print(f"Success: {result.success}")
    print(f"Output:\n{result.output}")

    # Test scratchpad
    print("\n--- Testing scratchpad ---")
    result = scratchpad.execute(action="add", text="Test finding: X works well", tag="finding")
    print(f"Add: {result.output}")

    result = scratchpad.execute(action="read")
    print(f"Read:\n{result.output}")

    print("\n" + "=" * 50)
    print("  All tests completed!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
