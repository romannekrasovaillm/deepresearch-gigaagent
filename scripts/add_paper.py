#!/usr/bin/env python3
"""
Quick script to add a single paper to the library.

Usage:
    python scripts/add_paper.py path/to/paper.docx
    python scripts/add_paper.py path/to/paper.docx --library ./data/library
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.library.converter import add_paper


def main():
    parser = argparse.ArgumentParser(description="Add a paper to the library")
    parser.add_argument("docx_path", help="Path to DOCX file")
    parser.add_argument(
        "--library",
        default="./data/library",
        help="Path to library directory"
    )

    args = parser.parse_args()

    docx_path = Path(args.docx_path)
    if not docx_path.exists():
        print(f"Error: File not found: {docx_path}")
        sys.exit(1)

    if not docx_path.suffix.lower() == '.docx':
        print(f"Warning: File does not have .docx extension: {docx_path}")

    try:
        metadata = add_paper(docx_path, args.library)
        print(f"Successfully added: {metadata.paper_id}")
        print(f"  Title: {metadata.title}")
        print(f"  Sections: {metadata.section_count}")
        print(f"  Words: {metadata.word_count}")
    except Exception as e:
        print(f"Error adding paper: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
