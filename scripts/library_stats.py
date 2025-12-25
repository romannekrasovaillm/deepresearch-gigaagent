#!/usr/bin/env python3
"""
Display library statistics.

Usage:
    python scripts/library_stats.py
    python scripts/library_stats.py --library ./data/library
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.library.indexer import LibraryIndexer


def main():
    parser = argparse.ArgumentParser(description="Show library statistics")
    parser.add_argument(
        "--library",
        default="./data/library",
        help="Path to library directory"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="Search papers by title"
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all papers"
    )

    args = parser.parse_args()

    library_path = Path(args.library)
    if not library_path.exists():
        print(f"Error: Library not found: {library_path}")
        sys.exit(1)

    indexer = LibraryIndexer(library_path)

    if args.search:
        print(f"\nSearch results for '{args.search}':\n")
        results = indexer.search_by_title(args.search)
        for r in results:
            print(f"  [{r.score:.2f}] {r.paper_id}: {r.title}")
        print()
        return

    if args.list_all:
        print("\nAll papers:\n")
        for paper in indexer.list_papers(limit=1000):
            print(f"  {paper.get('paper_id')}: {paper.get('title')}")
        print()
        return

    # Default: show stats
    stats = indexer.get_statistics()

    print("\n" + "=" * 50)
    print("  Library Statistics")
    print("=" * 50)
    print(f"  Total papers:      {stats['total_papers']}")
    print(f"  Total words:       {stats['total_words']:,}")
    print(f"  Total sections:    {stats['total_sections']}")
    print(f"  Avg words/paper:   {stats['avg_words_per_paper']:,}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
