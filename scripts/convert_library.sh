#!/bin/bash
# Convert DOCX papers to structured Markdown library
#
# Usage: ./scripts/convert_library.sh [INPUT_DIR] [OUTPUT_DIR]
#
# Example:
#   ./scripts/convert_library.sh ./papers /mnt/library

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT_DIR="${1:-$PROJECT_ROOT/data/raw}"
OUTPUT_DIR="${2:-/mnt/library}"

echo "Converting DOCX files from $INPUT_DIR to $OUTPUT_DIR"

python3 -m src.library.convert convert \
    "$INPUT_DIR" \
    --output "$OUTPUT_DIR" \
    --pattern "*.docx"

echo "Done! Library info:"
python3 -m src.library.convert info --library "$OUTPUT_DIR"
