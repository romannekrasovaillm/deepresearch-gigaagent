"""
DOCX to Markdown converter for scientific papers library.

Converts DOCX files to structured Markdown with metadata extraction,
section splitting, and indexing for efficient agent navigation.
"""

import json
import re
import subprocess
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    print("Warning: python-docx not installed. Some features will be limited.")


@dataclass
class PaperMetadata:
    """Metadata for a scientific paper."""
    paper_id: str
    title: str
    author: Optional[str] = None
    created: Optional[str] = None
    source_file: str = ""
    abstract_preview: str = ""
    word_count: int = 0
    section_count: int = 0
    file_hash: str = ""
    processed_at: str = ""


class DocxConverter:
    """Converts DOCX files to structured Markdown."""

    # Common section header patterns (English and Russian)
    SECTION_PATTERNS = [
        r'^#+\s+',  # Markdown headers
        r'^(Abstract|Аннотация|Introduction|Введение|Methods?|Методы|'
        r'Results?|Результаты|Discussion|Обсуждение|Conclusion|Заключение|'
        r'References|Литература|Ссылки|Background|Предпосылки|'
        r'Related Work|Обзор литературы|Experiments?|Эксперименты|'
        r'Implementation|Реализация|Evaluation|Оценка|Analysis|Анализ)',
    ]

    def __init__(self, pandoc_path: str = "pandoc"):
        self.pandoc_path = pandoc_path
        self._check_pandoc()

    def _check_pandoc(self) -> bool:
        """Check if pandoc is available."""
        try:
            subprocess.run(
                [self.pandoc_path, "--version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"Warning: pandoc not found at {self.pandoc_path}")
            return False

    def extract_metadata(self, docx_path: Path) -> dict:
        """Extract metadata from DOCX using python-docx."""
        if not DOCX_AVAILABLE:
            return {
                "title": docx_path.stem,
                "author": None,
                "created": None,
            }

        try:
            doc = Document(docx_path)
            props = doc.core_properties

            # First non-empty paragraph is usually the title
            title = ""
            for para in doc.paragraphs[:5]:
                text = para.text.strip()
                if text and len(text) > 5:
                    title = text
                    break

            return {
                "title": title or props.title or docx_path.stem,
                "author": props.author,
                "created": str(props.created) if props.created else None,
            }
        except Exception as e:
            print(f"Error extracting metadata from {docx_path}: {e}")
            return {
                "title": docx_path.stem,
                "author": None,
                "created": None,
            }

    def convert_to_markdown(self, docx_path: Path, extract_media: bool = False) -> str:
        """Convert DOCX to Markdown using pandoc."""
        cmd = [
            self.pandoc_path,
            str(docx_path),
            "-t", "markdown",
            "--wrap=none",
            "--standalone",
        ]

        if extract_media:
            cmd.extend(["--extract-media=."])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            if result.returncode != 0:
                print(f"Pandoc warning for {docx_path}: {result.stderr}")
            return result.stdout
        except subprocess.TimeoutExpired:
            print(f"Pandoc timeout for {docx_path}")
            return ""
        except Exception as e:
            print(f"Error converting {docx_path}: {e}")
            return ""

    def split_by_sections(self, markdown: str) -> dict[str, str]:
        """Split Markdown into sections by headers."""
        sections = {}
        current_section = "preamble"
        sections[current_section] = []

        for line in markdown.split('\n'):
            # Check for markdown headers (## Header)
            if line.startswith('## '):
                section_name = line[3:].strip()
                # Normalize section name for filename
                current_section = self._normalize_section_name(section_name)
                sections[current_section] = []
            elif line.startswith('# ') and current_section == "preamble":
                # First level header is usually title
                section_name = line[2:].strip()
                current_section = "title"
                sections[current_section] = [section_name]
            else:
                sections[current_section].append(line)

        # Join lines and filter empty sections
        return {
            k: '\n'.join(v).strip()
            for k, v in sections.items()
            if '\n'.join(v).strip()
        }

    def _normalize_section_name(self, name: str) -> str:
        """Normalize section name for use as filename."""
        # Convert to lowercase and replace non-alphanumeric with underscore
        normalized = re.sub(r'[^a-zа-яё0-9]+', '_', name.lower())
        # Remove leading/trailing underscores
        normalized = normalized.strip('_')
        # Limit length
        return normalized[:50] if normalized else "unnamed_section"

    def compute_file_hash(self, path: Path) -> str:
        """Compute MD5 hash of file for deduplication."""
        hasher = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]


def process_paper(
    docx_path: Path,
    output_dir: Path,
    paper_id: str,
    converter: Optional[DocxConverter] = None
) -> PaperMetadata:
    """Process a single paper: extract, convert, split, and save."""
    if converter is None:
        converter = DocxConverter()

    paper_dir = output_dir / paper_id
    paper_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract metadata
    raw_metadata = converter.extract_metadata(docx_path)

    # 2. Convert to markdown
    markdown = converter.convert_to_markdown(docx_path)
    if not markdown:
        raise ValueError(f"Failed to convert {docx_path}")

    (paper_dir / "full_text.md").write_text(markdown, encoding='utf-8')

    # 3. Split into sections
    sections = converter.split_by_sections(markdown)
    sections_dir = paper_dir / "sections"
    sections_dir.mkdir(exist_ok=True)

    for section_name, content in sections.items():
        section_file = sections_dir / f"{section_name}.txt"
        section_file.write_text(content, encoding='utf-8')

    # 4. Extract abstract separately
    abstract = sections.get('abstract',
               sections.get('аннотация',
               sections.get('preamble', '')))[:500]
    (paper_dir / "abstract.txt").write_text(abstract, encoding='utf-8')

    # 5. Build metadata
    metadata = PaperMetadata(
        paper_id=paper_id,
        title=raw_metadata.get('title', docx_path.stem),
        author=raw_metadata.get('author'),
        created=raw_metadata.get('created'),
        source_file=docx_path.name,
        abstract_preview=abstract[:200] + "..." if len(abstract) > 200 else abstract,
        word_count=len(markdown.split()),
        section_count=len(sections),
        file_hash=converter.compute_file_hash(docx_path),
        processed_at=datetime.now().isoformat(),
    )

    # 6. Save metadata
    (paper_dir / "metadata.json").write_text(
        json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    return metadata


def build_library(
    input_dir: Path,
    output_dir: Path,
    skip_existing: bool = True
) -> list[PaperMetadata]:
    """
    Convert all DOCX files in input_dir to structured library.

    Args:
        input_dir: Directory containing DOCX files
        output_dir: Output directory for structured library
        skip_existing: Skip already processed files (by hash)

    Returns:
        List of processed paper metadata
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Create output structure
    by_id_dir = output_dir / "by_id"
    by_id_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index if present
    index_path = output_dir / "index.json"
    existing_hashes = set()
    existing_index = []

    if index_path.exists() and skip_existing:
        try:
            existing_index = json.loads(index_path.read_text(encoding='utf-8'))
            existing_hashes = {p.get('file_hash') for p in existing_index}
            print(f"Found {len(existing_index)} existing papers")
        except Exception as e:
            print(f"Warning: Could not load existing index: {e}")

    converter = DocxConverter()
    new_papers = []

    # Find all DOCX files
    docx_files = list(input_dir.glob("**/*.docx"))
    print(f"Found {len(docx_files)} DOCX files")

    # Determine starting paper_id
    existing_ids = [p.get('paper_id', '') for p in existing_index]
    max_id = 0
    for pid in existing_ids:
        if pid.startswith('paper_'):
            try:
                max_id = max(max_id, int(pid.split('_')[1]))
            except ValueError:
                pass

    paper_num = max_id + 1

    for docx_path in docx_files:
        # Check if already processed
        file_hash = converter.compute_file_hash(docx_path)
        if file_hash in existing_hashes:
            print(f"Skipping (already processed): {docx_path.name}")
            continue

        paper_id = f"paper_{paper_num:04d}"
        print(f"Processing: {docx_path.name} -> {paper_id}")

        try:
            metadata = process_paper(docx_path, by_id_dir, paper_id, converter)
            new_papers.append(metadata)
            paper_num += 1
        except Exception as e:
            print(f"Error processing {docx_path}: {e}")
            continue

    # Merge and save index
    all_papers = existing_index + [asdict(m) for m in new_papers]
    index_path.write_text(
        json.dumps(all_papers, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f"Done! Processed {len(new_papers)} new papers. Total: {len(all_papers)}")
    return new_papers


def add_paper(docx_path: str | Path, library_dir: str | Path) -> PaperMetadata:
    """
    Add a single paper to existing library.

    Args:
        docx_path: Path to DOCX file
        library_dir: Path to library directory

    Returns:
        Metadata of added paper
    """
    docx_path = Path(docx_path)
    library_dir = Path(library_dir)

    if not docx_path.exists():
        raise FileNotFoundError(f"DOCX file not found: {docx_path}")

    # Load existing index
    index_path = library_dir / "index.json"
    existing_index = []
    if index_path.exists():
        existing_index = json.loads(index_path.read_text(encoding='utf-8'))

    # Check for duplicates
    converter = DocxConverter()
    file_hash = converter.compute_file_hash(docx_path)
    for paper in existing_index:
        if paper.get('file_hash') == file_hash:
            print(f"Paper already exists: {paper.get('paper_id')}")
            return PaperMetadata(**paper)

    # Determine new paper_id
    max_id = 0
    for p in existing_index:
        pid = p.get('paper_id', '')
        if pid.startswith('paper_'):
            try:
                max_id = max(max_id, int(pid.split('_')[1]))
            except ValueError:
                pass

    paper_id = f"paper_{max_id + 1:04d}"
    by_id_dir = library_dir / "by_id"
    by_id_dir.mkdir(parents=True, exist_ok=True)

    # Process paper
    metadata = process_paper(docx_path, by_id_dir, paper_id, converter)

    # Update index
    existing_index.append(asdict(metadata))
    index_path.write_text(
        json.dumps(existing_index, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    print(f"Added paper: {paper_id} - {metadata.title}")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DOCX papers to library")
    parser.add_argument("input_dir", help="Directory with DOCX files")
    parser.add_argument("output_dir", help="Output library directory")
    parser.add_argument("--no-skip", action="store_true", help="Re-process existing files")

    args = parser.parse_args()

    build_library(
        Path(args.input_dir),
        Path(args.output_dir),
        skip_existing=not args.no_skip
    )
