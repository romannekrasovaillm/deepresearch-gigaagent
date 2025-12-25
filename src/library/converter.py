"""DOCX to Markdown converter for scientific papers."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from docx import Document


@dataclass
class PaperMetadata:
    """Metadata extracted from a paper."""

    paper_id: str
    title: str
    author: Optional[str] = None
    created: Optional[str] = None
    source_file: str = ""
    abstract: str = ""
    sections: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "author": self.author,
            "created": self.created,
            "source_file": self.source_file,
            "abstract": self.abstract[:500] if self.abstract else "",
            "sections": self.sections,
        }


class DocxConverter:
    """Converts DOCX files to structured Markdown with sections."""

    # Common section headers in scientific papers (English + Russian)
    SECTION_PATTERNS = [
        r"^#+\s*(.+)$",  # Markdown headers
        r"^(Abstract|Аннотация|Введение|Introduction|Methods?|Методы|"
        r"Results?|Результаты|Discussion|Обсуждение|Conclusion|Заключение|"
        r"References|Литература|Appendix|Приложение)[\s:]*$",
    ]

    def __init__(self, pandoc_path: str = "pandoc"):
        self.pandoc_path = pandoc_path
        self._check_pandoc()

    def _check_pandoc(self) -> None:
        """Verify pandoc is installed."""
        try:
            subprocess.run(
                [self.pandoc_path, "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "pandoc not found. Install with: sudo apt install pandoc"
            )

    def extract_metadata(self, docx_path: Path) -> PaperMetadata:
        """Extract metadata from DOCX using python-docx."""
        doc = Document(str(docx_path))
        props = doc.core_properties

        # First non-empty paragraph is usually the title
        title = ""
        for para in doc.paragraphs:
            text = para.text.strip()
            if text and len(text) > 5:
                title = text
                break

        return PaperMetadata(
            paper_id="",  # Set later
            title=title or props.title or docx_path.stem,
            author=props.author,
            created=str(props.created) if props.created else None,
            source_file=docx_path.name,
        )

    def convert_to_markdown(
        self,
        docx_path: Path,
        extract_media: bool = True,
        media_dir: Optional[Path] = None,
    ) -> str:
        """Convert DOCX to Markdown using pandoc."""
        cmd = [
            self.pandoc_path,
            str(docx_path),
            "-t",
            "markdown",
            "--wrap=none",
            "--standalone",
        ]

        if extract_media and media_dir:
            media_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--extract-media", str(media_dir)])

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"pandoc failed: {result.stderr}")

        return result.stdout

    def split_by_sections(self, markdown: str) -> dict[str, str]:
        """Split markdown into sections by headers."""
        sections: dict[str, str] = {}
        current_section = "preamble"
        current_content: list[str] = []

        for line in markdown.split("\n"):
            # Check for section header
            is_header = False
            for pattern in self.SECTION_PATTERNS:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    # Save previous section
                    if current_content:
                        sections[current_section] = "\n".join(current_content).strip()

                    # Start new section
                    header = match.group(1) if match.groups() else line
                    current_section = self._normalize_section_name(header)
                    current_content = []
                    is_header = True
                    break

            if not is_header:
                current_content.append(line)

        # Save last section
        if current_content:
            sections[current_section] = "\n".join(current_content).strip()

        # Filter empty sections
        return {k: v for k, v in sections.items() if v.strip()}

    def _normalize_section_name(self, name: str) -> str:
        """Normalize section name for filesystem."""
        name = name.lower().strip()
        name = re.sub(r"^#+\s*", "", name)  # Remove markdown header markers
        name = re.sub(r"[^a-zа-яё0-9]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "untitled"

    def process_paper(
        self,
        docx_path: Path,
        output_dir: Path,
        paper_id: str,
    ) -> PaperMetadata:
        """Full pipeline: convert DOCX to structured library format."""
        paper_dir = output_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract metadata
        metadata = self.extract_metadata(docx_path)
        metadata.paper_id = paper_id

        # 2. Convert to markdown
        media_dir = paper_dir / "media"
        markdown = self.convert_to_markdown(docx_path, media_dir=media_dir)

        # 3. Save full text
        (paper_dir / "full_text.md").write_text(markdown, encoding="utf-8")

        # 4. Split into sections
        sections = self.split_by_sections(markdown)
        sections_dir = paper_dir / "sections"
        sections_dir.mkdir(exist_ok=True)

        metadata.sections = list(sections.keys())

        for name, content in sections.items():
            (sections_dir / f"{name}.txt").write_text(content, encoding="utf-8")

        # 5. Extract abstract separately
        abstract = sections.get(
            "abstract", sections.get("аннотация", sections.get("preamble", ""))
        )
        metadata.abstract = abstract[:2000]
        (paper_dir / "abstract.txt").write_text(abstract, encoding="utf-8")

        # 6. Save metadata
        (paper_dir / "metadata.json").write_text(
            json.dumps(metadata.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return metadata


def convert_to_markdown(docx_path: Path) -> str:
    """Convenience function for simple conversion."""
    converter = DocxConverter()
    return converter.convert_to_markdown(docx_path)


def extract_metadata(docx_path: Path) -> PaperMetadata:
    """Convenience function for metadata extraction."""
    converter = DocxConverter()
    return converter.extract_metadata(docx_path)
