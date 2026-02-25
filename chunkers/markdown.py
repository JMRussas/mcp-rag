#  mcp-rag - Markdown Chunker
#
#  Chunks markdown documentation files by heading level (h1-h3).
#  Preserves heading hierarchy as metadata. Strips YAML frontmatter.
#
#  Depends on: chunkers/base.py
#  Used by:    pipeline.py (via chunker registry)

import logging
import re
from pathlib import Path

from chunkers import register_chunker
from chunkers.base import BaseChunker

log = logging.getLogger("pipeline")

# Headings to split on
HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)")

# Default files to skip (overridable via repo_config "skip_files")
DEFAULT_SKIP_FILES = {"README.md", "SUMMARY.md", "LICENSE.md", "CONTRIBUTING.md"}

# Directories to skip
SKIP_DIRS = {"lib", ".git", "node_modules", "__pycache__"}


class MarkdownChunker(BaseChunker):
    """Chunks markdown files by heading, one chunk per section."""

    def chunk_directory(self, source_dir: Path, repo_config: dict) -> list[dict]:
        source_tag = repo_config.get("source_tag", "docs")
        include_filter = set(repo_config.get("include", []))
        skip_files = set(repo_config.get("skip_files", DEFAULT_SKIP_FILES))
        no_recurse = repo_config.get("no_recurse", False)

        if not source_dir.exists():
            log.warning(f"{source_dir} not found, skipping.")
            return []

        chunks = []
        glob_fn = source_dir.glob if no_recurse else source_dir.rglob

        for md_file in sorted(glob_fn("*.md")):
            if md_file.name in skip_files:
                continue
            if any(part.startswith(".") for part in md_file.relative_to(source_dir).parts[:-1]):
                continue
            if any(part in SKIP_DIRS for part in md_file.relative_to(source_dir).parts):
                continue
            if include_filter and md_file.name not in include_filter:
                continue

            file_chunks = _chunk_markdown_file(md_file, source_tag, source_dir)
            chunks.extend(file_chunks)

        log.info(f"  [{repo_config['name']}] {len(chunks)} chunks from {source_dir.name}")
        return chunks


def _chunk_markdown_file(
    file_path: Path,
    source_tag: str = "docs",
    source_dir: Path | None = None,
) -> list[dict]:
    """Split a markdown file into chunks by heading."""
    text = file_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    rel_path = str(file_path.relative_to(source_dir)).replace("\\", "/") if source_dir else file_path.name

    # Strip YAML frontmatter
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                lines = lines[i + 1 :]
                break

    chunks = []
    current_heading = file_path.stem.replace("-", " ").replace("_", " ")
    current_lines: list[str] = []
    section_idx = 0

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            if current_lines:
                chunk_text = "\n".join(current_lines).strip()
                if chunk_text and len(chunk_text) > 20:
                    chunks.append(
                        _make_chunk(file_path, current_heading, chunk_text, section_idx, source_tag, rel_path)
                    )
                    section_idx += 1

            current_heading = m.group(2).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        chunk_text = "\n".join(current_lines).strip()
        if chunk_text and len(chunk_text) > 20:
            chunks.append(_make_chunk(file_path, current_heading, chunk_text, section_idx, source_tag, rel_path))

    return chunks


def _make_chunk(
    file_path: Path,
    heading: str,
    text: str,
    idx: int,
    source_tag: str,
    rel_path: str = "",
) -> dict:
    """Create a chunk dict for a markdown section."""
    file_stem = file_path.stem
    safe_heading = re.sub(r"[^a-zA-Z0-9_-]", "_", heading.lower())[:50]
    chunk_id = f"{source_tag}:{file_stem}:{safe_heading}:{idx}"

    return {
        "id": chunk_id,
        "text": text,
        "source": source_tag,
        "module_path": "",
        "type_name": "",
        "category": file_stem,
        "heading": heading,
        "file_path": rel_path or str(file_path.name),
    }


register_chunker("markdown", MarkdownChunker)
