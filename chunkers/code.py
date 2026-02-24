#  mcp-rag - Generic Code Chunker
#
#  Chunks source code files with one chunk per file. Useful for languages
#  without a dedicated parser, or for code examples and snippets.
#  Extracts category from directory structure.
#
#  Depends on: chunkers/base.py
#  Used by:    pipeline.py (via chunker registry)

import re
from pathlib import Path

from chunkers import register_chunker
from chunkers.base import BaseChunker

# Default directories to skip
SKIP_DIRS = {".git", "node_modules", "__pycache__", "obj", "bin", ".venv", "venv"}

# Bytes to read when checking if a file is binary
_BINARY_CHECK_SIZE = 8192


def _is_binary(file_path: Path) -> bool:
    """Check if a file is likely binary by looking for null bytes (same heuristic as Git)."""
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(_BINARY_CHECK_SIZE)
        return b"\x00" in chunk
    except OSError:
        return True


class CodeChunker(BaseChunker):
    """Chunks source code files, one chunk per file.

    Config options:
        source_tag: Source identifier (default: "code")
        extensions: List of file extensions to include (default: all files)
        skip_patterns: List of filename patterns to skip
    """

    def chunk_directory(self, source_dir: Path, repo_config: dict) -> list[dict]:
        if not source_dir.exists():
            return []

        source = repo_config.get("source_tag", "code")
        extensions = repo_config.get("extensions", [])
        skip_patterns = set(repo_config.get("skip_patterns", []))
        chunks = []

        if extensions:
            files = []
            for ext in extensions:
                files.extend(source_dir.rglob(f"*.{ext.lstrip('.')}"))
            files = sorted(set(files))
        else:
            files = sorted(source_dir.rglob("*"))
            files = [f for f in files if f.is_file() and not _is_binary(f)]

        for source_file in files:
            rel = source_file.relative_to(source_dir)
            if any(part in SKIP_DIRS for part in rel.parts):
                continue
            if any(pat in source_file.name for pat in skip_patterns):
                continue

            chunk = _chunk_code_file(source_file, source_dir, source)
            if chunk:
                chunks.append(chunk)

        print(f"  [{repo_config['name']}] {len(chunks)} chunks")
        return chunks


def _chunk_code_file(file_path: Path, repo_dir: Path, source: str = "code") -> dict | None:
    """Create a single chunk from a source code file."""
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    except (UnicodeDecodeError, OSError):
        return None

    if not text or len(text) < 20:
        return None

    rel_path = file_path.relative_to(repo_dir)
    parts = rel_path.parts
    category = parts[0] if len(parts) > 1 else "general"

    file_stem = file_path.stem
    ext = file_path.suffix.lstrip(".").lower()
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", file_stem.lower())
    safe_cat = re.sub(r"[^a-zA-Z0-9_-]", "_", category.lower())
    chunk_id = f"{source}:{safe_cat}:{safe_name}_{ext}" if ext else f"{source}:{safe_cat}:{safe_name}"

    return {
        "id": chunk_id,
        "text": text,
        "source": source,
        "module_path": "",
        "type_name": "",
        "category": category,
        "heading": "",
        "file_path": str(rel_path).replace("\\", "/"),
    }


register_chunker("code", CodeChunker)
