#  mcp-rag - Base Chunker
#
#  Abstract base class for all chunkers. Every chunker must implement
#  chunk_directory() and return standardized chunk dicts.
#
#  Depends on: (none)
#  Used by:    chunkers/__init__.py, pipeline.py, chunkers/*.py

from abc import ABC, abstractmethod
from pathlib import Path


def count_braces(
    line: str,
    in_block_comment: bool = False,
    line_comment: str = "//",
) -> tuple[int, bool]:
    """Count net brace depth change for a line, skipping strings and comments.

    Handles: string literals ("..."), verbatim strings (@"..."), char literals ('{'),
    single-line comments, and block comments (/* */).

    Args:
        line: Source line to scan.
        in_block_comment: Whether we're inside a /* */ block comment.
        line_comment: Single-line comment prefix ("//", "#", etc.).

    Returns (net_brace_delta, still_in_block_comment).
    """
    delta = 0
    i = 0
    n = len(line)
    lc_len = len(line_comment)

    while i < n:
        ch = line[i]

        if in_block_comment:
            if ch == "*" and i + 1 < n and line[i + 1] == "/":
                in_block_comment = False
                i += 2
            else:
                i += 1
            continue

        # Single-line comment — rest of line is ignored
        if line_comment and line[i : i + lc_len] == line_comment:
            break

        # Block comment start (only for C-style languages)
        if lc_len == 2 and line_comment == "//" and ch == "/" and i + 1 < n and line[i + 1] == "*":
            in_block_comment = True
            i += 2
            continue

        # Char literal — skip contents (e.g. '{' or '\\')
        if ch == "'":
            i += 1
            if i < n and line[i] == "\\":
                i += 1  # skip escaped char
            if i < n:
                i += 1  # skip the char itself
            if i < n and line[i] == "'":
                i += 1  # skip closing quote
            continue

        # String literal — skip to closing quote
        if ch == '"':
            # Check for verbatim string (@"...")
            verbatim = i > 0 and line[i - 1] == "@"
            i += 1
            while i < n:
                if line[i] == "\\" and not verbatim:
                    i += 2  # skip escape sequence
                    continue
                if line[i] == '"':
                    if verbatim and i + 1 < n and line[i + 1] == '"':
                        i += 2  # doubled quote inside verbatim
                        continue
                    i += 1  # closing quote
                    break
                i += 1
            continue

        if ch == "{":
            delta += 1
        elif ch == "}":
            delta -= 1

        i += 1

    return delta, in_block_comment


def derive_category(rel_path: str) -> str:
    """Derive a category label from a relative file path.

    Uses the parent directory name as the category.
    e.g. "src/utils/helpers.py" → "utils", "helpers.py" → "core"
    """
    parts = rel_path.split("/")
    if len(parts) > 1:
        return parts[-2].lower()
    return "core"


class BaseChunker(ABC):
    """Abstract base class for source code and document chunkers.

    Every chunker takes a source directory and repo config, and returns
    a list of standardized chunk dicts with these keys:

        id:          Unique chunk identifier (format: "source:category:name")
        text:        The chunk content
        source:      Source tag (e.g. "engine", "docs", "examples")
        module_path: Namespace, module path, or package (language-dependent)
        type_name:   Class, function, or type name (if applicable)
        category:    Subdirectory or grouping label
        heading:     Section heading (for markdown chunks)
        file_path:   Relative path to the source file
    """

    @abstractmethod
    def chunk_directory(self, source_dir: Path, repo_config: dict) -> list[dict]:
        """Chunk all relevant files in a directory.

        Args:
            source_dir: Root directory containing source files.
            repo_config: Repo configuration dict from config.json.

        Returns:
            List of chunk dicts with the standard keys.
        """
        ...
