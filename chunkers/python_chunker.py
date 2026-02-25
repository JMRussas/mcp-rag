#  mcp-rag - Python Source Chunker
#
#  Parses Python files using the ast module to extract top-level classes
#  and functions as individual chunks. Preserves docstrings, decorators,
#  and module-level context.
#
#  Depends on: chunkers/base.py
#  Used by:    pipeline.py (via chunker registry)

import ast
import logging
from pathlib import Path

from chunkers import register_chunker
from chunkers.base import BaseChunker, derive_category

log = logging.getLogger("pipeline")


class PythonChunker(BaseChunker):
    """Chunks Python source files by top-level class and function definitions.

    Uses the ast module for reliable parsing. Falls back to whole-file
    chunking if the file has syntax errors or no top-level definitions.
    """

    def chunk_directory(self, source_dir: Path, repo_config: dict) -> list[dict]:
        source_tag = repo_config.get("source_tag", "python")

        if not source_dir.exists():
            log.warning(f"{source_dir} not found, skipping.")
            return []

        chunks = []
        py_files = sorted(source_dir.rglob("*.py"))

        # Filter out common non-source directories
        skip_dirs = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".tox", ".mypy_cache"}
        py_files = [f for f in py_files if not any(part in skip_dirs for part in f.relative_to(source_dir).parts)]

        for py_file in py_files:
            file_chunks = _chunk_python_file(py_file, source_tag, source_dir)
            chunks.extend(file_chunks)

        log.info(f"  [{repo_config['name']}] {len(chunks)} chunks from {len(py_files)} .py files")
        return chunks


def _chunk_python_file(
    file_path: Path,
    source_tag: str = "python",
    base_dir: Path | None = None,
) -> list[dict]:
    """Parse a Python file into chunks, one per top-level class or function."""
    try:
        text = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    if not text.strip():
        return []

    rel_path = str(file_path.relative_to(base_dir)) if base_dir else file_path.name
    rel_path = rel_path.replace("\\", "/")
    category = derive_category(rel_path)
    module_path = _derive_module_path(rel_path)

    # Try to parse the AST
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Fall back to whole-file chunk
        if len(text.strip()) < 20:
            return []
        chunk_id = f"python:{source_tag}:{module_path}.{file_path.stem}"
        return [
            {
                "id": chunk_id,
                "text": text.strip(),
                "source": source_tag,
                "module_path": module_path,
                "type_name": file_path.stem,
                "category": category,
                "heading": "",
                "file_path": rel_path,
            }
        ]

    lines = text.splitlines()

    # Find top-level class and function definitions
    top_level_defs = [
        node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not top_level_defs:
        # No definitions — chunk the whole file (might be a script or config)
        if len(text.strip()) < 20:
            return []
        chunk_id = f"python:{source_tag}:{module_path}.{file_path.stem}"
        return [
            {
                "id": chunk_id,
                "text": text.strip(),
                "source": source_tag,
                "module_path": module_path,
                "type_name": file_path.stem,
                "category": category,
                "heading": "",
                "file_path": rel_path,
            }
        ]

    # Single definition — chunk the whole file to preserve imports and context
    if len(top_level_defs) == 1:
        node = top_level_defs[0]
        chunk_id = f"python:{source_tag}:{module_path}.{node.name}"
        return [
            {
                "id": chunk_id,
                "text": text.strip(),
                "source": source_tag,
                "module_path": module_path,
                "type_name": node.name,
                "category": category,
                "heading": _get_docstring_summary(node),
                "file_path": rel_path,
            }
        ]

    # Multiple definitions — emit one chunk per definition
    # Include module header (imports, constants) as context
    header_end = _find_header_end(lines, top_level_defs[0])
    header_text = "\n".join(lines[:header_end]).rstrip()

    chunks = []
    for i, node in enumerate(top_level_defs):
        # Include decorators in the start line
        start = _node_start_with_decorators(node) - 1  # 0-indexed

        if i + 1 < len(top_level_defs):
            next_start = _node_start_with_decorators(top_level_defs[i + 1]) - 1
            # Walk backwards from next definition to skip blank lines
            end = next_start
            while end > start and not lines[end - 1].strip():
                end -= 1
        else:
            end = len(lines)

        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines).strip()

        if not chunk_text or len(chunk_text) < 20:
            continue

        # Prepend header for context
        if header_text:
            chunk_text = header_text + "\n\n" + chunk_text

        chunk_id = f"python:{source_tag}:{module_path}.{node.name}"
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "source": source_tag,
                "module_path": module_path,
                "type_name": node.name,
                "category": category,
                "heading": _get_docstring_summary(node),
                "file_path": rel_path,
            }
        )

    return chunks


def _derive_module_path(rel_path: str) -> str:
    """Derive a Python-style module path from relative file path."""
    # Remove .py extension and convert slashes to dots
    module = rel_path.replace("/", ".").replace("\\", ".")
    if module.endswith(".py"):
        module = module[:-3]
    if module.endswith(".__init__"):
        module = module[:-9]
    return module


def _get_docstring_summary(node: ast.AST) -> str:
    """Extract the first line of a docstring from a class or function."""
    docstring = ast.get_docstring(node)
    if docstring:
        return docstring.split("\n")[0].strip()
    return ""


def _node_start_with_decorators(node: ast.AST) -> int:
    """Get the starting line number including decorators."""
    if hasattr(node, "decorator_list") and node.decorator_list:
        return node.decorator_list[0].lineno
    return node.lineno


def _find_header_end(lines: list[str], first_def: ast.AST) -> int:
    """Find where the module header ends (before the first definition).

    The header includes imports, module docstrings, and constants.
    """
    first_line = _node_start_with_decorators(first_def) - 1  # 0-indexed

    # Walk backwards from first definition to skip blank lines
    end = first_line
    while end > 0 and not lines[end - 1].strip():
        end -= 1

    return end


register_chunker("python", PythonChunker)
