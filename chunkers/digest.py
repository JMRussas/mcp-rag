#  mcp-rag - Digest File Chunker
#
#  Parses .digest files that contain nested module and type definitions.
#  One chunk per top-level type (class, struct, enum, interface) with
#  depth-aware module path tracking via a stack-based resolver.
#
#  Originally built for .digest.verse API definition files but applicable
#  to any format with nested module/type hierarchies and brace-delimited blocks.
#
#  Depends on: chunkers/base.py
#  Used by:    pipeline.py (via chunker registry)

import logging
import re
from pathlib import Path

from chunkers import register_chunker
from chunkers.base import BaseChunker, count_braces

log = logging.getLogger("pipeline")

# Patterns for type definitions
TYPE_DEF_RE = re.compile(
    r"^(\s*)"  # leading whitespace (capture indent)
    r"(\w+)"  # type name
    r"(<[^>]*>)*"  # optional tags like <native><public>
    r"\s*:=\s*"  # assignment operator
    r"(class|struct|enum|interface)"  # type keyword
)

# Pattern for module declarations (with opening brace on same line)
MODULE_DEF_RE = re.compile(
    r"^(\s*)"  # leading whitespace
    r"(\w+)"  # module name
    r"(<[^>]*>)*"  # optional tags
    r"\s*:=\s*module\s*\{"  # module assignment with brace
)

# Pattern for module import path comments
MODULE_PATH_RE = re.compile(r"^\s*#\s*Module import path:\s*(.+)")

# Pattern for standalone functions (extension methods or free functions)
FUNC_DEF_RE = re.compile(
    r"^(\s*)"  # leading whitespace
    r"(?:\([^)]+\)\.)?"  # optional receiver like (InAgent: agent).
    r"(\w+)"  # function name
    r"(<[^>]*>)*"  # optional tags
    r"\s*\("  # opening paren
)


class DigestChunker(BaseChunker):
    """Chunks .digest definition files by type boundary.

    Handles nested module structures with depth-aware module path tracking.
    Produces one chunk per type definition or module-level function group.
    """

    def chunk_directory(self, source_dir: Path, repo_config: dict) -> list[dict]:
        # Find digest files based on repo config
        digest_dirs = repo_config.get("digest_dirs", {})
        include_internal = repo_config.get("include_internal", False)
        extension = repo_config.get("extension", ".digest.verse")

        if not digest_dirs:
            # Default: scan for all digest files
            return self._chunk_all_digest_files(source_dir, repo_config, extension)

        chunks = []
        for sub_name, base_path in digest_dirs.items():
            digest_file = source_dir / sub_name / f"{sub_name}{extension}"
            if digest_file.exists():
                source_name = sub_name.lower()
                file_chunks = _chunk_digest_file(digest_file, source_name, base_path)
                chunks.extend(file_chunks)
                log.info(f"  [{sub_name}] {len(file_chunks)} chunks from {digest_file.name}")

            if include_internal:
                internal_file = source_dir / sub_name / f"{sub_name}Internal{extension}"
                if internal_file.exists():
                    source_name = f"{sub_name.lower()}_internal"
                    file_chunks = _chunk_digest_file(internal_file, source_name, base_path)
                    chunks.extend(file_chunks)
                    log.info(f"  [{sub_name}Internal] {len(file_chunks)} chunks")

        return chunks

    def _chunk_all_digest_files(self, source_dir: Path, repo_config: dict, extension: str) -> list[dict]:
        """Fallback: chunk all digest files found in the directory."""
        source = repo_config.get("source_tag", "digest")
        chunks = []

        for digest_file in sorted(source_dir.rglob(f"*{extension}")):
            file_chunks = _chunk_digest_file(digest_file, source)
            chunks.extend(file_chunks)

        log.info(f"  [{repo_config['name']}] {len(chunks)} chunks")
        return chunks


def _build_cumulative_depth(lines: list[str]) -> list[int]:
    """Pre-compute cumulative brace depth after each line (comment-aware).

    cum_depth[i] is the brace depth after processing line i.
    Depth before line i is cum_depth[i-1] (or 0 for i == 0).
    """
    cum_depth = []
    depth = 0
    in_block = False
    for line in lines:
        delta, in_block = count_braces(line, in_block, line_comment="#")
        depth += delta
        cum_depth.append(depth)
    return cum_depth


def _chunk_digest_file(
    file_path: Path,
    source_name: str = "digest",
    base_module_path: str = "",
) -> list[dict]:
    """Parse a digest file into chunks, one per type definition."""
    text = file_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # First pass: compute module path at each line
    module_path_at = _build_module_path_map(lines, base_module_path)

    # Pre-compute cumulative brace depth for O(1) lookups
    cum_depth = _build_cumulative_depth(lines)

    chunks = []
    i = 0
    while i < len(lines):
        line = lines[i]

        if MODULE_PATH_RE.match(line):
            i += 1
            continue

        if MODULE_DEF_RE.match(line):
            i += 1
            continue

        # Check for type definition
        tm = TYPE_DEF_RE.match(line)
        if tm:
            type_name = tm.group(2)
            module_path = module_path_at.get(i, "")

            doc_start = i
            while doc_start > 0 and _is_doc_or_annotation(lines[doc_start - 1]):
                doc_start -= 1

            end = _find_block_end(lines, i)

            chunk_lines = lines[doc_start : end + 1]
            chunk_text = "\n".join(chunk_lines)

            if len(chunk_lines) >= 1:
                chunk_id = f"digest:{source_name}:{module_path}:{type_name}"
                chunks.append(
                    {
                        "id": chunk_id,
                        "text": chunk_text,
                        "source": source_name,
                        "module_path": module_path,
                        "type_name": type_name,
                        "category": "",
                        "heading": "",
                        "file_path": str(file_path.name),
                    }
                )

            i = end + 1
            continue

        # Check for standalone functions at module level
        fm = FUNC_DEF_RE.match(line)
        if fm:
            depth = cum_depth[i - 1] if i > 0 else 0
            if depth >= 1 and not _inside_type_def(lines, i, cum_depth):
                module_path = module_path_at.get(i, "")

                block_start = i
                while block_start > 0 and _is_doc_or_annotation(lines[block_start - 1]):
                    block_start -= 1

                block_end = i
                base_indent = len(fm.group(1))
                j = i + 1
                while j < len(lines):
                    next_line = lines[j]
                    if not next_line.strip():
                        j += 1
                        continue
                    if next_line.strip().startswith("#") or next_line.strip().startswith("@"):
                        j += 1
                        continue
                    next_fm = FUNC_DEF_RE.match(next_line)
                    if next_fm and len(next_fm.group(1)) == base_indent:
                        block_end = j
                        j += 1
                        continue
                    break

                chunk_lines = lines[block_start : block_end + 1]
                chunk_text = "\n".join(chunk_lines)
                func_name = fm.group(2)

                if chunk_text.strip():
                    chunk_id = f"digest:{source_name}:{module_path}:func:{func_name}"
                    chunks.append(
                        {
                            "id": chunk_id,
                            "text": chunk_text,
                            "source": source_name,
                            "module_path": module_path,
                            "type_name": func_name,
                            "category": "function",
                            "heading": "",
                            "file_path": str(file_path.name),
                        }
                    )

                i = block_end + 1
                continue

        i += 1

    return chunks


def _build_module_path_map(lines: list[str], base_module_path: str = "") -> dict[int, str]:
    """Build a mapping of line_number -> module_path using depth-aware tracking."""
    path_stack: list[tuple[int, str]] = []
    pending_path = ""
    depth = 0
    in_block = False
    result: dict[int, str] = {}

    for i, line in enumerate(lines):
        m = MODULE_PATH_RE.match(line)
        if m:
            pending_path = m.group(1).strip()
            result[i] = _current_path(path_stack)
            continue

        mm = MODULE_DEF_RE.match(line)
        if mm:
            module_name = mm.group(2)
            if pending_path:
                mod_path = pending_path
                pending_path = ""
            else:
                parent = _current_path(path_stack) or base_module_path
                if parent:
                    mod_path = f"{parent}/{module_name}"
                else:
                    mod_path = f"/{module_name}"

            delta, in_block = count_braces(line, in_block, line_comment="#")
            depth += delta
            # Pop before push so closing braces on the same line are handled
            _pop_stack_to_depth(path_stack, depth)

            path_stack.append((depth, mod_path))
            result[i] = mod_path
            continue

        delta, in_block = count_braces(line, in_block, line_comment="#")
        depth += delta
        if delta < 0:
            _pop_stack_to_depth(path_stack, depth)

        result[i] = _current_path(path_stack)

    return result


def _current_path(path_stack: list[tuple[int, str]]) -> str:
    if path_stack:
        return path_stack[-1][1]
    return ""


def _pop_stack_to_depth(path_stack: list[tuple[int, str]], depth: int):
    while path_stack and path_stack[-1][0] > depth:
        path_stack.pop()


def _is_doc_or_annotation(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.startswith("@"):
        return True
    if stripped.startswith("using"):
        return True
    return False


def _find_block_end(lines: list[str], start: int) -> int:
    """Find the line where a brace-delimited block closes (comment-aware)."""
    brace_depth = 0
    found_open = False
    in_block = False

    for i in range(start, len(lines)):
        delta, in_block = count_braces(lines[i], in_block, line_comment="#")
        for _ in range(delta):
            brace_depth += 1
            found_open = True
        for _ in range(-delta):
            brace_depth -= 1
            if found_open and brace_depth == 0:
                return i

    return start


def _inside_type_def(lines: list[str], line_idx: int, cum_depth: list[int]) -> bool:
    """Check if line_idx is inside a type definition block.

    Uses the cumulative depth array to find the line that opened the
    enclosing brace, then checks if that line is a type definition.
    """
    target_depth = cum_depth[line_idx - 1] if line_idx > 0 else 0
    if target_depth < 1:
        return False

    # Walk backward to find the line where depth increased to target_depth
    for i in range(line_idx - 1, -1, -1):
        depth_before = cum_depth[i - 1] if i > 0 else 0
        depth_after = cum_depth[i]
        # This line opened a brace that contributes to our current depth
        if depth_before < target_depth <= depth_after:
            if TYPE_DEF_RE.match(lines[i]):
                return True
            return False

    return False


register_chunker("digest", DigestChunker)
