#  mcp-rag - C# Source Chunker
#
#  Parses .cs files into chunks, one per top-level type definition
#  (class, struct, enum, interface, record). Extracts namespace, file header
#  comments (description, Depends on, Used by), and XML doc comments.
#  Uses brace-depth tracking to find type boundaries.
#
#  Depends on: chunkers/base.py
#  Used by:    pipeline.py (via chunker registry)

import re
from pathlib import Path

from chunkers import register_chunker
from chunkers.base import BaseChunker, count_braces, derive_category

# Top-level type declarations (handles modifiers like public, static, partial, etc.)
TYPE_DEF_RE = re.compile(
    r"^(\s*)"  # leading whitespace
    r"(?:(?:public|internal|private|protected|static|abstract|sealed|partial|readonly|unsafe|new|file)\s+)*"
    r"(class|struct|enum|interface|record)\s+"  # type keyword
    r"(\w+)"  # type name
)

# Namespace declaration (file-scoped: "namespace X;")
NAMESPACE_RE = re.compile(r"^\s*namespace\s+([\w.]+)\s*;")

# Block-scoped namespace: "namespace X {"
NAMESPACE_BLOCK_RE = re.compile(r"^\s*namespace\s+([\w.]+)\s*\{")

# File header comment block (lines starting with //)
HEADER_COMMENT_RE = re.compile(r"^\s*//")

# Dependency header patterns
DEPENDS_RE = re.compile(r"//\s*Depends on:\s*(.+)", re.IGNORECASE)
USED_BY_RE = re.compile(r"//\s*Used by:\s*(.+)", re.IGNORECASE)

# XML doc comment
XML_DOC_RE = re.compile(r"^\s*///")


def _qualified_name(namespace: str, name: str) -> str:
    """Build a dotted qualified name, omitting the dot if namespace is empty."""
    return f"{namespace}.{name}" if namespace else name


class CSharpChunker(BaseChunker):
    """Chunks C# source files by top-level type definition."""

    def chunk_directory(self, source_dir: Path, repo_config: dict) -> list[dict]:
        source_tag = repo_config.get("source_tag", "csharp")

        if not source_dir.exists():
            print(f"  Warning: {source_dir} not found, skipping.")
            return []

        chunks = []
        cs_files = sorted(source_dir.rglob("*.cs"))

        # Filter out obj/, bin/, .git/ directories
        cs_files = [
            f for f in cs_files if not any(part in ("obj", "bin", ".git") for part in f.relative_to(source_dir).parts)
        ]

        for cs_file in cs_files:
            file_chunks = _chunk_csharp_file(cs_file, source_tag, source_dir)
            chunks.extend(file_chunks)

        print(f"  [{repo_config['name']}] {len(chunks)} chunks from {len(cs_files)} .cs files")
        return chunks


def _chunk_csharp_file(
    file_path: Path,
    source_tag: str = "engine",
    base_dir: Path | None = None,
) -> list[dict]:
    """Parse a .cs file into chunks, one per top-level type definition."""
    try:
        text = file_path.read_text(encoding="utf-8-sig")
    except (UnicodeDecodeError, OSError):
        return []

    lines = text.splitlines()
    if not lines:
        return []

    # Extract file-level metadata
    namespace = _extract_namespace(lines)
    rel_path = str(file_path.relative_to(base_dir)) if base_dir else file_path.name
    rel_path = rel_path.replace("\\", "/")
    category = derive_category(rel_path)
    header_info = _extract_header_info(lines)

    # Find all top-level type definitions
    type_defs = _find_type_definitions(lines)

    if not type_defs:
        # No types found — chunk the whole file
        chunk_text = text.strip()
        if len(chunk_text) < 20:
            return []
        chunk_id = f"csharp:{source_tag}:{_qualified_name(namespace, file_path.stem)}"
        return [
            {
                "id": chunk_id,
                "text": chunk_text,
                "source": source_tag,
                "module_path": namespace,
                "type_name": file_path.stem,
                "category": category,
                "heading": header_info.get("description", ""),
                "file_path": rel_path,
            }
        ]

    # Single type — chunk the whole file as one unit
    if len(type_defs) == 1:
        td = type_defs[0]
        chunk_id = f"csharp:{source_tag}:{_qualified_name(namespace, td['name'])}"
        return [
            {
                "id": chunk_id,
                "text": text.strip(),
                "source": source_tag,
                "module_path": namespace,
                "type_name": td["name"],
                "category": category,
                "heading": header_info.get("description", ""),
                "file_path": rel_path,
            }
        ]

    # Multiple types — emit one chunk per type, plus file header as context
    header_text = _extract_file_header(lines)
    chunks = []

    for i, td in enumerate(type_defs):
        start = td["doc_start"]
        if i + 1 < len(type_defs):
            end = type_defs[i + 1]["doc_start"]
        else:
            end = len(lines)

        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines).strip()

        if not chunk_text or len(chunk_text) < 20:
            continue

        if header_text:
            chunk_text = header_text + "\n\n" + chunk_text

        chunk_id = f"csharp:{source_tag}:{_qualified_name(namespace, td['name'])}"
        chunks.append(
            {
                "id": chunk_id,
                "text": chunk_text,
                "source": source_tag,
                "module_path": namespace,
                "type_name": td["name"],
                "category": category,
                "heading": header_info.get("description", ""),
                "file_path": rel_path,
            }
        )

    return chunks


def _extract_namespace(lines: list[str]) -> str:
    """Extract namespace from file-scoped or block-scoped declaration."""
    for line in lines:
        m = NAMESPACE_RE.match(line)
        if m:
            return m.group(1)
        m = NAMESPACE_BLOCK_RE.match(line)
        if m:
            return m.group(1)
    return ""


def _extract_header_info(lines: list[str]) -> dict:
    """Extract file header comments including description and dependency info."""
    info: dict[str, str] = {}
    header_lines = []
    dep_line_indices: set[int] = set()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("//"):
            idx = len(header_lines)
            header_lines.append(stripped.lstrip("/ ").strip())

            m = DEPENDS_RE.search(line)
            if m:
                info["depends_on"] = m.group(1).strip()
                dep_line_indices.add(idx)
            m = USED_BY_RE.search(line)
            if m:
                info["used_by"] = m.group(1).strip()
                dep_line_indices.add(idx)
        elif stripped.startswith("using") or stripped.startswith("namespace") or not stripped:
            continue
        else:
            break

    desc_lines = [line for idx, line in enumerate(header_lines) if line and idx not in dep_line_indices]
    if desc_lines:
        info["description"] = " ".join(desc_lines[:4])

    return info


def _extract_file_header(lines: list[str]) -> str:
    """Extract everything before the first type definition."""
    result = []
    for line in lines:
        if TYPE_DEF_RE.match(line):
            break
        if XML_DOC_RE.match(line):
            break
        result.append(line)

    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)


def _find_type_definitions(lines: list[str]) -> list[dict]:
    """Find all top-level type definitions with their boundaries."""
    types = []
    brace_depth = 0
    in_block_comment = False
    in_namespace_block = False

    for line in lines:
        if NAMESPACE_BLOCK_RE.match(line):
            in_namespace_block = True
            break
        if NAMESPACE_RE.match(line):
            break

    top_level_depth = 1 if in_namespace_block else 0

    i = 0
    while i < len(lines):
        line = lines[i]

        if brace_depth == top_level_depth:
            m = TYPE_DEF_RE.match(line)
            if m:
                type_name = m.group(3)

                doc_start = i
                while doc_start > 0:
                    prev = lines[doc_start - 1].strip()
                    if prev.startswith("///") or prev.startswith("[") or prev.startswith("//"):
                        doc_start -= 1
                    elif not prev:
                        doc_start -= 1
                    else:
                        break

                while doc_start < i and not lines[doc_start].strip():
                    doc_start += 1

                types.append(
                    {
                        "name": type_name,
                        "keyword": m.group(2),
                        "line": i,
                        "doc_start": doc_start,
                    }
                )

        delta, in_block_comment = count_braces(line, in_block_comment)
        brace_depth += delta

        i += 1

    return types


register_chunker("csharp", CSharpChunker)
