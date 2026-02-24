#  mcp-rag - Markdown Chunker Tests

from chunkers.markdown import MarkdownChunker, _chunk_markdown_file

REQUIRED_KEYS = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path"}


def test_heading_split(tmp_markdown_file, tmp_path):
    """File with 3 headings produces 3 chunks with correct heading metadata."""
    chunks = _chunk_markdown_file(tmp_markdown_file, "docs", tmp_path)
    assert len(chunks) == 3
    headings = [c["heading"] for c in chunks]
    assert headings == ["Getting Started", "Installation", "Configuration"]
    for c in chunks:
        assert set(c.keys()) == REQUIRED_KEYS


def test_file_path_is_relative(tmp_path):
    """file_path uses relative path from source_dir, not just the filename."""
    subdir = tmp_path / "guides"
    subdir.mkdir()
    md = """\
# Topic

This is a guide section with enough text to pass minimum length.
"""
    f = subdir / "topic.md"
    f.write_text(md, encoding="utf-8")

    chunks = _chunk_markdown_file(f, "docs", tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["file_path"] == "guides/topic.md"


def test_yaml_frontmatter_stripped(tmp_path):
    """YAML frontmatter is stripped before chunking."""
    md = """\
---
title: Test Doc
date: 2025-01-01
---

# Main Content

This is the main content after frontmatter.
It should be the only chunk returned here.
"""
    f = tmp_path / "frontmatter.md"
    f.write_text(md, encoding="utf-8")

    chunks = _chunk_markdown_file(f, "docs")
    assert len(chunks) == 1
    assert chunks[0]["heading"] == "Main Content"
    assert "title:" not in chunks[0]["text"]
    assert "date:" not in chunks[0]["text"]


def test_readme_skipped(tmp_path):
    """README.md is in SKIP_FILES and should not produce chunks from chunk_directory."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# README\n\nThis is the readme with enough text to pass minimum length checks.", encoding="utf-8"
    )

    chunker = MarkdownChunker()
    repo_config = {"name": "test", "source_tag": "docs"}
    chunks = chunker.chunk_directory(tmp_path, repo_config)
    assert chunks == []
