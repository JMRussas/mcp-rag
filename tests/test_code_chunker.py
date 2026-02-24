#  mcp-rag - Code Chunker Tests

from chunkers.code import CodeChunker, _chunk_code_file, _is_binary

REQUIRED_KEYS = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path"}


def test_single_file_chunk(tmp_path):
    """A code file produces one chunk with all required keys."""
    f = tmp_path / "hello.py"
    f.write_text("print('hello world')\n# some more content here to pass min length", encoding="utf-8")

    chunk = _chunk_code_file(f, tmp_path, "test")
    assert chunk is not None
    assert set(chunk.keys()) == REQUIRED_KEYS
    assert chunk["source"] == "test"


def test_empty_file_skipped(tmp_path):
    """Empty file returns None."""
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")

    chunk = _chunk_code_file(f, tmp_path, "test")
    assert chunk is None


def test_small_file_skipped(tmp_path):
    """File under 20 chars returns None."""
    f = tmp_path / "tiny.py"
    f.write_text("x = 1", encoding="utf-8")

    chunk = _chunk_code_file(f, tmp_path, "test")
    assert chunk is None


def test_different_extensions_unique_ids(tmp_path):
    """Files with same stem but different extensions get unique IDs."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "utils.py").write_text("# Python utils file with enough content here", encoding="utf-8")
    (src / "utils.js").write_text("// JavaScript utils file with enough content here", encoding="utf-8")

    chunk_py = _chunk_code_file(src / "utils.py", tmp_path, "test")
    chunk_js = _chunk_code_file(src / "utils.js", tmp_path, "test")

    assert chunk_py is not None
    assert chunk_js is not None
    assert chunk_py["id"] != chunk_js["id"]


def test_chunk_directory(tmp_path):
    """CodeChunker.chunk_directory finds files and returns chunks."""
    (tmp_path / "sample.py").write_text("# A sample file with enough text to pass checks", encoding="utf-8")

    chunker = CodeChunker()
    repo_config = {"name": "test", "source_tag": "test", "extensions": ["py"]}
    chunks = chunker.chunk_directory(tmp_path, repo_config)
    assert len(chunks) >= 1
    for c in chunks:
        assert set(c.keys()) == REQUIRED_KEYS


def test_skip_dirs(tmp_path):
    """Files in SKIP_DIRS (e.g. .git) are excluded."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config.py").write_text("# Should be skipped even though enough content", encoding="utf-8")

    chunker = CodeChunker()
    repo_config = {"name": "test", "source_tag": "test", "extensions": ["py"]}
    chunks = chunker.chunk_directory(tmp_path, repo_config)
    assert len(chunks) == 0


# ---------------------------------------------------------------------------
# Binary file detection tests
# ---------------------------------------------------------------------------


def test_is_binary_detects_null_bytes(tmp_path):
    """Files with null bytes are detected as binary."""
    f = tmp_path / "image.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
    assert _is_binary(f) is True


def test_is_binary_passes_text(tmp_path):
    """Normal text files are not binary."""
    f = tmp_path / "code.py"
    f.write_text("print('hello world')\n", encoding="utf-8")
    assert _is_binary(f) is False


def test_code_chunker_skips_binary_without_extensions(tmp_path):
    """CodeChunker with no extensions filter skips binary files."""
    (tmp_path / "code.py").write_text("# valid code file with enough content here", encoding="utf-8")
    (tmp_path / "image.bin").write_bytes(b"\x00\x01\x02" * 100)

    chunker = CodeChunker()
    repo_config = {"name": "test", "source_tag": "test"}  # no extensions filter
    chunks = chunker.chunk_directory(tmp_path, repo_config)

    chunk_files = [c["file_path"] for c in chunks]
    assert any("code.py" in f for f in chunk_files)
    assert not any("image.bin" in f for f in chunk_files)
