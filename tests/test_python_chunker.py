#  mcp-rag - Python Chunker Tests

from chunkers.python_chunker import PythonChunker, _chunk_python_file, _derive_module_path

REQUIRED_KEYS = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path"}


def test_single_class_file(tmp_path):
    """Single-class file produces one whole-file chunk with correct metadata."""
    code = '''\
class Greeter:
    """Says hello."""

    def greet(self, name):
        return f"Hello, {name}"
'''
    f = tmp_path / "greeter.py"
    f.write_text(code, encoding="utf-8")

    chunks = _chunk_python_file(f, "test", tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["type_name"] == "Greeter"
    assert chunks[0]["source"] == "test"
    assert set(chunks[0].keys()) == REQUIRED_KEYS


def test_multiple_functions(tmp_python_file, tmp_path):
    """File with class + 2 functions produces 3 chunks, each with imports header."""
    # tmp_python_file has MyService, helper_one, helper_two
    chunks = _chunk_python_file(tmp_python_file, "test", tmp_path)
    assert len(chunks) == 3
    names = {c["type_name"] for c in chunks}
    assert names == {"MyService", "helper_one", "helper_two"}
    # Each chunk should contain the import header
    for c in chunks:
        assert "import os" in c["text"]


def test_syntax_error_fallback(tmp_path):
    """File with syntax errors falls back to whole-file chunk."""
    code = "def broken(\n    this is not valid python"
    f = tmp_path / "broken.py"
    f.write_text(code, encoding="utf-8")

    chunks = _chunk_python_file(f, "test", tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["type_name"] == "broken"
    assert set(chunks[0].keys()) == REQUIRED_KEYS


def test_empty_file(tmp_path):
    """Empty file returns no chunks."""
    f = tmp_path / "empty.py"
    f.write_text("", encoding="utf-8")

    chunks = _chunk_python_file(f, "test", tmp_path)
    assert chunks == []


def test_derive_module_path():
    """_derive_module_path converts file paths to dotted module notation."""
    assert _derive_module_path("src/utils/helpers.py") == "src.utils.helpers"
    assert _derive_module_path("chunkers/__init__.py") == "chunkers"
    assert _derive_module_path("main.py") == "main"


def test_chunk_directory(tmp_python_file, tmp_path):
    """PythonChunker.chunk_directory finds .py files and returns chunks."""
    chunker = PythonChunker()
    repo_config = {"name": "test", "source_tag": "test"}
    chunks = chunker.chunk_directory(tmp_path, repo_config)
    assert len(chunks) > 0
    for c in chunks:
        assert set(c.keys()) == REQUIRED_KEYS
