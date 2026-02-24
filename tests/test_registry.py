#  mcp-rag - Chunker Registry Tests

import pytest

from chunkers import CHUNKER_REGISTRY, get_chunker
from chunkers.base import BaseChunker
from chunkers.python_chunker import PythonChunker


def test_get_known_chunker():
    """get_chunker('python') returns a PythonChunker instance."""
    chunker = get_chunker("python")
    assert isinstance(chunker, PythonChunker)
    assert isinstance(chunker, BaseChunker)


def test_get_unknown_chunker():
    """get_chunker('unknown') raises ValueError listing available chunkers."""
    with pytest.raises(ValueError, match="Unknown chunker type"):
        get_chunker("unknown_type_xyz")


def test_all_builtins_registered():
    """All 5 built-in chunkers are registered on import."""
    expected = {"python", "csharp", "digest", "markdown", "code"}
    assert expected == set(CHUNKER_REGISTRY.keys())
