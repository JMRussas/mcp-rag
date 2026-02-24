#  mcp-rag - Digest Chunker Tests

from chunkers.digest import _chunk_digest_file

REQUIRED_KEYS = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path"}


def test_single_type_chunk(tmp_path):
    """A digest file with one type produces one chunk."""
    content = """\
my_module<public> := module {
    MyClass<native><public> := class {
        Name<public> : string
        DoWork<public>() : void
    }
}
"""
    f = tmp_path / "test.digest.verse"
    f.write_text(content, encoding="utf-8")

    chunks = _chunk_digest_file(f, "test_source", "/TestModule")
    assert len(chunks) >= 1
    assert all(set(c.keys()) == REQUIRED_KEYS for c in chunks)


def test_source_name_propagated(tmp_path):
    """source_name parameter is used in chunk source field, not hardcoded 'digest'."""
    content = """\
my_module<public> := module {
    MyType<native><public> := class {
        Value<public> : int
    }
}
"""
    f = tmp_path / "test.digest.verse"
    f.write_text(content, encoding="utf-8")

    chunks = _chunk_digest_file(f, "custom_source", "/Base")
    assert len(chunks) >= 1
    for c in chunks:
        assert c["source"] == "custom_source", f"Expected source='custom_source', got '{c['source']}'"


def test_empty_file(tmp_path):
    """Empty digest file produces no chunks."""
    f = tmp_path / "empty.digest.verse"
    f.write_text("", encoding="utf-8")

    chunks = _chunk_digest_file(f, "test")
    assert chunks == []


def test_nested_modules(tmp_path):
    """Nested modules produce correct module_path in chunks."""
    content = """\
outer<public> := module {
    inner<public> := module {
        Widget<native><public> := class {
            Draw<public>() : void
        }
    }
}
"""
    f = tmp_path / "nested.digest.verse"
    f.write_text(content, encoding="utf-8")

    chunks = _chunk_digest_file(f, "test", "/Root")
    type_chunks = [c for c in chunks if c["type_name"] == "Widget"]
    assert len(type_chunks) == 1
    # Module path should reflect nesting
    assert "inner" in type_chunks[0]["module_path"]
