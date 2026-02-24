#  mcp-rag - C# Chunker Tests

from chunkers.csharp import _chunk_csharp_file

REQUIRED_KEYS = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path"}


def test_single_class_file(tmp_csharp_file, tmp_path):
    """Single-class file produces one whole-file chunk with correct namespace and type."""
    chunks = _chunk_csharp_file(tmp_csharp_file, "test", tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["type_name"] == "SampleClass"
    assert chunks[0]["module_path"] == "TestProject"
    assert chunks[0]["source"] == "test"
    assert set(chunks[0].keys()) == REQUIRED_KEYS


def test_multi_type_file(tmp_path):
    """File with multiple types produces one chunk per type."""
    code = """\
namespace MultiTest;

public class Alpha
{
    public int Value { get; set; }
}

public struct Beta
{
    public float X;
    public float Y;
}

public enum Gamma
{
    One,
    Two,
    Three
}
"""
    f = tmp_path / "Multi.cs"
    f.write_text(code, encoding="utf-8")

    chunks = _chunk_csharp_file(f, "test", tmp_path)
    assert len(chunks) == 3
    names = {c["type_name"] for c in chunks}
    assert names == {"Alpha", "Beta", "Gamma"}


def test_file_scoped_namespace(tmp_path):
    """File-scoped namespace (namespace Foo;) is extracted correctly."""
    code = """\
namespace MyApp.Services;

public class UserService
{
    public void Save() { }
}
"""
    f = tmp_path / "UserService.cs"
    f.write_text(code, encoding="utf-8")

    chunks = _chunk_csharp_file(f, "test", tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["module_path"] == "MyApp.Services"


def test_empty_file(tmp_path):
    """Empty or unreadable file returns no chunks."""
    f = tmp_path / "Empty.cs"
    f.write_text("", encoding="utf-8")

    chunks = _chunk_csharp_file(f, "test", tmp_path)
    assert chunks == []


def test_no_namespace(tmp_path):
    """File with no namespace produces IDs without a leading dot."""
    code = """\
public class GlobalHelper
{
    public static void DoStuff() { }
}
"""
    f = tmp_path / "GlobalHelper.cs"
    f.write_text(code, encoding="utf-8")

    chunks = _chunk_csharp_file(f, "test", tmp_path)
    assert len(chunks) == 1
    assert chunks[0]["module_path"] == ""
    # ID should NOT have ":." (empty namespace + dot)
    assert ":." not in chunks[0]["id"]
    assert chunks[0]["id"] == "csharp:test:GlobalHelper"
