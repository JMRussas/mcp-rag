#  mcp-rag - Pipeline Tests

import json
import sqlite3

import pytest


def test_validate_config_rejects_missing_keys():
    """_validate_config rejects config missing required keys."""
    from pipeline import ConfigError, _validate_config

    with pytest.raises(ConfigError):
        _validate_config({})

    with pytest.raises(ConfigError):
        _validate_config({"ollama": {}, "database": {}, "search": {}, "sources": {}})


def test_validate_config_rejects_bad_ollama_host():
    """_validate_config rejects non-URL ollama host."""
    from pipeline import ConfigError, _validate_config

    config = {
        "ollama": {"host": "not-a-url", "embed_model": "test"},
        "database": {"path": "test.db"},
        "search": {"embed_dimensions": 768},
        "sources": {},
        "repos": [{"name": "test", "type": "python", "path": "/tmp"}],
    }
    with pytest.raises(ConfigError):
        _validate_config(config)


def test_cmd_chunk_produces_valid_jsonl(tmp_path, tmp_config):
    """cmd_chunk with test config writes valid JSONL with required keys."""
    from pipeline import cmd_chunk

    # Create a source directory with a Python file
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "example.py").write_text(
        'class Example:\n    """An example class."""\n    pass\n',
        encoding="utf-8",
    )

    # Point config at our temp directories
    chunks_path = tmp_path / "chunks.jsonl"
    tmp_config["sources"]["chunks_path"] = str(chunks_path)
    tmp_config["repos"] = [
        {
            "name": "test-src",
            "path": str(src_dir),
            "type": "python",
            "source_tag": "test",
        }
    ]

    # Monkey-patch SCRIPT_DIR so relative path resolution works
    import pipeline

    original_script_dir = pipeline.SCRIPT_DIR
    pipeline.SCRIPT_DIR = tmp_path
    try:
        cmd_chunk(tmp_config)
    finally:
        pipeline.SCRIPT_DIR = original_script_dir

    assert chunks_path.exists()

    required_keys = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path"}
    with open(chunks_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0
    for line in lines:
        chunk = json.loads(line)
        assert required_keys <= set(chunk.keys()), f"Missing keys in chunk: {chunk.get('id', 'unknown')}"


def test_db_source_produces_chunks(tmp_path, tmp_config):
    """db_sources config produces valid chunks from a SQLite database."""
    from pipeline import cmd_chunk

    # Create a SQLite database with some coding standards
    db_file = tmp_path / "standards.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute("CREATE TABLE standards (id TEXT, title TEXT, body TEXT, category TEXT)")
    conn.executemany(
        "INSERT INTO standards VALUES (?, ?, ?, ?)",
        [
            ("std-001", "Naming Conventions", "Use PascalCase for classes.", "style"),
            ("std-002", "Error Handling", "Always catch specific exceptions.", "reliability"),
        ],
    )
    conn.commit()
    conn.close()

    chunks_path = tmp_path / "chunks.jsonl"
    tmp_config["sources"]["chunks_path"] = str(chunks_path)
    tmp_config["repos"] = []
    tmp_config["db_sources"] = [
        {
            "name": "standards",
            "type": "sqlite",
            "path": str(db_file),
            "query": "SELECT id, title, body, category FROM standards",
            "text_column": "body",
            "id_column": "id",
            "heading_column": "title",
            "category_column": "category",
            "source_tag": "standards",
        }
    ]

    import pipeline

    original_script_dir = pipeline.SCRIPT_DIR
    pipeline.SCRIPT_DIR = tmp_path
    try:
        cmd_chunk(tmp_config)
    finally:
        pipeline.SCRIPT_DIR = original_script_dir

    assert chunks_path.exists()

    required_keys = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path"}
    with open(chunks_path, encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) == 2
    for line in lines:
        chunk = json.loads(line)
        assert required_keys <= set(chunk.keys())
        assert chunk["source"] == "standards"

    # Verify content mapping
    chunks = [json.loads(line) for line in lines]
    by_id = {c["id"]: c for c in chunks}
    assert "standards:std-001" in by_id
    assert by_id["standards:std-001"]["heading"] == "Naming Conventions"
    assert by_id["standards:std-001"]["category"] == "style"
    assert "PascalCase" in by_id["standards:std-001"]["text"]


def test_validate_config_accepts_db_sources_only():
    """Config with db_sources but no repos is valid."""
    from pipeline import _validate_config

    config = {
        "ollama": {"host": "http://localhost:11434", "embed_model": "test"},
        "database": {"path": "test.db"},
        "search": {"embed_dimensions": 768},
        "sources": {},
        "db_sources": [{"name": "test", "path": "/tmp/test.db", "query": "SELECT * FROM t"}],
    }
    _validate_config(config)  # Should not raise


def test_validate_config_rejects_no_sources():
    """Config with neither repos nor db_sources is rejected."""
    from pipeline import ConfigError, _validate_config

    config = {
        "ollama": {"host": "http://localhost:11434", "embed_model": "test"},
        "database": {"path": "test.db"},
        "search": {"embed_dimensions": 768},
        "sources": {},
    }
    with pytest.raises(ConfigError, match="repos.*db_sources"):
        _validate_config(config)
