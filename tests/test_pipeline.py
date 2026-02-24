#  mcp-rag - Pipeline Tests

import json

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
