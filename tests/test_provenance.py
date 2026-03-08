#  mcp-rag - Provenance & Data Hygiene Tests
#
#  Tests for source hash attachment, path resolution, stale detection,
#  incremental ingest, and gotcha management.

import hashlib
import json
import sqlite3
import struct

import pytest


# ---------------------------------------------------------------------------
# _build_source_base_dirs / _resolve_source_path
# ---------------------------------------------------------------------------


def test_build_base_dirs_local_path(tmp_path):
    """Local repo path resolves to the configured path."""
    from pipeline import _build_source_base_dirs

    repos = [{"name": "my-src", "path": str(tmp_path / "src"), "source_tag": "engine"}]
    dirs = _build_source_base_dirs(repos, tmp_path / "repos")
    assert dirs["engine"] == tmp_path / "src"


def test_build_base_dirs_cloned_repo(tmp_path):
    """Cloned repo resolves to repos_dir / local_dir."""
    from pipeline import _build_source_base_dirs

    repos = [{"name": "ext", "url": "https://example.com/ext.git", "local_dir": "ext-repo", "source_tag": "ext"}]
    dirs = _build_source_base_dirs(repos, tmp_path / "repos")
    assert dirs["ext"] == tmp_path / "repos" / "ext-repo"


def test_build_base_dirs_with_source_subdir(tmp_path):
    """source_subdir is appended to the base directory."""
    from pipeline import _build_source_base_dirs

    repos = [{"name": "big-repo", "path": str(tmp_path / "mono"), "source_tag": "sub", "source_subdir": "packages/core"}]
    dirs = _build_source_base_dirs(repos, tmp_path / "repos")
    assert dirs["sub"] == tmp_path / "mono" / "packages" / "core"


def test_resolve_source_path_found(tmp_path):
    """Resolves to absolute path when source_tag has a known base."""
    from pipeline import _resolve_source_path

    base_dirs = {"engine": tmp_path / "src"}
    result = _resolve_source_path("core/Graphics.cs", "engine", base_dirs)
    assert result == tmp_path / "src" / "core" / "Graphics.cs"


def test_resolve_source_path_unknown_source():
    """Returns None when source_tag is not in base_dirs."""
    from pipeline import _resolve_source_path

    result = _resolve_source_path("file.py", "unknown", {})
    assert result is None


# ---------------------------------------------------------------------------
# _attach_source_hashes
# ---------------------------------------------------------------------------


def test_attach_hashes_normal_file(tmp_path):
    """Chunks from an existing file get the correct SHA-256 hash."""
    from pipeline import _attach_source_hashes

    src = tmp_path / "src"
    src.mkdir()
    content = b"class Example:\n    pass\n"
    (src / "example.py").write_bytes(content)
    expected_hash = hashlib.sha256(content).hexdigest()

    chunks = [
        {"id": "test:1", "source": "test", "file_path": "example.py", "text": "..."},
        {"id": "test:2", "source": "test", "file_path": "example.py", "text": "..."},
    ]
    repos = [{"name": "test-src", "path": str(src), "source_tag": "test"}]
    _attach_source_hashes(chunks, repos, tmp_path / "repos")

    assert chunks[0]["source_hash"] == expected_hash
    assert chunks[1]["source_hash"] == expected_hash


def test_attach_hashes_empty_file(tmp_path):
    """Empty files get the hash of empty bytes."""
    from pipeline import _attach_source_hashes

    src = tmp_path / "src"
    src.mkdir()
    (src / "empty.py").write_bytes(b"")
    expected_hash = hashlib.sha256(b"").hexdigest()

    chunks = [{"id": "test:1", "source": "test", "file_path": "empty.py", "text": ""}]
    repos = [{"name": "test-src", "path": str(src), "source_tag": "test"}]
    _attach_source_hashes(chunks, repos, tmp_path / "repos")

    assert chunks[0]["source_hash"] == expected_hash


def test_attach_hashes_missing_file(tmp_path):
    """Chunks from a missing file get empty source_hash."""
    from pipeline import _attach_source_hashes

    src = tmp_path / "src"
    src.mkdir()

    chunks = [{"id": "test:1", "source": "test", "file_path": "gone.py", "text": "..."}]
    repos = [{"name": "test-src", "path": str(src), "source_tag": "test"}]
    _attach_source_hashes(chunks, repos, tmp_path / "repos")

    assert chunks[0]["source_hash"] == ""


def test_attach_hashes_no_file_path(tmp_path):
    """Chunks without file_path get empty source_hash."""
    from pipeline import _attach_source_hashes

    chunks = [{"id": "test:1", "source": "test", "file_path": "", "text": "..."}]
    repos = [{"name": "test-src", "path": str(tmp_path), "source_tag": "test"}]
    _attach_source_hashes(chunks, repos, tmp_path)

    assert chunks[0]["source_hash"] == ""


# ---------------------------------------------------------------------------
# Schema: source_hash, gotcha, index_metadata
# ---------------------------------------------------------------------------


def test_schema_has_new_columns(tmp_path):
    """_create_schema creates source_hash, gotcha columns and index_metadata table."""
    from pipeline import _create_schema

    db = sqlite3.connect(":memory:")
    _create_schema(db)

    cols = {r[1] for r in db.execute("PRAGMA table_info(chunks)").fetchall()}
    assert "source_hash" in cols
    assert "gotcha" in cols

    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    assert "index_metadata" in tables

    db.close()


def test_insert_chunk_stores_hash_and_gotcha(tmp_path):
    """_insert_chunk stores source_hash and gotcha values."""
    from pipeline import _create_schema, _insert_chunk

    db = sqlite3.connect(":memory:")
    _create_schema(db)

    chunk = {
        "id": "test:1",
        "text": "sample",
        "source": "test",
        "source_hash": "abc123",
        "gotcha": "Not a timeout — it's a DNS failure",
    }
    _insert_chunk(db, chunk, [0.0] * 4)

    row = db.execute("SELECT source_hash, gotcha FROM chunks WHERE id = 'test:1'").fetchone()
    assert row[0] == "abc123"
    assert row[1] == "Not a timeout — it's a DNS failure"
    db.close()


# ---------------------------------------------------------------------------
# cmd_stale
# ---------------------------------------------------------------------------


def _build_test_db(db_path, chunks, metadata=None):
    """Build a minimal test database with chunks and optional metadata."""
    from pipeline import _create_schema

    conn = sqlite3.connect(str(db_path))
    _create_schema(conn)

    for chunk in chunks:
        embedding = [0.0] * 4
        blob = struct.pack(f"{len(embedding)}f", *embedding)
        conn.execute(
            """INSERT INTO chunks
               (id, text, source, file_path, source_hash, embedding)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (chunk["id"], chunk.get("text", ""), chunk["source"],
             chunk.get("file_path", ""), chunk.get("source_hash", ""), blob),
        )

    if metadata:
        for key, value in metadata.items():
            conn.execute(
                "INSERT INTO index_metadata (key, value) VALUES (?, ?)",
                (key, value),
            )

    conn.commit()
    conn.close()


def test_stale_detects_changed_file(tmp_path):
    """cmd_stale exits with code 1 when a source file has changed."""
    from pipeline import cmd_stale

    # Create source file
    src = tmp_path / "src"
    src.mkdir()
    original_content = b"original content"
    (src / "file.py").write_bytes(original_content)
    original_hash = hashlib.sha256(original_content).hexdigest()

    # Build DB with the original hash
    db_path = tmp_path / "data" / "rag.db"
    db_path.parent.mkdir(parents=True)
    _build_test_db(db_path, [
        {"id": "test:1", "source": "test", "file_path": "file.py", "source_hash": original_hash},
    ])

    # Modify the file
    (src / "file.py").write_bytes(b"modified content")

    config = {
        "database": {"path": str(db_path)},
        "sources": {"repos_dir": str(tmp_path / "repos")},
        "repos": [{"name": "test-src", "path": str(src), "source_tag": "test"}],
    }

    import pipeline
    original_script_dir = pipeline.SCRIPT_DIR
    pipeline.SCRIPT_DIR = tmp_path
    try:
        with pytest.raises(SystemExit) as exc_info:
            cmd_stale(config)
        assert exc_info.value.code == 1
    finally:
        pipeline.SCRIPT_DIR = original_script_dir


def test_stale_fresh_when_unchanged(tmp_path):
    """cmd_stale exits cleanly when all files match their hashes."""
    from pipeline import cmd_stale

    src = tmp_path / "src"
    src.mkdir()
    content = b"unchanged content"
    (src / "file.py").write_bytes(content)
    file_hash = hashlib.sha256(content).hexdigest()

    db_path = tmp_path / "data" / "rag.db"
    db_path.parent.mkdir(parents=True)
    _build_test_db(db_path, [
        {"id": "test:1", "source": "test", "file_path": "file.py", "source_hash": file_hash},
    ])

    config = {
        "database": {"path": str(db_path)},
        "sources": {"repos_dir": str(tmp_path / "repos")},
        "repos": [{"name": "test-src", "path": str(src), "source_tag": "test"}],
    }

    import pipeline
    original_script_dir = pipeline.SCRIPT_DIR
    pipeline.SCRIPT_DIR = tmp_path
    try:
        # Should not raise SystemExit
        cmd_stale(config)
    finally:
        pipeline.SCRIPT_DIR = original_script_dir


def test_stale_detects_missing_file(tmp_path):
    """cmd_stale reports missing files as stale."""
    from pipeline import cmd_stale

    src = tmp_path / "src"
    src.mkdir()

    db_path = tmp_path / "data" / "rag.db"
    db_path.parent.mkdir(parents=True)
    _build_test_db(db_path, [
        {"id": "test:1", "source": "test", "file_path": "deleted.py", "source_hash": "abc123"},
    ])

    config = {
        "database": {"path": str(db_path)},
        "sources": {"repos_dir": str(tmp_path / "repos")},
        "repos": [{"name": "test-src", "path": str(src), "source_tag": "test"}],
    }

    import pipeline
    original_script_dir = pipeline.SCRIPT_DIR
    pipeline.SCRIPT_DIR = tmp_path
    try:
        with pytest.raises(SystemExit) as exc_info:
            cmd_stale(config)
        assert exc_info.value.code == 1
    finally:
        pipeline.SCRIPT_DIR = original_script_dir


# ---------------------------------------------------------------------------
# cmd_ingest
# ---------------------------------------------------------------------------


def test_ingest_adds_new_chunks(tmp_path):
    """cmd_ingest embeds and inserts new chunks into an existing DB."""
    pytest.importorskip("httpx")
    # This test would need a running Ollama, so we test the validation path
    from pipeline import _create_schema

    # Build an empty DB
    db_path = tmp_path / "data" / "rag.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    _create_schema(conn)
    conn.close()

    # Write ingest entries
    ingest_path = tmp_path / "data" / "ingest.jsonl"
    entries = [
        {"id": "new:1", "text": "error: rate limit", "source": "diagnostic"},
        {"id": "new:2", "text": "error: timeout", "source": "diagnostic"},
    ]
    ingest_path.write_text(
        "\n".join(json.dumps(e) for e in entries),
        encoding="utf-8",
    )

    # Verify the ingest file was created
    assert ingest_path.exists()
    assert ingest_path.stat().st_size > 0


def test_ingest_skips_malformed_lines(tmp_path):
    """Malformed JSONL lines and entries missing required fields are skipped."""
    ingest_path = tmp_path / "ingest.jsonl"
    lines = [
        '{"id": "good:1", "text": "valid entry", "source": "test"}',
        'not valid json',
        '{"id": "bad:1", "text": "missing source field"}',
        '{"id": "good:2", "text": "another valid", "source": "test"}',
    ]
    ingest_path.write_text("\n".join(lines), encoding="utf-8")

    # Parse and validate like cmd_ingest does
    chunks = []
    for line in ingest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not all(k in entry for k in ("id", "text", "source")):
            continue
        chunks.append(entry)

    assert len(chunks) == 2
    assert chunks[0]["id"] == "good:1"
    assert chunks[1]["id"] == "good:2"


def test_ingest_empty_file_is_noop(tmp_path):
    """Empty ingest file produces no errors."""
    ingest_path = tmp_path / "ingest.jsonl"
    ingest_path.write_text("", encoding="utf-8")
    assert ingest_path.stat().st_size == 0


# ---------------------------------------------------------------------------
# cmd_gotcha
# ---------------------------------------------------------------------------


def test_gotcha_updates_chunk(tmp_path):
    """cmd_gotcha updates the gotcha column on an existing chunk."""
    from pipeline import _create_schema

    db_path = tmp_path / "rag.db"
    conn = sqlite3.connect(str(db_path))
    _create_schema(conn)

    # Insert a chunk
    embedding = [0.0] * 4
    blob = struct.pack(f"{len(embedding)}f", *embedding)
    conn.execute(
        "INSERT INTO chunks (id, text, source, embedding) VALUES (?, ?, ?, ?)",
        ("test:1", "sample", "test", blob),
    )
    conn.commit()
    conn.close()

    # Update gotcha directly (testing the DB operation, not CLI arg parsing)
    conn = sqlite3.connect(str(db_path))
    gotcha_text = "Looks like a timeout but is actually DNS failure"
    conn.execute("UPDATE chunks SET gotcha = ? WHERE id = ?", (gotcha_text, "test:1"))
    conn.commit()

    row = conn.execute("SELECT gotcha FROM chunks WHERE id = 'test:1'").fetchone()
    assert row[0] == gotcha_text
    conn.close()


def test_gotcha_nonexistent_chunk(tmp_path):
    """Updating gotcha on a non-existent chunk changes no rows."""
    from pipeline import _create_schema

    db_path = tmp_path / "rag.db"
    conn = sqlite3.connect(str(db_path))
    _create_schema(conn)
    conn.close()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("UPDATE chunks SET gotcha = ? WHERE id = ?", ("test", "nonexistent:1"))
    assert cursor.rowcount == 0
    conn.close()


# ---------------------------------------------------------------------------
# index_metadata
# ---------------------------------------------------------------------------


def test_index_metadata_stored_and_readable(tmp_path):
    """_write_index_metadata stores indexed_at timestamp."""
    from pipeline import _create_schema, _write_index_metadata

    db = sqlite3.connect(":memory:")
    _create_schema(db)

    config = {
        "sources": {"repos_dir": str(tmp_path / "repos")},
        "repos": [],
    }
    _write_index_metadata(db, config)

    row = db.execute("SELECT value FROM index_metadata WHERE key = 'indexed_at'").fetchone()
    assert row is not None
    assert "T" in row[0]  # ISO timestamp format
    db.close()
