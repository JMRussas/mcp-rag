#!/usr/bin/env python3
#  mcp-rag - Data Pipeline
#
#  Chunks source code and documents, generates embeddings via Ollama, and
#  builds the SQLite vector store for the MCP server.
#
#  Supports both local directories and git-cloneable repositories.
#  Chunker selection is config-driven via the chunker registry.
#
#  Depends on: config.json, chunkers/, httpx (for Ollama embedding API)
#  Used by:    Manual CLI invocation (python pipeline.py rebuild)

import asyncio
import hashlib
import json
import logging
import os
import shutil
import sqlite3
import struct
import subprocess
import sys
import time
from pathlib import Path

import httpx

from chunkers import get_chunker

SCRIPT_DIR = Path(__file__).parent
CONFIG_PATH = SCRIPT_DIR / "config.json"

log = logging.getLogger("pipeline")


class ConfigError(ValueError):
    """Raised when config.json is missing or invalid."""


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        log.error(f"{CONFIG_PATH} not found. Copy config.example.json to config.json.")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as f:
        config = json.load(f)
    try:
        _validate_config(config)
    except ConfigError as e:
        log.error(str(e))
        sys.exit(1)
    return config


def _validate_config(config: dict):
    """Validate that the config has all required fields.

    Raises ConfigError on invalid config.
    """
    required_sections = ["ollama", "database", "search", "sources"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"config.json missing required section '{section}'.")

    if not isinstance(config["ollama"].get("host"), str) or not config["ollama"]["host"].startswith("http"):
        raise ConfigError("config.json 'ollama.host' must be a URL starting with http.")

    dims = config["search"].get("embed_dimensions")
    if not isinstance(dims, int) or dims <= 0:
        raise ConfigError("config.json 'search.embed_dimensions' must be a positive integer.")

    has_repos = isinstance(config.get("repos"), list) and len(config["repos"]) > 0
    has_db_sources = isinstance(config.get("db_sources"), list) and len(config["db_sources"]) > 0

    if not has_repos and not has_db_sources:
        raise ConfigError("config.json needs at least one of 'repos' (non-empty list) or 'db_sources' (non-empty list).")

    for i, repo in enumerate(config.get("repos", [])):
        if "name" not in repo:
            raise ConfigError(f"config.json repos[{i}] missing 'name'.")
        if "type" not in repo:
            raise ConfigError(f"config.json repos[{i}] ('{repo['name']}') missing 'type'.")
        if "path" not in repo and "url" not in repo:
            raise ConfigError(f"config.json repos[{i}] ('{repo['name']}') needs 'path' or 'url'.")

    for i, db_src in enumerate(config.get("db_sources", [])):
        if "name" not in db_src:
            raise ConfigError(f"config.json db_sources[{i}] missing 'name'.")
        if "path" not in db_src:
            raise ConfigError(f"config.json db_sources[{i}] ('{db_src.get('name', '?')}') missing 'path'.")
        if "query" not in db_src:
            raise ConfigError(f"config.json db_sources[{i}] ('{db_src['name']}') missing 'query'.")


# ---------------------------------------------------------------------------
# Clone (for git-based sources)
# ---------------------------------------------------------------------------


def cmd_clone(config: dict):
    """Clone or update git-based source repos."""
    import subprocess

    repos_dir = SCRIPT_DIR / config["sources"].get("repos_dir", "data/repos")
    repos_dir.mkdir(parents=True, exist_ok=True)

    for repo in config.get("repos", []):
        url = repo.get("url")
        if not url:
            # Local path — no cloning needed
            continue

        local_dir = repo.get("local_dir", repo["name"])
        local = repos_dir / local_dir

        if local.exists():
            log.info(f"[clone] Updating {repo['name']}...")
            try:
                result = subprocess.run(
                    ["git", "-C", str(local), "pull", "--ff-only"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    log.warning(f"[clone] git pull failed for {repo['name']} (exit {result.returncode})")
                    if result.stderr.strip():
                        log.warning(f"  {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                log.warning(f"[clone] git pull timed out for {repo['name']} (120s)")
        else:
            log.info(f"[clone] Cloning {repo['name']}...")
            try:
                subprocess.run(
                    ["git", "clone", "--depth", "1", url, str(local)],
                    check=True,
                    timeout=120,
                )
            except subprocess.TimeoutExpired:
                log.warning(f"[clone] git clone timed out for {repo['name']} (120s)")

    log.info("[clone] Done.")


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------


def _chunk_db_source(db_src: dict) -> list[dict]:
    """Read chunks from a database source.

    Each row from the configured query becomes one chunk.  Column mapping
    is controlled by the db_source config entry.
    """
    name = db_src["name"]
    db_type = db_src.get("type", "sqlite")
    source_tag = db_src.get("source_tag", name)
    query = db_src["query"]

    text_col = db_src.get("text_column", "text")
    id_col = db_src.get("id_column", "")
    heading_col = db_src.get("heading_column", "")
    category_col = db_src.get("category_column", "")
    module_col = db_src.get("module_column", "")
    type_name_col = db_src.get("type_name_column", "")

    log.info(f"[db] Chunking '{name}' ({db_type})...")

    if db_type != "sqlite":
        log.warning(f"[db] Unsupported db type '{db_type}' for '{name}', skipping.")
        return []

    db_path = db_src["path"]
    if not Path(db_path).exists():
        log.warning(f"[db] Database not found: {db_path}, skipping '{name}'.")
        return []

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query).fetchall()
    except sqlite3.OperationalError as e:
        log.error(f"[db] Query failed for '{name}': {e}")
        conn.close()
        return []

    columns = set(rows[0].keys()) if rows else set()
    chunks = []

    for row in rows:
        text = str(row[text_col]) if text_col in columns else ""
        if not text.strip():
            continue

        # Build chunk ID from source row ID or hash the text
        if id_col and id_col in columns:
            chunk_id = f"{source_tag}:{row[id_col]}"
        else:
            chunk_id = f"{source_tag}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"

        chunk = {
            "id": chunk_id,
            "text": text,
            "source": source_tag,
            "module_path": str(row[module_col]) if module_col and module_col in columns else "",
            "type_name": str(row[type_name_col]) if type_name_col and type_name_col in columns else "",
            "category": str(row[category_col]) if category_col and category_col in columns else "",
            "heading": str(row[heading_col]) if heading_col and heading_col in columns else "",
            "file_path": "",
        }
        chunks.append(chunk)

    conn.close()
    log.info(f"[db] '{name}': {len(chunks)} chunks from {len(rows)} rows")
    return chunks


def cmd_chunk(config: dict):
    """Run all chunkers and write chunks.jsonl."""
    chunks_path = SCRIPT_DIR / config["sources"]["chunks_path"]
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    repos_dir = SCRIPT_DIR / config["sources"].get("repos_dir", "data/repos")

    all_chunks = []

    for repo in config.get("repos", []):
        chunker_type = repo["type"]
        log.info(f"Chunking {repo['name']} ({chunker_type})...")

        # Resolve source directory
        if repo.get("path"):
            source_dir = Path(repo["path"])
        elif repo.get("local_dir"):
            source_dir = repos_dir / repo["local_dir"]
        else:
            log.warning(f"No path or local_dir for {repo['name']}, skipping.")
            continue

        # Allow diving into a subdirectory (e.g. "Modules/FortniteGame")
        if repo.get("source_subdir"):
            source_dir = source_dir / repo["source_subdir"]

        try:
            chunker = get_chunker(chunker_type)
        except ValueError as e:
            log.warning(str(e))
            continue

        chunks = chunker.chunk_directory(source_dir, repo)
        all_chunks.extend(chunks)

    # Chunk from database sources (if configured)
    for db_src in config.get("db_sources", []):
        db_chunks = _chunk_db_source(db_src)
        all_chunks.extend(db_chunks)

    # Deduplicate by ID (keep first occurrence)
    seen = set()
    deduped = []
    for chunk in all_chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            deduped.append(chunk)

    # Attach source file hashes for provenance tracking
    _attach_source_hashes(deduped, config.get("repos", []), repos_dir)

    # Write chunks.jsonl
    with open(chunks_path, "w", encoding="utf-8") as f:
        for chunk in deduped:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    log.info(f"[chunk] Wrote {len(deduped)} chunks to {chunks_path}")
    _log_source_stats(deduped)


def _log_source_stats(chunks: list[dict]):
    """Log chunk count by source."""
    from collections import Counter

    counts = Counter(c["source"] for c in chunks)
    for source, count in sorted(counts.items()):
        log.info(f"  {source}: {count}")


# ---------------------------------------------------------------------------
# Source provenance
# ---------------------------------------------------------------------------


def _build_source_base_dirs(repos_config: list[dict], repos_dir: Path) -> dict[str, Path]:
    """Build a mapping from source_tag → base directory for file path resolution.

    Resolution rules:
    1. If repo has 'path': base = Path(repo['path'])
    2. If repo has 'url':  base = repos_dir / repo.get('local_dir', repo['name'])
    3. If repo has 'source_subdir': base = base / source_subdir
    """
    base_dirs: dict[str, Path] = {}
    for repo in repos_config:
        source_tag = repo.get("source_tag", repo["name"])
        if repo.get("path"):
            base = Path(repo["path"])
        elif repo.get("url"):
            local_dir = repo.get("local_dir", repo["name"])
            base = repos_dir / local_dir
        else:
            continue
        if repo.get("source_subdir"):
            base = base / repo["source_subdir"]
        base_dirs[source_tag] = base
    return base_dirs


def _resolve_source_path(
    file_path: str,
    source_tag: str,
    base_dirs: dict[str, Path],
) -> Path | None:
    """Resolve a chunk's file_path to an absolute path on disk.

    Returns None if the source_tag has no known base directory.
    """
    base = base_dirs.get(source_tag)
    if base is None:
        return None
    return base / file_path


def _attach_source_hashes(chunks: list[dict], repos_config: list[dict], repos_dir: Path):
    """Attach SHA-256 source hashes to chunks grouped by (source, file_path).

    Modifies chunks in-place. Sets source_hash to '' for unresolvable or
    missing files (logged as warnings).
    """
    base_dirs = _build_source_base_dirs(repos_config, repos_dir)

    # Group chunks by (source, file_path) to hash each file once
    from collections import defaultdict

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk in chunks:
        fp = chunk.get("file_path", "")
        source = chunk.get("source", "")
        if fp:
            groups[(source, fp)].append(chunk)
        else:
            chunk["source_hash"] = ""

    hashed = 0
    skipped = 0
    for (source, fp), group in groups.items():
        resolved = _resolve_source_path(fp, source, base_dirs)
        if resolved is None:
            for c in group:
                c["source_hash"] = ""
            skipped += len(group)
            continue

        try:
            content = resolved.read_bytes()
            file_hash = hashlib.sha256(content).hexdigest()
            for c in group:
                c["source_hash"] = file_hash
            hashed += len(group)
        except FileNotFoundError:
            log.warning(f"[hash] File missing (deleted after chunking?): {resolved}")
            for c in group:
                c["source_hash"] = ""
            skipped += len(group)
        except OSError as e:
            log.warning(f"[hash] Could not read {resolved}: {e}")
            for c in group:
                c["source_hash"] = ""
            skipped += len(group)

    log.info(f"[hash] {hashed} chunks hashed, {skipped} skipped")


# ---------------------------------------------------------------------------
# Embed
# ---------------------------------------------------------------------------


def cmd_embed(config: dict):
    """Embed all chunks and build the SQLite database."""
    asyncio.run(_embed_async(config))


async def _embed_async(config: dict):
    chunks_path = SCRIPT_DIR / config["sources"]["chunks_path"]
    db_path = SCRIPT_DIR / config["database"]["path"]
    db_path.parent.mkdir(parents=True, exist_ok=True)

    ollama_host = config["ollama"]["host"]
    embed_model = config["ollama"]["embed_model"]
    embed_timeout = config["ollama"].get("embed_timeout", 30.0)
    dimensions = config["search"]["embed_dimensions"]

    # Load chunks
    if not chunks_path.exists():
        log.error(f"{chunks_path} not found. Run 'chunk' first.")
        return

    chunks = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    log.info(f"[embed] Loading {len(chunks)} chunks...")

    # Build into a temp file, then swap — avoids locking issues with MCP server
    tmp_path = db_path.with_suffix(".db.tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    conn = sqlite3.connect(str(tmp_path), isolation_level=None)
    _create_schema(conn)

    # Pipeline tuning (configurable with sensible defaults)
    pipeline_config = config.get("pipeline", {})
    concurrency = pipeline_config.get("concurrency", 4)
    batch_size = pipeline_config.get("batch_size", 50)
    progress_interval = pipeline_config.get("progress_interval", 100)
    max_retries = max(1, pipeline_config.get("max_retries", 3))
    max_embed_chars = pipeline_config.get("max_embed_chars", 6000)

    semaphore = asyncio.Semaphore(concurrency)
    embed_url = f"{ollama_host}/api/embeddings"
    completed = 0
    errors = 0
    dim_mismatches = 0
    start_time = time.time()

    async with httpx.AsyncClient(timeout=embed_timeout) as client:

        async def embed_one(chunk: dict) -> tuple[dict, list[float] | None]:
            nonlocal completed, errors
            async with semaphore:
                # Prefix for better retrieval quality with nomic-embed-text
                text = "search_document: " + chunk["text"][:max_embed_chars]
                body = {"model": embed_model, "prompt": text}
                for attempt in range(max_retries):
                    try:
                        resp = await client.post(embed_url, json=body)
                        resp.raise_for_status()
                        data = resp.json()
                        embedding = data.get("embedding", [])
                        completed += 1
                        if completed % progress_interval == 0:
                            elapsed = time.time() - start_time
                            rate = completed / elapsed
                            log.info(f"  [{completed}/{len(chunks)}] {rate:.1f} chunks/sec")
                        return chunk, embedding
                    except Exception as e:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0)
                            continue
                        errors += 1
                        if errors <= 5:
                            log.error(f"Error embedding {chunk['id']}: {e}")
                        return chunk, None

        # Process in batches to avoid overwhelming Ollama
        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            tasks = [embed_one(chunk) for chunk in batch]
            results = await asyncio.gather(*tasks)

            conn.execute("BEGIN")
            for chunk, embedding in results:
                if embedding is None:
                    continue
                if len(embedding) != dimensions:
                    dim_mismatches += 1
                    if dim_mismatches <= 3:
                        log.warning(f"{chunk['id']} returned {len(embedding)} dims, expected {dimensions}")
                    continue
                _insert_chunk(conn, chunk, embedding)
            conn.commit()

    if dim_mismatches > 0:
        log.warning(f"{dim_mismatches} chunks skipped due to dimension mismatch ({dimensions} expected)")

    conn.execute("BEGIN")
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()

    # Write index metadata (provenance tracking)
    _write_index_metadata(conn, config)

    conn.close()

    # Swap temp file into place (handles Windows file locking)
    final_path = tmp_path
    swapped = False
    if db_path.exists():
        try:
            db_path.unlink()
        except PermissionError:
            # Windows: MCP server may hold the old DB open
            try:
                shutil.copy2(str(tmp_path), str(db_path))
                tmp_path.unlink()
                swapped = True
                final_path = db_path
                log.info("[embed] Swapped DB via copy (server had old file open)")
            except PermissionError:
                log.warning(f"[embed] Could not replace {db_path}")
                log.warning(f"[embed] New DB is at: {tmp_path}")
                log.warning("[embed] Restart the MCP server, then rename .db.tmp -> .db")
                swapped = True
    if not swapped:
        os.rename(str(tmp_path), str(db_path))
        final_path = db_path

    elapsed = time.time() - start_time
    log.info(f"[embed] Done in {elapsed:.1f}s -- {completed} embedded, {errors} errors")
    log.info(f"[embed] Database: {final_path} ({final_path.stat().st_size / 1024 / 1024:.1f} MB)")


def _create_schema(conn: sqlite3.Connection):
    """Create the SQLite schema for chunks + FTS5.

    The FTS5 table uses content=chunks (external content mode) for reduced
    disk usage. This requires manual sync -- inserts into chunks must be
    followed by corresponding inserts into chunks_fts. This is safe here
    because the database is always built from scratch (never incrementally
    updated). If incremental updates are ever added, DELETE from chunks_fts
    before UPDATE/DELETE on chunks.
    """
    conn.executescript("""
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            module_path TEXT DEFAULT '',
            type_name TEXT DEFAULT '',
            category TEXT DEFAULT '',
            heading TEXT DEFAULT '',
            file_path TEXT DEFAULT '',
            source_hash TEXT DEFAULT '',
            gotcha TEXT DEFAULT '',
            embedding BLOB
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            id,
            text,
            type_name,
            module_path,
            heading,
            category,
            content=chunks,
            content_rowid=rowid
        );

        CREATE TABLE IF NOT EXISTS index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE INDEX idx_chunks_source ON chunks(source);
        CREATE INDEX idx_chunks_module ON chunks(module_path);
        CREATE INDEX idx_chunks_type_name ON chunks(type_name);
    """)


def _insert_chunk(conn: sqlite3.Connection, chunk: dict, embedding: list[float]):
    """Insert a chunk with its embedding into the database.

    Uses INSERT OR IGNORE because chunk IDs are already deduplicated upstream
    in cmd_chunk(). Duplicates here would indicate a bug, so ignoring them is
    safer than REPLACE (which deletes then re-inserts, leaving stale FTS5
    entries in external-content mode).
    """
    blob = struct.pack(f"{len(embedding)}f", *embedding)

    conn.execute(
        """INSERT OR IGNORE INTO chunks
           (id, text, source, module_path, type_name, category, heading, file_path,
            source_hash, gotcha, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk["id"],
            chunk["text"],
            chunk["source"],
            chunk.get("module_path", ""),
            chunk.get("type_name", ""),
            chunk.get("category", ""),
            chunk.get("heading", ""),
            chunk.get("file_path", ""),
            chunk.get("source_hash", ""),
            chunk.get("gotcha", ""),
            blob,
        ),
    )

    conn.execute(
        """INSERT INTO chunks_fts(rowid, id, text, type_name, module_path, heading, category)
           SELECT rowid, id, text, type_name, module_path, heading, category
           FROM chunks WHERE id = ?""",
        (chunk["id"],),
    )


def _write_index_metadata(conn: sqlite3.Connection, config: dict):
    """Write provenance metadata to the index_metadata table.

    Stores the index timestamp and git commit hashes for each repo that
    has a local .git directory.
    """
    from datetime import datetime, timezone

    repos_dir = SCRIPT_DIR / config["sources"].get("repos_dir", "data/repos")
    now = datetime.now(timezone.utc).isoformat()

    conn.execute("BEGIN")
    conn.execute(
        "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)",
        ("indexed_at", now),
    )

    for repo in config.get("repos", []):
        name = repo["name"]
        # Resolve the repo directory
        if repo.get("path"):
            repo_dir = Path(repo["path"])
        elif repo.get("url"):
            local_dir = repo.get("local_dir", name)
            repo_dir = repos_dir / local_dir
        else:
            continue

        # Check for git repo and capture commit hash
        git_dir = repo_dir / ".git"
        if git_dir.exists():
            try:
                result = subprocess.run(
                    ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    commit = result.stdout.strip()
                    conn.execute(
                        "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)",
                        (f"repo:{name}:commit", commit),
                    )
            except (subprocess.TimeoutExpired, OSError) as e:
                log.warning(f"[metadata] Could not get git commit for {name}: {e}")

    conn.commit()
    log.info(f"[metadata] Index metadata written (indexed_at: {now})")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def cmd_stats(config: dict):
    """Print statistics about the current database."""
    db_path = SCRIPT_DIR / config["database"]["path"]
    chunks_path = SCRIPT_DIR / config["sources"]["chunks_path"]

    if chunks_path.exists():
        with open(chunks_path, encoding="utf-8") as f:
            count = sum(1 for _ in f)
        log.info(f"[chunks.jsonl] {count} chunks")

    if not db_path.exists():
        log.info("[db] No database found. Run 'embed' first.")
        return

    conn = sqlite3.connect(str(db_path))

    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    log.info(f"[db] {total} chunks in database")
    log.info(f"[db] Size: {db_path.stat().st_size / 1024 / 1024:.1f} MB")

    # By source
    rows = conn.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source").fetchall()
    for source, count in rows:
        log.info(f"  {source}: {count}")

    # Top modules
    rows = conn.execute(
        """SELECT module_path, COUNT(*) FROM chunks
           WHERE module_path != '' GROUP BY module_path
           ORDER BY COUNT(*) DESC LIMIT 10"""
    ).fetchall()
    if rows:
        log.info("Top modules:")
        for mod, count in rows:
            log.info(f"  {mod}: {count}")

    # Top types
    rows = conn.execute(
        """SELECT type_name, source, file_path FROM chunks
           WHERE type_name != '' ORDER BY type_name LIMIT 20"""
    ).fetchall()
    if rows:
        log.info("Types indexed:")
        for name, source, fp in rows:
            log.info(f"  {name} ({source}) -- {fp}")

    conn.close()


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def cmd_verify(config: dict):
    """Verify index health and search quality.

    Runs automatic checks (schema, sources, embeddings, FTS5) and
    optional user-defined test queries from config.json "verify" section.
    Returns exit code 0 if all checks pass, 1 if any fail.
    """
    asyncio.run(_verify_async(config))


async def _verify_async(config: dict):
    db_path = SCRIPT_DIR / config["database"]["path"]
    if not db_path.exists():
        log.error(f"[verify] Database not found: {db_path}")
        log.error("[verify] Run 'rebuild' first.")
        sys.exit(1)

    dimensions = config["search"]["embed_dimensions"]
    expected_blob_size = dimensions * 4  # float32

    conn = sqlite3.connect(str(db_path))
    passed = 0
    failed = 0
    warnings = 0

    def check(name: str, ok: bool, detail: str = ""):
        nonlocal passed, failed
        if ok:
            passed += 1
            log.info(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
        else:
            failed += 1
            log.error(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))

    def warn(name: str, detail: str):
        nonlocal warnings
        warnings += 1
        log.warning(f"  WARN  {name} — {detail}")

    log.info("[verify] Running index health checks...")
    log.info("")

    # --- Schema checks ---
    log.info("Schema:")
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    check("chunks table exists", "chunks" in tables)
    check("chunks_fts table exists", "chunks_fts" in tables)

    # Column check
    cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)").fetchall()}
    expected_cols = {"id", "text", "source", "module_path", "type_name", "category", "heading", "file_path", "embedding"}
    missing_cols = expected_cols - cols
    check("chunks columns complete", not missing_cols, f"missing: {missing_cols}" if missing_cols else "")

    # --- Content checks ---
    log.info("")
    log.info("Content:")
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    check("chunks not empty", total > 0, f"{total} chunks")

    # Source distribution
    sources = conn.execute("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source").fetchall()
    for source, count in sources:
        log.info(f"         {source}: {count}")

    # Check all configured source_tags are represented
    expected_sources = {r.get("source_tag", r["name"]) for r in config.get("repos", [])}
    expected_sources |= {r.get("source_tag", r["name"]) for r in config.get("db_sources", [])}
    actual_sources = {r[0] for r in sources}
    missing_sources = expected_sources - actual_sources
    if missing_sources:
        warn("source coverage", f"configured sources not in DB: {missing_sources}")
    else:
        check("all configured sources indexed", True, f"{len(actual_sources)} sources")

    # --- Embedding checks ---
    log.info("")
    log.info("Embeddings:")
    null_embeddings = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL").fetchone()[0]
    check("no null embeddings", null_embeddings == 0, f"{null_embeddings} null" if null_embeddings else "")

    # Spot-check dimensions on a sample
    sample = conn.execute("SELECT id, LENGTH(embedding) FROM chunks WHERE embedding IS NOT NULL LIMIT 20").fetchall()
    bad_dims = [(cid, blen) for cid, blen in sample if blen != expected_blob_size]
    check(
        f"embedding dimensions ({dimensions}d = {expected_blob_size} bytes)",
        not bad_dims,
        f"{len(bad_dims)} mismatched: {bad_dims[:3]}" if bad_dims else f"checked {len(sample)} samples",
    )

    # --- FTS5 integrity ---
    log.info("")
    log.info("FTS5:")
    try:
        fts_count = conn.execute(
            "SELECT COUNT(*) FROM chunks_fts"
        ).fetchone()[0]
        check("FTS5 populated", fts_count > 0, f"{fts_count} entries")
        check("FTS5 matches chunks", fts_count == total, f"FTS5={fts_count} vs chunks={total}")
    except sqlite3.OperationalError as e:
        check("FTS5 readable", False, str(e))

    # --- Search quality (requires Ollama) ---
    verify_config = config.get("verify", {})
    test_queries = verify_config.get("queries", [])

    if test_queries:
        log.info("")
        log.info("Search quality:")

        ollama_host = config["ollama"]["host"]
        embed_model = config["ollama"]["embed_model"]
        embed_timeout = config["ollama"].get("embed_timeout", 30.0)

        # Load embeddings for similarity search
        import numpy as np

        rows = conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
        chunk_ids = [r[0] for r in rows]
        embeddings = np.array(
            [struct.unpack(f"{dimensions}f", r[1]) for r in rows],
            dtype=np.float32,
        )

        # Pre-load metadata for filtering
        meta_rows = conn.execute("SELECT id, source, module_path, type_name FROM chunks").fetchall()
        meta_by_id = {r[0]: {"source": r[1], "module_path": r[2], "type_name": r[3]} for r in meta_rows}

        ollama_available = True

        async with httpx.AsyncClient(timeout=embed_timeout) as client:
            for tq in test_queries:
                min_results = tq.get("min_results", 1)

                if "query" in tq:
                    # Semantic search test (requires Ollama)
                    if not ollama_available:
                        continue
                    query_text = tq["query"]
                    try:
                        resp = await client.post(
                            f"{ollama_host}/api/embeddings",
                            json={"model": embed_model, "prompt": f"search_query: {query_text}"},
                        )
                        resp.raise_for_status()
                        query_vec = np.array(resp.json()["embedding"], dtype=np.float32)
                        # Normalize
                        norm = np.linalg.norm(query_vec)
                        if norm > 0:
                            query_vec /= norm

                        similarities = embeddings @ query_vec
                        top_k = min(tq.get("top_k", 5), len(chunk_ids))
                        top_indices = np.argsort(similarities)[-top_k:][::-1]

                        results = []
                        for idx in top_indices:
                            cid = chunk_ids[idx]
                            score = float(similarities[idx])
                            meta = meta_by_id.get(cid, {})
                            results.append({"id": cid, "score": score, **meta})

                        # Apply filters
                        if tq.get("expect_source"):
                            results = [r for r in results if r.get("source") == tq["expect_source"]]

                        check(
                            f"search \"{query_text}\"",
                            len(results) >= min_results,
                            f"{len(results)} results (need {min_results}), "
                            f"top: {results[0]['id']} ({results[0]['score']:.3f})" if results else "no results",
                        )
                    except httpx.ConnectError:
                        warn(f"search \"{query_text}\"", "Ollama not reachable — skipping semantic tests")
                        ollama_available = False
                    except Exception as e:
                        check(f"search \"{query_text}\"", False, str(e))

                elif "lookup" in tq:
                    # Keyword lookup test (pure SQLite, no Ollama needed)
                    name = tq["lookup"]
                    lookup_rows = conn.execute(
                        "SELECT id, source, type_name FROM chunks WHERE type_name = ? LIMIT 5",
                        (name,),
                    ).fetchall()
                    if not lookup_rows:
                        # Fallback to LIKE
                        safe = name.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                        lookup_rows = conn.execute(
                            "SELECT id, source, type_name FROM chunks WHERE type_name LIKE ? ESCAPE '\\' LIMIT 5",
                            (f"%{safe}%",),
                        ).fetchall()

                    check(
                        f"lookup \"{name}\"",
                        len(lookup_rows) >= min_results,
                        f"{len(lookup_rows)} results" + (f", first: {lookup_rows[0][0]}" if lookup_rows else ""),
                    )

    conn.close()

    # --- Summary ---
    log.info("")
    total_checks = passed + failed
    if failed == 0:
        log.info(f"[verify] ALL PASSED ({passed} checks, {warnings} warnings)")
    else:
        log.error(f"[verify] {failed} FAILED / {passed} passed / {warnings} warnings")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Stale
# ---------------------------------------------------------------------------


def cmd_stale(config: dict):
    """Check for stale chunks by comparing source hashes and git commits.

    Two-level check:
    1. Repo-level (fast): compare stored git commit vs current HEAD.
    2. File-level (thorough): compare stored source_hash vs current file SHA-256.

    Exit code: 0 if all fresh, 1 if any stale or missing.
    """
    db_path = SCRIPT_DIR / config["database"]["path"]
    if not db_path.exists():
        log.error(f"[stale] Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    repos_dir = SCRIPT_DIR / config["sources"].get("repos_dir", "data/repos")
    base_dirs = _build_source_base_dirs(config.get("repos", []), repos_dir)

    stale_repos = []
    fresh_repos = []
    skipped_repos = []
    stale_files = 0
    missing_files = 0
    fresh_files = 0

    # --- Repo-level check (git commit comparison) ---
    log.info("[stale] Checking repo versions...")
    metadata_rows = conn.execute("SELECT key, value FROM index_metadata").fetchall()
    metadata = {r["key"]: r["value"] for r in metadata_rows}

    indexed_at = metadata.get("indexed_at", "unknown")
    log.info(f"[stale] Index built at: {indexed_at}")

    stale_source_tags = set()
    for repo in config.get("repos", []):
        name = repo["name"]
        stored_commit = metadata.get(f"repo:{name}:commit")
        if not stored_commit:
            skipped_repos.append(name)
            continue

        # Resolve repo directory
        if repo.get("path"):
            repo_dir = Path(repo["path"])
        elif repo.get("url"):
            local_dir = repo.get("local_dir", name)
            repo_dir = repos_dir / local_dir
        else:
            skipped_repos.append(name)
            continue

        try:
            result = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                current_commit = result.stdout.strip()
                if current_commit != stored_commit:
                    log.info(
                        f"[stale] Repo '{name}': commit changed "
                        f"({stored_commit[:7]} -> {current_commit[:7]})"
                    )
                    stale_repos.append(name)
                    source_tag = repo.get("source_tag", name)
                    stale_source_tags.add(source_tag)
                else:
                    fresh_repos.append(name)
            else:
                skipped_repos.append(name)
        except (subprocess.TimeoutExpired, OSError):
            skipped_repos.append(name)

    # --- File-level check (source hash comparison) ---
    log.info("[stale] Checking file hashes...")
    rows = conn.execute(
        "SELECT DISTINCT file_path, source, source_hash FROM chunks WHERE source_hash != ''"
    ).fetchall()

    for row in rows:
        fp = row["file_path"]
        source = row["source"]
        stored_hash = row["source_hash"]

        resolved = _resolve_source_path(fp, source, base_dirs)
        if resolved is None:
            missing_files += 1
            continue

        try:
            current_hash = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if current_hash != stored_hash:
                log.info(f"[stale]   {fp}: content changed")
                stale_files += 1
            else:
                fresh_files += 1
        except FileNotFoundError:
            log.info(f"[stale]   {fp}: file missing")
            missing_files += 1
        except OSError:
            missing_files += 1

    conn.close()

    # --- Summary ---
    log.info("")
    if fresh_repos:
        log.info(f"[stale] Fresh repos: {', '.join(fresh_repos)}")
    if stale_repos:
        log.info(f"[stale] Stale repos: {', '.join(stale_repos)}")
    if skipped_repos:
        log.info(f"[stale] Skipped repos (no git): {', '.join(skipped_repos)}")
    log.info(f"[stale] Files: {fresh_files} fresh, {stale_files} stale, {missing_files} missing")

    if stale_repos or stale_files or missing_files:
        log.info("[stale] Index is STALE — run 'rebuild' to update")
        sys.exit(1)
    else:
        log.info("[stale] Index is FRESH")


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def cmd_ingest(config: dict):
    """Incrementally ingest new chunks from an ingest queue file.

    Reads pre-formatted chunk dicts from the ingest path, embeds them,
    and appends to the existing database. Does NOT rebuild from scratch.

    The ingest file is atomically replaced with an empty file after
    processing, so concurrent writers (e.g. Orchestration) won't lose
    entries that arrive during processing.
    """
    asyncio.run(_ingest_async(config))


async def _ingest_async(config: dict):
    ingest_path = SCRIPT_DIR / config["sources"].get("ingest_path", "data/ingest.jsonl")
    db_path = SCRIPT_DIR / config["database"]["path"]

    if not ingest_path.exists() or ingest_path.stat().st_size == 0:
        log.info("[ingest] No entries to ingest.")
        return

    if not db_path.exists():
        log.error(f"[ingest] Database not found: {db_path}. Run 'rebuild' first.")
        return

    # Read all entries into memory, then replace file with empty one
    raw_lines = ingest_path.read_text(encoding="utf-8").splitlines()
    tmp_empty = ingest_path.with_suffix(".jsonl.tmp")
    tmp_empty.write_text("", encoding="utf-8")
    try:
        os.replace(str(tmp_empty), str(ingest_path))
    except OSError:
        # Windows fallback
        try:
            ingest_path.write_text("", encoding="utf-8")
            tmp_empty.unlink(missing_ok=True)
        except OSError:
            pass

    # Parse and validate entries
    chunks = []
    for i, line in enumerate(raw_lines):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            log.warning(f"[ingest] Skipping malformed JSON on line {i + 1}")
            continue
        if not all(k in entry for k in ("id", "text", "source")):
            log.warning(f"[ingest] Skipping entry missing required fields on line {i + 1}")
            continue
        chunks.append(entry)

    if not chunks:
        log.info("[ingest] No valid entries to ingest.")
        return

    # Deduplicate against existing DB
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    existing_ids = {
        r[0]
        for r in conn.execute("SELECT id FROM chunks").fetchall()
    }
    new_chunks = [c for c in chunks if c["id"] not in existing_ids]
    if not new_chunks:
        log.info(f"[ingest] All {len(chunks)} entries already exist in DB.")
        conn.close()
        return

    log.info(f"[ingest] {len(new_chunks)} new entries to embed (skipped {len(chunks) - len(new_chunks)} duplicates)")

    # Embed and insert
    ollama_host = config["ollama"]["host"]
    embed_model = config["ollama"]["embed_model"]
    embed_timeout = config["ollama"].get("embed_timeout", 30.0)
    dimensions = config["search"]["embed_dimensions"]
    pipeline_config = config.get("pipeline", {})
    concurrency = pipeline_config.get("concurrency", 4)
    max_retries = max(1, pipeline_config.get("max_retries", 3))
    max_embed_chars = pipeline_config.get("max_embed_chars", 6000)

    semaphore = asyncio.Semaphore(concurrency)
    embed_url = f"{ollama_host}/api/embeddings"
    embedded = 0
    errors = 0

    async with httpx.AsyncClient(timeout=embed_timeout) as client:

        async def embed_one(chunk: dict) -> tuple[dict, list[float] | None]:
            nonlocal embedded, errors
            async with semaphore:
                text = "search_document: " + chunk["text"][:max_embed_chars]
                body = {"model": embed_model, "prompt": text}
                for attempt in range(max_retries):
                    try:
                        resp = await client.post(embed_url, json=body)
                        resp.raise_for_status()
                        embedding = resp.json().get("embedding", [])
                        embedded += 1
                        return chunk, embedding
                    except Exception as e:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(1.0)
                            continue
                        errors += 1
                        if errors <= 5:
                            log.error(f"[ingest] Error embedding {chunk['id']}: {e}")
                        return chunk, None

        tasks = [embed_one(c) for c in new_chunks]
        results = await asyncio.gather(*tasks)

        conn.execute("BEGIN")
        for chunk, embedding in results:
            if embedding is None:
                continue
            if len(embedding) != dimensions:
                continue
            _insert_chunk(conn, chunk, embedding)
        conn.commit()

    # Write ingest metadata
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("BEGIN")
    conn.execute(
        "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)",
        ("last_ingest_at", now),
    )
    conn.execute(
        "INSERT OR REPLACE INTO index_metadata (key, value) VALUES (?, ?)",
        ("last_ingest_count", str(embedded)),
    )
    conn.commit()

    conn.close()
    log.info(f"[ingest] Done — {embedded} embedded, {errors} errors")


# ---------------------------------------------------------------------------
# Gotcha
# ---------------------------------------------------------------------------


def cmd_gotcha(config: dict):
    """Add or update a gotcha note on an existing chunk.

    Usage: python pipeline.py gotcha <chunk_id> "gotcha text"

    Gotcha notes are anti-hallucination annotations displayed alongside
    search results to warn about common misdiagnoses.
    """
    if len(sys.argv) < 4:
        print('Usage: python pipeline.py gotcha <chunk_id> "gotcha text"')
        sys.exit(1)

    chunk_id = sys.argv[2]
    gotcha_text = sys.argv[3]
    db_path = SCRIPT_DIR / config["database"]["path"]

    if not db_path.exists():
        log.error(f"[gotcha] Database not found: {db_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute("SELECT id FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
    if not row:
        log.error(f"[gotcha] Chunk not found: {chunk_id}")
        conn.close()
        sys.exit(1)

    conn.execute("UPDATE chunks SET gotcha = ? WHERE id = ?", (gotcha_text, chunk_id))
    conn.commit()
    conn.close()

    log.info(f"[gotcha] Updated chunk {chunk_id}")
    log.info(f"[gotcha] Text: {gotcha_text}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <command>")
        print("Commands: clone, chunk, embed, rebuild, stats, verify, stale, ingest, gotcha")
        sys.exit(1)

    command = sys.argv[1]
    config = load_config()

    if command == "clone":
        cmd_clone(config)
    elif command == "chunk":
        cmd_chunk(config)
    elif command == "embed":
        cmd_embed(config)
    elif command == "rebuild":
        cmd_clone(config)
        cmd_chunk(config)
        cmd_embed(config)
    elif command == "stats":
        cmd_stats(config)
    elif command == "verify":
        cmd_verify(config)
    elif command == "stale":
        cmd_stale(config)
    elif command == "ingest":
        cmd_ingest(config)
    elif command == "gotcha":
        cmd_gotcha(config)
    else:
        log.error(f"Unknown command: {command}")
        print("Commands: clone, chunk, embed, rebuild, stats, verify, stale, ingest, gotcha")
        sys.exit(1)


if __name__ == "__main__":
    main()
