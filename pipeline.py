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
import json
import logging
import os
import shutil
import sqlite3
import struct
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
    required_sections = ["ollama", "database", "search", "sources", "repos"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"config.json missing required section '{section}'.")

    if not isinstance(config["ollama"].get("host"), str) or not config["ollama"]["host"].startswith("http"):
        raise ConfigError("config.json 'ollama.host' must be a URL starting with http.")

    dims = config["search"].get("embed_dimensions")
    if not isinstance(dims, int) or dims <= 0:
        raise ConfigError("config.json 'search.embed_dimensions' must be a positive integer.")

    if not isinstance(config["repos"], list) or len(config["repos"]) == 0:
        raise ConfigError("config.json 'repos' must be a non-empty list.")

    for i, repo in enumerate(config["repos"]):
        if "name" not in repo:
            raise ConfigError(f"config.json repos[{i}] missing 'name'.")
        if "type" not in repo:
            raise ConfigError(f"config.json repos[{i}] ('{repo['name']}') missing 'type'.")
        if "path" not in repo and "url" not in repo:
            raise ConfigError(f"config.json repos[{i}] ('{repo['name']}') needs 'path' or 'url'.")


# ---------------------------------------------------------------------------
# Clone (for git-based sources)
# ---------------------------------------------------------------------------


def cmd_clone(config: dict):
    """Clone or update git-based source repos."""
    import subprocess

    repos_dir = SCRIPT_DIR / config["sources"].get("repos_dir", "data/repos")
    repos_dir.mkdir(parents=True, exist_ok=True)

    for repo in config["repos"]:
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


def cmd_chunk(config: dict):
    """Run all chunkers and write chunks.jsonl."""
    chunks_path = SCRIPT_DIR / config["sources"]["chunks_path"]
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    repos_dir = SCRIPT_DIR / config["sources"].get("repos_dir", "data/repos")

    all_chunks = []

    for repo in config["repos"]:
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

        try:
            chunker = get_chunker(chunker_type)
        except ValueError as e:
            log.warning(str(e))
            continue

        chunks = chunker.chunk_directory(source_dir, repo)
        all_chunks.extend(chunks)

    # Deduplicate by ID (keep first occurrence)
    seen = set()
    deduped = []
    for chunk in all_chunks:
        if chunk["id"] not in seen:
            seen.add(chunk["id"])
            deduped.append(chunk)

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
           (id, text, source, module_path, type_name, category, heading, file_path, embedding)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chunk["id"],
            chunk["text"],
            chunk["source"],
            chunk.get("module_path", ""),
            chunk.get("type_name", ""),
            chunk.get("category", ""),
            chunk.get("heading", ""),
            chunk.get("file_path", ""),
            blob,
        ),
    )

    conn.execute(
        """INSERT INTO chunks_fts(rowid, id, text, type_name, module_path, heading, category)
           SELECT rowid, id, text, type_name, module_path, heading, category
           FROM chunks WHERE id = ?""",
        (chunk["id"],),
    )


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
# Main
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <command>")
        print("Commands: clone, chunk, embed, rebuild, stats")
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
    else:
        log.error(f"Unknown command: {command}")
        print("Commands: clone, chunk, embed, rebuild, stats")
        sys.exit(1)


if __name__ == "__main__":
    main()
