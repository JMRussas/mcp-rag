#!/usr/bin/env python3
#  mcp-rag - RAG MCP Server
#
#  Exposes semantic and keyword search over an indexed codebase via the
#  Model Context Protocol (MCP). Loads a pre-built SQLite vector store
#  and serves search results to any MCP client (e.g. Claude Code).
#
#  Tool names, descriptions, and server identity are all config-driven,
#  so one codebase serves any project.
#
#  Supports hybrid search (BM25 + vector via RRF) and optional
#  cross-encoder reranking, both config-gated.
#
#  Depends on: config.json, data/*.db, numpy, httpx, mcp
#  Used by:    MCP clients (registered via `claude mcp add`)

import asyncio
import atexit
import json
import logging
import sqlite3
import struct
import sys
from pathlib import Path

import httpx
import numpy as np
from mcp.server.fastmcp import FastMCP

SCRIPT_DIR = Path(__file__).parent

log = logging.getLogger("mcp-rag-server")


def _escape_like(value: str) -> str:
    """Escape SQL LIKE wildcards so they match literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _sanitize_fts(query: str) -> str:
    """Sanitize a string for use inside an FTS5 double-quoted phrase query.

    Strips characters that are special to FTS5 even inside quotes.
    The caller wraps the result in "..." so most operators are already
    neutralized; this handles the remaining edge cases.
    """
    result = query.replace('"', "").replace("*", "").replace("^", "")
    result = result.lstrip("+-")
    return result.strip()


def _rrf_fuse(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion across multiple ranked ID lists.

    Combines rankings from different retrieval methods (e.g. vector + BM25)
    without requiring score normalization. Higher RRF score = better.

    Args:
        ranked_lists: List of ranked ID lists (best-first).
        k: RRF constant (default 60, standard value from the paper).

    Returns:
        List of (chunk_id, rrf_score) tuples, sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def format_results(rows: list, scores: dict[str, float] | None = None) -> str:
    """Format search results as readable text."""
    if not rows:
        return "No results found."

    parts = []
    for row in rows:
        header_parts = []
        if row["source"]:
            header_parts.append(f"Source: {row['source']}")
        if row["module_path"]:
            header_parts.append(f"Module: {row['module_path']}")
        if row["category"]:
            header_parts.append(f"Category: {row['category']}")
        if row["type_name"]:
            header_parts.append(f"Type: {row['type_name']}")
        if row["heading"]:
            header_parts.append(f"Section: {row['heading']}")
        if row["file_path"]:
            header_parts.append(f"File: {row['file_path']}")

        score_str = ""
        if scores and row["id"] in scores:
            score_str = f" (score: {scores[row['id']]:.3f})"

        header = " | ".join(header_parts)
        parts.append(f"--- [{header}]{score_str} ---\n{row['text']}")

    return "\n\n".join(parts)


def create_server(config_path: Path | None = None) -> FastMCP:
    """Create and configure the MCP server from a config file.

    Loads config, connects to the database, loads embeddings into memory,
    and registers search + lookup tools. Returns the FastMCP instance.
    """
    if config_path is None:
        config_path = SCRIPT_DIR / "config.json"

    if not config_path.exists():
        log.error(f"{config_path} not found. Copy config.example.json.")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # MCP server identity (all configurable)
    mcp_config = config.get("mcp", {})
    server_name = mcp_config.get("server_name", "mcp-rag")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[{server_name}] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    search_tool = mcp_config.get("search_tool", {})
    lookup_tool = mcp_config.get("lookup_tool", {})

    ollama_host = config["ollama"]["host"]
    embed_model = config["ollama"]["embed_model"]
    embed_timeout = config["ollama"].get("embed_timeout", 30.0)
    dimensions = config["search"]["embed_dimensions"]
    default_top_k = config["search"]["default_top_k"]
    max_top_k = config["search"]["max_top_k"]
    min_score = config["search"].get("min_score", 0.0)
    hybrid_enabled = config["search"].get("hybrid", False)
    retrieval_depth = config["search"].get("retrieval_depth", 20)
    rrf_k = config["search"].get("rrf_k", 60)
    reranker_config = config.get("reranker", {})
    reranker_enabled = reranker_config.get("enabled", False)
    db_path = SCRIPT_DIR / config["database"]["path"]

    mcp_server = FastMCP(server_name)

    # -------------------------------------------------------------------
    # Load embeddings into memory
    # -------------------------------------------------------------------

    conn: sqlite3.Connection | None = None
    embeddings: np.ndarray | None = None
    chunk_ids: list[str] = []
    # Pre-loaded metadata for search filtering.
    # chunk_sources is numpy for vectorized exact-match (==).
    # chunk_modules is a plain list because partial-match ("in") can't be vectorized.
    chunk_sources: np.ndarray = np.array([], dtype=object)
    chunk_modules: list[str] = []

    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        atexit.register(conn.close)

        rows = conn.execute(
            "SELECT id, source, module_path, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        if rows:
            # Validate embedding dimensions match config
            expected_blob_size = dimensions * 4  # 4 bytes per float32
            actual_blob_size = len(rows[0]["embedding"])
            if actual_blob_size != expected_blob_size:
                actual_dims = actual_blob_size // 4
                log.error(
                    f"Database embeddings have {actual_dims} dimensions "
                    f"but config specifies {dimensions}. "
                    f"Rebuild with 'python pipeline.py rebuild'."
                )
                sys.exit(1)
            else:
                sources_list = []
                modules_list = []
                vectors = []
                for row in rows:
                    chunk_ids.append(row["id"])
                    sources_list.append(row["source"] or "")
                    modules_list.append((row["module_path"] or "").lower())
                    floats = struct.unpack(f"{dimensions}f", row["embedding"])
                    vectors.append(floats)

                chunk_sources = np.array(sources_list, dtype=object)
                chunk_modules = modules_list

                embeddings = np.array(vectors, dtype=np.float32)
                # Normalize for cosine similarity via dot product
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms

                mem_mb = embeddings.nbytes / 1024 / 1024
                log.info(f"Loaded {len(chunk_ids)} embeddings ({embeddings.shape}, ~{mem_mb:.0f} MB)")
                if len(chunk_ids) > 50_000:
                    log.warning(
                        f"Large embedding set ({len(chunk_ids)} chunks, ~{mem_mb:.0f} MB). "
                        f"Memory usage may be high. Consider splitting into multiple databases."
                    )
        else:
            log.warning("No embeddings found in database.")
    else:
        log.warning(f"{db_path} not found. Run 'python pipeline.py rebuild' first.")

    # -------------------------------------------------------------------
    # Embedding helper (shared client for connection reuse)
    # -------------------------------------------------------------------

    _http_client = httpx.AsyncClient(timeout=embed_timeout)

    async def get_query_embedding(text: str) -> np.ndarray | None:
        """Embed a query string via Ollama."""
        url = f"{ollama_host}/api/embeddings"
        body = {"model": embed_model, "prompt": "search_query: " + text}
        max_retries = 2

        for attempt in range(max_retries):
            try:
                resp = await _http_client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
                embedding = data.get("embedding", [])
                if len(embedding) != dimensions:
                    return None
                vec = np.array(embedding, dtype=np.float32)
                vec = vec / (np.linalg.norm(vec) or 1)
                return vec
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
                    continue
                log.error(f"Embedding error: {e}")
                return None
        return None

    # -------------------------------------------------------------------
    # MCP Tools — names and descriptions are config-driven
    # -------------------------------------------------------------------

    search_name = search_tool.get("name", "search")
    search_desc = search_tool.get("description", "Search the indexed codebase using semantic similarity.")
    lookup_name = lookup_tool.get("name", "lookup")
    lookup_desc = lookup_tool.get("description", "Look up a specific type or API by exact name.")

    @mcp_server.tool(name=search_name, description=search_desc)
    async def search(
        query: str,
        top_k: int = default_top_k,
        source_filter: str = "",
        module_filter: str = "",
    ) -> str:
        """Semantic search over the indexed codebase.

        Args:
            query: Natural language description of what you're looking for.
            top_k: Number of results to return (default 8, max 20).
            source_filter: Optional. Filter by source tag. Empty returns all.
            module_filter: Optional. Filter by module path (partial match).
        """
        if embeddings is None or conn is None:
            return "Error: Database not loaded. Run 'python pipeline.py rebuild' first."

        top_k = min(max(1, top_k), max_top_k)
        depth = retrieval_depth if hybrid_enabled else top_k

        query_vec = await get_query_embedding(query)
        if query_vec is None:
            return "Error: Failed to generate query embedding. Is Ollama running?"

        # --- Vector retrieval ---
        similarities = embeddings @ query_vec

        # Apply filters using pre-loaded metadata (vectorized where possible)
        filter_mask = None
        if source_filter or module_filter:
            filter_mask = np.ones(len(chunk_ids), dtype=bool)
            if source_filter:
                filter_mask &= chunk_sources == source_filter
            if module_filter:
                module_filter_lower = module_filter.lower()
                filter_mask &= np.array([module_filter_lower in m for m in chunk_modules])
            similarities[~filter_mask] = -1

        top_indices = np.argsort(similarities)[::-1][:depth]
        vector_ranked = [
            chunk_ids[idx] for idx in top_indices
            if similarities[idx] >= min_score
        ]

        # --- BM25 retrieval (hybrid mode) ---
        if hybrid_enabled:
            bm25_ranked = _bm25_retrieve(
                conn, query, depth * 2, source_filter, module_filter,
                chunk_ids, chunk_sources, chunk_modules, filter_mask,
            )
            # Fuse with RRF
            fused = _rrf_fuse([vector_ranked, bm25_ranked], k=rrf_k)
            ranked_ids = [doc_id for doc_id, _ in fused[:depth]]
            result_scores = {doc_id: score for doc_id, score in fused[:depth]}
        else:
            ranked_ids = vector_ranked[:top_k]
            result_scores = {chunk_ids[idx]: float(similarities[idx]) for idx in top_indices
                             if chunk_ids[idx] in ranked_ids}

        if not ranked_ids:
            return format_results([])

        # Fetch full rows
        placeholders = ",".join("?" for _ in ranked_ids)
        rows = conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({placeholders})",
            ranked_ids,
        ).fetchall()
        row_map = {row["id"]: row for row in rows}
        results = [row_map[cid] for cid in ranked_ids if cid in row_map]

        # --- Reranking (optional) ---
        if reranker_enabled and len(results) > 1:
            try:
                from reranker import rerank
                results = rerank(query, results, config)
                # Reassign scores based on reranked order
                result_scores = {r["id"]: 1.0 / (i + 1) for i, r in enumerate(results)}
            except Exception as e:
                log.warning(f"Reranker failed, using original ranking: {e}")

        # Apply final top_k
        results = results[:top_k]
        final_scores = {r["id"]: result_scores.get(r["id"], 0.0) for r in results}

        return format_results(results, final_scores)

    @mcp_server.tool(name=lookup_name, description=lookup_desc)
    async def lookup(
        name: str,
        top_k: int = 5,
    ) -> str:
        """Look up a specific type, class, or function by exact name.

        Args:
            name: The type, class, function, or keyword to look up.
            top_k: Number of results to return (default 5, max 20).
        """
        if conn is None:
            return "Error: Database not loaded. Run 'python pipeline.py rebuild' first."

        top_k = min(max(1, top_k), max_top_k)

        # First try exact type_name match
        rows = conn.execute(
            "SELECT * FROM chunks WHERE type_name = ? LIMIT ?",
            (name, top_k),
        ).fetchall()

        if rows:
            return format_results(rows)

        # Try case-insensitive partial type_name match
        safe_like = _escape_like(name)
        rows = conn.execute(
            "SELECT * FROM chunks WHERE type_name LIKE ? ESCAPE '\\' LIMIT ?",
            (f"%{safe_like}%", top_k),
        ).fetchall()

        if rows:
            return format_results(rows)

        # Fall back to FTS5 full-text search
        try:
            safe_name = _sanitize_fts(name)
            if not safe_name:
                return format_results([])
            rows = conn.execute(
                "SELECT chunks.* FROM chunks_fts JOIN chunks ON chunks.rowid = chunks_fts.rowid WHERE chunks_fts MATCH ? LIMIT ?",
                (f'"{safe_name}"', top_k),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log.warning(f"FTS5 query failed for '{name}', falling back to LIKE: {e}")
            safe_like = _escape_like(name)
            rows = conn.execute(
                "SELECT * FROM chunks WHERE text LIKE ? ESCAPE '\\' LIMIT ?",
                (f"%{safe_like}%", top_k),
            ).fetchall()

        return format_results(rows)

    def _bm25_retrieve(
        db_conn: sqlite3.Connection,
        query: str,
        limit: int,
        source_filter: str,
        module_filter: str,
        all_chunk_ids: list[str],
        all_chunk_sources: np.ndarray,
        all_chunk_modules: list[str],
        precomputed_mask: np.ndarray | None,
    ) -> list[str]:
        """Retrieve chunk IDs ranked by BM25 relevance via FTS5.

        Filters are applied post-query in Python (consistent with vector path).
        Returns a ranked list of chunk IDs (best-first).
        """
        safe_query = _sanitize_fts(query)
        if not safe_query:
            return []

        try:
            # FTS5 bm25() returns negative scores (lower = better match)
            bm25_rows = db_conn.execute(
                """SELECT chunks.id, bm25(chunks_fts) as bm25_score
                   FROM chunks_fts
                   JOIN chunks ON chunks.rowid = chunks_fts.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY bm25(chunks_fts)
                   LIMIT ?""",
                (f'"{safe_query}"', limit),
            ).fetchall()
        except sqlite3.OperationalError as e:
            log.warning(f"BM25 query failed: {e}")
            return []

        if not bm25_rows:
            return []

        # Build ID-to-index lookup for filter checking
        if source_filter or module_filter:
            id_to_idx = {cid: i for i, cid in enumerate(all_chunk_ids)}
            filtered = []
            for row in bm25_rows:
                idx = id_to_idx.get(row[0])
                if idx is None:
                    continue
                if precomputed_mask is not None and not precomputed_mask[idx]:
                    continue
                filtered.append(row[0])
            return filtered
        else:
            return [row[0] for row in bm25_rows]

    return mcp_server


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server = create_server()
    server.run(transport="stdio")
