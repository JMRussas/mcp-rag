# mcp-rag

Configurable RAG pipeline + MCP server that indexes any codebase for AI-assisted search.

## Build / Run

```bash
pip install -r requirements.txt       # Install dependencies
cp config.example.json config.json    # Create config (edit paths)
python pipeline.py rebuild            # Build the index
python server.py                      # Start MCP server (stdio transport)
```

## Pipeline Commands

| Command | What it does |
|---------|-------------|
| `python pipeline.py clone` | Clone/update git-based repos |
| `python pipeline.py chunk` | Run chunkers, write `data/chunks.jsonl` |
| `python pipeline.py embed` | Generate embeddings, build SQLite DB |
| `python pipeline.py rebuild` | clone + chunk + embed |
| `python pipeline.py stats` | Print database statistics |

## Project Structure

| File | Role | Depends On | Used By |
|------|------|-----------|---------|
| `server.py` | MCP server — exposes search + lookup tools | config.json, data/*.db, numpy, httpx, mcp | Claude Code (MCP client) |
| `pipeline.py` | CLI pipeline — chunk, embed, rebuild | config.json, chunkers/, httpx | Manual invocation |
| `chunkers/base.py` | Abstract base class for chunkers | — | All chunker implementations |
| `chunkers/__init__.py` | Chunker registry | base.py, all chunkers | pipeline.py |
| `chunkers/python_chunker.py` | Python AST chunker | base.py | pipeline.py |
| `chunkers/csharp.py` | C# type-boundary chunker | base.py | pipeline.py |
| `chunkers/digest.py` | Nested module definition parser (.digest) | base.py | pipeline.py |
| `chunkers/markdown.py` | Markdown heading splitter | base.py | pipeline.py |
| `chunkers/code.py` | Generic whole-file chunker | base.py | pipeline.py |
| `config.example.json` | Configuration template | — | User copies to config.json |
| `tests/` | Pytest test suite | chunkers/, pipeline.py | CI, manual verification |

## Conventions

- **Chunk format:** Every chunker returns dicts with keys: `id`, `text`, `source`, `module_path`, `type_name`, `category`, `heading`, `file_path`
- **Chunker registration:** Each chunker calls `register_chunker("name", ClassName)` at module level
- **Config-driven tools:** MCP tool names and descriptions come from `config.json`, not code
- **Embedding prefix:** Documents get `"search_document: "`, queries get `"search_query: "` (nomic-embed-text convention)
- **Temp DB swap:** Pipeline writes to `.db.tmp` then swaps, so the MCP server can stay running during rebuilds
- **Minimum similarity threshold:** Configurable via `search.min_score` in config.json (default 0.0)
- **HTTP client reuse:** The MCP server creates one `httpx.AsyncClient` per server instance, reused across all queries (cleaned up via `atexit`)
- **Embed char limit:** `pipeline.max_embed_chars` (default 6000) — max characters sent to embedding model per chunk
- **FTS5 sanitization:** `_sanitize_fts()` in server.py strips special FTS5 characters before MATCH queries; `_escape_like()` handles LIKE queries
- **Binary file detection:** Code chunker skips binary files (null-byte heuristic) when no `extensions` filter is set
- **Dimension validation:** Server validates embedding blob size on startup; pipeline warns on dimension mismatches during embedding
- **Logging:** Both `pipeline.py` and `server.py` use Python's `logging` module with module-level loggers (`log = logging.getLogger(...)`). Pipeline configures logging in `main()`. Server logs to stderr (MCP uses stdout for protocol). CLI usage/help text stays as `print()`.
- **Transaction batching:** Pipeline wraps each embedding batch in an explicit `BEGIN`/`COMMIT` transaction. Uses `isolation_level=None` for manual control.
- **Git timeouts:** Clone and pull operations have a 120-second timeout to prevent hung pipelines
