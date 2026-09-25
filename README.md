# mcp-rag

[![CI](https://github.com/JMRussas/mcp-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/JMRussas/mcp-rag/actions/workflows/ci.yml)

**Turn any codebase into a searchable knowledge base for AI coding assistants.**

A local code-search pipeline: language-aware chunking, Ollama embeddings, SQLite storage, and MCP tools for semantic search and exact lookup. Optional BM25/vector fusion and cross-encoder reranking extend retrieval without requiring a hosted vector database.

```
Your Code ──→ Chunkers ──→ Ollama Embeddings ──→ SQLite + FTS5 ──→ MCP Server ──→ Claude Code
              (AST-aware)   (local, private)      (vector + text)    (search + lookup tools)
```

## Engineering focus

- **Inspectable storage:** SQLite and FTS5 keep chunks, metadata, and keyword search together.
- **Explicit retrieval modes:** semantic search by default; optional BM25 + reciprocal rank fusion via `search.hybrid`; optional local cross-encoder reranking.
- **Source provenance:** source hashes, freshness checks, and embedding-model drift detection help identify an index that needs rebuilding.
- **Known limits:** embeddings are loaded into memory, and confidence tiers are score heuristics rather than calibrated probabilities.

See [server.py](server.py), [pipeline.py](pipeline.py), and [provenance tests](tests/test_provenance.py) for the implementation and checks.

## Quick Start

```bash
# Clone and install
git clone https://github.com/JMRussas/mcp-rag.git
cd mcp-rag
pip install -r requirements.txt

# Pull the embedding model (one time)
ollama pull nomic-embed-text

# Configure — point at your codebase
cp config.example.json config.json
# Edit config.json: set repo paths, source tags, chunker types.
# Remove the example db_sources entry unless you have that SQLite source.

# Build the index
python pipeline.py rebuild

# Register with Claude Code
claude mcp add my-project-rag -s user -- python /path/to/mcp-rag/server.py
```

Or use Docker (set `ollama.host` to `http://ollama:11434` in your config.json):

```bash
cp config.example.json config.json
# Edit config.json: set repo paths and ollama.host to http://ollama:11434
docker compose up --build
```

## Architecture

```
Source Files          Pipeline                       MCP Server
============    ======================    ==============================

  .py files  \                           ┌─ search_<project> (semantic)
  .cs files   ├─→ Chunkers ──→ JSONL ──→ │   Ollama embed query
  .md files  /    (pluggable)            │   ──→ cosine similarity over numpy matrix
  any code                               │   ──→ source/module filtering
                                         │
                                         └─ lookup_<project> (keyword)
                                             exact type_name match
                                             ──→ case-insensitive LIKE
                                             ──→ FTS5 full-text search
                                                       │
                                                       ▼
                                             Claude Code / any MCP client
```

**Pipeline:** Parses source files with language-aware chunkers, generates embeddings via Ollama, and stores vectors + metadata in SQLite with FTS5 full-text indexing.

**Server:** Loads all embeddings into a pre-normalized numpy matrix at startup. Semantic search computes cosine similarity via dot product. Keyword lookup uses a three-tier fallback (exact match, partial match, FTS5). Tool names, descriptions, and server identity are entirely config-driven.

## Key Design Decisions

- **Config-driven tool identity.** MCP tool names and descriptions come from `config.json`, not code. The same server binary serves any project without modification.
- **Hybrid search.** Semantic similarity for "find code that does X" + three-tier keyword fallback for "show me class Y". Both available as separate MCP tools.
- **Pluggable chunker registry.** Language-specific parsers (Python AST, C# brace-depth tracking) self-register at import time. Adding a new language is one file.
- **Atomic rebuilds.** Pipeline writes to a temp DB then swaps, so the MCP server stays running during re-indexing.
- **Zero-copy embeddings.** Stored as binary blobs in SQLite, loaded once into a numpy array. No JSON serialization overhead at query time.

## Built-in Chunkers

| Type | Language | Strategy |
|------|----------|----------|
| `python` | Python (.py) | AST-based: one chunk per top-level class/function, preserves imports as context |
| `csharp` | C# (.cs) | Brace-depth tracking: one chunk per type definition, extracts namespace + doc headers |
| `digest` | Module definitions (.digest) | Two-pass depth-aware module path tracking for nested hierarchies |
| `markdown` | Markdown (.md) | Split on headings (h1-h3), strips YAML frontmatter |
| `code` | Any text file | One chunk per file, binary detection, category from directory structure |

### Writing a Custom Chunker

```python
# chunkers/rust_chunker.py
from pathlib import Path
from chunkers import register_chunker
from chunkers.base import BaseChunker


class RustChunker(BaseChunker):
    def chunk_directory(self, source_dir: Path, repo_config: dict) -> list[dict]:
        source_tag = repo_config.get("source_tag", "rust")
        chunks = []
        for rs_file in sorted(source_dir.rglob("*.rs")):
            chunks.append(
                {
                    "id": f"rust:{source_tag}:{rs_file.stem}",
                    "text": rs_file.read_text(),
                    "source": source_tag,
                    "module_path": "",
                    "type_name": "",
                    "category": "",
                    "heading": "",
                    "file_path": str(rs_file.relative_to(source_dir)),
                }
            )
        return chunks


register_chunker("rust", RustChunker)
```

Add the import to `chunkers/__init__.py` and use `"type": "rust"` in your config.

## Configuration

All tool names, descriptions, and server identity are config-driven:

```jsonc
{
  "mcp": {
    "server_name": "my-project-rag",
    "search_tool": {
      "name": "search_myproject",           // tool name (unique across MCP servers)
      "description": "Search my project."   // shown to the AI as tool description
    },
    "lookup_tool": {
      "name": "lookup_myproject",
      "description": "Look up a type by name."
    }
  },
  "ollama": {
    "host": "http://localhost:11434",
    "embed_model": "nomic-embed-text",      // 768-dimensional embeddings
    "embed_timeout": 30.0
  },
  "database": { "path": "data/rag.db" },
  "search": {
    "default_top_k": 8,
    "max_top_k": 20,
    "embed_dimensions": 768
  },
  "sources": {
    "repos_dir": "data/repos",
    "chunks_path": "data/chunks.jsonl"
  },
  "repos": [
    {
      "name": "src",
      "path": "/path/to/source",            // local path
      "type": "python",                     // chunker type
      "source_tag": "src"                   // for filtering results
    },
    {
      "name": "wiki",
      "url": "https://github.com/org/wiki.git",  // or a git URL
      "local_dir": "wiki",
      "type": "markdown",
      "source_tag": "wiki"
    }
  ]
}
```

See `examples/` for complete Python, C#, and docs-only configurations.

## Pipeline Commands

```bash
python pipeline.py clone     # Clone/update git-based repos
python pipeline.py chunk     # Run chunkers → data/chunks.jsonl
python pipeline.py embed     # Generate embeddings → data/rag.db
python pipeline.py rebuild   # All three steps in sequence
python pipeline.py stats     # Print database statistics
```

## Project Structure

```
mcp-rag/
├── server.py              MCP server — semantic search + keyword lookup
├── pipeline.py            CLI pipeline — chunk, embed, rebuild, stats
├── config.example.json    Configuration template
├── chunkers/
│   ├── __init__.py        Chunker registry (register/lookup by name)
│   ├── base.py            Abstract base class + shared utilities
│   ├── python_chunker.py  Python AST chunker
│   ├── csharp.py          C# brace-depth chunker
│   ├── digest.py          Nested module definition parser
│   ├── markdown.py        Heading-based markdown splitter
│   └── code.py            Generic whole-file chunker
├── tests/                 Pytest suite
├── examples/              Ready-to-use configs (Python, C#, docs)
├── Dockerfile             Container build
├── docker-compose.yml     One-command setup with Ollama
└── .github/workflows/     CI (lint + test)
```

## Development

Built as part of a local AI development infrastructure, extracted and open-sourced as a standalone tool. Development uses a structured review process — each commit addresses specific findings from code review passes (SQL injection safety, transaction correctness, logging hygiene). CI runs lint ([ruff](https://github.com/astral-sh/ruff)) and pytest on pushes and pull requests.

See [commit history](https://github.com/JMRussas/mcp-rag/commits/main) for the review-driven development trail.

## Limitations

- All embeddings are loaded into memory at startup; capacity depends on chunk count and embedding dimensions
- File-source updates require rebuilding; `ingest` appends new queued chunk IDs, not a general incremental file synchronizer
- Chunkers use AST/regex parsing, not full language servers
- Single Ollama instance for embedding

## Tech Stack

Python 3.11+ · [FastMCP](https://github.com/jlowin/fastmcp) · SQLite + FTS5 · [Ollama](https://ollama.com) · nomic-embed-text · numpy

## License

[AGPL-3.0-or-later](LICENSE)

## Run the checks

```bash
pip install -r requirements.txt ruff==0.16.9
ruff check .
ruff format --check .
python -m pytest tests/ -q
```

The MCP dependency is constrained to v1 because the server imports its FastMCP API. Migrating to MCP v2 requires an explicit compatibility change. Tests use fixtures and mocks; they do not establish retrieval quality on an unseen codebase.
