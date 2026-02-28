# mcp-rag

**Turn any codebase into a searchable knowledge base for AI coding assistants.**

Most RAG frameworks require heavy infrastructure — vector databases, LangChain, cloud APIs. mcp-rag takes a different approach: a single Python pipeline that chunks your code, embeds it locally via [Ollama](https://ollama.com), stores everything in SQLite, and serves hybrid semantic + keyword search over the [Model Context Protocol (MCP)](https://modelcontextprotocol.io).

One config file. No cloud dependencies. Works with any MCP client.

```
Your Code ──→ Chunkers ──→ Ollama Embeddings ──→ SQLite + FTS5 ──→ MCP Server ──→ Claude Code
              (AST-aware)   (local, private)      (vector + text)    (search + lookup tools)
```

## Why mcp-rag?

| | mcp-rag | LangChain / LlamaIndex | Cloud RAG (Pinecone, etc.) |
|---|---|---|---|
| **Dependencies** | 3 Python packages + Ollama | Dozens of packages, complex chains | Cloud account, API keys, billing |
| **Infrastructure** | SQLite (zero config) | Requires external vector DB | Managed service |
| **Privacy** | 100% local, nothing leaves your machine | Depends on provider choices | Data sent to cloud |
| **MCP native** | Built as an MCP server from the ground up | Requires adapter/wrapper | Requires adapter/wrapper |
| **Config model** | One JSON file drives everything | Code changes per project | Dashboard + code changes |

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
# Edit config.json: set repo paths, source tags, chunker types

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
            chunks.append({
                "id": f"rust:{source_tag}:{rs_file.stem}",
                "text": rs_file.read_text(),
                "source": source_tag,
                "module_path": "",
                "type_name": "",
                "category": "",
                "heading": "",
                "file_path": str(rs_file.relative_to(source_dir)),
            })
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
├── tests/                 Pytest suite (61 tests)
├── examples/              Ready-to-use configs (Python, C#, docs)
├── Dockerfile             Container build
├── docker-compose.yml     One-command setup with Ollama
└── .github/workflows/     CI (lint + test)
```

## Development

Built as part of a local AI development infrastructure, extracted and open-sourced as a standalone tool. Development uses a structured review process — each commit addresses specific findings from code review passes (SQL injection safety, transaction correctness, logging hygiene). 61 tests with CI running lint ([ruff](https://github.com/astral-sh/ruff)) and pytest on every push.

See [commit history](https://github.com/JMRussas/mcp-rag/commits/main) for the review-driven development trail.

## Limitations

- All embeddings loaded into memory at startup — practical up to ~50k chunks (~150 MB)
- Full rebuild each time (no incremental re-indexing)
- Chunkers use AST/regex parsing, not full language servers
- Single Ollama instance for embedding

## Tech Stack

Python 3.11+ · [FastMCP](https://github.com/jlowin/fastmcp) · SQLite + FTS5 · [Ollama](https://ollama.com) · nomic-embed-text · numpy

## License

MIT
