#!/usr/bin/env bash
#  mcp-rag - Fork Sync Script
#
#  Copies shared files from mcp-rag to fork projects (noz-rag, verse-rag).
#  Only copies infrastructure files — fork-specific chunkers, configs, and
#  docs are left untouched.
#
#  Usage:
#    ./tools/sync-forks.sh                          # Sync all defaults
#    ./tools/sync-forks.sh ~/Git/noz-rag            # Sync one fork
#    ./tools/sync-forks.sh ~/Git/noz-rag ~/Git/verse-rag  # Explicit list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCP_RAG_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default fork paths
DEFAULT_FORKS=(
    "$HOME/Git/noz-rag"
    "$HOME/Git/verse-rag"
)

# Files to sync (relative to project root)
SHARED_FILES=(
    "server.py"
    "pipeline.py"
    "reranker.py"
    "chunkers/base.py"
    "requirements.txt"
    "ruff.toml"
)

# Files NOT synced (fork-specific):
#   chunkers/__init__.py    — different chunker imports per fork
#   chunkers/*.py           — fork-specific chunkers (csharp, digest, etc.)
#   config.json             — fork-specific paths and tool names
#   config.example.json     — fork-specific examples
#   CLAUDE.md               — fork-specific docs
#   data/                   — fork-specific indexed data
#   tests/                  — fork-specific test fixtures

# Note: chunkers like csharp.py, markdown.py, digest.py, code.py ARE shared
# but they live in mcp-rag as the canonical source. Forks that use them will
# already have identical copies. If a fork adds a new chunker, it stays local.
SHARED_CHUNKERS=(
    "chunkers/csharp.py"
    "chunkers/markdown.py"
    "chunkers/digest.py"
    "chunkers/code.py"
)

forks=("${@:-${DEFAULT_FORKS[@]}}")

for fork_dir in "${forks[@]}"; do
    # Normalize path
    fork_dir="$(cd "$fork_dir" 2>/dev/null && pwd)" || {
        echo "SKIP: $fork_dir does not exist"
        continue
    }
    fork_name="$(basename "$fork_dir")"
    echo ""
    echo "=== Syncing $fork_name ==="

    changed=0

    # Sync infrastructure files
    for file in "${SHARED_FILES[@]}"; do
        src="$MCP_RAG_DIR/$file"
        dst="$fork_dir/$file"

        if [ ! -f "$src" ]; then
            echo "  WARN: $file missing from mcp-rag"
            continue
        fi

        if [ -f "$dst" ] && diff -q "$src" "$dst" > /dev/null 2>&1; then
            continue  # Already identical
        fi

        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
        echo "  UPDATED: $file"
        changed=$((changed + 1))
    done

    # Sync chunkers that exist in both places
    for file in "${SHARED_CHUNKERS[@]}"; do
        src="$MCP_RAG_DIR/$file"
        dst="$fork_dir/$file"

        if [ ! -f "$src" ]; then
            continue
        fi
        if [ ! -f "$dst" ]; then
            continue  # Fork doesn't use this chunker
        fi

        if diff -q "$src" "$dst" > /dev/null 2>&1; then
            continue  # Already identical
        fi

        cp "$src" "$dst"
        echo "  UPDATED: $file"
        changed=$((changed + 1))
    done

    if [ "$changed" -eq 0 ]; then
        echo "  All shared files already up to date."
    else
        echo "  $changed file(s) updated."
        echo "  >> Run 'python pipeline.py rebuild' in $fork_name to re-index."
    fi
done

echo ""
echo "Done. Shared files synced from mcp-rag."
