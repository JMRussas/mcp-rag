#  mcp-rag - Chunker Registry
#
#  Maps chunker type names (from config.json) to chunker classes.
#  Built-in chunkers self-register on import.
#
#  Depends on: chunkers/base.py
#  Used by:    pipeline.py

from chunkers.base import BaseChunker

CHUNKER_REGISTRY: dict[str, type[BaseChunker]] = {}


def register_chunker(name: str, cls: type[BaseChunker]):
    """Register a chunker class under the given type name."""
    CHUNKER_REGISTRY[name] = cls


def get_chunker(name: str) -> BaseChunker:
    """Get an instance of the chunker registered under the given type name."""
    if name not in CHUNKER_REGISTRY:
        available = ", ".join(sorted(CHUNKER_REGISTRY.keys()))
        raise ValueError(f"Unknown chunker type: '{name}'. Available: {available}")
    return CHUNKER_REGISTRY[name]()


# Import built-in chunkers to trigger self-registration
from chunkers import code as _code  # noqa: F401, E402
from chunkers import csharp as _csharp  # noqa: F401, E402
from chunkers import digest as _digest  # noqa: F401, E402
from chunkers import markdown as _markdown  # noqa: F401, E402
from chunkers import python_chunker as _python  # noqa: F401, E402
