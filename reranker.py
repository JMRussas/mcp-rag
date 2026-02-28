#  mcp-rag - Cross-Encoder Reranker
#
#  Reranks search candidates using a cross-encoder model for improved
#  relevance. Lazy-loads the model on first use to avoid startup cost
#  when reranking is disabled.
#
#  Uses sentence-transformers CrossEncoder with ONNX backend by default
#  (avoids pulling in full PyTorch for inference).
#
#  Depends on: sentence-transformers[onnx] (optional, only when enabled)
#  Used by:    server.py (when config reranker.enabled = true)

import logging

log = logging.getLogger("mcp-rag-server")

_reranker = None


def get_reranker(config: dict):
    """Load the cross-encoder model on first use.

    Subsequent calls return the cached instance. The model is downloaded
    from HuggingFace on first load if not cached locally.

    Args:
        config: Full config dict (uses 'reranker' section).

    Returns:
        A CrossEncoder instance.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        reranker_config = config.get("reranker", {})
        model_name = reranker_config.get("model", "cross-encoder/ms-marco-MiniLM-L6-v2")
        backend = reranker_config.get("backend", "onnx")

        log.info(f"Loading reranker: {model_name} (backend={backend})")
        try:
            _reranker = CrossEncoder(model_name, backend=backend)
        except Exception as e:
            # Fall back to default backend if ONNX fails
            log.warning(f"ONNX backend failed ({e}), falling back to default backend")
            _reranker = CrossEncoder(model_name)
        log.info("Reranker loaded")

    return _reranker


def rerank(query: str, candidates: list, config: dict) -> list:
    """Rerank candidate chunks by cross-encoder relevance.

    Args:
        query: The user's search query.
        candidates: List of sqlite3.Row or dict objects (must have 'id' and 'text' keys).
        config: Full config dict.

    Returns:
        Candidates reordered by cross-encoder score (best first).
    """
    if len(candidates) <= 1:
        return candidates

    reranker = get_reranker(config)
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scored]
