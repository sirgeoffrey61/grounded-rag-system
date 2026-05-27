"""
JSON-safe conversion for API responses.

Why JSON serialization fails with numpy types:
    FastAPI uses jsonable_encoder on responses. numpy.float32 / numpy.float64 are
    not native JSON types — encoding raises ValueError even when Pydantic accepted
    the model at construction time.

Why observability matters in ML systems:
    Surfacing the real exception (not a generic 500) saves hours when retrieval,
    reranking, or Ollama fail in production.

Why backend tracebacks are critical:
    Log full tracebacks server-side; return safe detail strings to clients.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_python_float(value: Any) -> float | None:
    """Coerce numpy / numeric scalars to native float for JSON."""
    if value is None:
        return None
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_python_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def sanitize_confidence(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": to_python_float(data.get("score")) or 0.0,
        "level": str(data.get("level", "low")),
        "mean_rerank_score": to_python_float(data.get("mean_rerank_score")) or 0.0,
        "top_rerank_score": to_python_float(data.get("top_rerank_score")) or 0.0,
        "hybrid_source_ratio": to_python_float(data.get("hybrid_source_ratio")) or 0.0,
        "dual_channel_ratio": to_python_float(data.get("dual_channel_ratio")) or 0.0,
        "notes": list(data.get("notes") or []),
    }


def sanitize_latency(
    retrieval_rerank_seconds: float,
    generation_seconds: float | None,
    total_seconds: float,
) -> dict[str, Any]:
    return {
        "retrieval_rerank_seconds": to_python_float(retrieval_rerank_seconds) or 0.0,
        "generation_seconds": to_python_float(generation_seconds),
        "total_seconds": to_python_float(total_seconds) or 0.0,
    }
