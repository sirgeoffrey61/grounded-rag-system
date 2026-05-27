"""GET /debug/status — detailed dependency diagnostics."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import RAGServiceDep

router = APIRouter(tags=["debug"])


@router.get(
    "/debug/status",
    summary="Debug status",
    description="LLM, Chroma, embedding model, reranker, and collection diagnostics.",
)
def debug_status(service: RAGServiceDep) -> dict:
    return service.get_debug_status()
