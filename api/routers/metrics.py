"""GET /metrics — in-process request statistics."""

from __future__ import annotations

from fastapi import APIRouter

from api.dependencies import RAGServiceDep
from api.schemas import MetricsResponse

router = APIRouter(tags=["metrics"])


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Service metrics",
    description="Request counts, latency averages, errors, and confidence distribution.",
)
def get_metrics(service: RAGServiceDep) -> MetricsResponse:
    return service.metrics.snapshot()
