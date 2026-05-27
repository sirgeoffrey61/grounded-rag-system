"""GET /health — dependency probes."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Response

from api.dependencies import RAGServiceDep
from api.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Probe Chroma, embedding model, and LLM (Groq) availability.",
)
def health_check(service: RAGServiceDep, response: Response) -> HealthResponse:
    report = service.check_health()
    if report.status == "unhealthy":
        response.status_code = 503
    elif report.status == "degraded":
        response.status_code = 200
    logger.debug("health status=%s", report.status)
    return report
