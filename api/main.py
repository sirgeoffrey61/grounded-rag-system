#!/usr/bin/env python3
"""
Phase 10 — Production FastAPI backend for grounded RAG.

Why singleton loading matters:
    See api/services.py — models load once at startup, not per HTTP request.

Why APIs are critical for ML systems:
    REST/OpenAPI contracts let frontends, agents, and eval pipelines share one
    inference path with versioning, auth (TODO), and monitoring.

Why observability matters in production AI:
    Middleware attaches request IDs; logs include stage latency; /metrics tracks
    drift in confidence and errors — required for SLOs and incident response.

Startup:
    uvicorn api.main:app --reload

Example requests:
    curl http://localhost:8000/health

    curl -X POST http://localhost:8000/retrieve \\
      -H "Content-Type: application/json" \\
      -d '{"question": "What did Keynes teach about saving?", "top_k": 5}'

    curl -X POST http://localhost:8000/ask \\
      -H "Content-Type: application/json" \\
      -d '{"question": "What did Keynes teach about saving?", "top_k": 5, "verbose": true}'

    curl http://localhost:8000/metrics
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.routers import ask, debug, health, metrics, retrieve
from api.schemas import ErrorResponse
from api.services import get_rag_service_instance

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load ML resources at startup; release on shutdown."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    service = get_rag_service_instance()
    try:
        service.initialize()
        app.state.rag_service = service
    except Exception as exc:
        logger.exception("Startup failed: %s", exc)
        raise

    yield

    logger.info("Shutting down RAG API")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production grounded RAG API (hybrid retrieval + rerank + citations).",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))[:64]
        request.state.request_id = request_id
        start = time.perf_counter()
        logger = logging.getLogger("api.request")
        logger.info(
            "start request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "error request_id=%s duration_ms=%.1f",
                request_id,
                (time.perf_counter() - start) * 1000,
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "end request_id=%s status=%s duration_ms=%.1f",
            request_id,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "unknown")
        logging.getLogger(__name__).exception(
            "unhandled request_id=%s: %s", request_id, exc
        )
        service = getattr(request.app.state, "rag_service", None)
        if service is not None:
            service.metrics.record_error()
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                request_id=request_id,
                error="Internal server error",
                detail=str(exc) if settings.debug else None,
            ).model_dump(),
        )

    app.include_router(health.router)
    app.include_router(metrics.router)
    app.include_router(debug.router)
    app.include_router(ask.router)
    app.include_router(retrieve.router)

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
        }

    return app


app = create_app()


# =============================================================================
# TODO — Production hardening
# =============================================================================
# TODO: Authentication — API keys or OAuth2 for multi-tenant deployments.
# TODO: Rate limiting — per-IP / per-key token bucket (slowapi or reverse proxy).
# TODO: Async inference — offload LLM to thread pool or task queue (Celery/ARQ).
# TODO: Streaming responses — Server-Sent Events for token-by-token answers.
# TODO: Prometheus metrics — replace in-process counters with prometheus_client.
#
# =============================================================================
# How to run
# =============================================================================
# pip install -r RAG_REQUIREMENTS.txt
# ollama pull mistral
# uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
