#!/usr/bin/env python3
"""
Production FastAPI backend for grounded RAG.

Render / Docker: bind HTTP immediately; ML loads on first /ask or /retrieve.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

print(
    f"[boot] api.main import start PORT={os.environ.get('PORT', '(unset)')}",
    flush=True,
)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.config import get_settings
from api.schemas import ErrorResponse

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=LOG_FORMAT)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """No ML work here — port must open before any model loading."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logging.getLogger(__name__).info(
        "HTTP server starting %s v%s (ML loads lazily on first query)",
        settings.app_name,
        settings.app_version,
    )
    print("[boot] FastAPI lifespan: no ML preload", flush=True)
    yield
    logging.getLogger(__name__).info("Shutting down RAG API")


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
        try:
            from api.services import get_rag_service_instance

            service = get_rag_service_instance()
            service.metrics.record_error()
        except Exception:
            pass
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                request_id=request_id,
                error="Internal server error",
                detail=str(exc) if settings.debug else None,
            ).model_dump(),
        )

    # Routers imported here (after FastAPI exists) to keep top-level import graph small.
    from api.routers import ask, debug, health, metrics, retrieve

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

    print("[boot] FastAPI app created", flush=True)
    return app


app = create_app()
print("[boot] api.main import complete", flush=True)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    port = int(os.environ.get("PORT", settings.api_port))
    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=port,
        reload=settings.debug,
    )
