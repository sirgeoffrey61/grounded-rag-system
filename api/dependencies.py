"""
FastAPI dependency injection — lightweight only (no ML imports).
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import Depends, Request

from api.config import Settings, get_settings


def get_settings_dep() -> Settings:
    return get_settings()


def get_rag_service() -> Any:
    """Return singleton RAGService without loading ML stacks at import time."""
    from api.services import get_rag_service_instance

    return get_rag_service_instance()


def get_request_id(request: Request) -> str:
    existing = request.headers.get("X-Request-ID")
    if existing and existing.strip():
        return existing.strip()[:64]
    rid = getattr(request.state, "request_id", None)
    if rid:
        return rid
    return str(uuid.uuid4())


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
RAGServiceDep = Annotated[Any, Depends(get_rag_service)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
