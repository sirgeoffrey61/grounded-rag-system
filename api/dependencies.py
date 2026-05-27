"""
FastAPI dependency injection for shared resources and request context.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Request

from api.config import Settings, get_settings
from api.services import RAGService, get_rag_service_instance


def get_settings_dep() -> Settings:
    return get_settings()


def get_rag_service() -> RAGService:
    return get_rag_service_instance()


def get_request_id(request: Request) -> str:
    """
  Request ID for log correlation.

  Clients may pass X-Request-ID; otherwise a UUID is generated.
  """
    existing = request.headers.get("X-Request-ID")
    if existing and existing.strip():
        return existing.strip()[:64]
    rid = getattr(request.state, "request_id", None)
    if rid:
        return rid
    return str(uuid.uuid4())


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
RAGServiceDep = Annotated[RAGService, Depends(get_rag_service)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
