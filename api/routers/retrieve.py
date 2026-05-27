"""POST /retrieve — retrieval and reranking without generation."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from api.dependencies import RAGServiceDep, RequestIdDep
from api.schemas import ErrorResponse, RetrieveRequest, RetrieveResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["retrieve"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    responses={500: {"model": ErrorResponse}},
    summary="Retrieve and rerank chunks",
    description="Return top reranked chunks with similarity and reranker scores (no LLM).",
)
async def retrieve_chunks(
    body: RetrieveRequest,
    service: RAGServiceDep,
    request_id: RequestIdDep,
) -> RetrieveResponse:
    await service.ensure_initialized()
    logger.info("retrieve request_id=%s top_k=%d", request_id, body.top_k)
    try:
        return await asyncio.to_thread(
            service.retrieve,
            body.question,
            body.top_k,
            body.candidate_k,
            body.include_text,
            request_id,
        )
    except RuntimeError as exc:
        service.metrics.record_error()
        logger.exception("retrieve failed request_id=%s", request_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        service.metrics.record_error()
        logger.exception("Retrieve endpoint failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
