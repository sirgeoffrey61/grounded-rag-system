"""POST /ask — grounded question answering."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from api.dependencies import RAGServiceDep, RequestIdDep
from api.schemas import AskRequest, AskResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    responses={500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    summary="Grounded question answering",
    description=(
        "Run hybrid retrieval, cross-encoder reranking, and grounded LLM generation. "
        "Returns answer with validated citations and confidence."
    ),
)
def ask_question(
    body: AskRequest,
    service: RAGServiceDep,
    request_id: RequestIdDep,
) -> AskResponse:
    logger.info(
        "ask endpoint request_id=%s question=%r top_k=%d",
        request_id,
        body.question[:80],
        body.top_k,
    )
    try:
        return service.ask(
            question=body.question,
            top_k=body.top_k,
            candidate_k=body.candidate_k,
            verbose=body.verbose,
            request_id=request_id,
        )
    except RuntimeError as exc:
        service.metrics.record_error()
        logger.exception("Ask endpoint failed (runtime) request_id=%s", request_id)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        service.metrics.record_error()
        logger.exception("Ask endpoint failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc
