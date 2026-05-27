"""
Pydantic request/response models for API validation and OpenAPI docs.

Strict schemas prevent malformed client payloads from reaching expensive
ML inference paths and document the contract for integrators.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------


class LatencyMetrics(BaseModel):
    """End-to-end and stage latency in seconds."""

    retrieval_rerank_seconds: float = Field(ge=0)
    generation_seconds: float | None = Field(default=None, ge=0)
    total_seconds: float = Field(ge=0)


class ConfidenceSchema(BaseModel):
    """Grounded confidence report exposed to clients."""

    score: float = Field(ge=0, le=1)
    level: str
    mean_rerank_score: float
    top_rerank_score: float
    hybrid_source_ratio: float
    dual_channel_ratio: float
    notes: list[str] = Field(default_factory=list)


class CitationSchema(BaseModel):
    """Validated citation from answer text — metadata only from retrieval."""

    citation_id: int
    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str = ""


class SourceChunkSchema(BaseModel):
    """One reranked source passage with scores and provenance."""

    citation_id: int
    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str = ""
    text: str = ""
    source_type: str
    dense_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float
    rerank_score: float
    rerank_rank: int
    hybrid_rank: int | None = None
    rank_delta: int | None = None


# ---------------------------------------------------------------------------
# POST /ask
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=25, ge=5, le=50)
    verbose: bool = False

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class AskResponse(BaseModel):
    request_id: str
    question: str
    answer: str
    confidence: ConfidenceSchema
    citations: list[CitationSchema]
    sources: list[SourceChunkSchema]
    latency: LatencyMetrics
    candidate_k: int
    top_k: int


# ---------------------------------------------------------------------------
# POST /retrieve
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=25, ge=5, le=50)
    include_text: bool = True

    @field_validator("question")
    @classmethod
    def strip_question(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        return stripped


class RetrievedChunkSchema(BaseModel):
    rank: int
    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str = ""
    text: str = ""
    source_type: str
    dense_score: float | None = None
    bm25_score: float | None = None
    hybrid_score: float
    rerank_score: float
    hybrid_rank: int
    rank_delta: int


class RetrieveResponse(BaseModel):
    request_id: str
    question: str
    chunks: list[RetrievedChunkSchema]
    latency: LatencyMetrics
    candidate_k: int
    top_k: int
    hybrid_latency_ms: float
    rerank_latency_ms: float


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class ComponentHealth(BaseModel):
    status: str  # ok | degraded | unavailable
    detail: str = ""
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: str  # healthy | degraded | unhealthy
    app_version: str
    chroma: ComponentHealth
    llm: ComponentHealth
    embeddings: ComponentHealth
    ollama: ComponentHealth | None = None  # deprecated; mirrors llm when present


# ---------------------------------------------------------------------------
# GET /metrics
# ---------------------------------------------------------------------------


class ConfidenceDistribution(BaseModel):
    very_low: int = 0
    low: int = 0
    medium: int = 0
    high: int = 0


class MetricsResponse(BaseModel):
    total_requests: int
    total_ask_requests: int
    total_retrieve_requests: int
    error_count: int
    avg_latency_seconds: float
    avg_ask_latency_seconds: float
    avg_retrieve_latency_seconds: float
    confidence_distribution: ConfidenceDistribution
    uptime_seconds: float


class ErrorResponse(BaseModel):
    request_id: str
    error: str
    detail: str | None = None
