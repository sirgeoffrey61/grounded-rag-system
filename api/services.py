"""
Service layer — lazy ML loading for Render / Docker fast port bind.

Heavy imports (Chroma, sentence-transformers, BM25, reranker) run only inside
``ensure_initialized()`` on the first /ask or /retrieve request.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from typing import Any

from api.config import Settings, get_settings
from api.serialization import (
    sanitize_confidence,
    sanitize_latency,
    to_python_float,
    to_python_int,
)
from api.schemas import (
    AskResponse,
    CitationSchema,
    ComponentHealth,
    ConfidenceDistribution,
    ConfidenceSchema,
    HealthResponse,
    LatencyMetrics,
    MetricsResponse,
    RetrieveResponse,
    RetrievedChunkSchema,
    SourceChunkSchema,
)

logger = logging.getLogger(__name__)


def _confidence_bucket(score: float) -> str:
    if score < 0.2:
        return "very_low"
    if score < 0.35:
        return "low"
    if score < 0.55:
        return "medium"
    return "high"


class RequestMetrics:
    """Thread-safe in-process metrics."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start_time = time.perf_counter()
        self.total_requests = 0
        self.total_ask = 0
        self.total_retrieve = 0
        self.error_count = 0
        self._latency_sum = 0.0
        self._ask_latency_sum = 0.0
        self._retrieve_latency_sum = 0.0
        self._confidence_buckets: dict[str, int] = defaultdict(int)

    def record_success(
        self,
        endpoint: str,
        latency_seconds: float,
        confidence_score: float | None = None,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self._latency_sum += latency_seconds
            if endpoint == "ask":
                self.total_ask += 1
                self._ask_latency_sum += latency_seconds
                if confidence_score is not None:
                    self._confidence_buckets[_confidence_bucket(confidence_score)] += 1
            elif endpoint == "retrieve":
                self.total_retrieve += 1
                self._retrieve_latency_sum += latency_seconds

    def record_error(self) -> None:
        with self._lock:
            self.error_count += 1

    def snapshot(self) -> MetricsResponse:
        with self._lock:
            n = self.total_requests
            ask_n = self.total_ask
            ret_n = self.total_retrieve
            return MetricsResponse(
                total_requests=n,
                total_ask_requests=ask_n,
                total_retrieve_requests=ret_n,
                error_count=self.error_count,
                avg_latency_seconds=self._latency_sum / n if n else 0.0,
                avg_ask_latency_seconds=self._ask_latency_sum / ask_n if ask_n else 0.0,
                avg_retrieve_latency_seconds=(
                    self._retrieve_latency_sum / ret_n if ret_n else 0.0
                ),
                confidence_distribution=ConfidenceDistribution(
                    **{
                        k: self._confidence_buckets.get(k, 0)
                        for k in ("very_low", "low", "medium", "high")
                    }
                ),
                uptime_seconds=time.perf_counter() - self._start_time,
            )


class RAGService:
    """Lazy singleton: ML resources load on first query, not at import time."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.metrics = RequestMetrics()
        self.initialized = False
        self._init_lock = threading.Lock()
        self._async_init_lock = asyncio.Lock()
        self._init_error: str | None = None

        self.vector_store: Any = None
        self.embeddings: Any = None
        self.bm25_index: Any = None
        self.cross_encoder: Any = None
        self.llm_client: Any = None
        self._chroma_count: int = 0

    async def ensure_initialized(self) -> None:
        """Load ML stack once (thread pool); safe to await from async routes."""
        if self.initialized:
            return
        if self._init_error:
            raise RuntimeError(
                f"RAG service failed to initialize: {self._init_error}. "
                "Check chroma_db, processed_chunks.json, and /debug/status."
            )

        async with self._async_init_lock:
            if self.initialized:
                return
            if self._init_error:
                raise RuntimeError(self._init_error)
            try:
                await asyncio.to_thread(self._initialize_sync)
            except Exception as exc:
                self._init_error = str(exc)
                logger.exception("[lazy-init] failed: %s", exc)
                raise RuntimeError(self._init_error) from exc

    def _initialize_sync(self) -> None:
        with self._init_lock:
            if self.initialized:
                return

            print("[lazy-init] starting", flush=True)
            t0 = time.perf_counter()

            print("[lazy-init] loading embeddings", flush=True)
            from qa_pipeline import load_vectorstore

            print("[lazy-init] loading chroma", flush=True)
            self.vector_store, self.embeddings = load_vectorstore(
                self.settings.chroma_dir,
                collection_name=self.settings.collection_name,
                embedding_model=self.settings.embedding_model,
            )
            try:
                self._chroma_count = self.vector_store._collection.count()
            except Exception:
                self._chroma_count = -1

            from hybrid_retriever import build_bm25_index, load_chunks

            chunks = load_chunks(self.settings.chunks_path)
            self.bm25_index = build_bm25_index(chunks)

            print("[lazy-init] loading reranker", flush=True)
            from reranker import load_cross_encoder

            self.cross_encoder = load_cross_encoder(self.settings.cross_encoder_model)

            from llm_client import LLMClient

            self.llm_client = LLMClient(
                provider=self.settings.llm_provider,
                model=self.settings.model_name,
                base_url=self.settings.groq_api_base_url,
                timeout_seconds=self.settings.llm_timeout_seconds,
            )

            llm_health = self._check_llm()
            if llm_health.status == "unavailable":
                logger.warning(
                    "LLM unavailable after init: %s (POST /ask may return 503)",
                    llm_health.detail,
                )

            self.initialized = True
            elapsed = time.perf_counter() - t0
            print(f"[lazy-init] complete ({elapsed:.2f}s)", flush=True)
            logger.info(
                "RAG service ready in %.2fs (chroma_chunks=%d)",
                elapsed,
                self._chroma_count,
            )

    def _ensure_llm_ready(self) -> None:
        health = self._check_llm()
        if health.status == "unavailable":
            raise RuntimeError(
                f"LLM unavailable ({self.settings.llm_provider}/"
                f"{self.settings.model_name}): {health.detail}"
            )

    @staticmethod
    def _source_from_hit(src: Any, hit_by_chunk: dict[str, Any], verbose: bool) -> SourceChunkSchema:
        hit = hit_by_chunk.get(src.chunk_id)
        return SourceChunkSchema(
            citation_id=to_python_int(src.citation_id) or 0,
            chunk_id=str(src.chunk_id),
            document_id=str(src.document_id),
            article_id=str(src.article_id),
            split=str(src.split),
            title=str(src.title or ""),
            text=str(hit.text) if (verbose and hit) else "",
            source_type=str(src.source_type),
            dense_score=to_python_float(src.dense_score),
            bm25_score=to_python_float(src.bm25_score),
            hybrid_score=to_python_float(src.hybrid_score) or 0.0,
            rerank_score=to_python_float(src.rerank_score) or 0.0,
            rerank_rank=to_python_int(src.rerank_rank) or 0,
            hybrid_rank=to_python_int(hit.hybrid_rank) if hit else None,
            rank_delta=to_python_int(hit.rank_delta) if hit else None,
        )

    def ask(
        self,
        question: str,
        top_k: int,
        candidate_k: int,
        verbose: bool,
        request_id: str,
    ) -> AskResponse:
        """Full grounded QA (caller must await ensure_initialized first)."""
        if not self.initialized:
            raise RuntimeError("RAG service not initialized")

        from grounded_qa import build_grounded_prompt, calculate_confidence, format_citations
        from reranker import run_rerank_pipeline

        self._ensure_llm_ready()

        total_start = time.perf_counter()
        logger.info(
            "ask start request_id=%s query_len=%d top_k=%d candidate_k=%d",
            request_id,
            len(question),
            top_k,
            candidate_k,
        )

        rerank_start = time.perf_counter()
        rerank_result = run_rerank_pipeline(
            self.vector_store,
            self.bm25_index,
            self.cross_encoder,
            question,
            candidate_k=candidate_k,
            final_k=top_k,
        )
        retrieval_rerank_s = time.perf_counter() - rerank_start

        hits = rerank_result.reranked_hits
        messages, citation_map = build_grounded_prompt(question, hits)
        gen_start = time.perf_counter()
        answer = self._generate(messages, request_id=request_id)
        generation_s = time.perf_counter() - gen_start

        citations = format_citations(answer, citation_map)
        confidence = calculate_confidence(hits)
        hit_by_chunk = {h.chunk_id: h for h in hits}

        sources = [
            self._source_from_hit(src, hit_by_chunk, verbose)
            for src in citation_map.values()
        ]

        total_s = time.perf_counter() - total_start
        conf_score = to_python_float(confidence.score) or 0.0
        self.metrics.record_success("ask", float(total_s), conf_score)

        return AskResponse(
            request_id=request_id,
            question=question,
            answer=answer,
            confidence=ConfidenceSchema(**sanitize_confidence(asdict(confidence))),
            citations=[
                CitationSchema(
                    citation_id=to_python_int(c.citation_id) or 0,
                    chunk_id=str(c.chunk_id),
                    document_id=str(c.document_id),
                    article_id=str(c.article_id),
                    split=str(c.split),
                    title=str(c.title or ""),
                )
                for c in citations
            ],
            sources=sources,
            latency=LatencyMetrics(
                **sanitize_latency(retrieval_rerank_s, generation_s, total_s)
            ),
            candidate_k=candidate_k,
            top_k=top_k,
        )

    def retrieve(
        self,
        question: str,
        top_k: int,
        candidate_k: int,
        include_text: bool,
        request_id: str,
    ) -> RetrieveResponse:
        """Retrieval + rerank only (caller must await ensure_initialized first)."""
        if not self.initialized:
            raise RuntimeError("RAG service not initialized")

        from reranker import run_rerank_pipeline

        total_start = time.perf_counter()
        rerank_result = run_rerank_pipeline(
            self.vector_store,
            self.bm25_index,
            self.cross_encoder,
            question,
            candidate_k=candidate_k,
            final_k=top_k,
        )
        total_s = time.perf_counter() - total_start
        self.metrics.record_success("retrieve", total_s)

        chunks = [
            RetrievedChunkSchema(
                rank=to_python_int(h.rerank_rank) or 0,
                chunk_id=str(h.chunk_id),
                document_id=str(h.document_id),
                article_id=str(h.article_id),
                split=str(h.split),
                title=str(h.title or ""),
                text=str(h.text) if include_text else "",
                source_type=str(h.source_type),
                dense_score=to_python_float(h.dense_score),
                bm25_score=to_python_float(h.bm25_score),
                hybrid_score=to_python_float(h.hybrid_score) or 0.0,
                rerank_score=to_python_float(h.rerank_score) or 0.0,
                hybrid_rank=to_python_int(h.hybrid_rank) or 0,
                rank_delta=to_python_int(h.rank_delta) or 0,
            )
            for h in rerank_result.reranked_hits
        ]

        return RetrieveResponse(
            request_id=request_id,
            question=question,
            chunks=chunks,
            latency=LatencyMetrics(**sanitize_latency(total_s, None, total_s)),
            candidate_k=candidate_k,
            top_k=top_k,
            hybrid_latency_ms=to_python_float(rerank_result.hybrid_latency_ms) or 0.0,
            rerank_latency_ms=to_python_float(rerank_result.rerank_latency_ms) or 0.0,
        )

    def _generate(self, messages: list[Any], request_id: str = "") -> str:
        assert self.llm_client is not None
        from llm_client import LLMClientError

        try:
            result = self.llm_client.generate_with_metadata(messages)
        except LLMClientError as exc:
            logger.exception(
                "LLM invoke failed request_id=%s provider=%s model=%s",
                request_id,
                self.settings.llm_provider,
                self.settings.model_name,
            )
            raise RuntimeError(str(exc)) from exc
        return result.text

    def get_debug_status(self) -> dict[str, Any]:
        chroma_path = str(self.settings.chroma_dir.resolve())
        chroma_exists = self.settings.chroma_dir.is_dir()
        if not self.initialized:
            return {
                "initialized": False,
                "init_error": self._init_error,
                "chroma": {
                    "path": chroma_path,
                    "path_exists": chroma_exists,
                    "detail": "ML not loaded",
                },
                "chunks_path": str(self.settings.chunks_path.resolve()),
                "chunks_path_exists": self.settings.chunks_path.is_file(),
            }

        llm = self._check_llm()
        chroma = self._check_chroma()
        embed = self._check_embeddings()
        return {
            "initialized": True,
            "chroma": {
                "status": chroma.status,
                "path": chroma_path,
                "path_exists": chroma_exists,
                "collection": self.settings.collection_name,
                "chunk_count": to_python_int(self._chroma_count),
                "detail": chroma.detail,
            },
            "embeddings": {
                "status": embed.status,
                "model": self.settings.embedding_model,
                "loaded": self.embeddings is not None,
                "detail": embed.detail,
            },
            "reranker": {
                "model": self.settings.cross_encoder_model,
                "loaded": self.cross_encoder is not None,
            },
            "llm": {
                "status": llm.status,
                "provider": self.settings.llm_provider,
                "model": self.settings.model_name,
                "detail": llm.detail,
            },
            "bm25": {
                "chunks_path": str(self.settings.chunks_path.resolve()),
                "chunks_path_exists": self.settings.chunks_path.is_file(),
                "loaded": self.bm25_index is not None,
            },
        }

    def check_health(self) -> HealthResponse:
        """Fast when ML is not loaded — no embedding probes."""
        settings = self.settings
        if not self.initialized:
            pending = ComponentHealth(
                status="unavailable",
                detail="ML not loaded (lazy init on first /ask or /retrieve)",
            )
            return HealthResponse(
                status="degraded",
                app_version=settings.app_version,
                initialized=False,
                chroma=pending,
                llm=pending,
                embeddings=pending,
                ollama=pending,
            )

        chroma_health = self._check_chroma()
        embed_health = self._check_embeddings()
        llm_health = self._check_llm()

        statuses = [chroma_health.status, embed_health.status, llm_health.status]
        if all(s == "ok" for s in statuses):
            overall = "healthy"
        elif any(s == "unavailable" for s in statuses):
            overall = "unhealthy"
        else:
            overall = "degraded"

        return HealthResponse(
            status=overall,
            app_version=settings.app_version,
            initialized=True,
            chroma=chroma_health,
            llm=llm_health,
            embeddings=embed_health,
            ollama=llm_health,
        )

    def _check_chroma(self) -> ComponentHealth:
        if not self.initialized or self.vector_store is None:
            return ComponentHealth(status="unavailable", detail="Service not initialized")
        try:
            t0 = time.perf_counter()
            count = self.vector_store._collection.count()
            ms = (time.perf_counter() - t0) * 1000
            return ComponentHealth(
                status="ok" if count > 0 else "degraded",
                detail=f"collection={self.settings.collection_name} chunks={count}",
                latency_ms=round(ms, 2),
            )
        except Exception as exc:
            return ComponentHealth(status="unavailable", detail=str(exc))

    def _check_embeddings(self) -> ComponentHealth:
        if not self.initialized or self.embeddings is None:
            return ComponentHealth(status="unavailable", detail="Embeddings not loaded")
        try:
            t0 = time.perf_counter()
            _ = self.embeddings.embed_query("health check")
            ms = (time.perf_counter() - t0) * 1000
            return ComponentHealth(
                status="ok",
                detail=f"model={self.settings.embedding_model}",
                latency_ms=round(ms, 2),
            )
        except Exception as exc:
            return ComponentHealth(status="unavailable", detail=str(exc))

    def _check_llm(self) -> ComponentHealth:
        if not self.initialized or self.llm_client is None:
            return ComponentHealth(status="unavailable", detail="LLM client not initialized")
        try:
            probe = self.llm_client.health_check()
            return ComponentHealth(
                status=probe.status,
                detail=probe.detail,
                latency_ms=probe.latency_ms,
            )
        except Exception as exc:
            return ComponentHealth(status="unavailable", detail=str(exc))


_service: RAGService | None = None
_service_lock = threading.Lock()


def get_rag_service_instance() -> RAGService:
    global _service
    with _service_lock:
        if _service is None:
            _service = RAGService()
        return _service


def reset_rag_service() -> None:
    global _service
    with _service_lock:
        _service = None
