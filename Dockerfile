# =============================================================================
# Phase 11 — Grounded RAG API (production Docker image)
# Build context: grounded-rag-system/ (this directory)
# =============================================================================

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="grounded-rag-api"
LABEL org.opencontainers.image.description="FastAPI grounded RAG (hybrid + rerank + citations)"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY scripts/docker_entrypoint.sh ./docker_entrypoint.sh
COPY llm_client.py grounded_qa.py hybrid_retriever.py reranker.py \
     qa_pipeline.py ingest.py explore_dataset.py evaluate_rag.py ./

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache \
    && chmod +x /app/docker_entrypoint.sh \
    && chown -R appuser:appuser /app
USER appuser

# Render sets PORT (often 10000). Entrypoint reads $PORT at runtime.
ENV PORT=10000 \
    RAG_API_HOST=0.0.0.0

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://127.0.0.1:10000/ || exit 1

ENTRYPOINT ["/app/docker_entrypoint.sh"]
