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
COPY llm_client.py grounded_qa.py hybrid_retriever.py reranker.py \
     qa_pipeline.py ingest.py explore_dataset.py evaluate_rag.py ./

RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

# Render scans EXPOSE and expects the process to listen on this port (often PORT=10000).
ENV PORT=10000

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://127.0.0.1:10000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
