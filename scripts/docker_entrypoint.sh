#!/bin/sh
# Render / Docker entrypoint — bind 0.0.0.0 on $PORT before ML warmup completes.
set -e

PORT="${PORT:-10000}"
HOST="${RAG_API_HOST:-0.0.0.0}"

echo "[entrypoint] Grounded RAG API starting"
echo "[entrypoint] host=${HOST} port=${PORT}"
echo "[entrypoint] RAG_CHROMA_DIR=${RAG_CHROMA_DIR:-/app/chroma_db}"
echo "[entrypoint] RAG_CHUNKS_PATH=${RAG_CHUNKS_PATH:-/app/processed_chunks.json}"
echo "[entrypoint] launching: python -m uvicorn api.main:app"

exec python -m uvicorn api.main:app --host "${HOST}" --port "${PORT}"
