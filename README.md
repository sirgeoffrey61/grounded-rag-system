# Grounded RAG System

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)](https://docker.com)
[![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3-F55036?logo=meta)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Enterprise-grade Retrieval-Augmented Generation** with hybrid retrieval, cross-encoder reranking, citation-grounded answers, confidence scoring, and a full evaluation pipeline — deployed via FastAPI + Docker + Streamlit.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Components](#key-components)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Docker Deployment](#docker-deployment)
- [Streamlit Frontend](#streamlit-frontend)
- [Evaluation Pipeline](#evaluation-pipeline)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)

---

## Overview

This system answers complex questions over long-form text corpora using a **grounded** approach: every answer is tied to specific retrieved passages with inline citations, a grounding score, and a calibrated confidence level.

Built on the **QuALITY v1.0.1** reading comprehension dataset (~25,000 chunks, 500-token windows), it demonstrates production-ready RAG engineering beyond basic similarity search.

**Core capabilities:**

| Capability | Implementation |
|---|---|
| Hybrid retrieval | Dense (ChromaDB/MiniLM) + sparse (BM25) |
| Cross-encoder reranking | `ms-marco-MiniLM-L-6-v2` |
| Grounded generation | Citation extraction + grounding ratio scoring |
| Confidence scoring | Multi-signal: rerank score, grounding, answer length |
| LLM backend | Groq (Llama 3.1 8B / 70B) via OpenAI-compatible API |
| REST API | FastAPI with async, observability endpoints |
| Frontend | Streamlit — citations, confidence badges, latency |
| Evaluation | Hit rate, MRR, precision, grounding, confidence |
| Deployment | Docker, docker-compose, Render.com |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User / Client                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST /ask
                ┌───────────▼────────────┐
                │   FastAPI REST API      │  ← api/main.py
                │   /ask  /retrieve       │
                │   /health  /metrics     │
                └───────────┬────────────┘
                            │
              ┌─────────────▼──────────────┐
              │       RAG Pipeline          │
              │                             │
              │  1. Hybrid Retrieval        │
              │   ├─ Dense: ChromaDB query  │ ← MiniLM embeddings
              │   └─ Sparse: BM25 index     │ ← rank_bm25
              │                             │
              │  2. Score Fusion            │
              │   └─ RRF (0.5 dense/BM25)  │
              │                             │
              │  3. Cross-Encoder Rerank    │ ← ms-marco-MiniLM
              │   └─ Top-5 from 25 cands   │
              │                             │
              │  4. Grounded Generation     │
              │   ├─ LLM (Groq Llama 3.1)  │
              │   ├─ Citation extraction    │
              │   └─ Grounding validation  │
              │                             │
              │  5. Confidence Scoring      │
              │   └─ rerank + grounding +  │
              │      answer characteristics │
              └─────────────┬──────────────┘
                            │
              ┌─────────────▼──────────────┐
              │    Streamlit Frontend       │
              │  Citations · Scores · UI    │
              └─────────────────────────────┘
```

---

## Key Components

### Hybrid Retrieval (`hybrid_retriever.py`)

> **Why hybrid?** Dense retrieval (embeddings) excels at semantic similarity but misses exact keyword matches. BM25 excels at keyword precision but misses paraphrases. Combining both — via Reciprocal Rank Fusion — consistently outperforms either alone, especially on technical and proper-noun-heavy queries.

- **Dense**: ChromaDB with `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Sparse**: `rank_bm25` over preprocessed tokens
- **Fusion**: Reciprocal Rank Fusion with equal 0.5/0.5 weighting
- Retrieves `candidate_k=25` before reranking; configurable

### Cross-Encoder Reranking (`reranker.py`)

> **Why reranking?** Bi-encoder retrieval scores query and passage independently — cheap but approximate. A cross-encoder reads the (query, passage) pair jointly, giving much higher precision at the cost of speed. Running it on 25 candidates (not 25,000) keeps latency acceptable while dramatically improving top-5 precision.

- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Scores all 25 candidates together (batched)
- Outputs `final_k=5` best passages with rank and score delta metadata

### Grounded Generation (`grounded_qa.py`)

> **Why grounding?** Raw LLM answers are unreliable — they hallucinate facts not in the retrieved context. Grounding forces the LLM to cite specific passage IDs, then validates each citation programmatically. The **grounding ratio** (cited passages / answer sentences) is a direct measure of answer faithfulness.

- Builds a numbered context from reranked passages
- Prompts the LLM to answer with inline citations `[1]`, `[2]`
- Extracts and validates citation IDs against actual retrieved chunks
- Computes grounding ratio, citation coverage, unsupported sentence detection

### Confidence Scoring

A calibrated multi-signal confidence score combining:
- Top rerank score (retrieval quality)
- Grounding ratio (answer faithfulness)
- Answer length and completeness
- Citation coverage

Output: `very_low | low | medium | high` with numeric score `[0, 1]`

### Evaluation Pipeline (`evaluate_rag.py`)

> **Why a systematic evaluation pipeline?** In enterprise AI, retrieval and generation quality must be measured, not assumed. Without metrics like hit rate, MRR, and grounding ratio computed over a held-out set, there is no objective way to compare configuration changes or catch regressions.

Metrics computed per split:
- **Retrieval**: Hit Rate @5, MRR @5, Precision @5
- **Generation**: Answer length, grounding ratio, confidence distribution
- `--skip-llm` flag runs retrieval-only evaluation (no API key needed)
- Outputs: `evaluation_results.csv`, `confidence_analysis.csv`, `reliability_report.json`

---

## Quick Start

### Prerequisites

- Python 3.12+
- [Groq API key](https://console.groq.com) (free tier available)

### 1. Clone and install

```bash
git clone https://github.com/your-username/grounded-rag-system.git
cd grounded-rag-system
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=gsk_your_real_key_here
```

### 3. Download and ingest data

```bash
# Downloads QuALITY v1.0.1 and builds ChromaDB + BM25 index
python ingest.py
# Expected: ~25,722 chunks, ~10-15 min on first run (model download + indexing)
```

### 4. Run the API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Run the frontend

```bash
streamlit run frontend/app.py
# Open: http://localhost:8501
```

---

## API Reference

Base URL: `http://localhost:8000`

### `POST /ask`

Ask a question with grounded generation.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main economic arguments in the article?",
    "top_k": 5,
    "candidate_k": 25,
    "verbose": false
  }'
```

**Response:**
```json
{
  "request_id": "uuid",
  "question": "What are the main economic arguments...",
  "answer": "The article argues that [1] economic inequality...",
  "sources": [
    {
      "citation_id": 1,
      "chunk_id": "abc123",
      "title": "Article Title",
      "rerank_score": 0.94,
      "hybrid_score": 0.81
    }
  ],
  "grounding": {
    "grounding_ratio": 0.87,
    "cited_chunks": 3,
    "total_sources": 5,
    "unsupported_sentences": 0
  },
  "confidence": {
    "level": "high",
    "score": 0.83,
    "notes": []
  },
  "latency": {
    "total_seconds": 1.42,
    "retrieval_rerank_seconds": 0.38,
    "generation_seconds": 1.04
  }
}
```

### `POST /retrieve`

Retrieval + reranking only (no LLM, no API key needed).

```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "economic inequality", "top_k": 5}'
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "app_version": "1.0.0",
  "chroma": {"status": "ok", "detail": "collection=quality_articles chunks=25722"},
  "llm":    {"status": "ok", "detail": "provider=groq model=llama-3.1-8b-instant"},
  "embeddings": {"status": "ok", "detail": "model=all-MiniLM-L6-v2"}
}
```

### `GET /metrics`

Runtime request counts, latency percentiles, confidence distribution.

### `GET /debug/status`

Full diagnostics: all paths, model load status, Chroma counts.

---

## Docker Deployment

### Build and run

```bash
# From grounded-rag-system/
# Prerequisite: run python ingest.py first to generate chroma_db/ and processed_chunks.json

docker compose up --build
```

The `chroma_db/` and `processed_chunks.json` are volume-mounted into the container — no large files in the image.

### Environment variables for Docker

Set in `.env` or as `docker compose` overrides:

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required** |
| `RAG_LLM_PROVIDER` | `groq` | `groq` or `ollama` |
| `RAG_MODEL_NAME` | `llama-3.1-8b-instant` | Groq model ID |
| `RAG_CHROMA_DIR` | `/app/chroma_db` | Path inside container |
| `RAG_CHUNKS_PATH` | `/app/processed_chunks.json` | Path inside container |
| `RAG_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | HuggingFace model |
| `RAG_CROSS_ENCODER_MODEL` | `ms-marco-MiniLM-L-6-v2` | Reranker model |

### Render.com deployment

See [`deployment.md`](deployment.md) and [`render.yaml`](render.yaml) for Render configuration with persistent disk for `chroma_db`.

---

## Streamlit Frontend

```bash
streamlit run frontend/app.py
```

Features:
- Question input with configurable `top_k` / `candidate_k` / `verbose`
- Confidence badge (color-coded: green/yellow/orange/red)
- Inline citation rendering with source titles
- Retrieval score display (dense, BM25, rerank)
- Latency breakdown (retrieval vs generation)
- Health check tab
- Metrics dashboard tab

**Screenshot placeholder:**
> _(Add `docs/screenshots/frontend.png` after first run)_

---

## Evaluation Pipeline

```bash
# Retrieval-only (no LLM / API key needed — fast)
python evaluate_rag.py --split dev --max-queries 50 --skip-llm

# Full evaluation with generation
python evaluate_rag.py --split dev --max-queries 50

# Train split
python evaluate_rag.py --split train --max-queries 200 --skip-llm
```

**Output files** (excluded from git — regenerate locally):
- `evaluation_results.csv` — per-query metrics
- `confidence_analysis.csv` — confidence distribution
- `reliability_report.json` — aggregated summary

---

## Configuration

All settings are in `api/config.py` (Pydantic `BaseSettings`). Override via `.env` or environment variables — all prefixed `RAG_`.

```env
# .env.example
GROQ_API_KEY=gsk_your_key_here
RAG_LLM_PROVIDER=groq
RAG_MODEL_NAME=llama-3.1-8b-instant
RAG_CHROMA_DIR=./chroma_db
RAG_CHUNKS_PATH=./processed_chunks.json
RAG_COLLECTION_NAME=quality_articles
RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

---

## Project Structure

```
grounded-rag-system/
│
├── api/                        # FastAPI application
│   ├── main.py                 # App factory, lifespan, middleware
│   ├── config.py               # Pydantic settings (all RAG_ env vars)
│   ├── services.py             # RAGService singleton (retrieval + generation)
│   ├── schemas.py              # Request/response Pydantic models
│   ├── dependencies.py         # FastAPI dependency injection
│   ├── serialization.py        # numpy type sanitization
│   └── routers/
│       ├── ask.py              # POST /ask
│       ├── retrieve.py         # POST /retrieve
│       ├── health.py           # GET /health
│       ├── metrics.py          # GET /metrics
│       └── debug.py            # GET /debug/status
│
├── frontend/
│   └── app.py                  # Streamlit UI
│
├── ingest.py                   # Build ChromaDB + BM25 index from dataset
├── explore_dataset.py          # Dataset download + inspection
├── hybrid_retriever.py         # Dense + BM25 fusion retrieval
├── reranker.py                 # Cross-encoder reranking
├── grounded_qa.py              # Grounded generation + citation validation
├── qa_pipeline.py              # End-to-end QA pipeline
├── evaluate_rag.py             # Evaluation: hit rate, MRR, grounding metrics
├── retrieval_experiments.py    # Ablation: dense vs BM25 vs hybrid
├── llm_client.py               # LLM abstraction (Groq / Ollama)
│
├── Dockerfile                  # Production Docker image
├── docker-compose.yml          # Local/staging compose
├── .dockerignore
├── render.yaml                 # Render.com deployment config
├── deployment.md               # Step-by-step deployment guide
│
├── requirements.txt            # All Python dependencies
├── .env.example                # Environment variable template
├── production.env.example      # Cloud deployment template
│
├── PROJECT_REPORT.md           # Technical project report
├── README.md                   # This file
└── .gitignore
```

---

## Design Decisions

**Why not just vector search?** Pure embedding similarity misses exact matches for names, acronyms, and uncommon terms. Hybrid retrieval (dense + BM25) with RRF fusion consistently improves recall by 10–20% on diverse query types.

**Why cross-encoder reranking?** Bi-encoders trade precision for speed. By applying a cross-encoder only to the top-25 candidates, we recover most of the precision gap at a small latency cost (~200ms), not the full corpus O(n) cost.

**Why grounding validation?** LLMs hallucinate. A grounding ratio below 0.6 on a given answer is a strong signal to surface "low confidence" to the user rather than presenting the answer as fact.

**Why Groq?** Fast inference (500+ tok/s), free tier, OpenAI-compatible API, no GPU required for deployment.

---

## License

MIT — see [LICENSE](LICENSE).
