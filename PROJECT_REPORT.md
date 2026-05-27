# Production-Grade Grounded RAG for Long-Document QA

**Project Type:** End-to-end Retrieval-Augmented Generation (RAG) system with hybrid retrieval, cross-encoder reranking, citation-grounded generation, evaluation harness, REST API, containerization, and observability UI.

**Author:** *[Your Name]*

**Repository:** `fruit-freshness-detection` (QuALITY RAG pipeline)

**Primary Technologies:** Python 3.12, LangChain, ChromaDB, sentence-transformers, rank-bm25, cross-encoder (MS MARCO MiniLM), Ollama (Mistral), FastAPI, Uvicorn, Streamlit, Docker, pandas, scikit-learn, matplotlib

---

## Executive Summary

This project implements a **production-oriented grounded RAG stack** over the **QuALITY v1.0.1** long-document multiple-choice QA corpus. The system was built in twelve engineering phases: dataset exploration, ingestion and vector indexing, dense QA baseline, retrieval experimentation, hybrid sparse–dense retrieval, cross-encoder reranking, citation-grounded answer generation, automated evaluation, FastAPI serving, Docker packaging, and a Streamlit client that talks only to HTTP APIs.

The design prioritizes **trustworthiness over fluency**: answers are constrained to retrieved passages, inline citations are validated against chunk metadata (never invented), temperature is set to zero, and the system is encouraged to abstain when context is insufficient. A parallel **reliability engineering** track measures article-level recall@k, reranker rank movement, citation validity, abstention rate, and heuristic confidence calibration.

The result is not a notebook demo but a **deployable inference surface** (`POST /ask`, `POST /retrieve`, health and metrics endpoints) with structured logging, request IDs, dependency probes, and operational fixes discovered during integration (Chroma batch upserts, numpy JSON serialization, explicit Ollama health checks).

---

## Problem Statement

Large language models excel at language but **hallucinate** when asked about facts outside their training window or when prompts lack grounding. In enterprise settings, unsupported claims create legal, compliance, and user-trust risk.

Pure semantic retrieval on long documents introduces a second failure mode: **the right article may be indexed, but the wrong chunk surfaces** because embeddings average over 500-character windows, questions use different vocabulary than the source, or evidence spans multiple paragraphs. Keyword-only retrieval misses paraphrases.

QuALITY amplifies these issues: articles are thousands of tokens, questions require multi-sentence reasoning, and writers composed questions after reading full documents—while the RAG system only sees fragments.

This project addresses: **(1)** retrieval quality, **(2)** precision-oriented reranking, **(3)** grounded generation with traceable citations, and **(4)** measurable reliability rather than anecdotal demos.

---

## Objectives

| Objective | Implementation |
|-----------|----------------|
| Trustworthy answers | Grounded prompts, citation validation, abstention phrase |
| Improve retrieval | Hybrid dense + BM25, tunable top-k, chunking experiments |
| Reduce hallucinations | Context-only rules; no citation IDs beyond retrieval metadata |
| Explainability | Per-chunk `document_id`, `chunk_id`, scores, source type, confidence report |
| Production readiness | FastAPI singleton loading, health/metrics, Docker, structured logs |
| Evaluation | `evaluate_rag.py` with CSV/JSON outputs and calibration buckets |

---

## System Architecture

### End-to-end data flow

```
QuALITY JSONL (train/dev/test)
        │
        ▼
┌───────────────────┐
│  ingest.py        │  clean → chunk (500/50) → embed (MiniLM) → Chroma
│  explore_dataset  │  + processed_chunks.json (BM25 corpus mirror)
└─────────┬─────────┘
          │
          ▼
┌───────────────────────────────────────────────────────────┐
│                    INFERENCE PATH                          │
│  User query                                                │
│     → hybrid_retriever (Chroma dense + BM25, merge 0.5/0.5)│
│     → reranker (MS MARCO cross-encoder, pool 25 → top 5)   │
│     → grounded_qa (numbered sources + Ollama mistral)      │
│     → format_citations ([N] tags validated)                │
│     → calculate_confidence (rerank + channel agreement)    │
└───────────────────────────────────────────────────────────┘
          │
          ├─ CLI: grounded_qa.py, hybrid_retriever.py, reranker.py
          ├─ API: api/ (FastAPI + RAGService singleton)
          └─ UI: frontend/app.py → HTTP only (no in-process RAG imports)
```

### Layer responsibilities

| Layer | Role |
|-------|------|
| **Ingestion** | Deterministic chunking, embedding, persisted vectors + JSON chunk audit file |
| **Retrieval** | Candidate generation; hybrid fusion increases recall before reranking |
| **Reranking** | Cross-encoder scores (query, passage) pairs; promotes semantically relevant chunks |
| **Generation** | Ollama Mistral with strict system prompt; citations required when answering |
| **Serving** | FastAPI validates I/O with Pydantic; sanitizes numeric types for JSON |
| **Evaluation** | Batch QuALITY dev queries; aggregates reliability metrics |
| **Deployment** | Docker image for API; Chroma and BM25 corpus mounted as volumes |

### API architecture

- **`api/main.py`:** App factory, lifespan (warm models at startup), CORS, request-ID middleware, global exception handler.
- **`api/services.py`:** `RAGService` singleton—Chroma, BM25 index, cross-encoder, `ChatOllama` client reused per request.
- **`api/schemas.py`:** Request/response contracts (`AskRequest`, `AskResponse`, `RetrieveResponse`, `HealthResponse`, `MetricsResponse`).
- **`api/serialization.py`:** Converts numpy scalars to native Python floats (FastAPI `jsonable_encoder` fails on `numpy.float32`).
- **Routers:** `ask`, `retrieve`, `health`, `metrics`, `debug`.

Ollama runs **outside** the Docker container by default (`host.docker.internal:11434` in Compose).

---

## Dataset Analysis

### QuALITY v1.0.1

- **Format:** JSON Lines—one record per writer–article pair with nested `questions` (multiple choice, `gold_label`).
- **Location in repo:** `data/quality/v1.0.1/` (files such as `QuALITY.v1.0.1.train`, `.dev`, `.test`; extensionless JSONL supported).
- **Exploration:** `explore_dataset.py` discovers splits, flattens questions, reports document/question counts and length statistics.

### Why long-document QA is hard for RAG

1. **Evidence fragmentation:** A correct answer may require two non-adjacent paragraphs; a single 500-character chunk may contain no decisive sentence.
2. **Vocabulary mismatch:** Questions are paraphrased relative to article wording—dense retrieval helps; BM25 helps on rare terms.
3. **Low similarity scores:** When many chunks are weak matches, reranking and hybrid fusion matter more than raw top-1 dense score.
4. **No page metadata:** QuALITY does not provide page numbers; retrieval is chunk- and article-level only.

Ingestion **deduplicates by `article_id`** (optional) so duplicate writer lines do not multiply index size; `document_id` comes from `set_unique_id`.

---

## Detailed Implementation

### Phase 1 — `explore_dataset.py`

Loads train/dev/test JSONL, normalizes HTML-stripped vs raw articles, emits `QuestionRecord` rows for analytics. Establishes file-discovery logic reused by ingest (handles QuALITY’s extensionless `.train` filenames).

### Phase 2 — `ingest.py`

**Pipeline:** load → `clean_text` → `RecursiveCharacterTextSplitter` (**500** chars, **50** overlap) → `sentence-transformers/all-MiniLM-L6-v2` → Chroma collection `quality_articles`.

**Engineering decisions:**

- **Batched Chroma upserts (500 docs/batch):** Full 25k+ single insert crashed; batching stabilized ingest.
- **Scalar metadata only:** `document_id`, `chunk_id`, `split` (and related fields in chunk JSON)—keeps Chroma filters simple.
- **`processed_chunks.json`:** Mirror of chunk text for BM25 without round-tripping Chroma on every index build.

**Typical scale after ingest:** on the order of **~25,700 chunks** and **~380 deduplicated documents** (as logged at API startup in development runs).

### Phase 4 — `qa_pipeline.py` (dense baseline)

Early end-to-end path: Chroma similarity search → context block → Ollama Mistral (`temperature=0`). Exposes `load_vectorstore`, `retrieve_chunks`, `run_pipeline`. Serves as the embedding/Chroma integration layer reused by hybrid retrieval and the API.

*Note: There is no separate `retrieve.py`; dense retrieval lives here.*

### Phase 5 — `retrieval_experiments.py`

Systematic grid over **top-k** ∈ {3,5,10,15}, **chunk sizes** ∈ {256,512,1024}, **overlaps** ∈ {20,50,100,150}. Outputs `retrieval_results.csv` and `experiment_logs/`. Finding documented in code/comments: **higher k lowers average similarity**—a useful diagnostic for score calibration, not raw accuracy.

### Phase 6 — `hybrid_retriever.py`

- **Dense:** LangChain Chroma similarity with scores.
- **Sparse:** `BM25Okapi` over tokenized `processed_chunks.json`.
- **Fusion:** Normalize scores per channel; merge with **DENSE_WEIGHT = 0.5**, **BM25_WEIGHT = 0.5**; dedupe by `chunk_id`; tag `source_type` as `dense`, `bm25`, or `hybrid`.
- **CLI:** Single-query mode and `--compare` benchmark exporting `hybrid_comparison.csv`.

### Phase 7 — `reranker.py`

- Model: **`cross-encoder/ms-marco-MiniLM-L-6-v2`**
- Default **candidate_k = 25** (hybrid pool), **final_k = 5**
- Records `hybrid_rank`, `rerank_rank`, `rank_delta` for observability
- Outputs `reranked_results.csv`, logs under `experiment_logs/reranking/`

### Phase 8 — `grounded_qa.py`

- Builds numbered `[SOURCE N]` blocks with `document_id`, `chunk_id`, `article_id`, `split`, retrieval channel, rerank score
- **`format_citations`:** Regex `[N]`; ignores unknown IDs (anti-hallucination for references)
- **`calculate_confidence`:** Sigmoid-normalized rerank scores, spread, hybrid/dual-channel ratios → `low` / `medium` / `high`
- Persists `grounded_answers.json`, per-run logs under `experiment_logs/grounded_qa/`

### Phase 9 — `evaluate_rag.py`

Samples QuALITY **dev** questions (`load_benchmark_queries`), runs hybrid + rerank + optional grounded QA, computes:

- Article-level **recall@k** (gold `article_id` in top-k chunks—standard proxy without span labels)
- Reranking movement (median rank delta, promotions)
- Grounding: abstention, unsupported answers (non-abstention with zero valid citations), invalid citation tags
- Confidence buckets + Brier score vs recall (when sample size sufficient)

Outputs: `evaluation_results.csv`, `reliability_report.json`, `confidence_analysis.csv`, plots under `experiment_logs/evaluation/`.

**Sample result (3 dev queries, retrieval-only run, `reliability_report.json`):** mean recall@5 and @final_k = **1.0**; mean hybrid score ≈ **0.84**; median rank delta **8.0**; confidence mean ≈ **0.53** (medium bucket). Full grounding metrics require runs without `--skip-llm`.

### Phase 10 — FastAPI (`api/`)

| Endpoint | Purpose |
|----------|---------|
| `POST /ask` | Full grounded QA |
| `POST /retrieve` | Hybrid + rerank only |
| `GET /health` | Chroma, embeddings, Ollama probes |
| `GET /metrics` | In-process counters, latency averages, confidence histogram |
| `GET /debug/status` | Extended diagnostics (paths, model loaded flags, collection count) |

Startup loads all models once (~20–60s first boot). Request middleware adds **`X-Request-ID`** and latency logging.

### Phase 11 — Docker

- **`Dockerfile`:** `python:3.12-slim`, installs `RAG_REQUIREMENTS.txt` (excludes `tf-keras` in image), non-root user, healthcheck on `/health`
- **`docker-compose.yml`:** Mounts `./chroma_db`, `./processed_chunks.json`, HF cache volume; `RAG_OLLAMA_BASE_URL=http://host.docker.internal:11434`
- **`.env.example`:** `RAG_*` settings documented

### Phase 12 — `frontend/app.py`

Streamlit UI: chat-style assistant, citation/source expanders, confidence badges, retrieval score tables, Health and Metrics tabs. Communicates **only** via `requests` to the API (`RAG_UI_API_BASE_URL`, `RAG_UI_API_TIMEOUT`). Improved error display: HTTP status, backend `detail`, request ID.

---

## Retrieval Engineering

### Dense retrieval (Chroma + MiniLM)

**Strengths:** Paraphrase tolerance; good for conceptual questions.  
**Weaknesses:** Long chunks dilute vectors; uniformly low scores on hard queries.

### BM25 (rank-bm25)

Built from the same chunk inventory as Chroma. **Strengths:** Exact token overlap, rare entities. **Weaknesses:** No semantic generalization.

### Hybrid fusion

Parallel top-k from each channel (up to ~2k unique candidates before cut), score normalization, weighted sum, provenance in `source_type`. Tradeoff: tuning weights (currently 50/50) shifts precision/recall balance; no learned fusion in this codebase.

### Cross-encoder reranking

Re-scores up to 25 candidates with a transformer trained on MS MARCO passage ranking. **Effect:** Chunks that are query-relevant but poorly ranked by hybrid score can move into the top 5—observed **rank_delta** values in evaluation (e.g., median 8 on a 3-query sample). Cost: ~400ms CPU rerank latency per query in logged runs.

### Semantic drift and query formulation

Benchmark queries like *"What happened during the speed validation?"* may retrieve lexically similar but **wrong-article** chunks (e.g., “validation” in unrelated contexts). The system correctly **abstained** in grounded runs when evidence was insufficient—demonstrating grounding policy over creative completion.

### Top-k and chunking experiments

`retrieval_experiments.py` documents tradeoffs:

- Smaller chunks (256): sharper matches, boundary splits
- Larger chunks (1024): more context, noisier embeddings
- Overlap: reduces edge-cutting at cost of redundancy in top-k slots

Production ingest uses **500/50** as the balanced default baked into `chroma_db`.

---

## Grounding and Reliability

### Citation validation

Citations are **only** `[N]` tags present in the model output that match `citation_map` built from reranked hits. Unknown tags are dropped server-side— the API never fabricates bibliographic metadata.

### Confidence scoring

Heuristic (not calibrated probability): sigmoid of rerank logits, score spread, fraction of `hybrid` source types, dual-channel (dense + BM25) presence. Levels: **low** (&lt;0.35), **medium**, **high** (≥0.55). Intended for **gating and dashboards**, not automated legal decisions without further calibration.

### Abstention

System prompt mandates exact phrase *"I cannot answer from the provided context."* when evidence is missing. Evaluation tracks **abstention_rate** and **unsupported_answer_rate** (answered without valid citations).

### Hallucination reduction mechanisms

1. Retrieved-only context in prompt  
2. Temperature 0  
3. Post-hoc citation validation  
4. Refusal instruction  
5. Evaluation pipeline to measure violations  

---

## Evaluation

| Metric | Definition in this codebase |
|--------|----------------------------|
| **recall@k** | Fraction of queries where any top-k chunk’s `article_id` matches QuALITY gold article |
| **rerank movement** | `hybrid_rank - rerank_rank`; promotions into top-k |
| **citation validity** | Valid `[N]` / all tags; invalid tag rate |
| **abstention rate** | Heuristic phrase match on answer |
| **confidence calibration** | Bucketed confidence vs recall (Brier when n sufficient) |

**Limitation:** Article-level recall does not measure multiple-choice accuracy against `gold_label`; RAGAS / human eval are listed as TODOs in `evaluate_rag.py`.

---

## Backend and Deployment

### Operational model

- **Singleton `RAGService`:** One embedding model, one Chroma client, one BM25 index, one cross-encoder, one Ollama client per process.
- **Health:** 503 when overall unhealthy; probes Chroma count, embedding embed test, Ollama `/api/tags` + model name match.
- **Metrics:** Thread-safe counters—total/ask/retrieve requests, errors, mean latencies, confidence bucket counts.

### Environment variables (`RAG_` prefix)

Examples: `RAG_CHROMA_DIR`, `RAG_CHUNKS_PATH`, `RAG_OLLAMA_BASE_URL`, `RAG_OLLAMA_MODEL`, `RAG_LOG_LEVEL`, `RAG_API_PORT`.

### Run commands

```bash
# Ingest (once)
python ingest.py --data-dir data/quality/v1.0.1

# API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# UI
streamlit run frontend/app.py

# Evaluation
python evaluate_rag.py --split dev --max-queries 25
python evaluate_rag.py --max-queries 10 --skip-llm

# Docker
docker compose up --build
```

---

## Challenges Faced

| Challenge | Resolution |
|-----------|------------|
| **Chroma single bulk insert crash** | Batched upserts (500); simplified metadata |
| **QuALITY filename discovery** | Custom `_is_jsonl_file()` for `.train` suffix |
| **Windows console Unicode** | ASCII-only in CLI `print` (no →, ∩) |
| **Ollama not on PATH** | Document `%LOCALAPPDATA%\Programs\Ollama\ollama.exe` |
| **TensorFlow / transformers conflict** | Optional `tf-keras` in requirements on TF hosts |
| **Generic HTTP 500 on `/ask`** | Root cause: `numpy.float32` in JSON response; fixed via `api/serialization.py` and `float()` at source |
| **`CitationSource` has `text_preview` not `text`** | API used wrong field; fixed to use reranked hit text when `verbose=True` |
| **Hard QuALITY queries** | Low confidence and abstention—expected grounded behavior |

---

## Key Learnings

1. **Retrieval dominates the error budget**—no reranker or prompt fixes wrong-corpus retrieval.  
2. **Hybrid retrieval is cheap insurance** against sparse/dense failure modes on long-form text.  
3. **Production ML is mostly engineering:** batching, serialization, health checks, request IDs, and clear 503s for Ollama outages.  
4. **Grounding is a system property**, not a prompt trick—validation, abstention, and metrics must align.  
5. **Scores are not probabilities**—cross-encoder logits and hybrid scores need explicit calibration before user-facing trust labels.

---

## Future Improvements

Documented TODOs in code (not implemented):

- RAGAS / human evaluation (`evaluate_rag.py`)
- API authentication, rate limiting, streaming (`api/main.py`)
- Prometheus metrics, Redis caching, async inference queues
- GPU images, Kubernetes (Helm, PVC for Chroma), distributed retrieval vs generation services
- Graph RAG, agentic RAG, query rewriting, adaptive confidence thresholds
- Inline citation highlighting, conversational memory (Streamlit / grounded_qa)

---

## Conclusion

The repository delivers a **coherent, phase-documented RAG platform** from raw QuALITY JSONL through hybrid retrieval, reranking, grounded generation with citations, quantitative evaluation, and a deployable FastAPI + Streamlit surface. Engineering effort concentrated on **reliability and observability**: singleton serving, health probes, numpy-safe JSON, volume-mounted vector stores, and explicit abstention policy.

For enterprise GenAI teams, the project demonstrates familiarity with **retrieval fusion, cross-encoder reranking, grounded generation contracts, and ML ops basics**—the minimum bar for moving RAG from demo to diagnosable service.

---

## Appendix

### A. Repository layout (principal artifacts)

```
fruit-freshness-detection/
├── data/quality/v1.0.1/          # QuALITY JSONL (not always in git)
├── chroma_db/                     # Persisted vectors (generated)
├── processed_chunks.json           # BM25 + audit corpus (generated)
├── explore_dataset.py              # Phase 1
├── ingest.py                       # Phase 2
├── qa_pipeline.py                  # Phase 4 — dense QA + vectorstore helpers
├── retrieval_experiments.py        # Phase 5
├── hybrid_retriever.py             # Phase 6
├── reranker.py                     # Phase 7
├── grounded_qa.py                  # Phase 8
├── evaluate_rag.py                 # Phase 9
├── api/                            # Phase 10 — FastAPI
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   ├── services.py
│   ├── serialization.py
│   ├── dependencies.py
│   └── routers/ (ask, retrieve, health, metrics, debug)
├── frontend/app.py                 # Phase 12 — Streamlit
├── Dockerfile                      # Phase 11
├── docker-compose.yml
├── .env.example
├── RAG_REQUIREMENTS.txt
├── grounded_answers.json           # Grounded run history (generated)
├── evaluation_results.csv          # (generated)
├── reliability_report.json         # (generated)
└── experiment_logs/                # Per-phase run logs (generated)
```

### B. Model and index configuration

| Component | Value |
|-----------|--------|
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Chroma collection | `quality_articles` |
| Chunking (production ingest) | 500 chars, 50 overlap |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Hybrid weights | 0.5 dense / 0.5 BM25 |
| Default candidate_k / top_k | 25 / 5 |
| LLM | Ollama `mistral` @ `http://localhost:11434` |

### C. API reference

| Method | Path | Body / notes |
|--------|------|----------------|
| POST | `/ask` | `{ "question", "top_k", "candidate_k", "verbose" }` |
| POST | `/retrieve` | `{ "question", "top_k", "candidate_k", "include_text" }` |
| GET | `/health` | Component status |
| GET | `/metrics` | Request stats |
| GET | `/debug/status` | Extended diagnostics |
| GET | `/docs` | OpenAPI (Swagger) |

### D. NotebookLM / audio narrative hook

*Suggested story arc for audio overview:* Start with enterprise hallucination risk on long documents → introduce QuALITY as stress test → walk through ingest and why chunk overlap matters → explain why hybrid beats single-channel retrieval → reranker as precision layer → grounded citations and abstention as trust layer → evaluation and API as production closure → close with real bugs fixed (Chroma batching, numpy JSON) as evidence of operational maturity.

---

*Report generated from implemented source in this repository. Metrics cited from `reliability_report.json` and development logs unless otherwise noted.*
