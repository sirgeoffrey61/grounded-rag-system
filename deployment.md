# Grounded RAG — Cloud deployment (Phase 13)

This guide deploys the **FastAPI backend** (Render / Railway / Docker) and **Streamlit frontend** (Streamlit Cloud) using **Groq** for LLM inference instead of local Ollama.

## Why not Ollama in the cloud?

- PaaS containers (Render, Railway, Streamlit Cloud) cannot run a persistent `ollama serve` daemon or reach `host.docker.internal`.
- No GPU on typical free tiers; large local models are slow or unavailable.
- **Groq** exposes an OpenAI-compatible HTTPS API — set `GROQ_API_KEY` and go.

## Architecture

```text
Streamlit Cloud (frontend/app.py)
        │  RAG_API_URL
        ▼
Render / Railway (api.main:app)
        │  hybrid + rerank + llm_client (Groq)
        ├── chroma_db/ + processed_chunks.json (volume or baked image)
        └── GROQ_API_KEY (secret)
```

## Required environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | **Yes** (for `/ask`) | Groq API key from [console.groq.com](https://console.groq.com) |
| `RAG_LLM_PROVIDER` | No | Default `groq` |
| `RAG_MODEL_NAME` | No | Default `llama-3.1-8b-instant` |
| `RAG_LLM_TIMEOUT_SECONDS` | No | Default `60` |
| `RAG_CHROMA_DIR` | No | `/app/chroma_db` in Docker |
| `RAG_CHUNKS_PATH` | No | `/app/processed_chunks.json` |
| `RAG_API_HOST` | No | `0.0.0.0` |
| `RAG_API_PORT` | No | `8000` |

Streamlit frontend:

| Variable | Description |
|----------|-------------|
| `RAG_API_URL` | Public backend URL, e.g. `https://your-api.onrender.com` |

Copy [production.env.example](production.env.example) or [.env.example](.env.example) for local dev.

## Render (backend)

1. Push repo to GitHub; connect in Render dashboard.
2. Use [render.yaml](render.yaml) or create a **Web Service**:
   - **Root directory:** `grounded-rag-system/` (Blueprint [render.yaml](render.yaml))
   - **Build:** Docker (`Dockerfile`) or `pip install -r requirements.txt`
   - **Start:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT` (Render sets `PORT`)
3. **Environment:** add `GROQ_API_KEY`, `RAG_MODEL_NAME`, paths for Chroma.
4. **Disk:** attach a persistent disk mounted at `/app/chroma_db` *or* bake `chroma_db/` into the image after running `ingest.py` locally.
5. **Health check path:** `/health` (expects 200 when Chroma + embeddings + Groq are OK).

### curl examples (Render)

Replace `BASE` with your service URL.

```bash
# Health (503 if Groq key missing or Chroma empty)
curl -sS "$BASE/health" | jq .

# Retrieval only (no LLM)
curl -sS -X POST "$BASE/retrieve" \
  -H "Content-Type: application/json" \
  -d '{"question": "Why was the vocabulary limited?", "top_k": 5}' | jq .

# Grounded ask (uses Groq)
curl -sS -X POST "$BASE/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Why was the vocabulary limited?", "top_k": 5, "candidate_k": 25}' | jq .

# Debug LLM status
curl -sS "$BASE/debug/status" | jq .llm
```

## Railway

Similar to Render: Dockerfile deploy, set `GROQ_API_KEY`, mount volume for `chroma_db`, health check on `/health`.

## Streamlit Cloud (frontend)

1. App path: `frontend/app.py`
2. Secrets / env: `RAG_API_URL=https://your-api.onrender.com`
3. Requirements: include `streamlit`, `requests` (see `requirements.txt`).

## Local Docker

```bash
cp .env.example .env
# Set GROQ_API_KEY=gsk_...
docker compose up --build
curl -sS http://localhost:8000/health | jq .
```

## Startup commands

| Target | Command |
|--------|---------|
| Local | `uvicorn api.main:app --reload --host 0.0.0.0 --port 8000` |
| Docker | `uvicorn api.main:app --host 0.0.0.0 --port 8000` |
| Render | `uvicorn api.main:app --host 0.0.0.0 --port $PORT` |

First request after cold start may take 1–3 minutes while embedding and cross-encoder models load.

## Debugging

| Symptom | Check |
|---------|--------|
| `/health` → `llm.unavailable` | `GROQ_API_KEY` set? Valid key? |
| `/ask` → 503 | `curl $BASE/debug/status` → `llm` block |
| Empty answers | Groq rate limits; increase `RAG_LLM_TIMEOUT_SECONDS` |
| Chroma degraded | Run `python ingest.py`; mount `chroma_db` |
| Frontend cannot reach API | `RAG_API_URL`, CORS (API allows `*` in dev) |

Logs: set `RAG_LOG_LEVEL=DEBUG` for LLM latency and token usage lines from `llm_client`.

## Provider abstraction

All generation goes through `llm_client.py` so retrieval, reranking, citations, and confidence code stay unchanged when swapping providers.
