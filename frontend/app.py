#!/usr/bin/env python3
"""
Phase 12 — Streamlit frontend for the Grounded RAG API.

Why frontend/backend separation matters:
    The UI talks only to HTTP endpoints — never imports retrieval or LLM code.
    You can swap models, scale the API, or add auth without redeploying Streamlit.

Why observability is important in enterprise AI:
    Operators need health, latency, and retrieval scores visible alongside answers
    to debug failures (wrong chunk vs. weak generation vs. dependency outage).

Why grounded citations improve trust:
    Users verify claims against document_id / chunk_id from retrieval metadata,
    not against model fluency alone.

Backend:
    uvicorn api.main:app --reload

Frontend:
    streamlit run frontend/app.py

Environment:
    RAG_UI_API_BASE_URL   default http://localhost:8000
    RAG_UI_API_TIMEOUT    default 300 (seconds; LLM + model load can be slow)
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration (environment variables)
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_TIMEOUT = 300.0

API_BASE = os.getenv("RAG_UI_API_BASE_URL", DEFAULT_API_BASE).rstrip("/")
API_TIMEOUT = float(os.getenv("RAG_UI_API_TIMEOUT", str(DEFAULT_TIMEOUT)))

CONFIDENCE_STYLES = {
    "low": {"label": "LOW", "color": "#b91c1c", "bg": "#fef2f2"},
    "medium": {"label": "MEDIUM", "color": "#b45309", "bg": "#fffbeb"},
    "high": {"label": "HIGH", "color": "#15803d", "bg": "#f0fdf4"},
}

STATUS_COLORS = {
    "ok": "#15803d",
    "healthy": "#15803d",
    "degraded": "#b45309",
    "unavailable": "#b91c1c",
    "unhealthy": "#b91c1c",
}


# ---------------------------------------------------------------------------
# API client (HTTP only — no RAG imports)
# ---------------------------------------------------------------------------


class APIClientError(Exception):
    """Raised when the FastAPI backend returns an error or is unreachable."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.request_id = request_id


def _request(
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    url = f"{API_BASE}{path}"
    try:
        response = requests.request(
            method,
            url,
            json=json_body,
            timeout=timeout or API_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.ConnectionError as exc:
        raise APIClientError(
            f"Cannot reach API at {API_BASE}. Is uvicorn running?"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise APIClientError(
            f"Request timed out after {timeout or API_TIMEOUT}s."
        ) from exc

    if response.status_code >= 400:
        detail = response.text[:800]
        request_id = response.headers.get("X-Request-ID")
        try:
            payload = response.json()
            raw_detail = payload.get("detail", detail)
            if isinstance(raw_detail, list):
                detail = "; ".join(
                    str(item.get("msg", item)) if isinstance(item, dict) else str(item)
                    for item in raw_detail
                )
            else:
                detail = str(raw_detail)
            request_id = payload.get("request_id") or request_id
        except Exception:
            pass
        raise APIClientError(
            str(detail),
            status_code=response.status_code,
            request_id=request_id,
        )

    return response.json()


def format_api_error(exc: APIClientError) -> str:
    """Human-readable error for chat UI (status, detail, request ID)."""
    parts = [f"**HTTP {exc.status_code or 'n/a'}**"]
    if exc.request_id:
        parts.append(f"Request ID: `{exc.request_id}`")
    parts.append(str(exc))
    return "\n\n".join(parts)


def fetch_health() -> dict[str, Any]:
    return _request("GET", "/health", timeout=min(API_TIMEOUT, 30.0))


def fetch_metrics() -> dict[str, Any]:
    return _request("GET", "/metrics", timeout=min(API_TIMEOUT, 30.0))


def post_ask(
    question: str,
    top_k: int,
    candidate_k: int,
    verbose: bool = True,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/ask",
        json_body={
            "question": question,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "verbose": verbose,
        },
    )


def post_retrieve(
    question: str,
    top_k: int,
    candidate_k: int,
) -> dict[str, Any]:
    return _request(
        "POST",
        "/retrieve",
        json_body={
            "question": question,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "include_text": True,
        },
    )


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def confidence_badge_html(level: str, score: float) -> str:
    key = level.lower().strip()
    style = CONFIDENCE_STYLES.get(key, CONFIDENCE_STYLES["medium"])
    return (
        f'<span style="background:{style["bg"]};color:{style["color"]};'
        f'padding:4px 12px;border-radius:6px;font-weight:600;font-size:0.9rem;">'
        f'{style["label"]} &middot; {score:.2f}</span>'
    )


def status_dot(status: str) -> str:
    color = STATUS_COLORS.get(status.lower(), "#6b7280")
    return (
        f'<span style="color:{color};font-weight:600;">'
        f"&#9679; {status.upper()}</span>"
    )


def sources_to_dataframe(sources: list[dict[str, Any]]) -> pd.DataFrame:
    if not sources:
        return pd.DataFrame()
    rows = []
    for s in sources:
        rows.append(
            {
                "rank": s.get("rerank_rank", s.get("rank")),
                "source_type": s.get("source_type"),
                "dense": s.get("dense_score"),
                "bm25": s.get("bm25_score"),
                "hybrid": s.get("hybrid_score"),
                "rerank": s.get("rerank_score"),
                "rank_delta": s.get("rank_delta"),
                "document_id": s.get("document_id"),
                "chunk_id": s.get("chunk_id"),
                "split": s.get("split"),
            }
        )
    return pd.DataFrame(rows)


def chunks_to_dataframe(chunks: list[dict[str, Any]]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "rank": c.get("rank"),
                "source_type": c.get("source_type"),
                "dense": c.get("dense_score"),
                "bm25": c.get("bm25_score"),
                "hybrid": c.get("hybrid_score"),
                "rerank": c.get("rerank_score"),
                "hybrid_rank": c.get("hybrid_rank"),
                "rank_delta": c.get("rank_delta"),
                "document_id": c.get("document_id"),
                "chunk_id": c.get("chunk_id"),
            }
            for c in chunks
        ]
    )


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_ask_response" not in st.session_state:
        st.session_state.last_ask_response = None
    if "last_retrieve_response" not in st.session_state:
        st.session_state.last_retrieve_response = None


def last_user_question() -> str | None:
    for msg in reversed(st.session_state.messages):
        if msg.get("role") == "user":
            return msg.get("content")
    return None


def render_citations(citations: list[dict[str, Any]]) -> None:
    if not citations:
        st.caption("No inline citation tags [N] in the answer.")
        return
    for cite in citations:
        with st.expander(
            f"[{cite.get('citation_id')}] {cite.get('document_id')} / {cite.get('chunk_id')}",
            expanded=False,
        ):
            st.markdown(f"**Article ID:** `{cite.get('article_id')}`")
            st.markdown(f"**Split:** `{cite.get('split')}`")
            if cite.get("title"):
                st.markdown(f"**Title:** {cite.get('title')}")


def render_source_chunks(sources: list[dict[str, Any]]) -> None:
    if not sources:
        st.info("No source passages returned.")
        return
    for src in sources:
        cid = src.get("citation_id", "?")
        title = src.get("title") or src.get("document_id", "")
        with st.expander(
            f"Source [{cid}] — {title} ({src.get('source_type', 'n/a')})",
            expanded=cid == 1,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Rerank score", f"{src.get('rerank_score', 0):.4f}")
            c2.metric("Hybrid score", f"{src.get('hybrid_score', 0):.4f}")
            c3.metric("Rank delta", src.get("rank_delta", "n/a"))
            st.markdown(
                f"`document_id={src.get('document_id')}` · "
                f"`chunk_id={src.get('chunk_id')}` · "
                f"`split={src.get('split')}`"
            )
            dense = src.get("dense_score")
            bm25 = src.get("bm25_score")
            dense_str = f"Dense: {dense:.4f}" if dense is not None else "Dense: n/a"
            bm25_str = f"BM25: {bm25:.2f}" if bm25 is not None else "BM25: n/a"
            st.caption(f"{dense_str} | {bm25_str}")
            text = src.get("text", "")
            if text:
                st.text_area("Passage", text, height=160, disabled=True, key=f"src_{cid}_{src.get('chunk_id')}")
            else:
                st.caption("Enable verbose=true on /ask to load full passage text.")


def render_latency(latency: dict[str, Any]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Retrieval + rerank", f"{latency.get('retrieval_rerank_seconds', 0):.2f}s")
    gen = latency.get("generation_seconds")
    c2.metric("Generation", f"{gen:.2f}s" if gen is not None else "n/a")
    c3.metric("Total", f"{latency.get('total_seconds', 0):.2f}s")


def render_confidence_block(confidence: dict[str, Any]) -> None:
    level = confidence.get("level", "medium")
    score = confidence.get("score", 0.0)
    st.markdown(confidence_badge_html(level, score), unsafe_allow_html=True)
    if confidence.get("notes"):
        st.caption("Notes: " + ", ".join(confidence["notes"]))


# ---------------------------------------------------------------------------
# Pages / tabs
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[int, int]:
    with st.sidebar:
        st.title("Grounded RAG")
        st.caption("Enterprise QA with citations and observability")
        st.divider()

        st.subheader("API connection")
        st.text(f"Base URL: {API_BASE}")
        st.text(f"Timeout: {API_TIMEOUT:.0f}s")

        top_k = st.slider("Top K (final)", 1, 20, 5)
        candidate_k = st.slider("Candidate K (hybrid pool)", 5, 50, 25)

        if st.button("Check API health", use_container_width=True):
            try:
                health = fetch_health()
                st.session_state.sidebar_health = health
            except APIClientError as exc:
                st.error(format_api_error(exc))

        health = st.session_state.get("sidebar_health")
        if health:
            st.markdown(
                status_dot(health.get("status", "unknown")),
                unsafe_allow_html=True,
            )

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_ask_response = None
            st.session_state.last_retrieve_response = None
            st.rerun()

        st.divider()
        st.caption("Phase 12 · API-only client")

    return top_k, candidate_k


def render_assistant_tab(top_k: int, candidate_k: int) -> None:
    st.subheader("Grounded assistant")
    st.markdown(
        "Answers are generated **only** from retrieved passages. "
        "Citations `[N]` map to source metadata from the API — never invented by the UI."
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                render_confidence_block(meta.get("confidence", {}))
                render_latency(meta.get("latency", {}))

    question = st.chat_input("Ask a question about the document corpus…")

    if st.button(
        "Retrieve only (debug)",
        help="POST /retrieve on the last asked question (no LLM)",
    ):
        debug_q = question or last_user_question()
        if not debug_q:
            st.warning("Ask a question first, or type one in the chat input.")
        else:
            with st.spinner("Retrieving and reranking…"):
                try:
                    result = post_retrieve(debug_q, top_k, candidate_k)
                    st.session_state.last_retrieve_response = result
                    st.success(f"Retrieved {len(result.get('chunks', []))} chunks.")
                except APIClientError as exc:
                    st.error(str(exc))
            st.rerun()

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("Hybrid retrieval → rerank → grounded generation…"):
            try:
                response = post_ask(question, top_k, candidate_k, verbose=True)
                st.session_state.last_ask_response = response
                answer = response.get("answer", "")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "confidence": response.get("confidence", {}),
                            "latency": response.get("latency", {}),
                            "request_id": response.get("request_id"),
                        },
                    }
                )
            except APIClientError as exc:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": format_api_error(exc),
                        "meta": None,
                    }
                )
        st.rerun()

    response = st.session_state.last_ask_response
    if not response:
        return

    st.divider()
    st.subheader("Latest response detail")

    conf = response.get("confidence", {})
    c_left, c_right = st.columns([1, 2])
    with c_left:
        st.markdown("**Confidence**")
        render_confidence_block(conf)
        with st.expander("Confidence breakdown"):
            st.json(conf)
    with c_right:
        st.markdown("**Latency**")
        render_latency(response.get("latency", {}))
        st.caption(f"Request ID: `{response.get('request_id', 'n/a')}`")

    tab_cite, tab_src, tab_ret = st.tabs(
        ["Citations", "Source passages", "Retrieval scores"]
    )

    with tab_cite:
        render_citations(response.get("citations", []))

    with tab_src:
        render_source_chunks(response.get("sources", []))

    with tab_ret:
        df = sources_to_dataframe(response.get("sources", []))
        if df.empty:
            st.info("No retrieval data.")
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("rank")[["rerank", "hybrid"]])


def render_health_tab() -> None:
    st.subheader("System health")
    if st.button("Refresh health", type="primary"):
        try:
            st.session_state.health_data = fetch_health()
        except APIClientError as exc:
            st.error(str(exc))

    health = st.session_state.get("health_data")
    if not health:
        st.info("Click **Refresh health** to poll GET /health.")
        return

    overall = health.get("status", "unknown")
    st.markdown(
        f"### API status: {status_dot(overall)}",
        unsafe_allow_html=True,
    )
    st.caption(f"Version {health.get('app_version', 'n/a')}")

    component_order: list[tuple[str, str]] = [
        ("chroma", "Chroma"),
        ("embeddings", "Embeddings"),
    ]
    llm_comp = health.get("llm") or health.get("ollama")
    if llm_comp:
        component_order.append(("llm", "LLM"))

    for key, label in component_order:
        comp = llm_comp if key == "llm" else health.get(key, {})
        with st.container(border=True):
            st.markdown(
                f"**{label}** — {status_dot(comp.get('status', 'unknown'))}",
                unsafe_allow_html=True,
            )
            st.write(comp.get("detail", ""))
            if comp.get("latency_ms") is not None:
                st.caption(f"Probe latency: {comp['latency_ms']:.1f} ms")


def render_metrics_tab() -> None:
    st.subheader("Service metrics")
    if st.button("Refresh metrics", type="primary"):
        try:
            st.session_state.metrics_data = fetch_metrics()
        except APIClientError as exc:
            st.error(str(exc))

    metrics = st.session_state.get("metrics_data")
    if not metrics:
        st.info("Click **Refresh metrics** to poll GET /metrics.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total requests", metrics.get("total_requests", 0))
    m2.metric("Errors", metrics.get("error_count", 0))
    m3.metric("Avg latency", f"{metrics.get('avg_latency_seconds', 0):.2f}s")
    m4.metric("Uptime", f"{metrics.get('uptime_seconds', 0) / 60:.1f} min")

    st.markdown("**Ask vs retrieve**")
    c1, c2 = st.columns(2)
    c1.metric("Ask requests", metrics.get("total_ask_requests", 0))
    c2.metric("Retrieve requests", metrics.get("total_retrieve_requests", 0))
    c1.metric("Avg ask latency", f"{metrics.get('avg_ask_latency_seconds', 0):.2f}s")
    c2.metric("Avg retrieve latency", f"{metrics.get('avg_retrieve_latency_seconds', 0):.2f}s")

    dist = metrics.get("confidence_distribution", {})
    if dist:
        st.markdown("**Confidence distribution (ask requests)**")
        df_dist = pd.DataFrame(
            {"bucket": list(dist.keys()), "count": list(dist.values())}
        )
        st.bar_chart(df_dist.set_index("bucket"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Grounded RAG",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    top_k, candidate_k = render_sidebar()

    st.title("Grounded RAG Assistant")
    st.caption(
        "Document-grounded answers with citations, confidence scoring, and retrieval observability."
    )

    tab_assistant, tab_health, tab_metrics = st.tabs(
        ["Assistant", "Health", "Metrics"]
    )

    with tab_assistant:
        render_assistant_tab(top_k, candidate_k)

        ret = st.session_state.last_retrieve_response
        if ret:
            with st.expander("Last retrieval-only debug run", expanded=False):
                st.caption(f"Request ID: `{ret.get('request_id')}`")
                render_latency(ret.get("latency", {}))
                df = chunks_to_dataframe(ret.get("chunks", []))
                if not df.empty:
                    st.dataframe(df, use_container_width=True, hide_index=True)
                for chunk in ret.get("chunks", []):
                    with st.expander(
                        f"Rank {chunk.get('rank')} — {chunk.get('chunk_id', '')[:40]}"
                    ):
                        st.text(chunk.get("text", "")[:2000])

    with tab_health:
        render_health_tab()

    with tab_metrics:
        render_metrics_tab()

    st.divider()
    st.caption(
        "Backend: POST /ask · POST /retrieve · GET /health · GET /metrics"
    )


# =============================================================================
# TODO — Frontend enhancements
# =============================================================================
# TODO: Authentication — login / API key header from Streamlit secrets.
# TODO: Chat history persistence — store threads in DB or local JSON.
# TODO: Streaming responses — SSE from API for token-by-token answers.
# TODO: Feedback collection — thumbs up/down per answer linked to request_id.
# TODO: User analytics — session metrics, popular queries, abstention rate charts.
#
# =============================================================================
# How to run
# =============================================================================
# Terminal 1: uvicorn api.main:app --reload
# Terminal 2: streamlit run frontend/app.py
# Optional: set RAG_UI_API_BASE_URL=http://localhost:8000
# =============================================================================

if __name__ == "__main__":
    main()
