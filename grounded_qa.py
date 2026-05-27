#!/usr/bin/env python3
"""
Phase 8 — Grounded QA with explicit citations from retrieved chunk metadata.

Why citation grounding matters:
    Users and auditors must verify claims. Citations tie each statement to
    document_id / chunk_id that came from retrieval — not from model memory.

Why hallucinations happen:
    LLMs complete patterns from pretraining. Without strict context boundaries
    and post-hoc citation validation, models invent facts or source IDs.

Why source traceability is important:
    Enterprise AI requires explainability: which passage supported which claim,
    which split/document, and what retrieval channel surfaced it.

Why prioritize correctness over creativity:
    temperature=0, refusal when context is insufficient, and citations only from
    metadata we pass in — never fabricated by the model.

Pipeline:
    Query -> hybrid retrieval -> cross-encoder rerank -> grounded prompt -> LLM

Run:
    python grounded_qa.py --query "What happened during the speed validation?"

Requires: ingest.py, chroma_db/, processed_chunks.json, GROQ_API_KEY (or --use-ollama).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder

from llm_client import LLMClient, LLMClientError

from hybrid_retriever import DEFAULT_CHUNKS_PATH, build_bm25_index, load_chunks
from qa_pipeline import (
    DEFAULT_CHROMA_DIR,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    load_vectorstore,
)
from reranker import (
    CROSS_ENCODER_MODEL,
    DEFAULT_CANDIDATE_K,
    DEFAULT_FINAL_K,
    RerankedHit,
    load_cross_encoder,
    run_rerank_pipeline,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_JSON = "grounded_answers.json"
GROUNDED_LOGS_DIR = "experiment_logs/grounded_qa"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logger = logging.getLogger(__name__)

GROUNDED_SYSTEM_PROMPT = """You are a careful question-answering assistant for long-document QA.

Rules:
1. Answer ONLY using the numbered SOURCE passages below.
2. When you use information from a passage, cite it inline using its tag exactly,
   e.g. [1] or [2]. Use ONLY the source numbers provided — never invent new ones.
3. Do NOT use outside knowledge, assumptions, or guesses.
4. If the sources do not contain enough information, reply exactly:
   "I cannot answer from the provided context."
5. Stay concise and faithful to the source wording.
6. Do not cite a source unless its text supports your statement."""

GROUNDED_USER_TEMPLATE = """Numbered sources (cite as [N] only):

{sources_block}

Question: {question}

Answer (cite sources as [N] where N is listed above):"""

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class CitationSource:
    """Traceable source bound to citation index [N] — from retrieval metadata only."""

    citation_id: int
    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str
    source_type: str
    dense_score: float | None
    bm25_score: float | None
    hybrid_score: float
    rerank_score: float
    rerank_rank: int
    text_preview: str


@dataclass
class Citation:
    """A citation reference extracted from the answer and validated against sources."""

    citation_id: int
    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str


@dataclass
class ConfidenceReport:
    """Heuristic confidence from reranker scores and retrieval agreement."""

    score: float  # 0.0 - 1.0
    level: str  # low | medium | high
    mean_rerank_score: float
    top_rerank_score: float
    hybrid_source_ratio: float
    dual_channel_ratio: float
    notes: list[str] = field(default_factory=list)


@dataclass
class GroundedTimings:
    retrieval_rerank_seconds: float = 0.0
    generation_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class GroundedResponse:
    """Full grounded QA output."""

    question: str
    answer: str
    citations: list[Citation]
    sources: list[CitationSource]
    confidence: ConfidenceReport
    timings: GroundedTimings = field(default_factory=GroundedTimings)
    candidate_k: int = DEFAULT_CANDIDATE_K
    final_k: int = DEFAULT_FINAL_K


# ---------------------------------------------------------------------------
# Prompting & generation
# ---------------------------------------------------------------------------


def build_grounded_prompt(
    question: str,
    reranked_hits: list[RerankedHit],
) -> tuple[list[SystemMessage | HumanMessage], dict[int, CitationSource]]:
    """
    Build messages with numbered sources from chunk metadata.

    citation_map keys are the only valid [N] tags the model may use.
    """
    sources_block_parts: list[str] = []
    citation_map: dict[int, CitationSource] = {}

    for idx, hit in enumerate(reranked_hits, start=1):
        citation_map[idx] = CitationSource(
            citation_id=idx,
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            article_id=hit.article_id,
            split=hit.split,
            title=hit.title,
            source_type=hit.source_type,
            dense_score=hit.dense_score,
            bm25_score=hit.bm25_score,
            hybrid_score=hit.hybrid_score,
            rerank_score=hit.rerank_score,
            rerank_rank=hit.rerank_rank,
            text_preview=hit.text[:400],
        )
        header = (
            f"[SOURCE {idx}]\n"
            f"document_id: {hit.document_id}\n"
            f"chunk_id: {hit.chunk_id}\n"
            f"article_id: {hit.article_id}\n"
            f"split: {hit.split}\n"
            f"retrieval_source: {hit.source_type}\n"
            f"rerank_score: {hit.rerank_score:.4f}\n"
        )
        if hit.title:
            header += f"title: {hit.title}\n"
        sources_block_parts.append(f"{header}text:\n{hit.text}")

    sources_block = "\n\n".join(sources_block_parts) if sources_block_parts else "(No sources.)"
    user_content = GROUNDED_USER_TEMPLATE.format(
        sources_block=sources_block,
        question=question.strip(),
    )
    messages = [
        SystemMessage(content=GROUNDED_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]
    return messages, citation_map


def generate_grounded_answer(
    messages: list[SystemMessage | HumanMessage],
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    *,
    use_ollama: bool = False,
) -> str:
    """Generate answer via Groq (default) or local Ollama (--use-ollama)."""
    if use_ollama:
        from langchain_ollama import ChatOllama

        ollama_model = model or DEFAULT_OLLAMA_MODEL
        ollama_url = base_url or DEFAULT_OLLAMA_BASE_URL
        llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_url,
            temperature=temperature,
        )
        try:
            response = llm.invoke(messages)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama generation failed. Is Ollama running? Try: ollama pull {ollama_model}"
            ) from exc
        content = response.content
        return content.strip() if isinstance(content, str) else str(content).strip()

    client = LLMClient(model=model) if model else LLMClient()
    try:
        return client.generate(messages)
    except LLMClientError as exc:
        raise RuntimeError(str(exc)) from exc


def format_citations(
    answer: str,
    citation_map: dict[int, CitationSource],
) -> list[Citation]:
    """
    Extract [N] tags from the answer and map ONLY to known citation_map entries.

    Unknown tags are ignored — we never invent citations not in retrieval metadata.
    """
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in CITATION_PATTERN.finditer(answer):
        cid = int(match.group(1))
        if cid in seen or cid not in citation_map:
            if cid not in citation_map:
                logger.debug("Ignoring invalid citation tag [%d] (not in sources)", cid)
            continue
        seen.add(cid)
        src = citation_map[cid]
        citations.append(
            Citation(
                citation_id=cid,
                chunk_id=src.chunk_id,
                document_id=src.document_id,
                article_id=src.article_id,
                split=src.split,
                title=src.title,
            )
        )
    return sorted(citations, key=lambda c: c.citation_id)


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def calculate_confidence(hits: list[RerankedHit]) -> ConfidenceReport:
    """
    Estimate confidence from reranker scores and retrieval channel agreement.

    Cross-encoder logits are unbounded; we use relative spread and source
    agreement (hybrid = both dense+BM25) as proxies — not a calibrated probability.
    """
    if not hits:
        return ConfidenceReport(
            score=0.0,
            level="low",
            mean_rerank_score=0.0,
            top_rerank_score=0.0,
            hybrid_source_ratio=0.0,
            dual_channel_ratio=0.0,
            notes=["no_retrieved_chunks"],
        )

    # Native floats — numpy scalars break FastAPI JSON encoding (see api/serialization.py).
    scores = [float(h.rerank_score) for h in hits]
    top = max(scores)
    mean = sum(scores) / len(scores)
    spread = top - min(scores)

    # Normalize logits via sigmoid for comparable 0-1 scale
    sig_scores = [_sigmoid(s) for s in scores]
    sig_mean = sum(sig_scores) / len(sig_scores)

    hybrid_ratio = sum(1 for h in hits if h.source_type == "hybrid") / len(hits)
    dual_ratio = sum(
        1 for h in hits if h.dense_score is not None and h.bm25_score is not None
    ) / len(hits)

    # Weighted heuristic — tuned for observability, not calibration
    raw = (
        0.45 * sig_mean
        + 0.25 * min(1.0, spread / 3.0)  # top chunk stands out from rest
        + 0.15 * hybrid_ratio
        + 0.15 * dual_ratio
    )
    score = max(0.0, min(1.0, raw))

    notes: list[str] = []
    if top < -6.0:
        notes.append("weak_absolute_rerank_scores")
    if spread < 0.5:
        notes.append("flat_rerank_distribution")
    if hybrid_ratio < 0.2:
        notes.append("mostly_single_channel_retrieval")

    if score >= 0.55:
        level = "high"
    elif score >= 0.35:
        level = "medium"
    else:
        level = "low"

    return ConfidenceReport(
        score=float(round(score, 4)),
        level=level,
        mean_rerank_score=float(round(mean, 4)),
        top_rerank_score=float(round(top, 4)),
        hybrid_source_ratio=float(round(hybrid_ratio, 4)),
        dual_channel_ratio=float(round(dual_ratio, 4)),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Display & persistence
# ---------------------------------------------------------------------------


def display_grounded_response(response: GroundedResponse) -> None:
    """Print answer, validated citations, sources, and confidence."""
    divider = "=" * 72
    thin = "-" * 72

    print(divider)
    print("Grounded QA Response")
    print(divider)
    print(f"\nQuestion:\n  {response.question}\n")

    print(thin)
    print(f"Confidence: {response.confidence.level.upper()} ({response.confidence.score:.2f})")
    if response.confidence.notes:
        print(f"  Notes: {', '.join(response.confidence.notes)}")
    print(
        f"  Rerank top={response.confidence.top_rerank_score:.4f} | "
        f"hybrid_ratio={response.confidence.hybrid_source_ratio:.2f}"
    )

    print(f"\n{thin}")
    print("Answer")
    print(thin)
    print(f"\n{response.answer}\n")

    print(thin)
    print(f"Citations used in answer ({len(response.citations)})")
    print(thin)
    if not response.citations:
        print("  (no valid [N] citation tags in answer)")
    for cite in response.citations:
        print(
            f"  [{cite.citation_id}] document_id={cite.document_id} | "
            f"chunk_id={cite.chunk_id} | split={cite.split}"
        )
        if cite.title:
            print(f"       title: {cite.title}")

    print(f"\n{thin}")
    print(f"Source passages ({len(response.sources)})")
    print(thin)
    for src in response.sources:
        dense_s = f"{src.dense_score:.4f}" if src.dense_score is not None else "n/a"
        bm25_s = f"{src.bm25_score:.2f}" if src.bm25_score is not None else "n/a"
        print(
            f"\n  [SOURCE {src.citation_id}] rerank_rank=#{src.rerank_rank} | "
            f"source_type={src.source_type}"
        )
        print(f"      document_id={src.document_id} | chunk_id={src.chunk_id}")
        print(
            f"      dense={dense_s} | bm25={bm25_s} | hybrid={src.hybrid_score:.4f} | "
            f"rerank={src.rerank_score:.4f}"
        )
        print(f"      preview: {src.text_preview[:220]}...")

    print(f"\n{thin}")
    print("Timings")
    print(thin)
    t = response.timings
    print(f"  Retrieval+rerank: {t.retrieval_rerank_seconds:.3f}s")
    print(f"  Generation:       {t.generation_seconds:.3f}s")
    print(f"  Total:            {t.total_seconds:.3f}s")
    print(divider)


def response_to_dict(response: GroundedResponse) -> dict[str, Any]:
    """Serialize for JSON export."""
    return {
        "question": response.question,
        "answer": response.answer,
        "confidence": asdict(response.confidence),
        "timings": asdict(response.timings),
        "candidate_k": response.candidate_k,
        "final_k": response.final_k,
        "citations": [asdict(c) for c in response.citations],
        "sources": [asdict(s) for s in response.sources],
    }


def save_grounded_json(
    response: GroundedResponse,
    path: Path,
    append: bool = True,
) -> None:
    """
    Save to grounded_answers.json (list of runs).

    append=True keeps a history of queries for experiment comparison.
    """
    path = path.resolve()
    records: list[dict[str, Any]] = []
    if append and path.is_file():
        with path.open(encoding="utf-8") as handle:
            try:
                records = json.load(handle)
                if not isinstance(records, list):
                    records = [records]
            except json.JSONDecodeError:
                records = []

    payload = response_to_dict(response)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    records.append(payload)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)
    logger.info("Saved grounded answer -> %s (%d runs)", path, len(records))


def save_run_log(response: GroundedResponse, logs_dir: Path) -> None:
    """Per-run detailed log under experiment_logs/grounded_qa/."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe = re.sub(r"[^\w]+", "_", response.question[:50]).strip("_")
    path = logs_dir / f"grounded_{ts}_{safe}.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(response_to_dict(response), handle, indent=2, ensure_ascii=False)
    logger.info("Run log -> %s", path)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_grounded_pipeline(
    query: str,
    vector_store: Any,
    bm25_index: Any,
    cross_encoder: CrossEncoder,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    use_ollama: bool = False,
) -> GroundedResponse:
    """Full pipeline: hybrid + rerank + grounded generation."""
    total_start = time.perf_counter()
    timings = GroundedTimings()

    rerank_start = time.perf_counter()
    rerank_result = run_rerank_pipeline(
        vector_store,
        bm25_index,
        cross_encoder,
        query,
        candidate_k=candidate_k,
        final_k=final_k,
    )
    timings.retrieval_rerank_seconds = time.perf_counter() - rerank_start

    hits = rerank_result.reranked_hits
    messages, citation_map = build_grounded_prompt(query, hits)

    gen_start = time.perf_counter()
    answer = generate_grounded_answer(
        messages,
        model=ollama_model if use_ollama else None,
        base_url=ollama_base_url if use_ollama else None,
        use_ollama=use_ollama,
    )
    timings.generation_seconds = time.perf_counter() - gen_start
    timings.total_seconds = time.perf_counter() - total_start

    citations = format_citations(answer, citation_map)
    confidence = calculate_confidence(hits)
    sources = list(citation_map.values())

    logger.info(
        "Grounded QA done: %d sources, %d citations, confidence=%s (%.2f)",
        len(sources),
        len(citations),
        confidence.level,
        confidence.score,
    )

    return GroundedResponse(
        question=query,
        answer=answer,
        citations=citations,
        sources=sources,
        confidence=confidence,
        timings=timings,
        candidate_k=candidate_k,
        final_k=final_k,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounded RAG QA with citations (hybrid + rerank + Ollama).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("-k", "--final-k", type=int, default=DEFAULT_FINAL_K, dest="final_k")
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use local Ollama instead of Groq (requires ollama serve)",
    )
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base-url", type=str, default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_OUTPUT_JSON,
    )
    parser.add_argument("--no-append-json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    logs_dir = Path(__file__).resolve().parent / GROUNDED_LOGS_DIR

    try:
        vector_store, _ = load_vectorstore(args.chroma_dir)
        chunks = load_chunks(args.chunks_path)
        bm25_index = build_bm25_index(chunks)
        cross_encoder = load_cross_encoder(CROSS_ENCODER_MODEL)

        response = run_grounded_pipeline(
            args.query,
            vector_store,
            bm25_index,
            cross_encoder,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            use_ollama=args.use_ollama,
        )

        display_grounded_response(response)
        save_grounded_json(response, args.output, append=not args.no_append_json)
        save_run_log(response, logs_dir)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# TODO — Phase 9+ enhancements
# =============================================================================
# TODO: Inline citation highlighting — link answer spans to chunk offsets in UI.
# TODO: Conversational memory — carry forward cited sources across turns.
# TODO: Metadata-aware prompting — boost same article_id in prompt ordering.
# TODO: Adaptive confidence thresholds — abstain when confidence.level == low.
#
# =============================================================================
# How to run
# =============================================================================
# ollama pull mistral
# python grounded_qa.py --query "What happened during the speed validation?"
# =============================================================================
