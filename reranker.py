#!/usr/bin/env python3
"""
Phase 7 — Cross-encoder reranking over hybrid retrieval candidates.

Retrieval vs reranking:
    Retrieval (bi-encoder / BM25) is fast: it embeds query and chunks separately
    and scores via vector distance or term overlap. Reranking (cross-encoder) is
    slow but precise: it reads query and chunk together and outputs a direct
    relevance score. Enterprise RAG almost always uses both stages.

Why cross-encoders improve precision:
    They model token-level interactions between question and passage, so
    "speed validation" in a QuALITY annotation paragraph ranks above a sci-fi
    chunk that only matches weakly in embedding space.

Latency vs quality tradeoff:
    Hybrid recall on 20–30 candidates (~100ms) + cross-encoder on pairs
    (~200–800ms on CPU) is cheaper than reranking the full 25k corpus and
    much more accurate than trusting hybrid order alone.

Why reranking is common in enterprise RAG:
    Production systems optimize recall first (hybrid), then precision (rerank),
    then generation. This stack mirrors that pattern for observability.

Pipeline:
    Query → dense + BM25 → merge candidates → cross-encoder rerank → top-k

Run:
    python reranker.py --query "What happened during the speed validation?"
    python reranker.py --compare --output reranked_results.csv

Install:
    pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentence_transformers import CrossEncoder

from hybrid_retriever import (
    DEFAULT_CHUNKS_PATH,
    BM25Index,
    RetrievalHit,
    build_bm25_index,
    dense_retrieve,
    bm25_retrieve,
    load_chunks,
    merge_results,
)
from qa_pipeline import DEFAULT_CHROMA_DIR, DEFAULT_TOP_K, load_vectorstore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_CANDIDATE_K = 25  # hybrid pool size before reranking
DEFAULT_FINAL_K = 5
DEFAULT_OUTPUT_CSV = "reranked_results.csv"
RERANK_LOGS_DIR = "experiment_logs/reranking"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logger = logging.getLogger(__name__)

DEFAULT_COMPARE_QUERIES = [
    "What did Keynes teach about saving?",
    "What happened during the speed validation?",
    "Why was the vocabulary limited?",
    "How did profanity serve social functions in the 1950s?",
    "What does the author say about literary style and sentence length?",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RerankedHit:
    """One chunk after cross-encoder scoring with rank movement metadata."""

    rerank_rank: int
    hybrid_rank: int
    rank_delta: int  # positive = moved up after reranking
    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str
    text: str
    dense_score: float | None
    bm25_score: float | None
    hybrid_score: float
    rerank_score: float
    source_type: str


@dataclass
class RerankResult:
    """Full reranking run for one query."""

    query: str
    candidate_k: int
    final_k: int
    hybrid_latency_ms: float
    rerank_latency_ms: float
    total_latency_ms: float
    hybrid_hits: list[RetrievalHit] = field(default_factory=list)
    reranked_hits: list[RerankedHit] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------


def load_cross_encoder(model_name: str = CROSS_ENCODER_MODEL) -> CrossEncoder:
    """
    Load a MS MARCO–trained cross-encoder for (query, passage) scoring.

    ms-marco-MiniLM-L-6-v2 is the standard lightweight reranker in open-source
    RAG stacks — good precision/latency balance on CPU.
    """
    logger.info("Loading cross-encoder: %s", model_name)
    try:
        model = CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load cross-encoder '{model_name}'. "
            "Install sentence-transformers and ensure network access on first run."
        ) from exc
    return model


def hybrid_candidate_pool(
    vector_store: Any,
    bm25_index: BM25Index,
    query: str,
    candidate_k: int,
) -> tuple[list[RetrievalHit], float]:
    """
    Build hybrid candidate pool (dense + BM25 merged).

    Uses a wider pool than final top-k so reranking can promote hidden gems.
    """
    start = time.perf_counter()
    chunk_by_id = bm25_index.chunk_by_id
    dense_hits, _ = dense_retrieve(vector_store, query, candidate_k, chunk_by_id)
    bm25_hits, _ = bm25_retrieve(bm25_index, query, candidate_k)
    merged = merge_results(
        dense_hits,
        bm25_hits,
        chunk_by_id,
        top_k=candidate_k,
    )
    return merged, (time.perf_counter() - start) * 1000.0


def rerank_chunks(
    model: CrossEncoder,
    query: str,
    candidates: list[RetrievalHit],
    final_k: int,
) -> tuple[list[RerankedHit], float]:
    """
    Score (query, chunk) pairs with the cross-encoder and return top final_k.

    Returns reranked hits and latency in milliseconds.
    """
    if not candidates:
        return [], 0.0

    start = time.perf_counter()
    pairs = [[query.strip(), hit.text] for hit in candidates]
    scores = model.predict(pairs, show_progress_bar=False)

    scored: list[tuple[RetrievalHit, float, int]] = []
    for hit, score in zip(candidates, scores):
        scored.append((hit, float(score), hit.rank))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:final_k]

    reranked: list[RerankedHit] = []
    for rerank_rank, (hit, rerank_score, hybrid_rank) in enumerate(top, start=1):
        reranked.append(
            RerankedHit(
                rerank_rank=rerank_rank,
                hybrid_rank=hybrid_rank,
                rank_delta=hybrid_rank - rerank_rank,
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                article_id=hit.article_id,
                split=hit.split,
                title=hit.title,
                text=hit.text,
                dense_score=hit.dense_score,
                bm25_score=hit.bm25_score,
                hybrid_score=float(hit.combined_score),
                rerank_score=float(rerank_score),
                source_type=hit.source_type,
            )
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info(
        "Reranked %d candidates -> top %d in %.1f ms",
        len(candidates),
        len(reranked),
        latency_ms,
    )
    return reranked, latency_ms


# ---------------------------------------------------------------------------
# Comparison & display
# ---------------------------------------------------------------------------


def compare_rankings(result: RerankResult) -> None:
    """Print hybrid vs reranked order with rank movement."""
    print("\n" + "-" * 72)
    print("Rank movement (hybrid -> reranked)")
    print("-" * 72)
    print(f"{'hybrid':>6}  {'rerank':>6}  {'delta':>5}  {'rerank_score':>12}  chunk_id")
    print("-" * 72)

    reranked_by_id = {h.chunk_id: h for h in result.reranked_hits}
    for hit in result.hybrid_hits[: result.final_k]:
        rr = reranked_by_id.get(hit.chunk_id)
        new_rank = rr.rerank_rank if rr else "-"
        delta = rr.rank_delta if rr else "-"
        score = f"{rr.rerank_score:.4f}" if rr else "n/a"
        marker = ""
        if isinstance(delta, int) and delta > 0:
            marker = " (up)"
        elif isinstance(delta, int) and delta < 0:
            marker = " (down)"
        print(
            f"{hit.rank:>6}  {str(new_rank):>6}  {str(delta):>5}  {score:>12}  "
            f"{hit.chunk_id[:40]}{marker}"
        )

    # Chunks that entered top-k only after reranking
    hybrid_top_ids = {h.chunk_id for h in result.hybrid_hits[: result.final_k]}
    promoted = [
        h for h in result.reranked_hits
        if h.chunk_id not in hybrid_top_ids
    ]
    if promoted:
        print("\nPromoted into top-k by reranker (not in hybrid top-k):")
        for h in promoted:
            print(f"  rerank={h.rerank_rank} hybrid_was={h.hybrid_rank} "
                  f"score={h.rerank_score:.4f} id={h.chunk_id}")


def display_reranked_results(result: RerankResult) -> None:
    """Human-readable reranked output with all score channels."""
    divider = "=" * 72
    print(divider)
    print("Cross-Encoder Reranking Results")
    print(divider)
    print(f"Query: {result.query}")
    print(
        f"Candidates: {len(result.hybrid_hits)} (pool k={result.candidate_k}) -> "
        f"Final k={result.final_k}"
    )
    print(
        f"Latency: hybrid={result.hybrid_latency_ms:.1f} ms | "
        f"rerank={result.rerank_latency_ms:.1f} ms | "
        f"total={result.total_latency_ms:.1f} ms"
    )

    if not result.reranked_hits:
        print("\n(no reranked results)\n")
        return

    for hit in result.reranked_hits:
        preview = hit.text[:260] + ("..." if len(hit.text) > 260 else "")
        dense_s = f"{hit.dense_score:.4f}" if hit.dense_score is not None else "n/a"
        bm25_s = f"{hit.bm25_score:.2f}" if hit.bm25_score is not None else "n/a"
        move = f"+{hit.rank_delta}" if hit.rank_delta > 0 else str(hit.rank_delta)

        print(f"\n  [rerank #{hit.rerank_rank}] hybrid was #{hit.hybrid_rank} ({move})")
        print(f"      source={hit.source_type} | document_id={hit.document_id}")
        if hit.title:
            print(f"      title: {hit.title}")
        print(f"      chunk_id={hit.chunk_id} | split={hit.split}")
        print(
            f"      dense={dense_s} | bm25={bm25_s} | hybrid={hit.hybrid_score:.4f} | "
            f"rerank={hit.rerank_score:.4f}"
        )
        print(f"      preview: {preview}")

    compare_rankings(result)
    print(divider)


# ---------------------------------------------------------------------------
# Pipeline & persistence
# ---------------------------------------------------------------------------


def run_rerank_pipeline(
    vector_store: Any,
    bm25_index: BM25Index,
    cross_encoder: CrossEncoder,
    query: str,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
) -> RerankResult:
    """Execute full hybrid → rerank pipeline for one query."""
    total_start = time.perf_counter()

    candidates, hybrid_ms = hybrid_candidate_pool(
        vector_store, bm25_index, query, candidate_k
    )
    reranked, rerank_ms = rerank_chunks(cross_encoder, query, candidates, final_k)

    return RerankResult(
        query=query,
        candidate_k=candidate_k,
        final_k=final_k,
        hybrid_latency_ms=hybrid_ms,
        rerank_latency_ms=rerank_ms,
        total_latency_ms=(time.perf_counter() - total_start) * 1000.0,
        hybrid_hits=candidates,
        reranked_hits=reranked,
    )


def result_to_csv_rows(result: RerankResult) -> list[dict[str, Any]]:
    """Flatten rerank result for CSV (one row per reranked hit)."""
    rows: list[dict[str, Any]] = []
    for hit in result.reranked_hits:
        rows.append(
            {
                "query": result.query,
                "candidate_k": result.candidate_k,
                "final_k": result.final_k,
                "hybrid_latency_ms": round(result.hybrid_latency_ms, 2),
                "rerank_latency_ms": round(result.rerank_latency_ms, 2),
                "total_latency_ms": round(result.total_latency_ms, 2),
                "rerank_rank": hit.rerank_rank,
                "hybrid_rank": hit.hybrid_rank,
                "rank_delta": hit.rank_delta,
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "article_id": hit.article_id,
                "split": hit.split,
                "title": hit.title,
                "source_type": hit.source_type,
                "dense_score": hit.dense_score if hit.dense_score is not None else "",
                "bm25_score": hit.bm25_score if hit.bm25_score is not None else "",
                "hybrid_score": round(hit.hybrid_score, 6),
                "rerank_score": round(hit.rerank_score, 6),
                "chunk_length": len(hit.text),
                "text_preview": hit.text[:200].replace("\n", " "),
            }
        )
    return rows


def save_results_csv(results: list[RerankResult], path: Path) -> None:
    """Write reranked_results.csv."""
    fieldnames = [
        "query",
        "candidate_k",
        "final_k",
        "hybrid_latency_ms",
        "rerank_latency_ms",
        "total_latency_ms",
        "rerank_rank",
        "hybrid_rank",
        "rank_delta",
        "chunk_id",
        "document_id",
        "article_id",
        "split",
        "title",
        "source_type",
        "dense_score",
        "bm25_score",
        "hybrid_score",
        "rerank_score",
        "chunk_length",
        "text_preview",
    ]
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(result_to_csv_rows(result))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows → %s", len(rows), path)


def save_run_json(result: RerankResult, logs_dir: Path) -> None:
    """Archive full run metadata under experiment_logs/reranking/."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_q = result.query[:40].replace(" ", "_").replace("?", "")
    path = logs_dir / f"rerank_{ts}_{safe_q}.json"
    payload = {
        "query": result.query,
        "candidate_k": result.candidate_k,
        "final_k": result.final_k,
        "latencies_ms": {
            "hybrid": result.hybrid_latency_ms,
            "rerank": result.rerank_latency_ms,
            "total": result.total_latency_ms,
        },
        "reranked": [
            {
                "rerank_rank": h.rerank_rank,
                "hybrid_rank": h.hybrid_rank,
                "rank_delta": h.rank_delta,
                "chunk_id": h.chunk_id,
                "rerank_score": h.rerank_score,
                "hybrid_score": h.hybrid_score,
            }
            for h in result.reranked_hits
        ],
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Run log → %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval + cross-encoder reranking (QuALITY RAG).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run benchmark queries and save CSV",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_CANDIDATE_K,
        help="Hybrid pool size before reranking",
    )
    parser.add_argument(
        "-k",
        "--final-k",
        type=int,
        default=DEFAULT_FINAL_K,
        dest="final_k",
        help="Final chunks after reranking",
    )
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--model",
        type=str,
        default=CROSS_ENCODER_MODEL,
        help="Cross-encoder model id",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    logs_dir = Path(__file__).resolve().parent / RERANK_LOGS_DIR

    try:
        vector_store, _ = load_vectorstore(args.chroma_dir)
        chunks = load_chunks(args.chunks_path)
        bm25_index = build_bm25_index(chunks)
        cross_encoder = load_cross_encoder(args.model)

        queries = DEFAULT_COMPARE_QUERIES if args.compare else [args.query]
        if not queries or queries == [None]:
            logger.error("Provide --query or use --compare")
            return 1

        results: list[RerankResult] = []
        for query in queries:
            if not query:
                continue
            logger.info("Reranking query: %s", query[:80])
            result = run_rerank_pipeline(
                vector_store,
                bm25_index,
                cross_encoder,
                query,
                candidate_k=args.candidate_k,
                final_k=args.final_k,
            )
            results.append(result)
            if not args.compare:
                display_reranked_results(result)
                save_run_json(result, logs_dir)

        if args.compare:
            save_results_csv(results, args.output.resolve())
            for result in results:
                print(f"\n>>> {result.query[:60]}...")
                compare_rankings(result)
            logger.info("Comparison saved → %s", args.output)
        elif results:
            save_results_csv(results, Path(__file__).resolve().parent / DEFAULT_OUTPUT_CSV)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# TODO — future ranking improvements
# =============================================================================
# TODO: Reciprocal Rank Fusion (RRF) — fuse dense/BM25 ranks before rerank.
# TODO: Query expansion — multi-query candidates fed into reranker.
# TODO: Metadata-aware reranking — boost same article_id or title match.
# TODO: Adaptive top-k — widen candidate pool when hybrid scores are flat.
#
# =============================================================================
# How to run
# =============================================================================
# python reranker.py --query "What happened during the speed validation?"
# python reranker.py --compare --candidate-k 30 --final-k 5
# =============================================================================
