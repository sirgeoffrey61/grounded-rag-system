#!/usr/bin/env python3
"""
Phase 6 — Hybrid retrieval: dense (Chroma) + BM25 (sparse).

Why hybrid retrieval improves RAG:
    Dense vectors capture paraphrases and conceptual similarity; BM25 excels
    at exact token overlap (names, rare terms, dataset-specific phrases). QuALITY
    questions often hinge on both — hybrid retrieval raises recall so the
    correct passage appears in the candidate set before any LLM step.

Dense retrieval — strengths:
    - Robust to rephrasing ("saving behavior" vs "propensity to save").
    - Works when questions do not share words with the article.

Dense retrieval — weaknesses:
    - Long chunks dilute embeddings; similarity scores can be uniformly low.
    - Misses rare proper nouns or exact phrases absent from the query embedding.

BM25 — strengths:
    - Strong keyword overlap signal; fast; interpretable scores.
    - Good for entity-like queries and technical vocabulary in QuALITY.

BM25 — weaknesses:
    - No semantic generalization — "automobile" won't match "car".
    - Sensitive to chunk length (long chunks accumulate term frequency noise).

Why QuALITY is hard for semantic-only retrieval:
    ~4k-token articles, reasoning across paragraphs, and questions written after
    full-document reading. A single 500-char chunk may lack the evidence span,
    producing low dense scores even when the article is relevant.

Run:
    python hybrid_retriever.py --query "What happened during the speed validation?"
    python hybrid_retriever.py --compare --output hybrid_comparison.csv

Install:
    pip install rank-bm25
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from qa_pipeline import DEFAULT_CHROMA_DIR, DEFAULT_TOP_K, load_vectorstore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent / "processed_chunks.json"
DEFAULT_OUTPUT_CSV = "hybrid_comparison.csv"
HYBRID_LOGS_DIR = "experiment_logs"

# Weight for merging dense + BM25 scores (both normalized to ~[0,1] first).
DENSE_WEIGHT = 0.5
BM25_WEIGHT = 0.5

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logger = logging.getLogger(__name__)

# Reuse benchmark queries from retrieval experiments for --compare mode.
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
class ChunkRecord:
    """One chunk row from processed_chunks.json (ingest audit file)."""

    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str
    text: str
    char_length: int


@dataclass
class RetrievalHit:
    """Unified hit with dense/BM25 scores and provenance for debugging."""

    rank: int
    chunk_id: str
    document_id: str
    article_id: str
    split: str
    title: str
    text: str
    dense_score: float | None
    bm25_score: float | None
    combined_score: float
    source_type: str  # dense | bm25 | hybrid (both channels contributed)
    retrieval_mode: str  # which experiment arm produced this row


@dataclass
class RetrievalBundle:
    """Results for one query and one retrieval mode."""

    query: str
    mode: str
    top_k: int
    latency_ms: float
    hits: list[RetrievalHit] = field(default_factory=list)


@dataclass
class BM25Index:
    """In-memory BM25 index aligned with processed_chunks.json."""

    bm25: BM25Okapi
    chunks: list[ChunkRecord]
    chunk_by_id: dict[str, ChunkRecord]


# ---------------------------------------------------------------------------
# Loading & indexing
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple word tokenizer for BM25 (lowercase alphanumeric tokens)."""
    return re.findall(r"\w+", text.lower())


def load_chunks(chunks_path: Path) -> list[ChunkRecord]:
    """
    Load chunk corpus from processed_chunks.json.

    This file mirrors the Chroma index content and provides a stable BM25
    corpus without querying Chroma for every token index build.
    """
    chunks_path = chunks_path.resolve()
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunks file not found: {chunks_path}. Run ingest.py first."
        )

    with chunks_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError(f"Expected JSON array in {chunks_path}")

    chunks: list[ChunkRecord] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        chunks.append(
            ChunkRecord(
                chunk_id=str(row.get("chunk_id", "")),
                document_id=str(row.get("document_id", "unknown")),
                article_id=str(row.get("article_id", "unknown")),
                split=str(row.get("split", "unknown")),
                title=str(row.get("title", "")),
                text=text,
                char_length=int(row.get("char_length", len(text))),
            )
        )

    if not chunks:
        raise ValueError(f"No chunks loaded from {chunks_path}")

    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path.name)
    return chunks


def build_bm25_index(chunks: list[ChunkRecord]) -> BM25Index:
    """Build BM25Okapi over the chunk corpus."""
    tokenized = [_tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    chunk_by_id = {c.chunk_id: c for c in chunks}
    logger.info("BM25 index ready (%d documents)", len(chunks))
    return BM25Index(bm25=bm25, chunks=chunks, chunk_by_id=chunk_by_id)


# ---------------------------------------------------------------------------
# Retrieval channels
# ---------------------------------------------------------------------------


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1] for fair hybrid merging."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def enrich_hits_from_index(
    hits: list[RetrievalHit],
    chunk_by_id: dict[str, ChunkRecord],
) -> None:
    """Fill title/article_id from processed_chunks when Chroma metadata is sparse."""
    for hit in hits:
        chunk = chunk_by_id.get(hit.chunk_id)
        if chunk:
            hit.article_id = chunk.article_id
            if chunk.title:
                hit.title = chunk.title


def dense_retrieve(
    vector_store: Chroma,
    query: str,
    k: int,
    chunk_by_id: dict[str, ChunkRecord] | None = None,
) -> tuple[list[RetrievalHit], float]:
    """
    Dense retrieval via Chroma embedding similarity.

    Returns hits and latency in seconds.
    """
    start = time.perf_counter()
    pairs = vector_store.similarity_search_with_relevance_scores(query.strip(), k=k)

    hits: list[RetrievalHit] = []
    for rank, (doc, score) in enumerate(pairs, start=1):
        hits.append(
            RetrievalHit(
                rank=rank,
                chunk_id=str(doc.metadata.get("chunk_id", "unknown")),
                document_id=str(doc.metadata.get("document_id", "unknown")),
                article_id=str(doc.metadata.get("article_id", "unknown")),
                split=str(doc.metadata.get("split", "unknown")),
                title=str(doc.metadata.get("title", "")),
                text=doc.page_content,
                dense_score=float(score),
                bm25_score=None,
                combined_score=float(score),
                source_type="dense",
                retrieval_mode="dense",
            )
        )
    if chunk_by_id:
        enrich_hits_from_index(hits, chunk_by_id)
    return hits, time.perf_counter() - start


def bm25_retrieve(
    index: BM25Index,
    query: str,
    k: int,
) -> tuple[list[RetrievalHit], float]:
    """
    Sparse retrieval via BM25 keyword overlap.

    BM25 scores are unbounded; we keep raw scores in bm25_score and normalize
    for combined_score within the candidate pool.
    """
    start = time.perf_counter()
    query_tokens = _tokenize(query)
    if not query_tokens:
        return [], time.perf_counter() - start

    raw_scores = index.bm25.get_scores(query_tokens)
    # Top-k indices by score
    k_eff = min(k, len(raw_scores))
    top_indices = sorted(
        range(len(raw_scores)),
        key=lambda i: raw_scores[i],
        reverse=True,
    )[:k_eff]

    score_map = {
        index.chunks[i].chunk_id: float(raw_scores[i]) for i in top_indices
    }
    norm = _normalize_scores(score_map)

    hits: list[RetrievalHit] = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = index.chunks[idx]
        cid = chunk.chunk_id
        hits.append(
            RetrievalHit(
                rank=rank,
                chunk_id=cid,
                document_id=chunk.document_id,
                article_id=chunk.article_id,
                split=chunk.split,
                title=chunk.title,
                text=chunk.text,
                dense_score=None,
                bm25_score=float(raw_scores[idx]),
                combined_score=norm.get(cid, 0.0),
                source_type="bm25",
                retrieval_mode="bm25",
            )
        )
    return hits, time.perf_counter() - start


def merge_results(
    dense_hits: list[RetrievalHit],
    bm25_hits: list[RetrievalHit],
    chunk_index: dict[str, ChunkRecord],
    top_k: int,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> list[RetrievalHit]:
    """
    Merge dense and BM25 lists: deduplicate by chunk_id, fuse scores.

    Chunks found by both channels get source_type='hybrid' and a weighted
    combined score. Single-channel hits are retained to improve recall.
    """
    dense_map = {h.chunk_id: h.dense_score for h in dense_hits if h.dense_score is not None}
    bm25_map = {h.chunk_id: h.bm25_score for h in bm25_hits if h.bm25_score is not None}

    all_ids = set(dense_map) | set(bm25_map)
    dense_norm = _normalize_scores({cid: dense_map[cid] for cid in dense_map})
    bm25_norm = _normalize_scores({cid: bm25_map[cid] for cid in bm25_map})

    fused: list[tuple[str, float, str, float | None, float | None]] = []
    for cid in all_ids:
        d = dense_norm.get(cid)
        b = bm25_norm.get(cid)
        d_raw = dense_map.get(cid)
        b_raw = bm25_map.get(cid)

        if d is not None and b is not None:
            combined = dense_weight * d + bm25_weight * b
            source = "hybrid"
        elif d is not None:
            combined = d
            source = "dense"
        else:
            combined = b or 0.0
            source = "bm25"

        fused.append((cid, combined, source, d_raw, b_raw))

    fused.sort(key=lambda x: x[1], reverse=True)
    fused = fused[:top_k]

    merged: list[RetrievalHit] = []
    for rank, (cid, combined, source, d_raw, b_raw) in enumerate(fused, start=1):
        chunk = chunk_index.get(cid)
        if chunk is None:
            # Fallback to dense/bm25 hit body
            ref = next((h for h in dense_hits + bm25_hits if h.chunk_id == cid), None)
            if ref is None:
                continue
            text, doc_id, art_id, split, title = (
                ref.text,
                ref.document_id,
                ref.article_id,
                ref.split,
                ref.title,
            )
        else:
            text, doc_id, art_id, split, title = (
                chunk.text,
                chunk.document_id,
                chunk.article_id,
                chunk.split,
                chunk.title,
            )

        merged.append(
            RetrievalHit(
                rank=rank,
                chunk_id=cid,
                document_id=doc_id,
                article_id=art_id,
                split=split,
                title=title,
                text=text,
                dense_score=float(d_raw) if d_raw is not None else None,
                bm25_score=float(b_raw) if b_raw is not None else None,
                combined_score=float(combined),
                source_type=source,
                retrieval_mode="hybrid",
            )
        )
    return merged


def hybrid_retrieve(
    vector_store: Chroma,
    bm25_index: BM25Index,
    query: str,
    k: int,
    dense_weight: float = DENSE_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> tuple[list[RetrievalHit], float]:
    """
    Run dense and BM25 in parallel (conceptually), merge with deduplication.

    Fetches top-k from each channel then merges — effective pool up to 2k
    unique chunks before cutting back to k.
    """
    start = time.perf_counter()
    chunk_by_id = bm25_index.chunk_by_id
    dense_hits, _ = dense_retrieve(vector_store, query, k, chunk_by_id)
    bm25_hits, _ = bm25_retrieve(bm25_index, query, k)
    chunk_by_id = bm25_index.chunk_by_id
    merged = merge_results(
        dense_hits,
        bm25_hits,
        chunk_by_id,
        top_k=k,
        dense_weight=dense_weight,
        bm25_weight=bm25_weight,
    )
    return merged, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Display & CSV logging
# ---------------------------------------------------------------------------


def display_results(bundle: RetrievalBundle) -> None:
    """Print retrieval hits with scores and source metadata."""
    divider = "=" * 72
    thin = "-" * 72

    print(divider)
    print(f"Retrieval mode: {bundle.mode.upper()}  |  k={bundle.top_k}  |  "
          f"latency={bundle.latency_ms:.1f} ms")
    print(f"Query: {bundle.query}")
    print(thin)

    if not bundle.hits:
        print("  (no results)\n")
        return

    for hit in bundle.hits:
        preview = hit.text[:280] + ("..." if len(hit.text) > 280 else "")
        dense_s = f"{hit.dense_score:.4f}" if hit.dense_score is not None else "n/a"
        bm25_s = f"{hit.bm25_score:.2f}" if hit.bm25_score is not None else "n/a"
        print(f"\n  [{hit.rank}] source={hit.source_type} | document_id={hit.document_id}")
        if hit.title:
            print(f"      title: {hit.title}")
        print(f"      chunk_id={hit.chunk_id} | split={hit.split}")
        print(f"      dense_score={dense_s} | bm25_score={bm25_s} | combined={hit.combined_score:.4f}")
        print(f"      preview: {preview}")

    print(f"\n{divider}")


def hits_to_csv_rows(bundle: RetrievalBundle) -> list[dict[str, Any]]:
    """Flatten a RetrievalBundle into CSV rows."""
    rows: list[dict[str, Any]] = []
    for hit in bundle.hits:
        rows.append(
            {
                "query": bundle.query,
                "retrieval_mode": bundle.mode,
                "top_k": bundle.top_k,
                "latency_ms": round(bundle.latency_ms, 3),
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "article_id": hit.article_id,
                "split": hit.split,
                "title": hit.title,
                "source_type": hit.source_type,
                "dense_score": hit.dense_score if hit.dense_score is not None else "",
                "bm25_score": hit.bm25_score if hit.bm25_score is not None else "",
                "combined_score": round(hit.combined_score, 6),
                "chunk_length": len(hit.text),
                "text_preview": hit.text[:200].replace("\n", " "),
            }
        )
    return rows


def save_comparison_csv(bundles: list[RetrievalBundle], path: Path) -> None:
    """Save dense vs BM25 vs hybrid comparison to CSV."""
    fieldnames = [
        "query",
        "retrieval_mode",
        "top_k",
        "latency_ms",
        "rank",
        "chunk_id",
        "document_id",
        "article_id",
        "split",
        "title",
        "source_type",
        "dense_score",
        "bm25_score",
        "combined_score",
        "chunk_length",
        "text_preview",
    ]
    rows: list[dict[str, Any]] = []
    for bundle in bundles:
        rows.extend(hits_to_csv_rows(bundle))

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved %d rows → %s", len(rows), path)


def run_compare_experiment(
    vector_store: Chroma,
    bm25_index: BM25Index,
    queries: list[str],
    k: int,
) -> list[RetrievalBundle]:
    """Run dense, BM25, and hybrid for each query (experiment logging)."""
    bundles: list[RetrievalBundle] = []

    for query in queries:
        logger.info("Comparing retrieval for: %s", query[:80])

        dense_hits, t_dense = dense_retrieve(
            vector_store, query, k, bm25_index.chunk_by_id
        )
        bundles.append(
            RetrievalBundle(
                query=query,
                mode="dense",
                top_k=k,
                latency_ms=t_dense * 1000,
                hits=dense_hits,
            )
        )

        bm25_hits, t_bm25 = bm25_retrieve(bm25_index, query, k)
        bundles.append(
            RetrievalBundle(
                query=query,
                mode="bm25",
                top_k=k,
                latency_ms=t_bm25 * 1000,
                hits=bm25_hits,
            )
        )

        hybrid_hits, t_hybrid = hybrid_retrieve(vector_store, bm25_index, query, k)
        bundles.append(
            RetrievalBundle(
                query=query,
                mode="hybrid",
                top_k=k,
                latency_ms=t_hybrid * 1000,
                hits=hybrid_hits,
            )
        )

    return bundles


def print_compare_summary(bundles: list[RetrievalBundle]) -> None:
    """Console summary: avg scores and overlap between modes per query."""
    print("\n" + "=" * 72)
    print("Comparison summary (dense vs BM25 vs hybrid)")
    print("=" * 72)

    queries = sorted({b.query for b in bundles})
    for query in queries:
        arms = {b.mode: b for b in bundles if b.query == query}
        print(f"\nQuery: {query[:70]}...")
        for mode in ("dense", "bm25", "hybrid"):
            b = arms.get(mode)
            if not b:
                continue
            avg = sum(h.combined_score for h in b.hits) / len(b.hits) if b.hits else 0
            ids = {h.chunk_id for h in b.hits}
            print(f"  {mode:6s}  avg_score={avg:.4f}  latency={b.latency_ms:.1f}ms  "
                  f"unique_chunks={len(ids)}")

        dense_ids = {h.chunk_id for h in arms.get("dense", RetrievalBundle("", "", 0, 0)).hits}
        bm25_ids = {h.chunk_id for h in arms.get("bm25", RetrievalBundle("", "", 0, 0)).hits}
        hybrid_ids = {h.chunk_id for h in arms.get("hybrid", RetrievalBundle("", "", 0, 0)).hits}
        overlap_dh = len(dense_ids & hybrid_ids)
        overlap_bh = len(bm25_ids & hybrid_ids)
        print(
            f"  overlap dense&hybrid={overlap_dh}  bm25&hybrid={overlap_bh}  "
            f"only_hybrid={len(hybrid_ids - dense_ids - bm25_ids)}"
        )

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hybrid dense + BM25 retrieval for QuALITY RAG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=None, help="Single test query")
    parser.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid"],
        default="hybrid",
        help="Retrieval mode for single-query run",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run dense, BM25, and hybrid for benchmark queries; save CSV",
    )
    parser.add_argument("-k", type=int, default=DEFAULT_TOP_K, help="Top-k per channel")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV path for --compare output",
    )
    parser.add_argument("--dense-weight", type=float, default=DENSE_WEIGHT)
    parser.add_argument("--bm25-weight", type=float, default=BM25_WEIGHT)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    global DENSE_WEIGHT, BM25_WEIGHT
    DENSE_WEIGHT = args.dense_weight
    BM25_WEIGHT = args.bm25_weight

    try:
        vector_store, _ = load_vectorstore(args.chroma_dir)
        chunks = load_chunks(args.chunks_path)
        bm25_index = build_bm25_index(chunks)

        if args.compare:
            bundles = run_compare_experiment(
                vector_store, bm25_index, DEFAULT_COMPARE_QUERIES, args.k
            )
            save_comparison_csv(bundles, args.output.resolve())
            print_compare_summary(bundles)
            for bundle in bundles:
                if args.query and bundle.query != args.query:
                    continue
            logger.info("Comparison complete → %s", args.output)
            return 0

        if not args.query:
            logger.error("Provide --query or use --compare")
            return 1

        if args.mode == "dense":
            hits, latency = dense_retrieve(
                vector_store, args.query, args.k, bm25_index.chunk_by_id
            )
        elif args.mode == "bm25":
            hits, latency = bm25_retrieve(bm25_index, args.query, args.k)
        else:
            hits, latency = hybrid_retrieve(
                vector_store,
                bm25_index,
                args.query,
                args.k,
                dense_weight=args.dense_weight,
                bm25_weight=args.bm25_weight,
            )

        bundle = RetrievalBundle(
            query=args.query,
            mode=args.mode,
            top_k=args.k,
            latency_ms=latency * 1000,
            hits=hits,
        )
        display_results(bundle)

        # Log single-query row to experiment_logs for traceability
        logs_dir = Path(__file__).resolve().parent / HYBRID_LOGS_DIR
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        single_csv = logs_dir / f"hybrid_single_{ts}.csv"
        save_comparison_csv([bundle], single_csv)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# TODO — Phase 7+ retrieval improvements
# =============================================================================
# TODO: Cross-encoder reranking — rescore hybrid pool with ms-marco-MiniLM.
# TODO: Metadata filtering — restrict to split/title/article_id before fusion.
# TODO: Query expansion — HyDE or multi-query retrieval for QuALITY paraphrases.
# TODO: Reciprocal Rank Fusion (RRF) — replace weighted linear score merge.
#
# =============================================================================
# requirements.txt
# =============================================================================
# rank-bm25>=0.2.2
#
# python hybrid_retriever.py --query "What happened during the speed validation?"
# python hybrid_retriever.py --compare --output hybrid_comparison.csv
# =============================================================================
