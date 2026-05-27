#!/usr/bin/env python3
"""
Phase 5 — Systematic retrieval experiments for the QuALITY RAG pipeline.

Retrieval quality dominates generation quality: if the wrong passages are
retrieved, even the best LLM will hallucinate or refuse. Optimize retrieval
before scaling models.

Why chunk size affects retrieval:
    Small chunks (256) are precise but may split answers across boundaries.
    Large chunks (1024) keep more context but dilute embeddings — similarity
    scores reflect a wider, noisier bag of words. QuALITY articles are ~4k
    tokens, so chunk size trades granularity vs. semantic focus.

Why overlap matters:
    Overlap preserves continuity at boundaries. Too little overlap loses
    bridge sentences; too much overlap duplicates near-identical chunks,
    wasting top-k slots and inflating duplicate detection counts.

Why QuALITY is hard for RAG:
    Long articles, multi-sentence reasoning, and questions that need broad
    context (not a single keyword hit). Writers also phrased questions after
    reading full articles — retrieval must approximate that with fragments.

Run:
    python retrieval_experiments.py --experiment all
    python retrieval_experiments.py --experiment top_k
    python retrieval_experiments.py --experiment chunk_grid --top-k 5

Outputs:
    retrieval_results.csv
    experiment_logs/
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from ingest import EMBEDDING_MODEL, clean_text, load_documents
from qa_pipeline import COLLECTION_NAME, load_vectorstore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "quality" / "v1.0.1"
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = "retrieval_results.csv"
LOGS_DIR = "experiment_logs"

TOP_K_VALUES = [3, 5, 10, 15]
CHUNK_SIZES = [256, 512, 1024]
CHUNK_OVERLAPS = [20, 50, 100, 150]
DEFAULT_CHUNK_EXPERIMENT_K = 5

# Benchmark questions — mix of QuALITY-style queries (some answerable, some hard).
DEFAULT_BENCHMARK_QUERIES = [
    "What did Keynes teach about saving?",
    "How does Robert view Koroby?",
    "What is the main theme of the article about Belgian politics?",
    "Why was the vocabulary limited?",
    "What happened during the speed validation?",
    "Who wrote the questions about the Gutenberg story?",
    "What does the author say about literary style and sentence length?",
    "How did profanity serve social functions in the 1950s?",
]

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RetrievedHit:
    """Single retrieval hit with metrics-friendly fields."""

    rank: int
    chunk_id: str
    document_id: str
    split: str
    text: str
    similarity_score: float
    chunk_length: int


@dataclass
class QueryRetrievalMetrics:
    """Per-query retrieval statistics for one experiment configuration."""

    experiment_id: str
    experiment_type: str
    query: str
    top_k: int
    chunk_size: int | None
    chunk_overlap: int | None
    num_hits: int
    similarity_scores: list[float]
    chunk_lengths: list[int]
    duplicate_chunk_ids: int
    duplicate_text_hashes: int
    avg_similarity: float
    max_similarity: float
    min_similarity: float
    retrieval_latency_ms: float
    hits: list[RetrievedHit] = field(default_factory=list)


@dataclass
class ExperimentRun:
    """Metadata for a full experiment batch."""

    run_id: str
    started_at: str
    experiment_types: list[str]
    num_queries: int
    num_result_rows: int


# ---------------------------------------------------------------------------
# Chunking & in-memory index (for chunk-size / overlap sweeps)
# ---------------------------------------------------------------------------


def documents_from_loaded(loaded: list[Any]) -> list[Document]:
    """Convert ingest LoadedDocument rows to LangChain documents."""
    docs: list[Document] = []
    for item in loaded:
        text = clean_text(item.text)
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": item.document_id,
                    "article_id": item.article_id,
                    "split": item.split,
                    "title": item.title,
                },
            )
        )
    return docs


def chunk_documents_with_config(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split articles with a specific size/overlap for controlled comparisons."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    for doc in documents:
        doc_id = str(doc.metadata.get("document_id", "unknown"))
        split = str(doc.metadata.get("split", "unknown"))
        for idx, piece in enumerate(splitter.split_documents([doc])):
            piece.metadata["document_id"] = doc_id
            piece.metadata["split"] = split
            piece.metadata["chunk_id"] = f"{doc_id}::cs{chunk_size}_ov{chunk_overlap}_c{idx:05d}"
            piece.metadata["chunk_size"] = chunk_size
            piece.metadata["chunk_overlap"] = chunk_overlap
            chunks.append(piece)
    return chunks


def _cache_path(logs_dir: Path, chunk_size: int, chunk_overlap: int) -> Path:
    return logs_dir / "cache" / f"embeddings_{chunk_size}_{chunk_overlap}.npz"


def build_or_load_chunk_index(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
    logs_dir: Path,
    batch_size: int = 128,
) -> tuple[list[Document], np.ndarray]:
    """
    Embed all chunks for a (size, overlap) config; cache vectors to disk.

    Caching avoids re-embedding ~25k chunks on every experiment rerun.
    """
    cache = _cache_path(logs_dir, chunk_size, chunk_overlap)
    if cache.is_file():
        logger.info("Loading cached embeddings: %s", cache.name)
        data = np.load(cache, allow_pickle=True)
        cached_chunks = data["chunks"].tolist()
        vectors = data["vectors"]
        return cached_chunks, vectors

    texts = [c.page_content for c in chunks]
    vectors_list: list[list[float]] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc=f"Embed cs={chunk_size} ov={chunk_overlap}",
        leave=False,
    ):
        batch = texts[start : start + batch_size]
        vectors_list.extend(embeddings.embed_documents(batch))

    vectors = np.array(vectors_list, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, chunks=np.array(chunks, dtype=object), vectors=vectors)
    logger.info("Cached %d vectors → %s", len(chunks), cache)
    return chunks, vectors


def retrieve_from_matrix(
    query: str,
    chunks: list[Document],
    vectors: np.ndarray,
    embeddings: HuggingFaceEmbeddings,
    k: int,
) -> tuple[list[RetrievedHit], float]:
    """
    Cosine retrieval over precomputed normalized embeddings.

    Returns hits and latency in seconds.
    """
    start = time.perf_counter()
    query_vec = np.array(embeddings.embed_query(query.strip()), dtype=np.float32)
    # Vectors are L2-normalized by HuggingFaceEmbeddings → dot product = cosine sim.
    scores = vectors @ query_vec
    k_eff = min(k, len(scores))
    if k_eff == 0:
        return [], time.perf_counter() - start

    top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    hits: list[RetrievedHit] = []
    for rank, idx in enumerate(top_idx, start=1):
        doc = chunks[int(idx)]
        hits.append(
            RetrievedHit(
                rank=rank,
                chunk_id=str(doc.metadata.get("chunk_id", f"idx_{idx}")),
                document_id=str(doc.metadata.get("document_id", "unknown")),
                split=str(doc.metadata.get("split", "unknown")),
                text=doc.page_content,
                similarity_score=float(scores[idx]),
                chunk_length=len(doc.page_content),
            )
        )
    latency = time.perf_counter() - start
    return hits, latency


# ---------------------------------------------------------------------------
# Metrics & duplicate detection
# ---------------------------------------------------------------------------


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def compute_query_metrics(
    experiment_id: str,
    experiment_type: str,
    query: str,
    top_k: int,
    chunk_size: int | None,
    chunk_overlap: int | None,
    hits: list[RetrievedHit],
    latency_sec: float,
) -> QueryRetrievalMetrics:
    """Aggregate per-query retrieval metrics including duplicate detection."""
    scores = [h.similarity_score for h in hits]
    lengths = [h.chunk_length for h in hits]

    chunk_ids = [h.chunk_id for h in hits]
    text_hashes = [_text_hash(h.text) for h in hits]
    duplicate_chunk_ids = len(chunk_ids) - len(set(chunk_ids))
    duplicate_text_hashes = len(text_hashes) - len(set(text_hashes))

    return QueryRetrievalMetrics(
        experiment_id=experiment_id,
        experiment_type=experiment_type,
        query=query,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        num_hits=len(hits),
        similarity_scores=scores,
        chunk_lengths=lengths,
        duplicate_chunk_ids=duplicate_chunk_ids,
        duplicate_text_hashes=duplicate_text_hashes,
        avg_similarity=float(np.mean(scores)) if scores else 0.0,
        max_similarity=float(max(scores)) if scores else 0.0,
        min_similarity=float(min(scores)) if scores else 0.0,
        retrieval_latency_ms=latency_sec * 1000.0,
        hits=hits,
    )


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


def retrieve_from_chroma(
    vector_store: Chroma,
    query: str,
    k: int,
) -> tuple[list[RetrievedHit], float]:
    """Retrieve from persisted Chroma (ingest.py index, typically 500/50 chunks)."""
    start = time.perf_counter()
    pairs = vector_store.similarity_search_with_relevance_scores(query.strip(), k=k)
    hits = [
        RetrievedHit(
            rank=rank,
            chunk_id=str(doc.metadata.get("chunk_id", "unknown")),
            document_id=str(doc.metadata.get("document_id", "unknown")),
            split=str(doc.metadata.get("split", "unknown")),
            text=doc.page_content,
            similarity_score=float(score),
            chunk_length=len(doc.page_content),
        )
        for rank, (doc, score) in enumerate(pairs, start=1)
    ]
    return hits, time.perf_counter() - start


def run_topk_experiment(
    vector_store: Chroma,
    queries: list[str],
    k_values: list[int],
) -> list[QueryRetrievalMetrics]:
    """
    Experiment 1 — top-k tuning on the production Chroma index.

    Tests whether increasing k surfaces better context or only adds noise.
    """
    results: list[QueryRetrievalMetrics] = []
    logger.info("=== Top-k experiment (Chroma index) ===")

    for k in k_values:
        exp_id = f"top_k_k{k}"
        for query in tqdm(queries, desc=f"  k={k}", leave=False):
            hits, latency = retrieve_from_chroma(vector_store, query, k=k)
            results.append(
                compute_query_metrics(
                    experiment_id=exp_id,
                    experiment_type="top_k",
                    query=query,
                    top_k=k,
                    chunk_size=500,  # ingest default (informational)
                    chunk_overlap=50,
                    hits=hits,
                    latency_sec=latency,
                )
            )
    return results


def run_chunk_size_experiment(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    queries: list[str],
    chunk_sizes: list[int],
    overlap: int,
    top_k: int,
    logs_dir: Path,
) -> list[QueryRetrievalMetrics]:
    """Experiment 2 — vary chunk size at fixed overlap."""
    results: list[QueryRetrievalMetrics] = []
    logger.info("=== Chunk size experiment (overlap=%d) ===", overlap)

    for size in chunk_sizes:
        chunks = chunk_documents_with_config(documents, size, overlap)
        chunks, vectors = build_or_load_chunk_index(
            chunks, embeddings, size, overlap, logs_dir
        )
        exp_id = f"chunk_size_{size}_ov{overlap}"
        logger.info("  size=%d → %d chunks", size, len(chunks))

        for query in tqdm(queries, desc=f"  size={size}", leave=False):
            hits, latency = retrieve_from_matrix(
                query, chunks, vectors, embeddings, top_k
            )
            results.append(
                compute_query_metrics(
                    experiment_id=exp_id,
                    experiment_type="chunk_size",
                    query=query,
                    top_k=top_k,
                    chunk_size=size,
                    chunk_overlap=overlap,
                    hits=hits,
                    latency_sec=latency,
                )
            )
    return results


def run_chunk_overlap_experiment(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    queries: list[str],
    chunk_size: int,
    overlaps: list[int],
    top_k: int,
    logs_dir: Path,
) -> list[QueryRetrievalMetrics]:
    """Experiment 3 — vary overlap at fixed chunk size."""
    results: list[QueryRetrievalMetrics] = []
    logger.info("=== Chunk overlap experiment (size=%d) ===", chunk_size)

    for overlap in overlaps:
        if overlap >= chunk_size:
            logger.warning("Skipping overlap=%d (>= chunk_size=%d)", overlap, chunk_size)
            continue
        chunks = chunk_documents_with_config(documents, chunk_size, overlap)
        chunks, vectors = build_or_load_chunk_index(
            chunks, embeddings, chunk_size, overlap, logs_dir
        )
        exp_id = f"chunk_overlap_{overlap}_cs{chunk_size}"
        logger.info("  overlap=%d → %d chunks", overlap, len(chunks))

        for query in tqdm(queries, desc=f"  ov={overlap}", leave=False):
            hits, latency = retrieve_from_matrix(
                query, chunks, vectors, embeddings, top_k
            )
            results.append(
                compute_query_metrics(
                    experiment_id=exp_id,
                    experiment_type="chunk_overlap",
                    query=query,
                    top_k=top_k,
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    hits=hits,
                    latency_sec=latency,
                )
            )
    return results


def run_chunk_grid_experiment(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    queries: list[str],
    chunk_sizes: list[int],
    overlaps: list[int],
    top_k: int,
    logs_dir: Path,
) -> list[QueryRetrievalMetrics]:
    """Full grid of chunk size × overlap (cached embeddings per cell)."""
    results: list[QueryRetrievalMetrics] = []
    logger.info("=== Chunk grid experiment ===")

    for size in chunk_sizes:
        for overlap in overlaps:
            if overlap >= size:
                continue
            chunks = chunk_documents_with_config(documents, size, overlap)
            chunks, vectors = build_or_load_chunk_index(
                chunks, embeddings, size, overlap, logs_dir
            )
            exp_id = f"grid_cs{size}_ov{overlap}"
            logger.info("  cs=%d ov=%d → %d chunks", size, overlap, len(chunks))

            for query in tqdm(queries, desc=f"  {size}/{overlap}", leave=False):
                hits, latency = retrieve_from_matrix(
                    query, chunks, vectors, embeddings, top_k
                )
                results.append(
                    compute_query_metrics(
                        experiment_id=exp_id,
                        experiment_type="chunk_grid",
                        query=query,
                        top_k=top_k,
                        chunk_size=size,
                        chunk_overlap=overlap,
                        hits=hits,
                        latency_sec=latency,
                    )
                )
    return results


# ---------------------------------------------------------------------------
# Persistence & visualization
# ---------------------------------------------------------------------------


def metrics_to_csv_rows(metrics_list: list[QueryRetrievalMetrics]) -> Iterator[dict[str, Any]]:
    """Flatten metrics to one CSV row per retrieved hit (detail level)."""
    for m in metrics_list:
        if not m.hits:
            yield {
                "experiment_id": m.experiment_id,
                "experiment_type": m.experiment_type,
                "query": m.query,
                "top_k": m.top_k,
                "chunk_size": m.chunk_size,
                "chunk_overlap": m.chunk_overlap,
                "rank": "",
                "chunk_id": "",
                "document_id": "",
                "split": "",
                "similarity_score": "",
                "chunk_length": "",
                "avg_similarity": m.avg_similarity,
                "max_similarity": m.max_similarity,
                "min_similarity": m.min_similarity,
                "duplicate_chunk_ids": m.duplicate_chunk_ids,
                "duplicate_text_hashes": m.duplicate_text_hashes,
                "retrieval_latency_ms": round(m.retrieval_latency_ms, 3),
                "num_hits": m.num_hits,
            }
            continue

        for hit in m.hits:
            yield {
                "experiment_id": m.experiment_id,
                "experiment_type": m.experiment_type,
                "query": m.query,
                "top_k": m.top_k,
                "chunk_size": m.chunk_size,
                "chunk_overlap": m.chunk_overlap,
                "rank": hit.rank,
                "chunk_id": hit.chunk_id,
                "document_id": hit.document_id,
                "split": hit.split,
                "similarity_score": round(hit.similarity_score, 6),
                "chunk_length": hit.chunk_length,
                "avg_similarity": round(m.avg_similarity, 6),
                "max_similarity": round(m.max_similarity, 6),
                "min_similarity": round(m.min_similarity, 6),
                "duplicate_chunk_ids": m.duplicate_chunk_ids,
                "duplicate_text_hashes": m.duplicate_text_hashes,
                "retrieval_latency_ms": round(m.retrieval_latency_ms, 3),
                "num_hits": m.num_hits,
            }


def save_results_csv(metrics_list: list[QueryRetrievalMetrics], path: Path) -> None:
    """Write retrieval_results.csv."""
    fieldnames = [
        "experiment_id",
        "experiment_type",
        "query",
        "top_k",
        "chunk_size",
        "chunk_overlap",
        "rank",
        "chunk_id",
        "document_id",
        "split",
        "similarity_score",
        "chunk_length",
        "avg_similarity",
        "max_similarity",
        "min_similarity",
        "duplicate_chunk_ids",
        "duplicate_text_hashes",
        "retrieval_latency_ms",
        "num_hits",
    ]
    rows = list(metrics_to_csv_rows(metrics_list))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows → %s", len(rows), path)


def save_summary_json(
    metrics_list: list[QueryRetrievalMetrics],
    run: ExperimentRun,
    logs_dir: Path,
) -> None:
    """Save per-query summary JSON for quick diffing between experiments."""
    summaries = []
    for m in metrics_list:
        summaries.append(
            {
                **{k: v for k, v in asdict(m).items() if k != "hits"},
                "similarity_scores": m.similarity_scores,
                "chunk_lengths": m.chunk_lengths,
            }
        )
    payload = {"run": asdict(run), "summaries": summaries}
    out = logs_dir / f"summary_{run.run_id}.json"
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    logger.info("Saved experiment summary → %s", out)


def plot_retrieval_summaries(
    metrics_list: list[QueryRetrievalMetrics],
    plots_dir: Path,
) -> None:
    """
    Plot similarity and chunk-length distributions per experiment type.

    Visuals help spot low-score retrieval clusters and overlap-induced duplication.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logger.warning("matplotlib not available — skipping plots (%s)", exc)
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[float] = []
    all_lengths: list[int] = []
    exp_labels: list[str] = []
    avg_by_exp: dict[str, list[float]] = {}

    for m in metrics_list:
        all_scores.extend(m.similarity_scores)
        all_lengths.extend(m.chunk_lengths)
        exp_labels.append(m.experiment_id)
        avg_by_exp.setdefault(m.experiment_id, []).append(m.avg_similarity)

    if all_scores:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_scores, bins=30, edgecolor="black", alpha=0.75)
        ax.set_xlabel("Similarity score (higher = better)")
        ax.set_ylabel("Count (hits across all queries)")
        ax.set_title("Retrieval similarity score distribution")
        fig.tight_layout()
        path = plots_dir / "similarity_distribution.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Plot saved → %s", path)

    if all_lengths:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(all_lengths, bins=30, edgecolor="black", alpha=0.75, color="steelblue")
        ax.set_xlabel("Chunk length (characters)")
        ax.set_ylabel("Count")
        ax.set_title("Retrieved chunk length distribution")
        fig.tight_layout()
        path = plots_dir / "chunk_length_distribution.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Plot saved → %s", path)

    if avg_by_exp:
        labels = sorted(avg_by_exp.keys())
        means = [float(np.mean(avg_by_exp[label])) for label in labels]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(labels)), means, color="seagreen", alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean avg similarity per query")
        ax.set_title("Average retrieval score by experiment")
        fig.tight_layout()
        path = plots_dir / "avg_similarity_by_experiment.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        logger.info("Plot saved → %s", path)


def print_console_summary(metrics_list: list[QueryRetrievalMetrics]) -> None:
    """Readable console table grouped by experiment."""
    divider = "=" * 72
    print(divider)
    print("Retrieval Experiment Summary")
    print(divider)

    by_exp: dict[str, list[QueryRetrievalMetrics]] = {}
    for m in metrics_list:
        by_exp.setdefault(m.experiment_id, []).append(m)

    for exp_id, items in sorted(by_exp.items()):
        avg_sim = float(np.mean([m.avg_similarity for m in items]))
        avg_lat = float(np.mean([m.retrieval_latency_ms for m in items]))
        dupes = sum(m.duplicate_chunk_ids for m in items)
        print(f"\n{exp_id} ({items[0].experiment_type})")
        print(f"  Queries:           {len(items)}")
        print(f"  Mean avg score:    {avg_sim:.4f}")
        print(f"  Mean latency (ms): {avg_lat:.2f}")
        print(f"  Duplicate chunk_ids (total): {dupes}")

        worst = min(items, key=lambda x: x.avg_similarity)
        best = max(items, key=lambda x: x.avg_similarity)
        print(f"  Lowest avg score:  {worst.avg_similarity:.4f} — {worst.query[:60]}...")
        print(f"  Highest avg score: {best.avg_similarity:.4f} — {best.query[:60]}...")

    print(divider)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_queries(path: Path | None) -> list[str]:
    """Load benchmark queries from a text file (one per line) or defaults."""
    if path is None:
        return list(DEFAULT_BENCHMARK_QUERIES)
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not lines:
        raise ValueError(f"No queries found in {path}")
    return lines


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval experiments for QuALITY RAG (no LLM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        choices=["all", "top_k", "chunk_size", "chunk_overlap", "chunk_grid"],
        default="all",
        help="Which experiment suite to run",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=None,
        help="Text file with one benchmark query per line",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_CHUNK_EXPERIMENT_K,
        help="k for chunk-size / overlap experiments",
    )
    parser.add_argument(
        "--overlap-for-size-test",
        type=int,
        default=50,
        help="Fixed overlap when running chunk_size experiment",
    )
    parser.add_argument(
        "--size-for-overlap-test",
        type=int,
        default=512,
        help="Fixed chunk size when running chunk_overlap experiment",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    queries = load_queries(args.queries_file)
    logs_dir = (args.output_dir / LOGS_DIR).resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / RESULTS_CSV

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_metrics: list[QueryRetrievalMetrics] = []
    experiment_types: list[str] = []

    try:
        if args.experiment in ("all", "top_k"):
            experiment_types.append("top_k")
            vector_store, _ = load_vectorstore(args.chroma_dir)
            all_metrics.extend(run_topk_experiment(vector_store, queries, TOP_K_VALUES))

        if args.experiment in ("all", "chunk_size", "chunk_overlap", "chunk_grid"):
            experiment_types.append("chunk_reindex")
            logger.info("Loading QuALITY documents for chunk experiments...")
            loaded = load_documents(args.data_dir)
            documents = documents_from_loaded(loaded)
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

            if args.experiment in ("all", "chunk_size"):
                experiment_types.append("chunk_size")
                all_metrics.extend(
                    run_chunk_size_experiment(
                        documents,
                        embeddings,
                        queries,
                        CHUNK_SIZES,
                        args.overlap_for_size_test,
                        args.top_k,
                        logs_dir,
                    )
                )

            if args.experiment in ("all", "chunk_overlap"):
                experiment_types.append("chunk_overlap")
                all_metrics.extend(
                    run_chunk_overlap_experiment(
                        documents,
                        embeddings,
                        queries,
                        args.size_for_overlap_test,
                        CHUNK_OVERLAPS,
                        args.top_k,
                        logs_dir,
                    )
                )

            if args.experiment == "chunk_grid":
                experiment_types.append("chunk_grid")
                all_metrics.extend(
                    run_chunk_grid_experiment(
                        documents,
                        embeddings,
                        queries,
                        CHUNK_SIZES,
                        CHUNK_OVERLAPS,
                        args.top_k,
                        logs_dir,
                    )
                )

        if not all_metrics:
            logger.error("No experiments produced results.")
            return 1

        run = ExperimentRun(
            run_id=run_id,
            started_at=run_id,
            experiment_types=experiment_types,
            num_queries=len(queries),
            num_result_rows=sum(len(m.hits) or 1 for m in all_metrics),
        )

        save_results_csv(all_metrics, csv_path)
        save_summary_json(all_metrics, run, logs_dir)
        print_console_summary(all_metrics)

        if not args.skip_plots:
            plot_retrieval_summaries(all_metrics, logs_dir / "plots")

        logger.info("Experiments complete. Results: %s", csv_path)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# TODO — future retrieval improvements (not in scope for Phase 5)
# =============================================================================
# TODO: BM25 hybrid retrieval — combine sparse keyword hits with dense vectors.
# TODO: Reranking — cross-encoder rescore top-20 before passing to the LLM.
# TODO: Metadata filtering — restrict search to train/dev or by article_id/title.
#
# =============================================================================
# How to run
# =============================================================================
# pip install -r RAG_REQUIREMENTS.txt
# python retrieval_experiments.py --experiment top_k
# python retrieval_experiments.py --experiment all
# python retrieval_experiments.py --experiment chunk_grid --top-k 5
# =============================================================================
