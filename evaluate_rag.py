#!/usr/bin/env python3
"""
Phase 9 — RAG evaluation and reliability engineering.

Why RAG evaluation is difficult:
    End-to-end quality depends on retrieval, reranking, and generation. Each stage
    has different failure modes; a single BLEU/ROUGE score on answers hides whether
    errors came from wrong chunks or hallucinated text.

Why confidence calibration matters:
    Uncalibrated scores cause false trust or unnecessary abstention. Bucketing
    predicted confidence against observed outcomes (recall, abstention) surfaces
    when to escalate to humans or widen retrieval.

Why grounded abstention is valuable:
    Refusing when context is insufficient beats confident wrong answers. Measuring
    abstention rate vs. confidence validates that the system knows when it does not know.

Why enterprise AI requires measurable reliability:
    Production systems need SLOs: recall@k, citation validity, unsupported-answer
    rate, and calibration drift — not demo anecdotes.

Pipeline per benchmark query:
    QuALITY question -> hybrid retrieval -> rerank -> grounded QA (optional)
    -> retrieval / reranking / grounding / confidence metrics

Run:
    python evaluate_rag.py --split dev --max-queries 25
    python evaluate_rag.py --max-queries 10 --skip-llm   # retrieval + rerank only

Outputs:
    evaluation_results.csv
    reliability_report.json
    confidence_analysis.csv
    experiment_logs/evaluation/
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from explore_dataset import DEFAULT_DATA_DIR, QuestionRecord, load_dataset
from grounded_qa import (
    CITATION_PATTERN,
    build_grounded_prompt,
    calculate_confidence,
    format_citations,
    generate_grounded_answer,
)
from hybrid_retriever import (
    DEFAULT_CHUNKS_PATH,
    RetrievalHit,
    build_bm25_index,
    hybrid_retrieve,
    load_chunks,
)
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
    RerankResult,
    load_cross_encoder,
    run_rerank_pipeline,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_CSV = "evaluation_results.csv"
RELIABILITY_JSON = "reliability_report.json"
CONFIDENCE_CSV = "confidence_analysis.csv"
EVAL_LOGS_DIR = "experiment_logs/evaluation"
PLOTS_SUBDIR = "plots"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"
logger = logging.getLogger(__name__)

ABSTENTION_MARKERS = (
    "i cannot answer from the provided context",
    "cannot answer from the provided context",
    "do not contain enough information",
    "sources do not contain",
    "not contain enough information",
)

CONFIDENCE_BUCKETS = [
    (0.0, 0.2, "very_low"),
    (0.2, 0.35, "low"),
    (0.35, 0.55, "medium"),
    (0.55, 1.01, "high"),
]

DEFAULT_EVAL_K = 5


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkCase:
    """One QuALITY question used as an evaluation query."""

    query_id: str
    question: str
    article_id: str
    split: str
    gold_label: int
    title: str = ""


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    recall_at_final_k: float
    mean_hybrid_score: float
    mean_dense_score: float | None
    mean_bm25_score: float | None
    hybrid_hit_count: int
    gold_in_hybrid_pool: bool


@dataclass
class RerankingMetrics:
    median_rank_delta: float
    mean_rank_delta: float
    promoted_count: int
    demoted_count: int
    unchanged_count: int
    max_rank_gain: int
    mean_rerank_score: float


@dataclass
class GroundingMetrics:
    abstained: bool
    unsupported_answer: bool
    citation_count: int
    invalid_citation_tags: int
    citation_validity_rate: float
    answer_length: int


@dataclass
class ConfidenceMetrics:
    score: float
    level: str
    bucket: str
    brier_contribution: float | None = None


@dataclass
class QueryEvaluation:
    """Per-query evaluation record."""

    query_id: str
    question: str
    article_id: str
    split: str
    gold_label: int
    retrieval: RetrievalMetrics
    reranking: RerankingMetrics
    grounding: GroundingMetrics
    confidence: ConfidenceMetrics
    retrieval_recall_binary: int
    latency_retrieval_ms: float
    latency_rerank_ms: float
    latency_generation_s: float | None
    skipped_generation: bool = False


# ---------------------------------------------------------------------------
# Benchmark loading
# ---------------------------------------------------------------------------


def load_benchmark_queries(
    data_dir: Path,
    split: str = "dev",
    max_queries: int | None = 50,
    seed: int = 42,
    difficult_only: bool = False,
) -> list[BenchmarkCase]:
    """
    Load QuALITY questions with gold labels for evaluation.

    Uses article_id as retrieval ground truth: success = at least one top-k chunk
    from the same article (standard article-level RAG recall proxy).
    """
    _, questions = load_dataset(data_dir)
    filtered = [q for q in questions if q.split == split and q.question.strip()]
    if difficult_only:
        filtered = [q for q in filtered if q.difficult == 1]

    if not filtered:
        raise ValueError(
            f"No questions for split={split!r}. Check --data-dir and --split."
        )

    rng = random.Random(seed)
    if max_queries is not None and len(filtered) > max_queries:
        filtered = rng.sample(filtered, max_queries)

    cases: list[BenchmarkCase] = []
    for idx, q in enumerate(filtered):
        qid = f"{q.split}_{q.article_id}_{idx}"
        cases.append(
            BenchmarkCase(
                query_id=qid,
                question=q.question.strip(),
                article_id=q.article_id,
                split=q.split,
                gold_label=q.gold_label,
                title=q.title,
            )
        )
    logger.info("Loaded %d benchmark queries (split=%s)", len(cases), split)
    return cases


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _article_recall(hits: list[Any], gold_article_id: str, k: int) -> float:
    if k <= 0 or not hits:
        return 0.0
    top = hits[:k]
    return 1.0 if any(h.article_id == gold_article_id for h in top) else 0.0


def _mean_optional(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def evaluate_retrieval(
    hybrid_hits: list[RetrievalHit],
    reranked_hits: list[Any],
    gold_article_id: str,
    eval_k: int = DEFAULT_EVAL_K,
) -> RetrievalMetrics:
    """Article-level recall@k and hybrid score summaries."""
    recall_k = _article_recall(hybrid_hits, gold_article_id, eval_k)
    recall_final = _article_recall(reranked_hits, gold_article_id, len(reranked_hits))

    hybrid_scores = [h.combined_score for h in hybrid_hits[:eval_k]]
    dense_vals = [h.dense_score for h in hybrid_hits[:eval_k]]
    bm25_vals = [h.bm25_score for h in hybrid_hits[:eval_k]]

    return RetrievalMetrics(
        recall_at_k=recall_k,
        recall_at_final_k=recall_final,
        mean_hybrid_score=float(np.mean(hybrid_scores)) if hybrid_scores else 0.0,
        mean_dense_score=_mean_optional(dense_vals),
        mean_bm25_score=_mean_optional(bm25_vals),
        hybrid_hit_count=len(hybrid_hits),
        gold_in_hybrid_pool=any(h.article_id == gold_article_id for h in hybrid_hits),
    )


def evaluate_reranking(result: RerankResult) -> RerankingMetrics:
    """Rank movement and promotion stats after cross-encoder reranking."""
    deltas = [h.rank_delta for h in result.reranked_hits]
    if not deltas:
        return RerankingMetrics(
            median_rank_delta=0.0,
            mean_rank_delta=0.0,
            promoted_count=0,
            demoted_count=0,
            unchanged_count=0,
            max_rank_gain=0,
            mean_rerank_score=0.0,
        )

    hybrid_top_ids = {h.chunk_id for h in result.hybrid_hits[: result.final_k]}
    promoted = sum(
        1 for h in result.reranked_hits if h.chunk_id not in hybrid_top_ids
    )

    return RerankingMetrics(
        median_rank_delta=float(np.median(deltas)),
        mean_rank_delta=float(np.mean(deltas)),
        promoted_count=promoted,
        demoted_count=sum(1 for d in deltas if d < 0),
        unchanged_count=sum(1 for d in deltas if d == 0),
        max_rank_gain=int(max(deltas)),
        mean_rerank_score=float(
            np.mean([h.rerank_score for h in result.reranked_hits])
        ),
    )


def _is_abstention(answer: str) -> bool:
    lower = answer.strip().lower()
    return any(marker in lower for marker in ABSTENTION_MARKERS)


def evaluate_grounding(
    answer: str,
    citation_map: dict[int, Any],
) -> GroundingMetrics:
    """
    Grounding reliability: abstention, unsupported answers, citation validity.

    Unsupported = non-abstention answer with zero valid citations (heuristic).
    """
    abstained = _is_abstention(answer)
    valid = format_citations(answer, citation_map)
    all_tags = [int(m.group(1)) for m in CITATION_PATTERN.finditer(answer)]
    invalid = sum(1 for t in all_tags if t not in citation_map)
    total_tags = len(all_tags)

    validity_rate = (
        len(valid) / total_tags if total_tags > 0 else (1.0 if abstained else 0.0)
    )
    unsupported = (not abstained) and len(valid) == 0 and len(answer.strip()) > 20

    return GroundingMetrics(
        abstained=abstained,
        unsupported_answer=unsupported,
        citation_count=len(valid),
        invalid_citation_tags=invalid,
        citation_validity_rate=round(validity_rate, 4),
        answer_length=len(answer),
    )


def _confidence_bucket(score: float) -> str:
    for low, high, name in CONFIDENCE_BUCKETS:
        if low <= score < high:
            return name
    return "unknown"


def evaluate_confidence(
    hits: list[Any],
    retrieval_recall: float,
    grounding: GroundingMetrics,
) -> ConfidenceMetrics:
    """Map rerank-based confidence to bucket; Brier vs recall filled in aggregate."""
    report = calculate_confidence(hits)
    return ConfidenceMetrics(
        score=report.score,
        level=report.level,
        bucket=_confidence_bucket(report.score),
    )


# ---------------------------------------------------------------------------
# Single-query pipeline
# ---------------------------------------------------------------------------


def evaluate_single_query(
    case: BenchmarkCase,
    vector_store: Any,
    bm25_index: Any,
    cross_encoder: CrossEncoder,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    eval_k: int = DEFAULT_EVAL_K,
    skip_generation: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    use_ollama: bool = False,
) -> tuple[QueryEvaluation, RerankResult]:
    """Run hybrid + rerank (+ optional grounded QA) and compute metrics."""
    t0 = time.perf_counter()
    hybrid_hits, _ = hybrid_retrieve(vector_store, bm25_index, case.question, eval_k)
    retrieval_ms = (time.perf_counter() - t0) * 1000.0

    rerank_result = run_rerank_pipeline(
        vector_store,
        bm25_index,
        cross_encoder,
        case.question,
        candidate_k=candidate_k,
        final_k=final_k,
    )

    retrieval_m = evaluate_retrieval(
        rerank_result.hybrid_hits,
        rerank_result.reranked_hits,
        case.article_id,
        eval_k=eval_k,
    )
    rerank_m = evaluate_reranking(rerank_result)

    gen_s: float | None = None
    if skip_generation:
        answer = ""
        citation_map: dict[int, Any] = {}
        grounding_m = GroundingMetrics(
            abstained=False,
            unsupported_answer=False,
            citation_count=0,
            invalid_citation_tags=0,
            citation_validity_rate=0.0,
            answer_length=0,
        )
    else:
        hits = rerank_result.reranked_hits
        messages, citation_map = build_grounded_prompt(case.question, hits)
        g0 = time.perf_counter()
        answer = generate_grounded_answer(
            messages,
            model=ollama_model if use_ollama else None,
            base_url=ollama_base_url if use_ollama else None,
            use_ollama=use_ollama,
        )
        gen_s = time.perf_counter() - g0
        grounding_m = evaluate_grounding(answer, citation_map)

    confidence_m = evaluate_confidence(
        rerank_result.reranked_hits,
        retrieval_m.recall_at_final_k,
        grounding_m,
    )

    recall_binary = int(retrieval_m.recall_at_final_k >= 1.0)

    eval_row = QueryEvaluation(
        query_id=case.query_id,
        question=case.question,
        article_id=case.article_id,
        split=case.split,
        gold_label=case.gold_label,
        retrieval=retrieval_m,
        reranking=rerank_m,
        grounding=grounding_m,
        confidence=confidence_m,
        retrieval_recall_binary=recall_binary,
        latency_retrieval_ms=retrieval_ms,
        latency_rerank_ms=rerank_result.rerank_latency_ms,
        latency_generation_s=gen_s,
        skipped_generation=skip_generation,
    )
    return eval_row, rerank_result


# ---------------------------------------------------------------------------
# Aggregation & reporting
# ---------------------------------------------------------------------------


def evaluations_to_dataframe(rows: list[QueryEvaluation]) -> pd.DataFrame:
    """Flatten per-query metrics for CSV export."""
    records = []
    for r in rows:
        records.append(
            {
                "query_id": r.query_id,
                "question": r.question,
                "article_id": r.article_id,
                "split": r.split,
                "gold_label": r.gold_label,
                "recall_at_k": r.retrieval.recall_at_k,
                "recall_at_final_k": r.retrieval.recall_at_final_k,
                "mean_hybrid_score": r.retrieval.mean_hybrid_score,
                "mean_dense_score": r.retrieval.mean_dense_score,
                "mean_bm25_score": r.retrieval.mean_bm25_score,
                "gold_in_hybrid_pool": r.retrieval.gold_in_hybrid_pool,
                "median_rank_delta": r.reranking.median_rank_delta,
                "mean_rank_delta": r.reranking.mean_rank_delta,
                "promoted_count": r.reranking.promoted_count,
                "max_rank_gain": r.reranking.max_rank_gain,
                "mean_rerank_score": r.reranking.mean_rerank_score,
                "abstained": r.grounding.abstained,
                "unsupported_answer": r.grounding.unsupported_answer,
                "citation_count": r.grounding.citation_count,
                "invalid_citation_tags": r.grounding.invalid_citation_tags,
                "citation_validity_rate": r.grounding.citation_validity_rate,
                "confidence_score": r.confidence.score,
                "confidence_level": r.confidence.level,
                "confidence_bucket": r.confidence.bucket,
                "retrieval_recall_binary": r.retrieval_recall_binary,
                "latency_retrieval_ms": r.latency_retrieval_ms,
                "latency_rerank_ms": r.latency_rerank_ms,
                "latency_generation_s": r.latency_generation_s,
                "skipped_generation": r.skipped_generation,
            }
        )
    return pd.DataFrame(records)


def build_confidence_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bucket calibration: confidence vs recall / abstention / unsupported."""
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for low, high, name in CONFIDENCE_BUCKETS:
        mask = (df["confidence_score"] >= low) & (df["confidence_score"] < high)
        subset = df[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "bucket": name,
                "score_min": low,
                "score_max": high,
                "query_count": len(subset),
                "mean_confidence": subset["confidence_score"].mean(),
                "mean_recall_at_final_k": subset["recall_at_final_k"].mean(),
                "abstention_rate": subset["abstained"].mean()
                if "abstained" in subset
                else np.nan,
                "unsupported_rate": subset["unsupported_answer"].mean()
                if "unsupported_answer" in subset
                else np.nan,
                "mean_citation_count": subset["citation_count"].mean()
                if "citation_count" in subset
                else np.nan,
                "invalid_citation_rate": (
                    subset["invalid_citation_tags"] > 0
                ).mean()
                if "invalid_citation_tags" in subset
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _sklearn_calibration(df: pd.DataFrame) -> dict[str, Any]:
    """Calibration curve and Brier score (confidence vs article recall)."""
    if len(df) < 5:
        return {"note": "insufficient_samples", "n": len(df)}

    y_true = df["retrieval_recall_binary"].astype(int).values
    y_prob = df["confidence_score"].astype(float).values

    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=min(5, max(2, len(df) // 3)), strategy="quantile"
        )
        brier = float(brier_score_loss(y_true, y_prob))
    except ValueError as exc:
        return {"note": str(exc), "n": len(df)}

    return {
        "brier_score": brier,
        "calibration_fraction_of_positives": prob_true.tolist(),
        "calibration_mean_predicted_value": prob_pred.tolist(),
        "n_samples": len(df),
    }


def generate_summary_report(
    df: pd.DataFrame,
    conf_df: pd.DataFrame,
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Build reliability_report.json payload and print metrics dashboard."""
    n = len(df)
    gen_df = df[~df["skipped_generation"]] if "skipped_generation" in df else df

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "sample_size": n,
        "retrieval": {
            "mean_recall_at_k": float(df["recall_at_k"].mean()) if n else 0.0,
            "mean_recall_at_final_k": float(df["recall_at_final_k"].mean())
            if n
            else 0.0,
            "mean_hybrid_score": float(df["mean_hybrid_score"].mean()) if n else 0.0,
            "gold_in_pool_rate": float(df["gold_in_hybrid_pool"].mean()) if n else 0.0,
        },
        "reranking": {
            "median_rank_delta": float(df["median_rank_delta"].median()) if n else 0.0,
            "mean_rank_delta": float(df["mean_rank_delta"].mean()) if n else 0.0,
            "total_promoted": int(df["promoted_count"].sum()) if n else 0,
            "mean_rerank_score": float(df["mean_rerank_score"].mean()) if n else 0.0,
        },
        "grounding": {},
        "confidence": {
            "mean_score": float(df["confidence_score"].mean()) if n else 0.0,
            "bucket_counts": df["confidence_bucket"].value_counts().to_dict()
            if n
            else {},
            "calibration": calibration,
        },
        "confidence_buckets": conf_df.to_dict(orient="records") if not conf_df.empty else [],
    }

    if len(gen_df) > 0:
        summary["grounding"] = {
            "abstention_rate": float(gen_df["abstained"].mean()),
            "unsupported_answer_rate": float(gen_df["unsupported_answer"].mean()),
            "mean_citation_count": float(gen_df["citation_count"].mean()),
            "invalid_citation_query_rate": float(
                (gen_df["invalid_citation_tags"] > 0).mean()
            ),
            "mean_citation_validity_rate": float(
                gen_df["citation_validity_rate"].mean()
            ),
        }
    elif config.get("skip_generation"):
        summary["grounding"] = {"note": "generation_skipped"}

    _print_dashboard(summary, df, conf_df)
    return summary


def _print_dashboard(
    summary: dict[str, Any],
    df: pd.DataFrame,
    conf_df: pd.DataFrame,
) -> None:
    """Readable stdout metrics dashboard."""
    divider = "=" * 72
    print(divider)
    print("RAG Evaluation Dashboard")
    print(divider)
    print(f"Queries evaluated: {summary['sample_size']}")
    r = summary["retrieval"]
    print("\nRetrieval (article-level recall proxy)")
    print(f"  Mean recall@{DEFAULT_EVAL_K}:     {r['mean_recall_at_k']:.3f}")
    print(f"  Mean recall@final_k:  {r['mean_recall_at_final_k']:.3f}")
    print(f"  Mean hybrid score:    {r['mean_hybrid_score']:.4f}")
    print(f"  Gold in pool rate:    {r['gold_in_pool_rate']:.3f}")

    rr = summary["reranking"]
    print("\nReranking")
    print(f"  Median rank delta:    {rr['median_rank_delta']:.2f}")
    print(f"  Mean rank delta:      {rr['mean_rank_delta']:.2f}")
    print(f"  Total promoted:       {rr['total_promoted']}")
    print(f"  Mean rerank score:    {rr['mean_rerank_score']:.4f}")

    g = summary.get("grounding", {})
    if g and "note" not in g:
        print("\nGrounding")
        print(f"  Abstention rate:        {g['abstention_rate']:.3f}")
        print(f"  Unsupported answer rate:{g['unsupported_answer_rate']:.3f}")
        print(f"  Mean citations:         {g['mean_citation_count']:.2f}")
        print(f"  Invalid citation rate:  {g['invalid_citation_query_rate']:.3f}")

    c = summary["confidence"]
    print("\nConfidence")
    print(f"  Mean score:           {c['mean_score']:.3f}")
    if c.get("bucket_counts"):
        for bucket, cnt in sorted(c["bucket_counts"].items()):
            print(f"    {bucket}: {cnt}")
    cal = c.get("calibration", {})
    if "brier_score" in cal:
        print(f"  Brier score (vs recall): {cal['brier_score']:.4f}")

    if not conf_df.empty:
        print("\nConfidence buckets (calibration table)")
        print(conf_df.to_string(index=False))

    print(divider)


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def generate_plots(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Save matplotlib charts for experiment review."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    if df.empty:
        return saved

    # 1. Confidence distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["confidence_score"], bins=15, color="steelblue", edgecolor="white")
    ax.set_xlabel("Confidence score")
    ax.set_ylabel("Query count")
    ax.set_title("Confidence score distribution")
    path = plots_dir / "confidence_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    # 2. Retrieval hybrid score histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["mean_hybrid_score"], bins=15, color="seagreen", edgecolor="white")
    ax.set_xlabel("Mean hybrid score (top-k)")
    ax.set_ylabel("Query count")
    ax.set_title("Retrieval score distribution")
    path = plots_dir / "retrieval_score_histogram.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    # 3. Reranking gain (rank delta)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(
        range(len(df)),
        df["median_rank_delta"],
        color="coral",
        alpha=0.85,
    )
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Query index")
    ax.set_ylabel("Median rank delta (hybrid -> rerank)")
    ax.set_title("Reranking rank gain per query")
    path = plots_dir / "reranking_gain_chart.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    # 4. Abstention vs confidence (generation runs only)
    gen = df[~df["skipped_generation"]] if "skipped_generation" in df.columns else df
    if len(gen) > 0 and gen["abstained"].any() or len(gen) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["crimson" if a else "steelblue" for a in gen["abstained"]]
        ax.scatter(
            gen["confidence_score"],
            gen["recall_at_final_k"],
            c=colors,
            alpha=0.75,
            s=60,
        )
        ax.set_xlabel("Confidence score")
        ax.set_ylabel("Recall@final_k")
        ax.set_title("Recall vs confidence (red=abstained)")
        path = plots_dir / "abstention_vs_confidence.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    logger.info("Saved %d plots to %s", len(saved), plots_dir)
    return saved


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_evaluation_outputs(
    df: pd.DataFrame,
    conf_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: Path,
    logs_dir: Path,
) -> None:
    """Write CSV, JSON, and per-run logs."""
    output_dir = output_dir.resolve()
    logs_dir = logs_dir.resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / RESULTS_CSV
    df.to_csv(csv_path, index=False)
    logger.info("Wrote %s", csv_path)

    conf_path = output_dir / CONFIDENCE_CSV
    conf_df.to_csv(conf_path, index=False)
    logger.info("Wrote %s", conf_path)

    json_path = output_dir / RELIABILITY_JSON
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", json_path)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log = logs_dir / f"eval_run_{ts}.json"
    with run_log.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "summary": summary,
                "per_query_count": len(df),
            },
            handle,
            indent=2,
        )
    logger.info("Run log -> %s", run_log)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def run_evaluation(
    cases: list[BenchmarkCase],
    vector_store: Any,
    bm25_index: Any,
    cross_encoder: CrossEncoder,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    final_k: int = DEFAULT_FINAL_K,
    eval_k: int = DEFAULT_EVAL_K,
    skip_generation: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    use_ollama: bool = False,
) -> list[QueryEvaluation]:
    """Evaluate all benchmark cases."""
    results: list[QueryEvaluation] = []
    for i, case in enumerate(cases, start=1):
        logger.info("[%d/%d] %s", i, len(cases), case.question[:60])
        try:
            row, _ = evaluate_single_query(
                case,
                vector_store,
                bm25_index,
                cross_encoder,
                candidate_k=candidate_k,
                final_k=final_k,
                eval_k=eval_k,
                skip_generation=skip_generation,
                ollama_model=ollama_model,
                ollama_base_url=ollama_base_url,
                use_ollama=use_ollama,
            )
            results.append(row)
        except Exception as exc:
            logger.error("Failed query %s: %s", case.query_id, exc)
    return results


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval, reranking, grounding, and confidence.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", type=str, default="dev", choices=["train", "dev", "test"])
    parser.add_argument("--max-queries", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--difficult-only", action="store_true")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    parser.add_argument("--chunks-path", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("-k", "--final-k", type=int, default=DEFAULT_FINAL_K, dest="final_k")
    parser.add_argument("--eval-k", type=int, default=DEFAULT_EVAL_K)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM; retrieval+rereank only")
    parser.add_argument("--use-ollama", action="store_true", help="Use local Ollama instead of Groq")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--ollama-base-url", type=str, default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)
    output_dir = args.output_dir.resolve()
    logs_dir = output_dir / EVAL_LOGS_DIR
    plots_dir = logs_dir / PLOTS_SUBDIR

    config = {
        "split": args.split,
        "max_queries": args.max_queries,
        "seed": args.seed,
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "eval_k": args.eval_k,
        "skip_generation": args.skip_llm,
        "difficult_only": args.difficult_only,
    }

    try:
        cases = load_benchmark_queries(
            args.data_dir,
            split=args.split,
            max_queries=args.max_queries,
            seed=args.seed,
            difficult_only=args.difficult_only,
        )
        vector_store, _ = load_vectorstore(args.chroma_dir)
        chunks = load_chunks(args.chunks_path)
        bm25_index = build_bm25_index(chunks)
        cross_encoder = load_cross_encoder(CROSS_ENCODER_MODEL)

        rows = run_evaluation(
            cases,
            vector_store,
            bm25_index,
            cross_encoder,
            candidate_k=args.candidate_k,
            final_k=args.final_k,
            eval_k=args.eval_k,
            skip_generation=args.skip_llm,
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            use_ollama=args.use_ollama,
        )

        if not rows:
            logger.error("No successful evaluations.")
            return 1

        df = evaluations_to_dataframe(rows)
        conf_df = build_confidence_analysis(df)
        calibration = _sklearn_calibration(df)
        summary = generate_summary_report(df, conf_df, calibration, config)

        save_evaluation_outputs(df, conf_df, summary, output_dir, logs_dir)

        if not args.no_plots:
            generate_plots(df, plots_dir)
            summary["plots_dir"] = str(plots_dir)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# TODO — Phase 10+ evaluation enhancements
# =============================================================================
# TODO: RAGAS integration — faithfulness, answer relevance, context precision.
# TODO: Human evaluation — expert rubric on grounded answers and citations.
# TODO: Online feedback loops — log thumbs-down / corrections into retraining.
# TODO: Adaptive retrieval thresholds — auto-tune k from calibration drift.
#
# =============================================================================
# How to run
# =============================================================================
# python evaluate_rag.py --split dev --max-queries 25
# python evaluate_rag.py --max-queries 10 --skip-llm
# =============================================================================
