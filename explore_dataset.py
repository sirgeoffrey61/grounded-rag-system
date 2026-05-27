#!/usr/bin/env python3
"""
Explore the QuALITY long-document QA dataset before building a RAG pipeline.

QuALITY ships as JSON Lines (one article + question set per line). This script
loads train/dev/test splits when present, prints aggregate statistics, and
shows random examples for manual inspection.

Why JSONL + nested questions?
    Each line is one writer's validated questions for one article. Statistics
    must flatten the nested ``questions`` list to count QA pairs correctly.

Run:
    python explore_dataset.py --data-dir path/to/quality/data/v1.0.1

Download data (not included in this repo):
    https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.zip
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Common filenames after extracting QuALITY.v1.0.1.zip
SPLIT_FILE_PATTERNS: dict[str, list[str]] = {
    "train": [
        "QuALITY.v1.0.1.train.jsonl",
        "QuALITY.v1.0.1.train",  # official zip uses extensionless JSONL
        "QuALITY.v1.0.1.htmlstripped.train.jsonl",
        "QuALITY.v1.0.1.htmlstripped.train",
        "train.jsonl",
        "train.json",
    ],
    "dev": [
        "QuALITY.v1.0.1.dev.jsonl",
        "QuALITY.v1.0.1.dev",
        "QuALITY.v1.0.1.htmlstripped.dev.jsonl",
        "QuALITY.v1.0.1.htmlstripped.dev",
        "dev.jsonl",
        "dev.json",
    ],
    "test": [
        "QuALITY.v1.0.1.test.jsonl",
        "QuALITY.v1.0.1.test",
        "QuALITY.v1.0.1.htmlstripped.test.jsonl",
        "QuALITY.v1.0.1.htmlstripped.test",
        "test.jsonl",
        "test.json",
    ],
}

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "quality" / "v1.0.1"
ARTICLE_PREVIEW_CHARS = 500
RANDOM_SAMPLE_COUNT = 5


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class QuestionRecord:
    """A single multiple-choice question tied to one article."""

    split: str
    article_id: str
    title: str
    article: str
    question: str
    options: list[str]
    gold_label: int
    difficult: int | None = None


@dataclass
class DatasetStatistics:
    """Aggregate metrics across all loaded splits."""

    num_documents: int = 0
    num_questions: int = 0
    article_char_lengths: list[int] = field(default_factory=list)
    article_token_lengths: list[int] = field(default_factory=list)
    question_char_lengths: list[int] = field(default_factory=list)
    option_char_lengths: list[int] = field(default_factory=list)
    splits_loaded: list[str] = field(default_factory=list)
    documents_per_split: dict[str, int] = field(default_factory=dict)
    questions_per_split: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def tokenize_simple(text: str) -> list[str]:
    """
    Whitespace tokenization for exploratory stats.

    We avoid heavyweight tokenizers here so the script runs without GPU/API
    dependencies. For RAG chunking later, switch to the same tokenizer as your
    embedding model.
    """
    return text.split()


def strip_html_basic(html: str) -> str:
    """Remove HTML tags for readable previews and length stats."""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_article_text(article: str, html_stripped: bool) -> str:
    """Return plain text; htmlstripped files are already plain."""
    if html_stripped:
        return article.strip()
    return strip_html_basic(article)


def discover_split_files(data_dir: Path) -> dict[str, Path]:
    """
    Locate train/dev/test files under ``data_dir`` (recursive search).

    Prefers non-htmlstripped JSONL when both exist so raw + stripped stats
    can be compared later; htmlstripped is used only if it is the only match.
    """
    if not data_dir.is_dir():
        return {}

    found: dict[str, Path] = {}
    all_jsonl = list(data_dir.rglob("*.jsonl"))
    all_json = list(data_dir.rglob("*.json"))

    for split, patterns in SPLIT_FILE_PATTERNS.items():
        for pattern in patterns:
            for candidate in data_dir.rglob(pattern):
                if candidate.is_file():
                    found[split] = candidate
                    break
            if split in found:
                break

    # QuALITY release files named QuALITY.v1.0.1.{train,dev,test}
    if len(found) < 3:
        for path in sorted(data_dir.rglob("QuALITY*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            if "htmlstripped" in name:
                continue
            for split in ("train", "dev", "test"):
                if name.endswith(f".{split}") and split not in found:
                    found[split] = path
                    break

    # Fallback: any *train*.jsonl etc. not yet matched
    if not found and (all_jsonl or all_json):
        for path in sorted(all_jsonl + all_json):
            name = path.name.lower()
            for split in ("train", "dev", "test"):
                if split in name and split not in found:
                    found[split] = path

    return found


def _parse_jsonl_line(line: str, line_no: int, source: Path) -> dict[str, Any]:
    line = line.strip()
    if not line:
        raise ValueError(f"empty line at {line_no}")
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON at {source}:{line_no}") from exc


def _is_jsonl_file(path: Path) -> bool:
    """
    True for JSON Lines files.

    QuALITY.v1.0.1.train has no .jsonl suffix; pathlib reports ``.train`` as the
    suffix, so we match known split endings and the QuALITY filename prefix.
    """
    name_lower = path.name.lower()
    if path.suffix.lower() == ".jsonl":
        return True
    if name_lower.startswith("quality") and path.suffix.lower() in (
        ".train",
        ".dev",
        ".test",
        "",
    ):
        return True
    return False


def _iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield one document record from JSONL or a JSON array file."""
    suffix = path.suffix.lower()
    if _is_jsonl_file(path):
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                yield _parse_jsonl_line(line, line_no, path)
    elif suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            yield data
        else:
            raise ValueError(f"unsupported JSON root type in {path}")
    else:
        raise ValueError(f"unsupported file type: {path}")


def _extract_questions(
    doc: dict[str, Any], split: str, html_stripped: bool
) -> list[QuestionRecord]:
    """Flatten nested questions into QuestionRecord rows."""
    article_raw = doc.get("article", "")
    article = normalize_article_text(article_raw, html_stripped)
    article_id = str(doc.get("article_id", "unknown"))
    title = str(doc.get("title", ""))

    questions = doc.get("questions")
    if not isinstance(questions, list):
        return []

    records: list[QuestionRecord] = []
    for item in questions:
        if not isinstance(item, dict):
            continue
        options = item.get("options", [])
        if not isinstance(options, list):
            options = []
        gold = item.get("gold_label")
        if gold is None:
            continue
        records.append(
            QuestionRecord(
                split=split,
                article_id=article_id,
                title=title,
                article=article,
                question=str(item.get("question", "")),
                options=[str(o) for o in options],
                gold_label=int(gold),
                difficult=item.get("difficult"),
            )
        )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    data_dir: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[QuestionRecord]]:
    """
    Load all available QuALITY splits from ``data_dir``.

    Returns:
        documents_by_split: raw document dicts per split name
        questions: flattened list of QuestionRecord for stats / sampling
    """
    split_files = discover_split_files(data_dir)
    if not split_files:
        raise FileNotFoundError(
            f"No QuALITY data files found under {data_dir}.\n"
            "Download QuALITY v1.0.1 from "
            "https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.zip "
            "and extract it, then pass --data-dir to the folder containing "
            "*.jsonl files."
        )

    documents_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_questions: list[QuestionRecord] = []

    print(f"\nLoading QuALITY from: {data_dir.resolve()}\n")
    for split, path in sorted(split_files.items()):
        html_stripped = "htmlstripped" in path.name.lower()
        print(f"  [{split}] {path.name}")
        try:
            records = list(
                tqdm(
                    _iter_records(path),
                    desc=f"  Reading {split}",
                    unit=" docs",
                    leave=False,
                )
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to load {path}") from exc

        documents_by_split[split].extend(records)
        for doc in records:
            all_questions.extend(_extract_questions(doc, split, html_stripped))

    return dict(documents_by_split), all_questions


def compute_statistics(
    documents_by_split: dict[str, list[dict[str, Any]]],
    questions: list[QuestionRecord],
) -> DatasetStatistics:
    """Compute document- and question-level metrics."""
    stats = DatasetStatistics()

    for split, docs in documents_by_split.items():
        stats.splits_loaded.append(split)
        stats.documents_per_split[split] = len(docs)
        stats.num_documents += len(docs)

    stats.num_questions = len(questions)
    for split in stats.splits_loaded:
        stats.questions_per_split[split] = sum(1 for q in questions if q.split == split)

    # Unique articles (same article_id can appear twice — two writers)
    seen_articles: set[tuple[str, str]] = set()
    for q in questions:
        key = (q.split, q.article_id)
        if key not in seen_articles:
            seen_articles.add(key)
            stats.article_char_lengths.append(len(q.article))
            stats.article_token_lengths.append(len(tokenize_simple(q.article)))

    for q in questions:
        stats.question_char_lengths.append(len(q.question))
        for opt in q.options:
            stats.option_char_lengths.append(len(opt))

    return stats


def _mean(values: list[int | float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _max(values: list[int]) -> int:
    return max(values) if values else 0


def print_statistics(stats: DatasetStatistics) -> None:
    """Pretty-print dataset statistics to stdout."""
    divider = "=" * 72
    print(divider)
    print("QuALITY Dataset Statistics")
    print(divider)

    print(f"\nSplits loaded: {', '.join(stats.splits_loaded) or '(none)'}")
    for split in stats.splits_loaded:
        print(
            f"  {split:5s}  documents: {stats.documents_per_split.get(split, 0):>5,}  "
            f"questions: {stats.questions_per_split.get(split, 0):>5,}"
        )

    print(f"\nTotal documents (JSONL lines):  {stats.num_documents:>8,}")
    print(f"Total questions (flattened):    {stats.num_questions:>8,}")
    if stats.questions_per_split.get("test", 0) == 0 and "test" in stats.splits_loaded:
        print(
            "\nNote: The public test split does not ship gold_label / question text "
            "for leaderboard evaluation — document counts still apply."
        )

    print("\nArticle length (characters, unique article_id per split):")
    print(f"  Average:  {_mean(stats.article_char_lengths):>10,.1f}")
    print(f"  Maximum:  {_max(stats.article_char_lengths):>10,}")

    print("\nArticle length (whitespace tokens, same scope):")
    print(f"  Average:  {_mean(stats.article_token_lengths):>10,.1f}")
    print(f"  Maximum:  {_max(stats.article_token_lengths):>10,}")

    print("\nQuestion length (characters):")
    print(f"  Average:  {_mean(stats.question_char_lengths):>10,.1f}")

    print("\nAnswer option length (characters, all options):")
    print(f"  Average:  {_mean(stats.option_char_lengths):>10,.1f}")

    print(divider)


def _format_correct_answer(record: QuestionRecord) -> str:
    """Map 1-indexed gold_label to the option text."""
    idx = record.gold_label - 1
    if 0 <= idx < len(record.options):
        letter = chr(ord("A") + idx)
        return f"{letter}) {record.options[idx]} (gold_label={record.gold_label})"
    return f"(invalid gold_label={record.gold_label})"


def show_random_examples(
    questions: list[QuestionRecord],
    n: int = RANDOM_SAMPLE_COUNT,
    seed: int | None = 42,
) -> None:
    """Print ``n`` random QA examples for manual inspection."""
    if not questions:
        print("\nNo questions available to sample.")
        return

    rng = random.Random(seed)
    sample = rng.sample(questions, min(n, len(questions)))

    print(f"\n{'=' * 72}")
    print(f"Random sample ({len(sample)} examples, seed={seed})")
    print("=" * 72)

    for i, record in enumerate(sample, start=1):
        preview = record.article[:ARTICLE_PREVIEW_CHARS]
        if len(record.article) > ARTICLE_PREVIEW_CHARS:
            preview += "..."

        token_len = len(tokenize_simple(record.article))
        difficult = (
            "yes" if record.difficult == 1 else "no" if record.difficult == 0 else "n/a"
        )

        print(f"\n--- Example {i} [{record.split}] article_id={record.article_id} ---")
        if record.title:
            print(f"Title: {record.title}")
        print(f"Hard subset (difficult=1): {difficult}")
        print(f"\nQuestion:\n  {record.question}")
        print("\nAnswer choices:")
        for j, option in enumerate(record.options):
            letter = chr(ord("A") + j)
            print(f"  {letter}) {option}")
        print(f"\nCorrect answer:\n  {_format_correct_answer(record)}")
        print(f"\nArticle preview (first {ARTICLE_PREVIEW_CHARS} chars):")
        print(f"  {preview}")
        print(f"\nArticle token length (whitespace): {token_len:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore the QuALITY long-document QA dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing QuALITY JSONL files (extracted zip)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible example sampling",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=RANDOM_SAMPLE_COUNT,
        help="Number of random examples to display",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir: Path = args.data_dir

    try:
        documents_by_split, questions = load_dataset(data_dir)
        stats = compute_statistics(documents_by_split, questions)
        print_statistics(stats)
        show_random_examples(questions, n=args.num_examples, seed=args.seed)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# requirements.txt (install before running)
# =============================================================================
# tqdm>=4.66.0
#
# Or: pip install -r RAG_REQUIREMENTS.txt
#
# =============================================================================
# How to run
# =============================================================================
# 1. Download QuALITY v1.0.1 (JSONL, not the PDF reference doc):
#    https://github.com/nyu-mll/quality/blob/main/data/v1.0.1/QuALITY.v1.0.1.zip
# 2. Extract so you have files like QuALITY.v1.0.1.train.jsonl
# 3. Install dependencies:
#    pip install -r RAG_REQUIREMENTS.txt
# 4. Run exploration (point --data-dir at the folder with the JSONL files):
#    python explore_dataset.py --data-dir ./data/quality/v1.0.1
# 5. Optional flags:
#    python explore_dataset.py --data-dir ./data/quality --seed 0 --num-examples 10
#
# Note: "C:\Users\asus\Downloads\Rag dataset.pdf" is dataset documentation;
# the script expects extracted .jsonl files from the GitHub release above.
# =============================================================================
