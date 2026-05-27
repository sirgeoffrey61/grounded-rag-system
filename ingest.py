#!/usr/bin/env python3
"""
Phase 2 — Ingest QuALITY articles into a local Chroma vector store.

Pipeline: load → clean → chunk → embed → persist.

Why chunk overlap (50 tokens on 500-token chunks)?
    Questions often sit on boundaries between chunks. Overlap gives the retriever
    a second chance to surface context that would otherwise be split across two
    chunks, reducing "lost in the middle" failures at section edges.

Why RecursiveCharacterTextSplitter instead of fixed-width splits?
    It tries paragraph → sentence → word breaks before hard-cutting mid-word.
    Naive fixed offsets break entities and sentences, which hurts both readability
    and embedding quality.

Why embeddings?
    Retrieval is similarity search in vector space: queries and chunks are mapped
    to the same space so "meaningfully similar" text is near even without keyword
    overlap. The rest of the RAG stack (rerank, LLM) depends on this step.

Run:
    python ingest.py --data-dir data/quality/v1.0.1

Install:
    pip install -r RAG_REQUIREMENTS.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Reuse battle-tested QuALITY file discovery from Phase 1 exploration.
from explore_dataset import _iter_records, discover_split_files, normalize_article_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "quality" / "v1.0.1"
DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_CHUNKS_PATH = Path(__file__).resolve().parent / "processed_chunks.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "quality_articles"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_BATCH_SIZE = 64
CHROMA_UPSERT_BATCH_SIZE = 500

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class IngestionStatistics:
    """Summary metrics printed after a successful ingest run."""

    num_documents: int = 0
    num_chunks: int = 0
    avg_chunk_length: float = 0.0
    embedding_dimension: int = 0
    documents_per_split: dict[str, int] | None = None
    chunks_per_split: dict[str, int] | None = None


@dataclass
class LoadedDocument:
    """One QuALITY article row ready for cleaning and chunking."""

    document_id: str
    article_id: str
    split: str
    title: str
    text: str


# ---------------------------------------------------------------------------
# Step 1 — Load
# ---------------------------------------------------------------------------


def _prefer_htmlstripped(path: Path) -> Path:
    """
    Use HTML-stripped QuALITY files when available.

    Cleaner text → better chunks and embeddings without running our own
    HTML parser on every ingest.
    """
    if "htmlstripped" in path.name.lower():
        return path
    stripped_name = path.name.replace("QuALITY.v1.0.1.", "QuALITY.v1.0.1.htmlstripped.")
    candidate = path.parent / stripped_name
    return candidate if candidate.is_file() else path


def load_documents(data_dir: Path, dedupe_articles: bool = True) -> list[LoadedDocument]:
    """
    Load article text from all QuALITY splits under ``data_dir``.

    Uses ``set_unique_id`` as ``document_id`` (one JSONL line = one document).
    Optionally deduplicates by ``article_id`` so the same article from two
    writers is only indexed once — important for retrieval precision.
    """
    split_files = discover_split_files(data_dir)
    if not split_files:
        raise FileNotFoundError(
            f"No QuALITY files under {data_dir}. "
            "Extract QuALITY.v1.0.1.zip (see explore_dataset.py header)."
        )

    documents: list[LoadedDocument] = []
    seen_article_ids: set[str] = set()

    for split, raw_path in sorted(split_files.items()):
        path = _prefer_htmlstripped(raw_path)
        html_stripped = "htmlstripped" in path.name.lower()
        logger.info("Loading split=%s from %s", split, path.name)

        try:
            records = list(_iter_records(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to read {path}") from exc

        split_count = 0
        for row in tqdm(records, desc=f"  Load {split}", unit=" docs", leave=False):
            article_raw = row.get("article", "")
            if not article_raw or not str(article_raw).strip():
                logger.debug("Skipping empty article in split=%s", split)
                continue

            article_id = str(row.get("article_id", "unknown"))
            if dedupe_articles and article_id in seen_article_ids:
                continue
            seen_article_ids.add(article_id)

            document_id = str(row.get("set_unique_id") or f"{article_id}_{split}_{split_count}")
            title = str(row.get("title", ""))
            text = normalize_article_text(str(article_raw), html_stripped)

            documents.append(
                LoadedDocument(
                    document_id=document_id,
                    article_id=article_id,
                    split=split,
                    title=title,
                    text=text,
                )
            )
            split_count += 1

        logger.info("  %s: %d documents kept", split, split_count)

    if not documents:
        raise ValueError("No documents loaded — check data_dir and file contents.")

    return documents


# ---------------------------------------------------------------------------
# Step 2 — Clean
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """
    Normalize whitespace and fix common OCR/HTML artifacts.

    Embedding models are sensitive to noisy repeated spaces and replacement
    characters; light cleaning here improves vector consistency.
    """
    if not text:
        return ""

    # Unicode replacement char often appears in Project Gutenberg extracts
    text = text.replace("\ufffd", "'")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _to_langchain_documents(loaded: list[LoadedDocument]) -> list[Document]:
    """Convert loaded rows into LangChain Document objects with metadata."""
    docs: list[Document] = []
    for item in loaded:
        cleaned = clean_text(item.text)
        if not cleaned:
            continue
        docs.append(
            Document(
                page_content=cleaned,
                metadata={
                    "document_id": item.document_id,
                    "article_id": item.article_id,
                    "split": item.split,
                    "title": item.title,
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Step 3 — Chunk
# ---------------------------------------------------------------------------


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split long articles with RecursiveCharacterTextSplitter.

    Each output chunk carries ``chunk_id`` plus parent ``document_id`` and
    ``split`` for filtered retrieval and debugging.
    """
    if not documents:
        raise ValueError("No documents to chunk.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Prefer natural boundaries before arbitrary cuts (see module docstring).
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Document] = []
    for doc in tqdm(documents, desc="Chunking documents", unit=" doc"):
        doc_id = doc.metadata.get("document_id", "unknown")
        split = doc.metadata.get("split", "unknown")
        split_chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(split_chunks):
            chunk.metadata["document_id"] = doc_id
            chunk.metadata["split"] = split
            chunk.metadata["chunk_id"] = f"{doc_id}::chunk_{idx:04d}"
            chunks.append(chunk)

    logger.info("Created %d chunks from %d documents", len(chunks), len(documents))
    return chunks


# ---------------------------------------------------------------------------
# Step 4 — Embeddings
# ---------------------------------------------------------------------------


def create_embeddings(
    chunks: list[Document],
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[HuggingFaceEmbeddings, list[list[float]]]:
    """
    Load the embedding model and vectorize all chunk texts.

    Returns the model (for Chroma) and vectors (for dimension checks / auditing).
    all-MiniLM-L6-v2 outputs 384-dimensional normalized vectors — ideal for
    cosine similarity search in Chroma.

    Embeddings are the bridge between language and retrieval: without them,
    chunks are just text files; with them, a query can be matched by meaning.
    """
    logger.info("Loading embedding model: %s", model_name)
    try:
        model = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load embedding model '{model_name}'. "
            "Install sentence-transformers and ensure network access for first download."
        ) from exc

    texts = [c.page_content for c in chunks]
    vectors: list[list[float]] = []
    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Embedding chunks",
        unit=" batch",
    ):
        batch = texts[start : start + batch_size]
        try:
            vectors.extend(model.embed_documents(batch))
        except Exception as exc:
            raise RuntimeError(f"Embedding failed at batch offset {start}") from exc

    return model, vectors


# ---------------------------------------------------------------------------
# Step 5 — Chroma persistence
# ---------------------------------------------------------------------------


def _chroma_metadata(chunk: Document) -> dict[str, str]:
    """
    Chroma accepts only flat scalar metadata — keep the fields required for RAG.

    We store only document_id, chunk_id, and split so filters stay simple and
    values are always strings (no None / nested dict failures).
    """
    return {
        "document_id": str(chunk.metadata.get("document_id", "unknown")),
        "chunk_id": str(chunk.metadata.get("chunk_id", "unknown")),
        "split": str(chunk.metadata.get("split", "unknown")),
    }


def store_in_chroma(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: Path,
    precomputed_vectors: list[list[float]] | None = None,
    collection_name: str = COLLECTION_NAME,
    reset: bool = False,
) -> Chroma:
    """
    Persist chunk embeddings in a local ChromaDB directory.

    Metadata (document_id, chunk_id, split) is stored alongside vectors so
    retrieval can filter by split or trace results back to source articles.

    When ``precomputed_vectors`` is supplied, we skip re-embedding inside Chroma
    (vectors were already produced in ``create_embeddings``).
    """
    import shutil

    persist_directory = persist_directory.resolve()
    persist_directory.mkdir(parents=True, exist_ok=True)

    if reset and persist_directory.exists():
        logger.warning("Resetting Chroma directory: %s", persist_directory)
        for child in persist_directory.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    logger.info("Writing %d chunks to Chroma at %s", len(chunks), persist_directory)
    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
        )
        if precomputed_vectors is not None and len(precomputed_vectors) != len(chunks):
            raise ValueError("precomputed_vectors length must match chunks")

        # Batch upserts — adding 25k+ rows in one call often exhausts memory on Windows.
        for start in tqdm(
            range(0, len(chunks), CHROMA_UPSERT_BATCH_SIZE),
            desc="Storing in Chroma",
            unit=" batch",
        ):
            end = start + CHROMA_UPSERT_BATCH_SIZE
            batch_chunks = chunks[start:end]
            texts = [c.page_content for c in batch_chunks]
            metadatas = [_chroma_metadata(c) for c in batch_chunks]
            ids = [str(c.metadata.get("chunk_id", f"chunk_{start + i}")) for i, c in enumerate(batch_chunks)]

            kwargs: dict[str, Any] = {
                "texts": texts,
                "metadatas": metadatas,
                "ids": ids,
            }
            if precomputed_vectors is not None:
                kwargs["embeddings"] = precomputed_vectors[start:end]

            vector_store.add_texts(**kwargs)

    except Exception as exc:
        raise RuntimeError(f"Chroma persistence failed under {persist_directory}") from exc

    return vector_store


# ---------------------------------------------------------------------------
# Outputs & statistics
# ---------------------------------------------------------------------------


def save_processed_chunks(chunks: list[Document], output_path: Path) -> None:
    """Serialize chunk text and metadata for inspection (no raw vectors)."""
    payload = [
        {
            "chunk_id": c.metadata.get("chunk_id"),
            "document_id": c.metadata.get("document_id"),
            "article_id": c.metadata.get("article_id"),
            "split": c.metadata.get("split"),
            "title": c.metadata.get("title"),
            "char_length": len(c.page_content),
            "text": c.page_content,
        }
        for c in chunks
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    logger.info("Saved processed chunks to %s", output_path)


def compute_ingestion_statistics(
    documents: list[Document],
    chunks: list[Document],
    embedding_dimension: int,
) -> IngestionStatistics:
    """Aggregate counts for the final ingestion report."""
    lengths = [len(c.page_content) for c in chunks]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    docs_per_split: dict[str, int] = {}
    chunks_per_split: dict[str, int] = {}
    for doc in documents:
        split = str(doc.metadata.get("split", "unknown"))
        docs_per_split[split] = docs_per_split.get(split, 0) + 1
    for chunk in chunks:
        split = str(chunk.metadata.get("split", "unknown"))
        chunks_per_split[split] = chunks_per_split.get(split, 0) + 1

    return IngestionStatistics(
        num_documents=len(documents),
        num_chunks=len(chunks),
        avg_chunk_length=avg_len,
        embedding_dimension=embedding_dimension,
        documents_per_split=docs_per_split,
        chunks_per_split=chunks_per_split,
    )


def print_ingestion_statistics(stats: IngestionStatistics) -> None:
    """Print a readable ingestion summary to stdout."""
    divider = "=" * 72
    print(divider)
    print("Ingestion Statistics")
    print(divider)
    print(f"Documents ingested:     {stats.num_documents:>8,}")
    print(f"Chunks created:         {stats.num_chunks:>8,}")
    print(f"Average chunk length:   {stats.avg_chunk_length:>8,.1f} characters")
    print(f"Embedding dimension:    {stats.embedding_dimension:>8,}")

    if stats.documents_per_split:
        print("\nDocuments per split:")
        for split, count in sorted(stats.documents_per_split.items()):
            print(f"  {split:6s}  {count:>6,}")

    if stats.chunks_per_split:
        print("\nChunks per split:")
        for split, count in sorted(stats.chunks_per_split.items()):
            print(f"  {split:6s}  {count:>6,}")

    print(divider)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest QuALITY articles into ChromaDB for RAG retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory with QuALITY train/dev/test files",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Persistent ChromaDB storage directory",
    )
    parser.add_argument(
        "--chunks-output",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="JSON file listing processed chunks (text + metadata)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear existing Chroma data before ingesting",
    )
    parser.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Keep duplicate article_id rows from multiple writers",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    try:
        loaded = load_documents(args.data_dir, dedupe_articles=not args.no_dedupe)
        documents = _to_langchain_documents(loaded)
        chunks = chunk_documents(documents)

        embeddings, vectors = create_embeddings(chunks)
        embedding_dim = len(vectors[0]) if vectors else 0

        store_in_chroma(
            chunks,
            embeddings,
            persist_directory=args.chroma_dir,
            precomputed_vectors=vectors,
            reset=args.reset,
        )
        save_processed_chunks(chunks, args.chunks_output)

        stats = compute_ingestion_statistics(documents, chunks, embedding_dim)
        print_ingestion_statistics(stats)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("%s", exc)
        return 1

    logger.info("Ingestion complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# requirements.txt — install before running
# =============================================================================
# langchain>=0.3.0
# langchain-core>=0.3.0
# langchain-text-splitters>=0.3.0
# langchain-huggingface>=0.1.0
# langchain-chroma>=0.2.0
# chromadb>=0.5.0
# sentence-transformers>=3.0.0
# tqdm>=4.66.0
#
# pip install -r RAG_REQUIREMENTS.txt
#
# =============================================================================
# How to run
# =============================================================================
# 1. Ensure QuALITY data is extracted (see explore_dataset.py).
# 2. pip install -r RAG_REQUIREMENTS.txt
# 3. python ingest.py --data-dir data/quality/v1.0.1
# 4. Optional: python ingest.py --reset --verbose
#
# Outputs:
#   chroma_db/              — persisted vectors + metadata
#   processed_chunks.json   — human-readable chunk audit trail
# =============================================================================
