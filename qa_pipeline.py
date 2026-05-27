#!/usr/bin/env python3
"""
Phase 4 — Retrieval-Augmented Generation (RAG) QA pipeline.

Connects Chroma retrieval to a local Ollama LLM with strict context grounding.

Why grounding matters:
    LLMs invent plausible text from pretraining. Supplying retrieved passages and
    instructing the model to answer ONLY from them ties outputs to evidence,
    which is the core value of RAG over raw chat.

Why hallucinations still happen in RAG:
    - Retrieved chunks may be irrelevant (bad retrieval).
    - The model may ignore instructions under pressure to answer.
    - Chunks may be incomplete relative to the question.
    Grounding reduces hallucinations; it does not eliminate them without
    reranking, better chunks, and answer validation.

Why retrieved context quality is critical:
    Garbage in → garbage out. Wrong or partial chunks cause confident wrong
    answers. Observability (scores, metadata, latencies) helps you debug retrieval
    before blaming the LLM.

Run:
    ollama pull mistral
    python qa_pipeline.py --query "Why was the vocabulary limited?"

Install:
    pip install -r RAG_REQUIREMENTS.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from llm_client import LLMClient, LLMClientError

# Match ingest.py so query vectors live in the same space as indexed chunks.
from ingest import COLLECTION_NAME, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"
DEFAULT_TOP_K = 5
DEFAULT_OLLAMA_MODEL = "mistral"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

logger = logging.getLogger(__name__)

# Grounded system prompt — correctness over fluency.
SYSTEM_PROMPT = """You are a careful question-answering assistant for long-document QA.

Rules:
1. Answer ONLY using the provided context passages below.
2. Do NOT use outside knowledge, assumptions, or guesses.
3. If the context does not contain enough information to answer, reply exactly:
   "I cannot answer from the provided context."
4. When you answer, stay concise and faithful to the wording in the context.
5. Do not invent citations, names, dates, or facts not present in the context."""

USER_PROMPT_TEMPLATE = """Context passages (use only these):
{context}

Question: {question}

Answer (from context only):"""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """One retrieved passage with provenance and similarity for inspection."""

    rank: int
    text: str
    document_id: str
    chunk_id: str
    split: str
    similarity_score: float  # higher = more relevant (LangChain relevance score)


@dataclass
class PipelineTimings:
    """Latency breakdown for observability."""

    retrieval_seconds: float = 0.0
    generation_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass
class QAResponse:
    """Full pipeline output for display and logging."""

    question: str
    answer: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    timings: PipelineTimings = field(default_factory=PipelineTimings)


# ---------------------------------------------------------------------------
# Vector store & retrieval
# ---------------------------------------------------------------------------


def load_vectorstore(
    chroma_dir: Path,
    collection_name: str = COLLECTION_NAME,
    embedding_model: str = EMBEDDING_MODEL,
) -> tuple[Chroma, HuggingFaceEmbeddings]:
    """
    Load the persisted Chroma index and the same embedding model used at ingest.

    Using the identical model is non-negotiable: mismatched embeddings make
    similarity search meaningless.
    """
    chroma_dir = chroma_dir.resolve()
    if not chroma_dir.is_dir():
        raise FileNotFoundError(
            f"Chroma directory not found: {chroma_dir}. Run ingest.py first."
        )

    logger.info("Loading embedding model: %s", embedding_model)
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    logger.info("Loading Chroma from %s (collection=%s)", chroma_dir, collection_name)
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(chroma_dir),
    )

    # Fail fast if ingest was never run
    collection = vector_store._collection  # noqa: SLF001 — intentional debug/health check
    count = collection.count()
    if count == 0:
        raise ValueError(f"Chroma collection '{collection_name}' is empty. Run ingest.py.")
    logger.info("Vector store ready (%d chunks indexed)", count)

    return vector_store, embeddings


def embed_query(query: str, embeddings: HuggingFaceEmbeddings) -> list[float]:
    """Embed the user question in the same vector space as document chunks."""
    if not query.strip():
        raise ValueError("Query must not be empty.")
    return embeddings.embed_query(query.strip())


def retrieve_chunks(
    vector_store: Chroma,
    query: str,
    k: int = DEFAULT_TOP_K,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks by semantic similarity.

    Uses the store's embedding function on the query text so relevance scores
    stay on a consistent 0–1 scale (higher = better match).
    """
    if k < 1:
        raise ValueError("k must be at least 1")

    pairs = vector_store.similarity_search_with_relevance_scores(query.strip(), k=k)

    if not pairs:
        logger.warning("No chunks retrieved for query")
        return []

    results: list[RetrievedChunk] = []
    for rank, (doc, score) in enumerate(pairs, start=1):
        results.append(
            RetrievedChunk(
                rank=rank,
                text=doc.page_content,
                document_id=str(doc.metadata.get("document_id", "unknown")),
                chunk_id=str(doc.metadata.get("chunk_id", "unknown")),
                split=str(doc.metadata.get("split", "unknown")),
                similarity_score=float(score),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Prompting & generation
# ---------------------------------------------------------------------------


def _format_context_block(chunks: list[RetrievedChunk]) -> str:
    """Number passages so the LLM can refer to them unambiguously."""
    if not chunks:
        return "(No context retrieved.)"
    parts: list[str] = []
    for chunk in chunks:
        header = (
            f"[Passage {chunk.rank} | split={chunk.split} | "
            f"doc={chunk.document_id} | chunk={chunk.chunk_id} | "
            f"score={chunk.similarity_score:.4f}]"
        )
        parts.append(f"{header}\n{chunk.text}")
    return "\n\n".join(parts)


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> list[SystemMessage | HumanMessage]:
    """
    Build chat messages with retrieved context inlined.

    Separating system rules from the user block keeps the grounding contract
    visible in logs and easy to tune without touching retrieval code.
    """
    context = _format_context_block(chunks)
    user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question.strip())
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]


def generate_answer(
    messages: list[SystemMessage | HumanMessage],
    model: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.0,
    *,
    use_ollama: bool = False,
) -> str:
    """
    Produce a grounded answer via Groq (default) or local Ollama (--use-ollama).
    """
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
                f"Ollama generation failed. Is Ollama running? Try: ollama pull {ollama_model}\n"
                f"Base URL: {ollama_url}"
            ) from exc
        content = response.content
        if isinstance(content, str):
            return content.strip()
        return str(content).strip()

    client = LLMClient(model=model) if model else LLMClient()
    try:
        return client.generate(messages)
    except LLMClientError as exc:
        raise RuntimeError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Display & orchestration
# ---------------------------------------------------------------------------


def display_response(result: QAResponse) -> None:
    """Print a human-readable trace of retrieval + generation for inspection."""
    divider = "=" * 72
    thin = "-" * 72

    print(divider)
    print("RAG Question Answering")
    print(divider)

    print(f"\nQuestion:\n  {result.question}\n")

    print(thin)
    print(f"Retrieved chunks ({len(result.chunks)})")
    print(thin)
    if not result.chunks:
        print("  (none — generation will likely refuse or hallucinate)\n")
    else:
        for chunk in result.chunks:
            preview = chunk.text[:300] + ("..." if len(chunk.text) > 300 else "")
            print(
                f"\n  [{chunk.rank}] similarity={chunk.similarity_score:.4f} | "
                f"split={chunk.split} | document_id={chunk.document_id}"
            )
            print(f"      chunk_id={chunk.chunk_id}")
            print(f"      preview: {preview}")

    print(f"\n{thin}")
    print("Generated answer")
    print(thin)
    print(f"\n{result.answer}\n")

    print(thin)
    print("Source metadata (retrieval)")
    print(thin)
    for chunk in result.chunks:
        print(
            f"  rank={chunk.rank} | score={chunk.similarity_score:.4f} | "
            f"split={chunk.split} | document_id={chunk.document_id} | "
            f"chunk_id={chunk.chunk_id}"
        )

    print(f"\n{thin}")
    print("Timings")
    print(thin)
    t = result.timings
    print(f"  Retrieval:  {t.retrieval_seconds:.3f}s")
    print(f"  Generation: {t.generation_seconds:.3f}s")
    print(f"  Total:      {t.total_seconds:.3f}s")
    print(divider)


def run_pipeline(
    question: str,
    chroma_dir: Path,
    k: int = DEFAULT_TOP_K,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    use_ollama: bool = False,
) -> QAResponse:
    """Execute retrieve → prompt → generate with timing instrumentation."""
    pipeline_start = time.perf_counter()
    timings = PipelineTimings()

    vector_store, embeddings = load_vectorstore(chroma_dir)

    retrieval_start = time.perf_counter()
    # Explicit embed step for observability (same model/dim as ingest).
    query_vector = embed_query(question, embeddings)
    logger.debug("Query embedding dimension: %d", len(query_vector))
    chunks = retrieve_chunks(vector_store, question, k=k)
    timings.retrieval_seconds = time.perf_counter() - retrieval_start
    logger.info(
        "Retrieved %d chunks in %.3fs (top score=%.4f)",
        len(chunks),
        timings.retrieval_seconds,
        chunks[0].similarity_score if chunks else 0.0,
    )

    messages = build_prompt(question, chunks)

    generation_start = time.perf_counter()
    answer = generate_answer(
        messages,
        model=ollama_model if use_ollama else None,
        base_url=ollama_base_url if use_ollama else None,
        use_ollama=use_ollama,
    )
    timings.generation_seconds = time.perf_counter() - generation_start
    logger.info("Generated answer in %.3fs", timings.generation_seconds)

    timings.total_seconds = time.perf_counter() - pipeline_start
    logger.info("Pipeline total: %.3fs", timings.total_seconds)

    return QAResponse(
        question=question,
        answer=answer,
        chunks=chunks,
        timings=timings,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format=LOG_FORMAT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grounded RAG QA over QuALITY articles (Chroma + Ollama).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help='User question, e.g. "Why was the vocabulary limited?"',
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=DEFAULT_CHROMA_DIR,
        help="Path to persisted ChromaDB directory",
    )
    parser.add_argument(
        "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve",
    )
    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Use local Ollama instead of Groq",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=DEFAULT_OLLAMA_MODEL,
        help="Ollama model name (run: ollama pull mistral)",
    )
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Ollama API base URL",
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
        result = run_pipeline(
            question=args.query,
            chroma_dir=args.chroma_dir,
            k=args.k,
            ollama_model=args.ollama_model,
            ollama_base_url=args.ollama_base_url,
            use_ollama=args.use_ollama,
        )
        display_response(result)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        logger.exception("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# =============================================================================
# Future improvements (TODO)
# =============================================================================
# TODO: Reranking — cross-encoder rerank top-20 → top-5 for sharper context.
# TODO: Conversational memory — track prior Q/A with session_id for follow-ups.
# TODO: Hybrid retrieval — combine BM25 keyword search with vector similarity.
# TODO: Citation highlighting — map answer spans back to chunk offsets in UI.
#
# =============================================================================
# requirements.txt
# =============================================================================
# langchain-ollama>=0.2.0
# (plus RAG_REQUIREMENTS.txt from ingest phase)
#
# Setup:
#   ollama serve
#   ollama pull mistral
#   pip install -r RAG_REQUIREMENTS.txt
#
# Run:
#   python qa_pipeline.py --query "Why was the vocabulary limited?"
#   python qa_pipeline.py --query "..." -k 8 --verbose
# =============================================================================
