"""
FastAPI Route Handlers.
Exposes REST endpoints for bilingual querying, retrieval inspection, and health checking.
"""

import time
import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, status

from src.config import settings
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    CitationDetail,
    RetrieveRequest,
    RetrieveResponse,
    PassageItem,
    HealthResponse,
)
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.baseline import BaselineRetriever
from src.retrieval.reranker import MultilingualReranker, RerankedRetriever
from src.generation.graph import build_rag_graph
from src.generation.llm import get_llm

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Bilingual Document Q&A"])

# Global component singletons
_vector_store: VectorStoreManager = None
_baseline_retriever: BaselineRetriever = None
_reranked_retriever: RerankedRetriever = None
_rag_graph = None


def get_dependencies():
    """Dependency provider for RAG components."""
    global _vector_store, _baseline_retriever, _reranked_retriever, _rag_graph
    if _vector_store is None:
        _vector_store = VectorStoreManager()
        _baseline_retriever = BaselineRetriever(vector_store=_vector_store)
        reranker = MultilingualReranker()
        _reranked_retriever = RerankedRetriever(
            baseline_retriever=_baseline_retriever,
            reranker=reranker
        )
        llm = get_llm()
        _rag_graph = build_rag_graph(
            baseline_retriever=_baseline_retriever,
            reranked_retriever=_reranked_retriever,
            llm=llm
        )
    return _vector_store, _baseline_retriever, _reranked_retriever, _rag_graph


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Bilingual Document Q&A",
    description="Answers questions in English or Arabic with citations and strict refusal when ungrounded."
)
async def query_endpoint(
    req: QueryRequest,
    deps=Depends(get_dependencies)
):
    """Executes full RAG workflow via LangGraph."""
    start_time = time.perf_counter()
    _, _, _, rag_graph = deps

    logger.info(
        f"Processing query: '{req.query[:60]}...' "
        f"[mode={req.retrieval_mode}, top_k={req.top_k}]"
    )

    init_state = {
        "question": req.query,
        "retrieval_mode": req.retrieval_mode,
        "top_k": req.top_k,
    }

    try:
        output = rag_graph.invoke(init_state)
    except Exception as e:
        logger.error(f"Error executing RAG graph: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing query: {str(e)}"
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    passages = output.get("retrieved_passages", [])
    citations = output.get("citations", [])

    # Build detailed citation objects
    citation_details: List[CitationDetail] = []
    for p in passages:
        citation_details.append(
            CitationDetail(
                filename=p.filename,
                page=p.page_number,
                text_snippet=p.text[:150] + "...",
                similarity_score=p.score
            )
        )

    return QueryResponse(
        query=req.query,
        language=output.get("language", "english"),
        retrieval_mode=req.retrieval_mode,
        answer=output.get("answer", ""),
        citations=citations,
        citation_details=citation_details,
        is_refusal=output.get("is_refusal", False),
        retrieved_count=len(passages),
        latency_ms=round(latency_ms, 2)
    )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Inspect Retrieved Passages",
    description="Direct retrieval endpoint for ranking and similarity debugging without LLM invocation."
)
async def retrieve_endpoint(
    req: RetrieveRequest,
    deps=Depends(get_dependencies)
):
    """Performs retrieval inspection."""
    start_time = time.perf_counter()
    _, baseline, reranked, _ = deps

    if req.retrieval_mode == "baseline":
        passages = baseline.retrieve(
            query=req.query,
            top_k=req.top_k,
            language_filter=req.language_filter
        )
    else:
        passages = reranked.retrieve(
            query=req.query,
            top_k=req.top_k,
            language_filter=req.language_filter
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    items = [
        PassageItem(
            chunk_id=p.chunk_id,
            filename=p.filename,
            page_number=p.page_number,
            language=p.language,
            text=p.text,
            score=p.score,
            rank=p.rank
        )
        for p in passages
    ]

    return RetrieveResponse(
        query=req.query,
        retrieval_mode=req.retrieval_mode,
        passages=items,
        total_retrieved=len(items),
        latency_ms=round(latency_ms, 2)
    )


@router.get(
    "/documents",
    summary="List Indexed Documents",
    description="Returns metadata on indexed documents and collection size."
)
async def list_documents(deps=Depends(get_dependencies)):
    """Returns vector collection statistics."""
    vector_store, _, _, _ = deps
    stats = vector_store.get_stats()
    return stats


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health and Readiness",
    description="Verifies Qdrant connection and service readiness."
)
async def health_check(deps=Depends(get_dependencies)):
    """Readiness probe."""
    vector_store, _, _, _ = deps
    stats = vector_store.get_stats()
    is_ok = "error" not in stats

    return HealthResponse(
        status="healthy" if is_ok else "degraded",
        qdrant_status=stats.get("status", "unknown"),
        collection_name=settings.QDRANT_COLLECTION,
        indexed_vectors=stats.get("vectors_count", 0),
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        reranker_model=settings.RERANKER_MODEL_NAME,
        llm_provider=settings.LLM_PROVIDER,
    )
