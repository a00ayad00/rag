"""
Reranked Retriever (Improvement Module).
Implements Two-Stage Retrieval:
1. High-recall candidate retrieval via dense multilingual bi-encoder (K=15).
2. High-precision cross-attention reranking via multilingual cross-encoder (K=5).
"""

import logging
from typing import List, Optional
from src.retrieval.vector_store import VectorStoreManager, RetrievedPassage
from src.retrieval.baseline import BaselineRetriever
from src.config import settings

logger = logging.getLogger(__name__)


class MultilingualReranker:
    """
    Cross-Encoder Reranker supporting Arabic and English.
    Applies deep cross-attention over [Query, Passage] pairs to eliminate
    bi-encoder false positives and rank relevant citations at rank 1.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.RERANKER_MODEL_NAME
        self._model = None
        self._backend = None

    def _load_model(self):
        """Loads cross-encoder model with automatic fallback."""
        if self._model is not None:
            return

        # Backend 1: FlashRank (Lightweight, ultra-fast CPU inference)
        try:
            from flashrank import Ranker
            logger.info("Initializing FlashRank cross-encoder reranker...")
            self._model = Ranker(model_name="ms-marco-MultiBERT-L-12", cache_dir="/tmp/flashrank")
            self._backend = "flashrank"
            return
        except Exception as e:
            logger.debug(f"FlashRank init note: {e}")

        # Backend 2: SentenceTransformers CrossEncoder
        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading CrossEncoder: {self.model_name}...")
            self._model = CrossEncoder(self.model_name, device=settings.EMBEDDING_DEVICE)
            self._backend = "cross_encoder"
            return
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder {self.model_name}: {e}. Trying fallback...")

        # Backend 3: Lightweight CrossEncoder fallback
        try:
            from sentence_transformers import CrossEncoder
            fallback = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self._model = CrossEncoder(fallback, device=settings.EMBEDDING_DEVICE)
            self._backend = "cross_encoder"
            return
        except Exception as e:
            logger.error(f"Could not load any cross-encoder: {e}. Falling back to passthrough.")
            self._backend = "passthrough"

    def rerank(self, query: str, passages: List[RetrievedPassage], top_k: int = 5) -> List[RetrievedPassage]:
        """
        Reranks a list of candidate passages for the given query.
        """
        if not passages:
            return []

        self._load_model()

        if self._backend == "flashrank":
            from flashrank import RerankRequest
            passages_data = [{"id": p.chunk_id, "text": p.text, "meta": p} for p in passages]
            req = RerankRequest(query=query, passages=passages_data)
            results = self._model.rerank(req)

            reranked: List[RetrievedPassage] = []
            for rank, r in enumerate(results[:top_k], start=1):
                p = r["meta"]
                p.score = float(r["score"])
                p.rank = rank
                reranked.append(p)
            return reranked

        elif self._backend == "cross_encoder":
            pairs = [[query, p.text] for p in passages]
            scores = self._model.predict(pairs)

            # Pair and sort
            scored_passages = list(zip(passages, scores))
            scored_passages.sort(key=lambda x: x[1], reverse=True)

            reranked: List[RetrievedPassage] = []
            for rank, (p, score) in enumerate(scored_passages[:top_k], start=1):
                p.score = float(score)
                p.rank = rank
                reranked.append(p)
            return reranked

        else:
            # Passthrough fallback: return original passages up to top_k
            return passages[:top_k]


class RerankedRetriever:
    """
    Two-stage retrieval pipeline:
    Stage 1: Retrieve candidate_k (15) candidates using dense bi-encoder.
    Stage 2: Rerank candidates using cross-encoder and return top_k (5).
    """

    def __init__(
        self,
        baseline_retriever: Optional[BaselineRetriever] = None,
        reranker: Optional[MultilingualReranker] = None
    ):
        self.baseline = baseline_retriever or BaselineRetriever()
        self.reranker = reranker or MultilingualReranker()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        language_filter: Optional[str] = None
    ) -> List[RetrievedPassage]:
        """
        Executes two-stage retrieval: broad recall then cross-encoder precision rerank.
        """
        final_k = top_k or settings.RETRIEVAL_TOP_K
        cand_k = candidate_k or settings.RETRIEVAL_CANDIDATE_K

        # Step 1: Candidate retrieval
        candidates = self.baseline.retrieve(
            query=query,
            top_k=cand_k,
            language_filter=language_filter
        )

        if not candidates:
            return []

        # Step 2: Cross-encoder rerank
        reranked_results = self.reranker.rerank(
            query=query,
            passages=candidates,
            top_k=final_k
        )

        return reranked_results
