"""
Baseline Dense Retriever.
Uses open-source multilingual embedding model (BAAI/bge-m3) with Qdrant Cosine Similarity.
"""

import logging
from typing import List, Optional
from src.retrieval.vector_store import VectorStoreManager, RetrievedPassage
from src.config import settings

logger = logging.getLogger(__name__)


class BaselineRetriever:
    """
    Baseline dense retriever.
    Encodes query with multilingual bi-encoder and retrieves top-k nearest neighbors.
    """

    def __init__(self, vector_store: Optional[VectorStoreManager] = None):
        self.vector_store = vector_store or VectorStoreManager()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        language_filter: Optional[str] = None
    ) -> List[RetrievedPassage]:
        """
        Executes baseline dense search.
        """
        k = top_k or settings.RETRIEVAL_TOP_K
        logger.debug(f"Baseline retrieval for query='{query[:40]}...' (k={k})")
        results = self.vector_store.search_dense(
            query=query,
            top_k=k,
            language_filter=language_filter
        )
        return results
