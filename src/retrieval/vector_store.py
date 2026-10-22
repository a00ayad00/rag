"""
Qdrant Vector Store Manager and Embedding Interface.
Manages vector collection creation, batch upserting, and vector similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.config import settings
from src.ingestion.chunker import TextChunk

logger = logging.getLogger(__name__)


class RetrievedPassage(BaseModel):
    """Represents a retrieved passage with similarity metadata and citations."""
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int
    language: str
    text: str
    score: float
    rank: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def citation(self) -> str:
        """Returns standard citation string."""
        return f"[{self.filename}, Page {self.page_number}]"


class VectorStoreManager:
    """
    Manages Qdrant client connection, embedding generation, collection lifecycle,
    and vector similarity search.
    """

    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION
        self._client: Optional[QdrantClient] = None
        self._embedder = None
        self.embedding_dim = settings.EMBEDDING_DIM

    @property
    def client(self) -> QdrantClient:
        """Lazy initialization of Qdrant client with automatic fallback to local storage."""
        if self._client is None:
            if settings.QDRANT_URL:
                logger.info(f"Connecting to Qdrant at URL: {settings.QDRANT_URL}")
                self._client = QdrantClient(
                    url=settings.QDRANT_URL,
                    api_key=settings.QDRANT_API_KEY,
                    timeout=settings.QDRANT_TIMEOUT,
                )
            else:
                try:
                    logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT} (timeout={settings.QDRANT_TIMEOUT}s)")
                    self._client = QdrantClient(
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT,
                        timeout=settings.QDRANT_TIMEOUT
                    )
                    # Test connection
                    self._client.get_collections()
                except Exception as e:
                    local_path = settings.BASE_DIR / "qdrant_storage"
                    local_path.mkdir(parents=True, exist_ok=True)
                    logger.warning(
                        f"Could not connect to Qdrant service ({e}). "
                        f"Falling back to embedded local storage at: {local_path}"
                    )
                    self._client = QdrantClient(path=str(local_path))

            self._ensure_collection()
        return self._client

    @property
    def embedder(self):
        """Lazy loading of multilingual sentence transformer embedder."""
        if self._embedder is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME} on {settings.EMBEDDING_DEVICE}")
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    settings.EMBEDDING_MODEL_NAME,
                    device=settings.EMBEDDING_DEVICE
                )
                dim = self._embedder.get_sentence_embedding_dimension()
                self.embedding_dim = int(dim) if dim is not None else settings.EMBEDDING_DIM
                logger.info(f"Successfully loaded {settings.EMBEDDING_MODEL_NAME} (dimension: {self.embedding_dim})")
            except Exception as e:
                logger.error(f"Failed to load embedding model {settings.EMBEDDING_MODEL_NAME}: {e}")
                # Lightweight fallback if primary model download or init fails
                try:
                    from sentence_transformers import SentenceTransformer
                    fallback_name = "paraphrase-multilingual-MiniLM-L12-v2"
                    logger.warning(f"Attempting fallback to lightweight model: {fallback_name}")
                    self._embedder = SentenceTransformer(fallback_name, device=settings.EMBEDDING_DEVICE)
                    dim = self._embedder.get_sentence_embedding_dimension()
                    self.embedding_dim = int(dim) if dim is not None else 384
                except Exception as fallback_err:
                    logger.critical(f"Fallback embedder also failed ({fallback_err}). Using mock offline embedder.")
                    # Bulletproof offline fallback so the service never crashes
                    class OfflineMockEmbedder:
                        def encode(self, texts, **kwargs):
                            import numpy as np
                            return np.zeros((len(texts), settings.EMBEDDING_DIM), dtype=np.float32)
                        def get_sentence_embedding_dimension(self):
                            return settings.EMBEDDING_DIM

                    self._embedder = OfflineMockEmbedder()
                    self.embedding_dim = settings.EMBEDDING_DIM

        return self._embedder

    def _ensure_collection(self):
        """Creates collection with Cosine metric and payload indexes if not present."""
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            logger.info(f"Creating Qdrant collection: {self.collection_name} (dim={self.embedding_dim})")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE
                )
            )
            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="language",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_id",
                field_schema=qmodels.PayloadSchemaType.KEYWORD
            )

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encodes text passages into dense embedding vectors."""
        embeddings = self.embedder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def _upsert_batch(self, points: List[qmodels.PointStruct]):
        """Upserts a batch of points with automatic sub-batch retry if timeout occurs."""
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )
        except Exception as e:
            logger.warning(f"Upsert timed out or encountered error ({e}). Retrying with smaller sub-batches of 8...")
            sub_size = 8
            for sub_i in range(0, len(points), sub_size):
                sub_pts = points[sub_i : sub_i + sub_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=sub_pts,
                    wait=True
                )

    def index_chunks(self, chunks: List[TextChunk], batch_size: int = 32) -> int:
        """Embeds and upserts chunks into Qdrant."""
        if not chunks:
            return 0

        self._ensure_collection()
        total = len(chunks)
        logger.info(f"Indexing {total} chunks in batches of {batch_size}...")

        indexed = 0
        for i in range(0, total, batch_size):
            batch = chunks[i: i + batch_size]
            texts = [c.text for c in batch]
            vectors = self.encode(texts)

            points = []
            for idx, (chunk, vector) in enumerate(zip(batch, vectors)):
                point_id = i + idx + 1
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "filename": chunk.filename,
                    "page_number": chunk.page_number,
                    "language": chunk.language,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                    "word_count": chunk.word_count,
                    **chunk.metadata
                }
                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                )

            self._upsert_batch(points)
            indexed += len(points)
            logger.info(f"Indexed {indexed}/{total} chunks...")

        logger.info(f"Successfully indexed total {indexed} chunks into {self.collection_name}")
        return indexed

    def search_dense(
        self,
        query: str,
        top_k: int = 5,
        language_filter: Optional[str] = None
    ) -> List[RetrievedPassage]:
        """Performs standard dense vector similarity search (Baseline)."""
        query_vector = self.encode([query])[0]

        query_filter = None
        if language_filter:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="language",
                        match=qmodels.MatchValue(value=language_filter)
                    )
                ]
            )

        # Compatible across qdrant-client versions (v1.10+ uses query_points, earlier uses search)
        search_result = []
        if hasattr(self.client, "query_points"):
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                )
                search_result = response.points if hasattr(response, "points") else response
            except Exception as q_err:
                logger.warning(f"query_points invocation failed ({q_err}), falling back to search...")
                if hasattr(self.client, "search"):
                    search_result = self.client.search(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=top_k
                    )
                else:
                    raise q_err
        elif hasattr(self.client, "search"):
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
        elif hasattr(self.client, "search_points"):
            search_result = self.client.search_points(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k
            )
        else:
            raise AttributeError(
                f"QdrantClient object '{type(self.client).__name__}' has neither 'query_points' nor 'search'."
            )

        passages: List[RetrievedPassage] = []
        for rank, hit in enumerate(search_result, start=1):
            p = hit.payload or {}
            passages.append(
                RetrievedPassage(
                    chunk_id=p.get("chunk_id", str(hit.id)),
                    doc_id=p.get("doc_id", "unknown"),
                    filename=p.get("filename", "unknown"),
                    page_number=p.get("page_number", 1),
                    language=p.get("language", "unknown"),
                    text=p.get("text", ""),
                    score=float(hit.score),
                    rank=rank,
                    metadata=p
                )
            )

        return passages

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection count and vector dimensions."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "vectors_count": info.points_count or 0,
                "vector_dimension": self.embedding_dim,
                "status": str(info.status),
            }
        except Exception as e:
            return {"error": str(e), "collection_name": self.collection_name}
