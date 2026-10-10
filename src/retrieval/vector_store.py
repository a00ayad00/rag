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
                )
            else:
                try:
                    logger.info(f"Connecting to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
                    self._client = QdrantClient(
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT,
                        timeout=5.0
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
                self.embedding_dim = self._embedder.get_sentence_embedding_dimension()
            except Exception as e:
                logger.error(f"Failed to load embedding model {settings.EMBEDDING_MODEL_NAME}: {e}")
                # Lightweight fallback if needed
                from sentence_transformers import SentenceTransformer
                fallback_name = "paraphrase-multilingual-MiniLM-L12-v2"
                logger.warning(f"Falling back to {fallback_name}")
                self._embedder = SentenceTransformer(fallback_name, device=settings.EMBEDDING_DEVICE)
                self.embedding_dim = self._embedder.get_sentence_embedding_dimension()

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

    def index_chunks(self, chunks: List[TextChunk], batch_size: int = 64) -> int:
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
                # Unique integer or UUID point ID
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

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            indexed += len(points)
            logger.debug(f"Indexed {indexed}/{total} chunks...")

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

        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k
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
