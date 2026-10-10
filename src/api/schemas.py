"""
Pydantic Request and Response Schemas.
Implements strict input validation, type coercion, and structured API documentation.
"""

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """User query request schema with validation."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=1500,
        description="The user question in either English or Arabic.",
        examples=["What was the total enterprise AI investment in 2024?", "كم بلغت نسبة نمو الناتج المحلي الإجمالي غير النفطي؟"]
    )
    language_hint: Optional[Literal["english", "arabic", "auto"]] = Field(
        default="auto",
        description="Optional language hint. If 'auto', language is automatically inferred."
    )
    retrieval_mode: Literal["baseline", "reranked"] = Field(
        default="reranked",
        description="Retrieval strategy: 'baseline' (dense BGE-M3) or 'reranked' (two-stage cross-encoder)."
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of passages to provide to the generation model."
    )

    @field_validator("query")
    @classmethod
    def clean_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or purely whitespace.")
        return cleaned


class CitationDetail(BaseModel):
    """Structured citation metadata."""
    filename: str
    page: int
    text_snippet: Optional[str] = None
    similarity_score: Optional[float] = None


class QueryResponse(BaseModel):
    """Grounded Q&A response payload."""
    query: str
    language: str
    retrieval_mode: str
    answer: str
    citations: List[str] = Field(
        default_factory=list,
        description="List of cited source strings [filename, Page X]"
    )
    citation_details: List[CitationDetail] = Field(
        default_factory=list,
        description="Detailed structured citation objects"
    )
    is_refusal: bool = Field(
        description="True if the system refused due to missing evidence in the context."
    )
    retrieved_count: int
    latency_ms: float = Field(description="Total end-to-end processing latency in milliseconds.")


class PassageItem(BaseModel):
    """Individual retrieved passage item."""
    chunk_id: str
    filename: str
    page_number: int
    language: str
    text: str
    score: float
    rank: int


class RetrieveRequest(BaseModel):
    """Request schema for direct retrieval inspection."""
    query: str = Field(..., min_length=3, max_length=1500)
    retrieval_mode: Literal["baseline", "reranked"] = "reranked"
    top_k: int = Field(default=5, ge=1, le=50)
    language_filter: Optional[Literal["english", "arabic"]] = None


class RetrieveResponse(BaseModel):
    """Direct retrieval response schema."""
    query: str
    retrieval_mode: str
    passages: List[PassageItem]
    total_retrieved: int
    latency_ms: float


class HealthResponse(BaseModel):
    """System health and readiness status."""
    status: str
    qdrant_status: str
    collection_name: str
    indexed_vectors: int
    embedding_model: str
    reranker_model: str
    llm_provider: str
