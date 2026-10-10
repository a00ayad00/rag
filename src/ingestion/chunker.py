"""
Bilingual Text Chunker with Morphological & Punctuation Awareness.
Splits text into semantically cohesive passages while respecting Arabic and English
sentence boundary conventions and punctuation.
"""

import hashlib
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from src.ingestion.loader import DocumentPage

logger = logging.getLogger(__name__)


class TextChunk(BaseModel):
    """Represents a chunk of text with rich retrieval and citation metadata."""
    chunk_id: str = Field(description="Deterministic hash of doc_id, page, and chunk_index")
    doc_id: str
    filename: str
    page_number: int
    language: str
    chunk_index: int
    text: str
    char_count: int
    word_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BilingualChunker:
    """
    Recursive character text chunker customized for bilingual Arabic and English corpora.

    Justification for Design & Parameters:
    1. Chunk Size (600 chars / ~90-120 words):
       - Arabic is a morphologically synthetic language where prefixes (conjunctions, prepositions,
         definite articles) and suffixes (pronouns, gender/number markers) fuse into single tokens.
       - A 600-character Arabic segment contains higher semantic density than an equivalent English
         string. Setting the window to ~600 chars balances dense vector capacity (BGE-M3) with
         propositional purity, avoiding multi-topic dilution.
    2. Overlap (120 chars / ~20%):
       - Prevents splitting key entity definitions, numerical values, and anaphora at chunk seams.
    3. Bilingual Boundary Separators:
       - Incorporates Arabic punctuation: Arabic comma (، \u060C), Arabic semicolon (؛ \u061B),
         and Arabic question mark (؟ \u061F) in addition to standard Latin delimiters.
    """

    # Hierarchical split characters preserving semantic units
    BILINGUAL_SEPARATORS = [
        "\n\n",   # Paragraph breaks
        "\n",     # Line breaks
        ". ",     # Latin sentence end
        ".\n",    # Sentence followed by newline
        "؟ ",     # Arabic question mark
        "؛ ",     # Arabic semicolon
        "! ",     # Exclamation mark
        "? ",     # Latin question mark
        "، ",     # Arabic comma / clause boundary
        ", ",     # Latin comma
        " ",      # Word space
        ""        # Character fallback
    ]

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 120,
        separators: Optional[List[str]] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.BILINGUAL_SEPARATORS

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting based on hierarchical separators."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators: List[str] = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if separator and s:
                s = s + separator if separator.strip() else s
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    sub_chunks = self._split_text(s, new_separators)
                    final_chunks.extend(sub_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits)
            final_chunks.extend(merged)

        return [c.strip() for c in final_chunks if c.strip()]

    def _merge_splits(self, splits: List[str]) -> List[str]:
        """Combines short splits into chunks of target size with sliding overlap."""
        docs: List[str] = []
        current_doc: List[str] = []
        total = 0

        for s in splits:
            s_len = len(s)
            if total + s_len > self.chunk_size:
                if current_doc:
                    doc = "".join(current_doc).strip()
                    if doc:
                        docs.append(doc)

                    # Retain overlap from end of current_doc
                    while total > self.chunk_overlap and current_doc:
                        popped = current_doc.pop(0)
                        total -= len(popped)

            current_doc.append(s)
            total += s_len

        if current_doc:
            doc = "".join(current_doc).strip()
            if doc:
                docs.append(doc)

        return docs

    def _generate_chunk_id(self, doc_id: str, page_number: int, chunk_index: int, text: str) -> str:
        """Generates a stable unique hash identifier for the chunk."""
        key = f"{doc_id}_p{page_number}_c{chunk_index}_{text[:50]}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    def split_page(self, page: DocumentPage) -> List[TextChunk]:
        """Splits a DocumentPage into multiple TextChunk objects with citations."""
        if not page.text.strip():
            return []

        raw_chunks = self._split_text(page.text, self.separators)
        chunks: List[TextChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = self._generate_chunk_id(page.doc_id, page.page_number, idx, chunk_text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    filename=page.filename,
                    page_number=page.page_number,
                    language=page.language,
                    chunk_index=idx,
                    text=chunk_text,
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    metadata={
                        **page.metadata,
                        "source_file": page.filename,
                        "page": page.page_number,
                    }
                )
            )

        return chunks

    def chunk_pages(self, pages: List[DocumentPage]) -> List[TextChunk]:
        """Chunks a collection of DocumentPage objects."""
        all_chunks: List[TextChunk] = []
        for page in pages:
            page_chunks = self.split_page(page)
            all_chunks.extend(page_chunks)

        logger.info(f"Chunked {len(pages)} pages into {len(all_chunks)} text chunks.")
        return all_chunks
