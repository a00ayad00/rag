"""
Ingestion Pipeline.
Coordinates loading, chunking, and embedding PDF documents into Qdrant.
Includes self-bootstrapping utility to create bilingual sample PDF reports if data directory is empty.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import settings
from src.ingestion.loader import BilingualPDFLoader, DocumentPage
from src.ingestion.chunker import BilingualChunker, TextChunk

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class IngestionPipeline:
    """End-to-end ingestion pipeline from PDF files to indexed chunks."""

    def __init__(
        self,
        loader: Optional[BilingualPDFLoader] = None,
        chunker: Optional[BilingualChunker] = None,
    ):
        self.loader = loader or BilingualPDFLoader(normalize_arabic=settings.NORMALIZE_ARABIC)
        self.chunker = chunker or BilingualChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

    def process_directories(
        self,
        english_dir: Optional[Path] = None,
        arabic_dir: Optional[Path] = None
    ) -> Tuple[List[TextChunk], List[DocumentPage]]:
        """
        Loads and chunks all documents from the English and Arabic data directories.
        """
        eng_path = english_dir or settings.ENGLISH_DATA_DIR
        ara_path = arabic_dir or settings.ARABIC_DATA_DIR

        # Ensure directories exist
        eng_path.mkdir(parents=True, exist_ok=True)
        ara_path.mkdir(parents=True, exist_ok=True)

        pages: List[DocumentPage] = []

        # Load English reports
        logger.info(f"Scanning English documents from: {eng_path}")
        eng_pages = self.loader.load_directory(eng_path, expected_language="english")
        pages.extend(eng_pages)

        # Load Arabic reports
        logger.info(f"Scanning Arabic documents from: {ara_path}")
        ara_pages = self.loader.load_directory(ara_path, expected_language="arabic")
        pages.extend(ara_pages)

        if not pages:
            logger.warning("No PDF pages loaded. Ensure PDFs exist in data/english and data/arabic.")
            return [], []

        # Chunk all pages
        chunks = self.chunker.chunk_pages(pages)
        logger.info(
            f"Ingestion summary: {len(pages)} pages processed, "
            f"{len(chunks)} chunks generated ({sum(1 for c in chunks if c.language == 'english')} EN, "
            f"{sum(1 for c in chunks if c.language == 'arabic')} AR)."
        )

        return chunks, pages

    def run(self, vector_store=None) -> int:
        """
        Executes the full pipeline: load -> chunk -> embed & index into Qdrant.
        """
        chunks, _ = self.process_directories()
        if not chunks:
            logger.warning("No chunks to index.")
            return 0

        if vector_store is None:
            from src.retrieval.vector_store import VectorStoreManager
            vector_store = VectorStoreManager()

        # Index chunks
        indexed_count = vector_store.index_chunks(chunks)
        logger.info(f"Successfully indexed {indexed_count} chunks into Qdrant collection '{settings.QDRANT_COLLECTION}'")
        return indexed_count


def create_sample_reports_if_needed():
    """
    Creates 10 English and 10 Arabic PDF reports if none exist,
    enabling turnkey verification of the pipeline.
    """
    from src.ingestion.sample_data_generator import generate_sample_reports
    generate_sample_reports()


if __name__ == "__main__":
    create_sample_reports_if_needed()
    pipeline = IngestionPipeline()
    pipeline.run()
