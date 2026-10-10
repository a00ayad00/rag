"""
Bilingual PDF Document Loader.
Extracts text and page-level metadata from Arabic and English PDF documents.
Handles Unicode normalization, language detection, and Arabic text orientation.
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentPage(BaseModel):
    """Represents a single extracted page from a PDF document."""
    doc_id: str
    filename: str
    filepath: str
    page_number: int
    text: str
    language: str = Field(description="'arabic' or 'english'")
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BilingualPDFLoader:
    """
    Robust PDF text extractor with bilingual support (Arabic & English).
    Applies unicode normalization and cleans common PDF extraction artifacts.
    """

    ARABIC_CHAR_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

    def __init__(self, normalize_arabic: bool = True):
        self.normalize_arabic = normalize_arabic

    def detect_language(self, text: str) -> str:
        """
        Determines if text is predominantly Arabic or English.
        Uses character frequency in the Arabic Unicode script block.
        """
        if not text.strip():
            return "english"

        arabic_chars = len(self.ARABIC_CHAR_PATTERN.findall(text))
        total_letters = sum(1 for c in text if c.isalpha())

        if total_letters == 0:
            return "english"

        ratio = arabic_chars / total_letters
        return "arabic" if ratio > 0.20 else "english"

    def clean_text(self, text: str, language: str) -> str:
        """
        Cleans extracted PDF text.
        - Normalizes Unicode representations (NFKC).
        - Strips invisible control characters.
        - Preserves Arabic diacritics and punctuation.
        """
        if not text:
            return ""

        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)

        # Remove zero-width spaces and non-printable characters
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)

        # Clean multiple spaces and irregular newlines
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)

        return text.strip()

    def load_pdf(self, file_path: Path) -> List[DocumentPage]:
        """
        Loads a single PDF and returns a list of DocumentPage objects.
        Tries PyMuPDF (fitz) first for high fidelity, then falls back to pypdf.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages: List[DocumentPage] = []
        doc_id = path.stem

        # Method 1: Try PyMuPDF (fitz)
        try:
            import fitz  # type: ignore

            doc = fitz.open(str(path))
            for page_idx, page in enumerate(doc):
                raw_text = page.get_text("text") or ""
                lang = self.detect_language(raw_text)
                cleaned = self.clean_text(raw_text, lang)

                if cleaned:
                    pages.append(
                        DocumentPage(
                            doc_id=doc_id,
                            filename=path.name,
                            filepath=str(path.resolve()),
                            page_number=page_idx + 1,
                            text=cleaned,
                            language=lang,
                            char_count=len(cleaned),
                            metadata={"total_pages": len(doc), "source_dir": path.parent.name}
                        )
                    )
            doc.close()
            logger.info(f"Loaded {len(pages)} pages from {path.name} via PyMuPDF")
            return pages
        except ImportError:
            logger.debug("PyMuPDF not installed, falling back to pypdf.")
        except Exception as e:
            logger.warning(f"PyMuPDF failed on {path.name}: {e}. Falling back to pypdf.")

        # Method 2: Fallback to pypdf
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            total_pages = len(reader.pages)
            for page_idx, page in enumerate(reader.pages):
                raw_text = page.extract_text() or ""
                lang = self.detect_language(raw_text)
                cleaned = self.clean_text(raw_text, lang)

                if cleaned:
                    pages.append(
                        DocumentPage(
                            doc_id=doc_id,
                            filename=path.name,
                            filepath=str(path.resolve()),
                            page_number=page_idx + 1,
                            text=cleaned,
                            language=lang,
                            char_count=len(cleaned),
                            metadata={"total_pages": total_pages, "source_dir": path.parent.name}
                        )
                    )
            logger.info(f"Loaded {len(pages)} pages from {path.name} via pypdf")
            return pages
        except Exception as e:
            logger.error(f"Failed to load PDF {path.name}: {e}")
            raise RuntimeError(f"Could not extract text from {path.name}: {e}")

    def load_directory(self, directory_path: Path, expected_language: Optional[str] = None) -> List[DocumentPage]:
        """Loads all PDF files in a given directory."""
        dir_path = Path(directory_path)
        if not dir_path.exists():
            logger.warning(f"Directory does not exist: {dir_path}")
            return []

        all_pages: List[DocumentPage] = []
        pdf_files = sorted(dir_path.glob("*.pdf"))

        for pdf_file in pdf_files:
            try:
                pages = self.load_pdf(pdf_file)
                if expected_language:
                    # Enforce expected directory language if specified
                    for p in pages:
                        p.language = expected_language
                all_pages.extend(pages)
            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {e}")

        logger.info(f"Loaded total {len(all_pages)} pages from {len(pdf_files)} PDFs in {dir_path.name}")
        return all_pages
