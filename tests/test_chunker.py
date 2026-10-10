"""
Unit tests for Bilingual Chunker and Language Detection.
"""

import pytest
from src.ingestion.loader import DocumentPage, BilingualPDFLoader
from src.ingestion.chunker import BilingualChunker, TextChunk


def test_language_detection():
    loader = BilingualPDFLoader()

    # English text
    eng_text = "The enterprise accelerated AI adoption across customer operations with 35% growth."
    assert loader.detect_language(eng_text) == "english"

    # Arabic text
    ara_text = "شهد عام 2024 قفزة نوعية في مسار التحول الرقمي والحوسبة السحابية."
    assert loader.detect_language(ara_text) == "arabic"

    # Empty text
    assert loader.detect_language("") == "english"


def test_chunker_basic_split():
    chunker = BilingualChunker(chunk_size=100, chunk_overlap=20)
    page = DocumentPage(
        doc_id="test_doc",
        filename="test.pdf",
        filepath="/tmp/test.pdf",
        page_number=1,
        text="Sentence one is here. Sentence two is longer and provides more detail. Sentence three concludes the section.",
        language="english",
        char_count=108,
    )

    chunks = chunker.split_page(page)
    assert len(chunks) >= 1
    for c in chunks:
        assert len(c.text) <= 120  # respects size within tolerance
        assert c.doc_id == "test_doc"
        assert c.page_number == 1
        assert c.language == "english"


def test_chunker_arabic_punctuation():
    chunker = BilingualChunker(chunk_size=120, chunk_overlap=20)
    ara_text = (
        "تم تدشين محطات طاقة شمسية بقدرة 2.5 غيغاوات، مما أسهم في خفض الانبعاثات الكربونية بمقدار 3.2 مليون طن؛ "
        "وهل سنحقق الاكتفاء الذاتي بحلول عام 2030؟ نعم إن الخطط تسير وفق الجدول الزمني المحدد."
    )
    page = DocumentPage(
        doc_id="ara_doc",
        filename="energy.pdf",
        filepath="/tmp/energy.pdf",
        page_number=2,
        text=ara_text,
        language="arabic",
        char_count=len(ara_text),
    )

    chunks = chunker.split_page(page)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.language == "arabic"
        assert c.page_number == 2


def test_chunker_overlap_validation():
    # overlap >= chunk_size should raise ValueError
    with pytest.raises(ValueError):
        BilingualChunker(chunk_size=100, chunk_overlap=100)
