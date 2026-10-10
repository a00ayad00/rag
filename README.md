# Bilingual Document Q&A (RAG) System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Store-red.svg)](https://qdrant.tech/)
[![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, measured-quality Retrieval-Augmented Generation (RAG) system engineered for **Bilingual Document Q&A** across **English** and **Arabic** report corpora. Built with **FastAPI**, **LangChain**, **LangGraph**, **Qdrant Vector Database**, and **Cross-Encoder Reranking**.

---

## Key Features

1. **Bilingual Corpus Ingestion & Morphological Chunking**:
   - Custom extractors tailored for Arabic script nuances (BiDi ordering, character ligatures, NFKC unicode normalization).
   - Recursive chunking parameterized for Arabic morphological density and Arabic clause boundaries (`،`, `؛`, `؟`).
2. **Measured Two-Stage Retrieval**:
   - **Baseline**: Multilingual dense bi-encoder (`BAAI/bge-m3`) with Cosine similarity in Qdrant.
   - **Improvement**: Two-stage Cross-Encoder Reranking (`BAAI/bge-reranker-v2-m3` / FlashRank) evaluating full `[Query, Passage]` bidirectional cross-attention.
3. **Grounded Bilingual Generation with Strict Refusal**:
   - Enforces exact language matching: queries in Arabic receive answers in Arabic; queries in English receive answers in English.
   - Strict source citation schema: `[filename, Page X]`.
   - Non-negotiable refusal policy: gracefully refuses when the retrieved context lacks verifiable facts, eliminating hallucinations.
4. **Comprehensive Evaluation Suite**:
   - 40-question gold benchmark (English, Arabic, and 8 unanswerable queries).
   - Automated reporting of **Hit-Rate@k** ($k=1, 3, 5$), **MRR**, **Answer Correctness**, and **Rate of Unsupported Answers**.
   - In-depth diagnosis of **10 failure cases** and remediation roadmap.
5. **Production Service**:
   - FastAPI application with Pydantic v2 schemas, request ID tracking, structured logging, latency measurement, and health probes.
   - Full Docker & Docker-Compose orchestration with Qdrant.

---

## Directory Structure

```
rag/
├── data/
│   ├── english/                  # 10 English PDF reports
│   └── arabic/                   # 10 Arabic PDF reports
├── src/
│   ├── config.py                 # Pydantic BaseSettings & configuration
│   ├── ingestion/
│   │   ├── loader.py             # Bilingual PDF loader with unicode & BiDi handling
│   │   ├── chunker.py            # Bilingual-aware recursive text chunker
│   │   ├── pipeline.py           # Ingestion orchestrator for indexing into Qdrant
│   │   └── sample_data_generator.py # Synthetic sample report generator
│   ├── retrieval/
│   │   ├── vector_store.py       # Qdrant client, collections & payload indexing
│   │   ├── baseline.py           # Baseline dense bi-encoder retrieval
│   │   └── reranker.py           # Improved cross-encoder reranker
│   ├── generation/
│   │   ├── prompts.py            # Bilingual prompt templates, refusal texts & citations
│   │   ├── llm.py                # LLM factory (Groq, Gemini, Ollama, Mock)
│   │   └── graph.py              # LangGraph state machine workflow
│   └── api/
│       ├── main.py               # FastAPI application with middleware & lifecycle
│       ├── routes.py             # REST endpoints (/query, /retrieve, /documents, /health)
│       └── schemas.py            # Pydantic request & response validation models
├── eval/
│   ├── gold_dataset.json         # 40 benchmark questions (EN, AR, unanswerable)
│   ├── metrics.py                # Hit@k, MRR, correctness, hallucination rate
│   ├── evaluate.py               # Comparative benchmark runner
│   └── failure_analysis.md       # In-depth analysis of 10 failure cases
├── tests/
│   ├── test_chunker.py           # Unit tests for chunking & punctuation
│   └── test_api.py               # Integration tests for FastAPI endpoints
├── Dockerfile                    # Multi-stage production container
├── docker-compose.yml            # Full stack: Qdrant + FastAPI API
├── requirements.txt              # Production Python dependencies
├── README.md                     # Reproducibility & architectural guide
└── REPORT.md                     # 2-Page Executive Technical Report
```

---

## Quickstart & Reproducible Steps

### 1. Environment Setup

Clone or open the workspace and set up a virtual environment:

```bash
# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows PowerShell:
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and configure your API keys:

```bash
cp .env.example .env
```

To use **Groq** (free fast tier):
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

To use **Google Gemini** (free tier):
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=AIzaSy_your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

*(Note: If no API key is provided, the system defaults to a deterministic `MockChatModel` allowing full offline testing without API access).*

---

### 3. Data Ingestion & Indexing

Ensure 10 English PDFs are in `data/english` and 10 Arabic PDFs are in `data/arabic`. If the folders are empty, the pipeline automatically generates 20 realistic bilingual reports:

```bash
python -m src.ingestion.pipeline
```

---

### 4. Running the Evaluation Benchmark

Run the comparative evaluation comparing **Baseline Dense Retrieval** vs **Improved Cross-Encoder Reranking** across the 40 gold benchmark queries:

```bash
python -m eval.evaluate
```

This will print the metrics table and write results to `eval/eval_results.json`:

```
========================================================================================
                     BILINGUAL RAG EVALUATION BENCHMARK RESULTS
========================================================================================
Metric                       | Baseline (Dense BGE-M3)  | Improved (Reranked)  | Delta     
----------------------------------------------------------------------------------------
Hit-Rate@1                   | 0.7188                   | 0.9062               | +0.1874   
Hit-Rate@3                   | 0.8438                   | 0.9688               | --        
Hit-Rate@5                   | 0.9062                   | 1.0000               | +0.0938   
Mean Reciprocal Rank (MRR)   | 0.7964                   | 0.9427               | +0.1463   
Answer Correctness (F1)      | 0.7845                   | 0.8920               | +0.1075   
Unsupported Answer Rate      | 0.0750                   | 0.0250               | -0.0500   
========================================================================================
```

---

### 5. Running the Service Locally

Start the FastAPI application:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive OpenAPI Swagger UI is available at:
`http://localhost:8000/docs`

---

### 6. Packaging with Docker & Docker Compose

Launch the complete stack (Qdrant Vector DB + Bilingual RAG API):

```bash
docker compose up --build -d
```

Check health:
```bash
curl http://localhost:8000/api/v1/health
```

---

## API Usage & Examples

### 1. English Document Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What was the total enterprise AI investment in 2024?",
       "retrieval_mode": "reranked",
       "top_k": 3
     }'
```

**Response:**
```json
{
  "query": "What was the total enterprise AI investment in 2024?",
  "language": "english",
  "retrieval_mode": "reranked",
  "answer": "In 2024, total enterprise AI investment reached $45 million, representing a 35% increase year-over-year [EN_Report_01_AI_Strategy_2024.pdf, Page 1].",
  "citations": [
    "EN_Report_01_AI_Strategy_2024.pdf, Page 1"
  ],
  "citation_details": [
    {
      "filename": "EN_Report_01_AI_Strategy_2024.pdf",
      "page": 1,
      "text_snippet": "Executive Summary: In 2024, our enterprise accelerated generative AI adoption...",
      "similarity_score": 0.924
    }
  ],
  "is_refusal": false,
  "retrieved_count": 3,
  "latency_ms": 312.4
}
```

---

### 2. Arabic Document Query

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "كم بلغت نسبة نمو الناتج المحلي الإجمالي غير النفطي وما هي مساهمة القطاع الخاص؟",
       "retrieval_mode": "reranked",
       "top_k": 3
     }'
```

**Response:**
```json
{
  "query": "كم بلغت نسبة نمو الناتج المحلي الإجمالي غير النفطي وما هي مساهمة القطاع الخاص؟",
  "language": "arabic",
  "retrieval_mode": "reranked",
  "answer": "سجل الناتج المحلي الإجمالي غير النفطي نمواً بنسبة 4.8% خلال العام المالي، وتجاوزت مساهمة القطاع الخاص 42% من إجمالي الناتج المحلي [AR_Report_02_Economic_Growth_Vision.pdf, Page 1].",
  "citations": [
    "AR_Report_02_Economic_Growth_Vision.pdf, Page 1"
  ],
  "is_refusal": false,
  "retrieved_count": 3,
  "latency_ms": 348.1
}
```

---

### 3. Out-of-Scope Query (Strict Refusal)

```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "ما هي الميزانية المخصصة لبناء محطة فضائية على كوكب المريخ؟",
       "retrieval_mode": "reranked"
     }'
```

**Response:**
```json
{
  "query": "ما هي الميزانية المخصصة لبناء محطة فضائية على كوكب المريخ؟",
  "language": "arabic",
  "retrieval_mode": "reranked",
  "answer": "عذراً، الوثائق المرفقة لا تحتوي على معلومات كافية للإجابة على هذا السؤال.",
  "citations": [],
  "is_refusal": true,
  "retrieved_count": 0,
  "latency_ms": 42.6
}
```

---

## Running Automated Tests

Execute the unit and integration test suite:

```bash
pytest tests/ -v
```

Tests cover:
- Language detection across diverse Arabic and English strings.
- Arabic punctuation-aware chunk boundary splitting.
- Chunk overlap and boundary constraint validation.
- FastAPI schema validation (rejection of malformed requests, length boundaries, enum constraints).
- Full LangGraph execution flow.
