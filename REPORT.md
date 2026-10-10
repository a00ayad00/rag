# Technical Report: Bilingual Document Q&A (RAG) System with Measured Quality
**Authors:** Senior AI Engineering & Research Team  
**Focus:** English & Arabic Report Corpora (20 Documents)  
**Stack:** FastAPI, LangChain, LangGraph, Qdrant, BGE-M3, Cross-Encoder Reranking

---

<!-- PAGE 1: SYSTEM DESIGN, INGESTION JUSTIFICATION & RETRIEVAL ARCHITECTURE -->

## 1. Executive Summary & Problem Formulation
Organizations operating in bilingual environments face significant challenges extracting factual insights from mixed Arabic and English regulatory, financial, and technical reports. Traditional monolingual RAG pipelines suffer from severe performance degradation when applied to Arabic text due to morphological complexity, character encoding artifacts, and poor cross-lingual semantic alignment.

This work presents an end-to-end Bilingual RAG architecture engineered to answer questions strictly in the language of the prompt, enforce exact citations (`[filename, Page X]`), and decisively refuse ungrounded queries. Using an experimental corpus of 20 PDF reports (10 English, 10 Arabic) and a 40-question gold evaluation benchmark, we quantitatively evaluate a dense retrieval baseline against an improved two-stage cross-encoder reranking pipeline.

```mermaid
flowchart LR
    A["User Query (EN/AR)"] --> B["Language Router & Normalizer"]
    B --> C["Qdrant Vector DB (Dense BGE-M3)"]
    C -->|"Top-15 Candidates"| D["Multilingual Cross-Encoder Reranker"]
    D -->|"Top-5 Chunks"| E["LangGraph Relevance Grader"]
    E -->|"Sufficient Evidence"| F["Grounded LLM Generator"]
    E -->|"Insufficient / Out-of-Scope"| G["Deterministic Refusal"]
    F --> H["Response with Exact Citations"]
    G --> H
```

---

## 2. Ingestion & Morphologically-Aware Chunking Justifications
Document ingestion of bilingual corpora cannot treat Arabic as equivalent to Latin-script text. Arabic is a synthetic, root-and-pattern language where proclitics (conjunctions like `و`, prepositions like `بـ`, `لـ`, and the definite article `الـ`) and enclitics (pronouns, gender/number suffixes) attach directly to the stem.

### Justifications for Ingestion & Chunking Parameters:
1. **Chunk Window Size (600 characters / ~90–120 words)**:
   - *Information Density*: A single Arabic word frequently encodes what requires 3 to 4 words in English (e.g., `وسيكتبونها` = "and they will write it"). An Arabic text chunk of 600 characters exhibits higher propositional density than an English chunk of identical length.
   - *Embedding Representation*: A 600-character window aligns cleanly with the dense representation capacity of `BAAI/bge-m3`, preventing multi-topic semantic dilution.
2. **Chunk Overlap (120 characters / ~20%)**:
   - Preserves numerical figures, monetary amounts, and multi-word named entities situated at chunk boundaries.
3. **Bilingual Boundary Delimiters**:
   - Standard sentence splitters only split on `. `, `?`, and `!`. Arabic narrative reports frequently chain clauses across long paragraphs using the coordinating conjunction `و` without full stops.
   - We implemented custom hierarchical separators incorporating Arabic punctuation:
     $$\text{Separators} = [\text{"\n\n"}, \text{"\n"}, \text{". "}, \text{"؟ "}, \text{"؛ "}, \text{"! "}, \text{"، "}, \text{" "}]$$
     where `،` (U+060C, Arabic comma) and `؛` (U+061B, Arabic semicolon) act as natural clause boundary delimiters.
4. **Unicode Normalization & BiDi Handling**:
   - Arabic PDF extractions frequently suffer from visual left-to-right (LTR) character stream reversals. We enforce Unicode NFKC normalization and bidirectional reshaping via `arabic-reshaper` and `python-bidi`, normalizing alef variants (`أ`, `إ`, `آ` $\rightarrow$ `ا`) for consistent indexing.

---

## 3. Retrieval Architecture: Baseline vs. One Improvement (Cross-Encoder Reranker)

### Baseline Pipeline:
- **Embedding Model**: `BAAI/bge-m3` (1024-dimensional multilingual bi-encoder, 8192 token context, supporting 100+ languages).
- **Vector Database**: Qdrant Vector DB with Cosine similarity metric.
- **Retrieval Mechanism**: Single-stage dense vector search retrieving Top-$k=5$ candidates.
- *Inherent Limitation*: Bi-encoders compress entire queries and passages into independent single-vector embeddings. This creates an information bottleneck that struggles with negation scope, subtle lexical distinctions, and cross-lingual score calibration.

### Improvement: Two-Stage Cross-Encoder Reranking:
- **Reranker Model**: `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder).
- **Mechanism**:
  1. Dense candidate retrieval fetches $K_{\text{cand}} = 15$ candidate passages from Qdrant.
  2. The Cross-Encoder evaluates the joint concatenated sequence `[CLS] Query [SEP] Passage [SEP]` across all transformer layers with full all-to-all cross-attention:
     $$\text{Score}(q, d) = \sigma(\mathbf{W} \cdot \text{Transformer}([q; d])_{[\text{CLS}]})$$
  3. Passages are sorted by relevance score, returning the top $K=5$ reranked passages.
- *Why Reranking over Hybrid / Query Rewriting*: Cross-encoders directly eliminate the bi-encoder independence assumption. In bilingual corpora, queries often contain synonyms or cross-lingual terms where sparse BM25 produces zero overlap. Joint cross-attention accurately ranks nuanced financial numbers and policy conditions at Rank 1.

---

<!-- PAGE 2: EVALUATION, FAILURE ANALYSIS, SERVICE & ROADMAP -->

## 4. Quantitative Evaluation & Benchmark Results
Evaluation was conducted on a curated gold benchmark of **40 bilingual questions**: 16 English answerable, 16 Arabic answerable, and 8 unanswerable questions (4 English, 4 Arabic) testing out-of-scope robustness.

### Retrieval & Generation Benchmark Comparison:

| Evaluation Metric | Baseline (Dense BGE-M3) | Improved (Cross-Encoder Reranked) | Absolute Delta ($\Delta$) | Relative Improvement |
| :--- | :---: | :---: | :---: | :---: |
| **Hit-Rate@1** | 0.7188 | **0.9062** | **+0.1874** | **+26.1%** |
| **Hit-Rate@3** | 0.8438 | **0.9688** | +0.1250 | +14.8% |
| **Hit-Rate@5** | 0.9062 | **1.0000** | +0.0938 | +10.3% |
| **Mean Reciprocal Rank (MRR)** | 0.7964 | **0.9427** | **+0.1463** | **+18.4%** |
| **Answer Correctness (Token F1)** | 0.7845 | **0.8920** | **+0.1075** | **+13.7%** |
| **Unsupported Answer Rate (Halluc.)**| 0.0750 | **0.0250** | **-0.0500** | **-66.7%** |

### Key Findings:
1. **MRR Surged to 0.9427**: The reranker moved the correct source passage to Rank 1 in over 90% of cases, directly improving generation quality by placing relevant context at the top of the LLM prompt.
2. **Hit@5 Reached 100%**: Across all 32 answerable queries, the reranked candidate set contained the ground truth passage.
3. **Hallucination Suppressed by 66.7%**: Strict refusal gating in LangGraph reduced the rate of unsupported answers from 7.5% to 2.5%, successfully refusing 100% of out-of-scope unanswerable queries.

---

## 5. Failure Case Diagnosis & Future Fixes (10 Analyzed Modes)

The following 10 failure modes were isolated during testing and assigned concrete engineering remediations:

1. **Orthographic Invariance**: Queries omitting Arabic hamzas (`الامن` vs `الأمن`) exhibited degraded bi-encoder similarity. *Fix*: Added NFKC unicode canonical decomposition and alef normalization.
2. **Cross-Lingual Metric Extraction**: English queries querying Arabic financial figures favored English reports. *Fix*: The cross-encoder resolved semantic equivalence across languages.
3. **Cross-Page Table Splitting**: Financial tables split across PDF page breaks led to missing EBITDA metrics. *Fix*: Expanded chunk overlap to 120 chars; planned structural table parsing.
4. **Negation Blindness**: Dense bi-encoder retrieved passages containing terms but asserting the exact opposite ("without approval"). *Fix*: Full-attention reranking correctly captured negation scope.
5. **Acronym Polysemy**: Ambiguity on 3-letter acronyms (`LPI` Logistics vs Linear Programming). *Fix*: Reranker rewarded contextual definitions; planned bilingual acronym expansion.
6. **Compound Clause Refusal**: Multi-part Arabic queries were falsely rejected if one sub-clause was missing. *Fix*: Refined prompts to deliver partial answers with explicit coverage caveats.
7. **Entity Overlap on Unanswerable Queries**: High term overlap on non-existent metrics ("crypto revenue in Q3") induced hallucinations. *Fix*: Zero temperature ($T=0.0$) and LangGraph citation verification.
8. **Reversed PDF Stream (BiDi)**: Arabic text extracted in reverse byte order produced gibberish. *Fix*: Applied PyMuPDF Unicode extraction with `arabic-reshaper` BiDi correction.
9. **Citation Header Drift**: When multiple consecutive chunks were combined, citations shifted by 1 page. *Fix*: Standardized delimited passage headers `--- [Passage X] Source: {file}, Page: {p} ---`.
10. **Code-Switching Language Drift**: Mixing English terms in an Arabic prompt caused responses to drift into English. *Fix*: Injected explicit language constraint enforcing output in prompt language.

---

## 6. Production Engineering & Roadmap

### Architecture & Service Delivery:
- **FastAPI Endpoint**: Fully typed Pydantic v2 schemas (`/api/v1/query`, `/api/v1/retrieve`, `/api/v1/health`), with request ID tracking and sub-millisecond middleware timing.
- **LangGraph State Graph**: Decoupled state machine separating language detection, retrieval, relevance grading, and generation.
- **Docker Orchestration**: Multi-stage `Dockerfile` (Python 3.11-slim) paired with `docker-compose.yml` deploying a self-healing Qdrant container with persistent volume storage.

### Strategic Roadmap:
1. **Document Layout Engine**: Migrate from raw PDF stream extraction to multimodal vision-based layout parsing (e.g., Docling / ColPali) to preserve multi-column tables.
2. **Hybrid Dense + Sparse Indexing**: Implement Qdrant native BM25 sparse vectors alongside BGE-M3 for hybrid lexical-semantic candidate generation.
3. **Self-RAG Reflection**: Add automated token-level reflection nodes to verify factuality before returning the response payload.
