# Failure Analysis & Architectural Remediation Report

This document presents an in-depth diagnosis of **10 failure cases** observed during the evaluation of the Bilingual Document Q&A (RAG) system across English and Arabic reports. Each case details the input query, observed vs. expected behavior, underlying root cause, implemented short-term mitigation, and the planned architectural fix.

---

## Failure 1: Arabic Orthographic & Diacritic Mismatch (Hamza & Taa Marbuta)
- **Query (AR)**: `ما هي نسبة تطبيق ضوابط الامن السيبراني الاساسية؟` (Query typed without hamzas: `الامن`, `الاساسية`).
- **Context in PDF**: `تم تطبيق ضوابط الأمن السيبراني الأساسية بنسبة 96%` (Document text contains hamzas: `الأمن`, `الأساسية`).
- **Observed Behavior (Baseline)**: Dense bi-encoder yielded a lower cosine score ($0.61$), placing the target passage at rank 4 behind less relevant security overviews.
- **Root Cause**: Sub-word tokenizers (WordPiece / Byte-Pair) tokenize `الأمن` and `الامن` into distinct token IDs. In dense embeddings, while semantic proximity remains close, exact lexical matching is degraded without character-level orthographic normalization.
- **Implemented Fix**: Introduced NFKC unicode normalization and an Arabic orthographic preprocessor in `BilingualPDFLoader` and query preprocessing that normalizes alefs (`أ`, `إ`, `آ` $\rightarrow$ `ا`) and taa marbuta (`ة` $\rightarrow$ `ه`) for lexical matching.
- **Next Architectural Step**: Implement a hybrid sparse lexical index (Qdrant BM25 sparse vectors) paired with an Arabic-specific stemmer (such as Farasa or ISRI stemmer) to index normalized root forms.

---

## Failure 2: Cross-Lingual Metric Extraction (English Query over Arabic Financials)
- **Query (EN)**: `What was the non-oil GDP growth rate in the financial report?`
- **Context in PDF (AR)**: `سجل الناتج المحلي الإجمالي غير النفطي نمواً بنسبة 4.8% خلال العام المالي`
- **Observed Behavior (Baseline)**: Baseline retrieved general English macroeconomic sections from other reports rather than the specific Arabic report.
- **Root Cause**: Cross-lingual dense embeddings align high-level conceptual embeddings across languages well, but exact financial terminology (`non-oil GDP` vs `الناتج المحلي الإجمالي غير النفطي`) suffered from language bias towards same-language documents.
- **Implemented Fix**: The cross-encoder reranker (`BAAI/bge-reranker-v2-m3`) computes cross-attention across both languages, successfully boosting the Arabic target chunk from rank 8 to rank 1.
- **Next Architectural Step**: Implement automated cross-lingual query expansion (HyDE / Multi-Query Expansion) that generates parallel Arabic and English query formulations prior to vector search.

---

## Failure 3: Tabular Numerical Data Boundary Fragmentation
- **Query (EN)**: `What was the enterprise free cash flow and EBITDA margin in Q3?`
- **Context in PDF**: The financial table placed Revenue and EBITDA at the bottom of Page 1, while Free Cash Flow details were placed at the top of Page 2.
- **Observed Behavior**: The chunker split the text at the page boundary. The retriever returned Page 1, answering EBITDA margin ($25.0\%$) but refusing or omitting the Free Cash Flow figure ($54.6 million).
- **Root Cause**: Rigid page-by-page chunking isolated interconnected balance sheet metrics across page transitions.
- **Implemented Fix**: Increased chunk overlap to 120 characters (~20%) and allowed cross-page boundary sliding windows for consecutive pages within the same document.
- **Next Architectural Step**: Incorporate structural document layout parsing (e.g., PyMuPDF table extraction or Docling / Unstructured) to detect tables and retain them as atomic Markdown table chunks rather than raw linear text.

---

## Failure 4: Bi-Encoder False Positives on Negation Constraints
- **Query (EN)**: `Which third-party vendors were approved without privacy impact assessments?`
- **Context in PDF**: `100% of external third-party data processors underwent privacy impact assessments (PIA). Five vendor contracts were terminated due to inadequate data governance controls.`
- **Observed Behavior (Baseline)**: Bi-encoder retrieved the chunk with high cosine similarity ($0.82$) due to dense lexical overlap ("third-party vendors", "privacy impact assessments"), leading the model to initially misread the sentence.
- **Root Cause**: Bi-encoders compress documents into single vectors and struggle with semantic inversion, negative qualifiers, and syntactic scope.
- **Implemented Fix**: The Cross-Encoder reranker scores `[Query, Chunk]` jointly via bidirectional full-layer attention, correctly identifying that the document asserts the exact opposite (100% underwent assessments). LangGraph relevance grader verified lack of evidence for "approved without" and refused.
- **Next Architectural Step**: Add an explicit NLI (Natural Language Inference) entailment verification node in LangGraph before generation.

---

## Failure 5: Acronym Ambiguity Across Disparate Domains
- **Query (AR)**: `ما هي معايير مؤشر الأداء LPI المذكورة في التقرير؟`
- **Context in PDF**: `ارتفع مؤشر الأداء اللوجستي (LPI) 14 مرتبة عالمياً` (Logistics Performance Index).
- **Observed Behavior**: Retrieval returned passages discussing LP / LPI in IT network infrastructure and linear programming from unrelated reports.
- **Root Cause**: Acronyms with 3 letters have identical dense representations when isolated from full contextual definitions.
- **Implemented Fix**: Reranker penalized passages lacking the surrounding Arabic phrase `مؤشر الأداء اللوجستي`.
- **Next Architectural Step**: Maintain an automated domain-specific glossary/acronym lookup table that expands acronyms to their full bilingual definitions before vectorization.

---

## Failure 6: Premature Refusal on Multi-Clause Arabic Questions
- **Query (AR)**: `ما هي نسبة الخدمات الحكومية المحولة إلى منصات رقمية، وكم بلغ حجم الاستثمار في البنية التحتية، وما هي أهداف برنامج 2025؟`
- **Context in PDF**: `AR_Report_01_Digital_Transformation.pdf` contains all three facts, but across Page 1 and Page 3.
- **Observed Behavior**: When `top_k=2`, the context contained answers to only the first two parts. The LLM produced a full refusal rather than providing a partial answer with caveat.
- **Root Cause**: The strict refusal prompt ("if the context does not contain enough facts to directly answer the question, refuse") caused conservative models to reject compound questions when any single clause was missing.
- **Implemented Fix**: Refined system prompts to permit partial answers: *"If only a sub-part of the question is answered in the context, provide the verified facts with citations and explicitly state which sub-parts are not mentioned."*
- **Next Architectural Step**: Implement query decomposition in LangGraph: decompose multi-clause questions into sub-queries, execute sub-retrievals, and aggregate findings.

---

## Failure 7: Entity Overlap Leading to Hallucinations on Out-of-Scope Queries
- **Query (EN)**: `What was the enterprise's total revenue generated from consumer crypto trading services in Q3 2024?`
- **Context in PDF**: `EN_Report_02_Financial_Performance_Q3.pdf` discusses Q3 2024 revenues ($312.4M), enterprise software, and consumer subscriptions.
- **Observed Behavior (Without Strict Guardrails)**: Standard LLM hallucinated that consumer crypto revenue was part of the $82 million consumer subscription segment.
- **Root Cause**: High lexical overlap with valid document entities ("enterprise", "total revenue", "Q3 2024") coupled with parametric knowledge bias of generative LLMs.
- **Implemented Fix**: Enforced zero temperature ($T=0.0$), strict negative prompt constraints, and explicit refusal templates (`REFUSAL_ENGLISH` / `REFUSAL_ARABIC`). LangGraph citation validation flags claims without direct verbatim support.
- **Next Architectural Step**: Integrate Self-RAG reflection tokens and automated hallucination scoring via Ragas Faithfulness metric in the API response pipeline.

---

## Failure 8: Bidirectional (BiDi) Text & Reversal in Arabic PDFs
- **Query (AR)**: `كم يبلغ الإنتاج اليومي للمياه المحلاة؟`
- **Context in PDF**: Text rendered as reverse order `م3 موي/رتيم نويلم 6.2` by naive ASCII PDF extractors.
- **Observed Behavior**: Embedding search completely failed because reversed characters formed nonsensical strings.
- **Root Cause**: Some legacy PDF generators store Arabic characters visually left-to-right (LTR) rather than logical right-to-left (RTL) unicode sequences.
- **Implemented Fix**: Integrated `arabic-reshaper` and `python-bidi` with PyMuPDF logical extraction heuristics in `BilingualPDFLoader` to restore correct bidirectional text order before chunking.
- **Next Architectural Step**: Implement an automated BiDi sanity check that detects reversed text streams using word frequency dictionaries and auto-corrects them during ingestion.

---

## Failure 9: Citation Drift & Page Number Offset
- **Query (EN)**: `What was the reduction in loan processing time achieved by automated underwriting copilots?`
- **Context in PDF**: Fact located on physical PDF Page 2.
- **Observed Behavior**: Model answered correctly ("from 4 days to 4 hours") but cited `[EN_Report_01_AI_Strategy_2024.pdf, Page 1]`.
- **Root Cause**: When multiple consecutive chunks from the same document were concatenated in the prompt, the model attributed the quote to the header of the preceding chunk.
- **Implemented Fix**: Formatted prompt passages with clear delimited blocks: `--- [Passage X] --- Source: {filename}, Page: {page_number}` and instructed the model to attach citations immediately adjacent to each fact.
- **Next Architectural Step**: Implement programmatic post-hoc citation anchoring: locate the exact substring span in the retrieved chunks and automatically attach the verified ground-truth chunk's page number.

---

## Failure 10: Code-Switching & Language Drift
- **Query (AR)**: `What are the zero trust cybersecurity policies?` (Input in Arabic script mixed with English technical terms: `ما هي سياسات Zero Trust للأمن السيبراني؟`).
- **Context in PDF**: Contains both English and Arabic technical terms.
- **Observed Behavior**: The LLM replied primarily in English with occasional Arabic sentences, violating the strict requirement to answer in the language of the prompt.
- **Root Cause**: The presence of English keywords in the context biased the LLM's autoregressive generation towards English.
- **Implemented Fix**: Added prompt-level language assertion: `BilingualPromptBuilder.detect_language()` identifies the dominant query language and injects an explicit constraint (`"You MUST respond entirely in Arabic"`).
- **Next Architectural Step**: Add a downstream language validation check that verifies language agreement via `langdetect` and re-prompts the model if language drift is detected.

---

## Summary of Fixes & Strategic Roadmap

| Area | Immediate Fix | Strategic Next Step |
| :--- | :--- | :--- |
| **Ingestion** | Arabic NFKC normalization & BiDi correction | Multi-modal document layout parsing (tables, charts) |
| **Retrieval** | Two-stage Cross-Encoder Reranking (`bge-reranker-v2-m3`) | Hybrid Sparse-Dense Search (BM25 + BGE-M3) + HyDE |
| **Reasoning** | LangGraph relevance grading & refusal branch | Query decomposition for multi-hop questions |
| **Generation** | Strict zero-temperature prompts + delimited citation headers | Programmatic span-level citation verification & Self-RAG |
| **Evaluation** | 40-query gold benchmark with Hit@k, MRR & hallucination rate | Continuous CI/CD automated Ragas evaluation pipeline |
