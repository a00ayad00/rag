"""
LangGraph State Machine for Bilingual RAG Execution.
Implements modular graph workflow: Language Detection -> Retrieval (Baseline / Reranked)
-> Relevance Grading -> Grounded Generation -> Citation Extraction & Validation.
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from src.retrieval.vector_store import RetrievedPassage
from src.retrieval.baseline import BaselineRetriever
from src.retrieval.reranker import RerankedRetriever
from src.generation.prompts import (
    BilingualPromptBuilder,
    REFUSAL_ENGLISH,
    REFUSAL_ARABIC,
)
from src.generation.llm import get_llm
from src.config import settings

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    """LangGraph execution state."""
    question: str
    language: str
    retrieval_mode: str
    top_k: int
    retrieved_passages: List[RetrievedPassage]
    context: str
    system_prompt: str
    user_prompt: str
    answer: str
    citations: List[str]
    is_refusal: bool
    error: Optional[str]


def detect_language_step(state: RAGState) -> Dict[str, Any]:
    """Detects input question language."""
    lang = BilingualPromptBuilder.detect_language(state["question"])
    return {"language": lang}


def retrieve_step(
    state: RAGState,
    baseline_retriever: BaselineRetriever,
    reranked_retriever: RerankedRetriever
) -> Dict[str, Any]:
    """Retrieves context passages using baseline or reranked retriever."""
    mode = state.get("retrieval_mode", "reranked")
    k = state.get("top_k", settings.RETRIEVAL_TOP_K)

    try:
        if mode == "baseline":
            passages = baseline_retriever.retrieve(query=state["question"], top_k=k)
        else:
            passages = reranked_retriever.retrieve(query=state["question"], top_k=k)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        passages = []

    context_str = BilingualPromptBuilder.format_context(passages)
    return {
        "retrieved_passages": passages,
        "context": context_str
    }


def grade_relevance_step(state: RAGState) -> Dict[str, Any]:
    """
    Grades retrieved passages.
    If no passages are retrieved or max similarity is zero, routes to refusal.
    """
    passages = state.get("retrieved_passages", [])
    if not passages:
        lang = state.get("language", "english")
        refusal = REFUSAL_ARABIC if lang == "arabic" else REFUSAL_ENGLISH
        return {
            "answer": refusal,
            "citations": [],
            "is_refusal": True
        }

    sys_prompt, user_prompt, _ = BilingualPromptBuilder.build_prompt_messages(
        question=state["question"],
        passages=passages
    )

    return {
        "system_prompt": sys_prompt,
        "user_prompt": user_prompt,
        "is_refusal": False
    }


def generate_step(state: RAGState, llm) -> Dict[str, Any]:
    """Generates grounded answer using configured LLM."""
    if state.get("is_refusal", False):
        return {}

    sys_prompt = state.get("system_prompt", "")
    user_prompt = state.get("user_prompt", "")

    messages = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        response = llm.invoke(messages)
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Primary LLM generation failed ({e}). Attempting fallback to Gemini...")
        try:
            from src.generation.llm import get_llm
            fallback_llm = get_llm(provider="gemini")
            response = fallback_llm.invoke(messages)
            answer_text = response.content if hasattr(response, "content") else str(response)
            logger.info("Successfully recovered generation using Gemini fallback.")
        except Exception as fb_err:
            logger.error(f"Fallback generation also failed ({fb_err}). Defaulting to refusal.")
            lang = state.get("language", "english")
            answer_text = REFUSAL_ARABIC if lang == "arabic" else REFUSAL_ENGLISH

    return {"answer": answer_text}


def validate_citations_step(state: RAGState) -> Dict[str, Any]:
    """Extracts citations from generated text and validates refusal flags."""
    answer = state.get("answer", "")
    lang = state.get("language", "english")

    refusal_phrase = REFUSAL_ARABIC if lang == "arabic" else REFUSAL_ENGLISH
    is_refusal = (refusal_phrase in answer) or (
        "لا تحتوي على معلومات كافية" in answer
        or "do not contain sufficient information" in answer
    )

    citations = BilingualPromptBuilder.extract_citations(answer)

    # If answer did not cite explicitly but we have retrieved passages,
    # collect top retrieved source as citation metadata if not a refusal
    if not citations and not is_refusal and state.get("retrieved_passages"):
        top_p = state["retrieved_passages"][0]
        citations = [f"{top_p.filename}, Page {top_p.page_number}"]

    return {
        "citations": citations,
        "is_refusal": is_refusal
    }


def should_generate(state: RAGState) -> str:
    """Conditional edge: skips LLM generation if already refused during grading."""
    return "validate_citations" if state.get("is_refusal") else "generate"


def build_rag_graph(
    baseline_retriever: Optional[BaselineRetriever] = None,
    reranked_retriever: Optional[RerankedRetriever] = None,
    llm=None
):
    """Constructs and compiles the executable LangGraph workflow."""
    baseline = baseline_retriever or BaselineRetriever()
    reranked = reranked_retriever or RerankedRetriever(baseline_retriever=baseline)
    model = llm or get_llm()

    workflow = StateGraph(RAGState)

    # Add Nodes
    workflow.add_node("detect_language", detect_language_step)
    workflow.add_node("retrieve", lambda s: retrieve_step(s, baseline, reranked))
    workflow.add_node("grade_relevance", grade_relevance_step)
    workflow.add_node("generate", lambda s: generate_step(s, model))
    workflow.add_node("validate_citations", validate_citations_step)

    # Set Entry Point
    workflow.set_entry_point("detect_language")

    # Add Edges
    workflow.add_edge("detect_language", "retrieve")
    workflow.add_edge("retrieve", "grade_relevance")
    workflow.add_conditional_edges(
        "grade_relevance",
        should_generate,
        {
            "generate": "generate",
            "validate_citations": "validate_citations"
        }
    )
    workflow.add_edge("generate", "validate_citations")
    workflow.add_edge("validate_citations", END)

    return workflow.compile()