"""
Evaluation Runner for Bilingual RAG Pipeline.
Benchmarks Baseline (Dense BGE-M3) vs Improved (Cross-Encoder Reranked).
Calculates Hit@1, Hit@3, Hit@5, MRR, Answer Correctness, and Unsupported Answer Rate.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.config import settings
from src.retrieval.vector_store import VectorStoreManager
from src.retrieval.baseline import BaselineRetriever
from src.retrieval.reranker import MultilingualReranker, RerankedRetriever
from src.generation.graph import build_rag_graph
from src.generation.llm import get_llm
from eval.metrics import calculate_retrieval_metrics, calculate_generation_metrics

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


def load_gold_dataset(path: Path) -> List[Dict[str, Any]]:
    """Loads gold benchmark questions from JSON."""
    if not path.exists():
        raise FileNotFoundError(f"Gold dataset not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_mode(
    dataset: List[Dict[str, Any]],
    retrieval_mode: str,
    rag_graph,
    top_k: int = 5
) -> Dict[str, Any]:
    """Runs full evaluation on the dataset for a specific retrieval mode."""
    logger.info(f"--- Starting evaluation for mode: {retrieval_mode.upper()} ---")

    records: List[Dict[str, Any]] = []

    for item in dataset:
        qid = item["id"]
        q_text = item["question"]
        is_ans = item["is_answerable"]
        expected_src = item.get("expected_sources", [])

        # Invoke LangGraph
        init_state = {
            "question": q_text,
            "retrieval_mode": retrieval_mode,
            "top_k": top_k,
        }

        try:
            output = rag_graph.invoke(init_state)
            retrieved = output.get("retrieved_passages", [])
            answer = output.get("answer", "")
            citations = output.get("citations", [])
            is_refusal = output.get("is_refusal", False)
        except Exception as e:
            logger.error(f"Error executing query {qid}: {e}")
            retrieved = []
            answer = "Error during execution."
            citations = []
            is_refusal = False

        record = {
            "id": qid,
            "question": q_text,
            "language": item.get("language", "english"),
            "is_answerable": is_ans,
            "expected_sources": expected_src,
            "ground_truth_answer": item.get("ground_truth_answer", ""),
            "key_entities": item.get("key_entities", []),
            "retrieved_passages": retrieved,
            "answer": answer,
            "citations": citations,
            "is_refusal": is_refusal,
        }
        records.append(record)

    # Compute metrics
    retrieval_metrics = calculate_retrieval_metrics(records, k_values=(1, 3, 5))
    generation_metrics = calculate_generation_metrics(records)

    return {
        "mode": retrieval_mode,
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
        "records": records,
    }


def run_full_evaluation(output_file: Path = Path("eval/eval_results.json")) -> Dict[str, Any]:
    """
    Executes comparative evaluation of Baseline vs Improved Reranked.
    """
    gold_path = settings.BASE_DIR / "eval" / "gold_dataset.json"
    dataset = load_gold_dataset(gold_path)
    logger.info(f"Loaded {len(dataset)} benchmark queries from {gold_path.name}")

    # Initialize components
    vector_store = VectorStoreManager()
    baseline = BaselineRetriever(vector_store=vector_store)
    reranker = MultilingualReranker()
    reranked = RerankedRetriever(baseline_retriever=baseline, reranker=reranker)
    llm = get_llm()

    rag_graph = build_rag_graph(
        baseline_retriever=baseline,
        reranked_retriever=reranked,
        llm=llm
    )

    # 1. Run Baseline
    baseline_results = evaluate_mode(dataset, retrieval_mode="baseline", rag_graph=rag_graph)

    # 2. Run Improved (Reranked)
    reranked_results = evaluate_mode(dataset, retrieval_mode="reranked", rag_graph=rag_graph)

    summary = {
        "baseline": {
            "hit_rate@1": baseline_results["retrieval"]["hit_rate@1"],
            "hit_rate@3": baseline_results["retrieval"]["hit_rate@3"],
            "hit_rate@5": baseline_results["retrieval"]["hit_rate@5"],
            "mrr": baseline_results["retrieval"]["mrr"],
            "answer_correctness": baseline_results["generation"]["answer_correctness"],
            "unsupported_rate": baseline_results["generation"]["unsupported_rate"],
        },
        "improved_reranked": {
            "hit_rate@1": reranked_results["retrieval"]["hit_rate@1"],
            "hit_rate@3": reranked_results["retrieval"]["hit_rate@3"],
            "hit_rate@5": reranked_results["retrieval"]["hit_rate@5"],
            "mrr": reranked_results["retrieval"]["mrr"],
            "answer_correctness": reranked_results["generation"]["answer_correctness"],
            "unsupported_rate": reranked_results["generation"]["unsupported_rate"],
        },
        "improvements_delta": {
            "hit_rate@1_diff": round(
                reranked_results["retrieval"]["hit_rate@1"] - baseline_results["retrieval"]["hit_rate@1"], 4
            ),
            "hit_rate@5_diff": round(
                reranked_results["retrieval"]["hit_rate@5"] - baseline_results["retrieval"]["hit_rate@5"], 4
            ),
            "mrr_diff": round(
                reranked_results["retrieval"]["mrr"] - baseline_results["retrieval"]["mrr"], 4
            ),
            "answer_correctness_diff": round(
                reranked_results["generation"]["answer_correctness"] - baseline_results["generation"]["answer_correctness"], 4
            ),
            "unsupported_rate_reduction": round(
                baseline_results["generation"]["unsupported_rate"] - reranked_results["generation"]["unsupported_rate"], 4
            ),
        }
    }

    # Print clean formatted markdown table
    print("\n" + "=" * 80)
    print("                     BILINGUAL RAG EVALUATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"{'Metric':<28} | {'Baseline (Dense BGE-M3)':<24} | {'Improved (Reranked)':<20} | {'Delta':<10}")
    print("-" * 88)
    print(f"{'Hit-Rate@1':<28} | {summary['baseline']['hit_rate@1']:<24.4f} | {summary['improved_reranked']['hit_rate@1']:<20.4f} | {summary['improvements_delta']['hit_rate@1_diff']:+<10.4f}")
    print(f"{'Hit-Rate@3':<28} | {summary['baseline']['hit_rate@3']:<24.4f} | {summary['improved_reranked']['hit_rate@3']:<20.4f} | {'--':<10}")
    print(f"{'Hit-Rate@5':<28} | {summary['baseline']['hit_rate@5']:<24.4f} | {summary['improved_reranked']['hit_rate@5']:<20.4f} | {summary['improvements_delta']['hit_rate@5_diff']:+<10.4f}")
    print(f"{'Mean Reciprocal Rank (MRR)':<28} | {summary['baseline']['mrr']:<24.4f} | {summary['improved_reranked']['mrr']:<20.4f} | {summary['improvements_delta']['mrr_diff']:+<10.4f}")
    print(f"{'Answer Correctness (F1/Overlap)':<28} | {summary['baseline']['answer_correctness']:<24.4f} | {summary['improved_reranked']['answer_correctness']:<20.4f} | {summary['improvements_delta']['answer_correctness_diff']:+<10.4f}")
    print(f"{'Unsupported Answer Rate (Halluc.)':<28} | {summary['baseline']['unsupported_rate']:<24.4f} | {summary['improved_reranked']['unsupported_rate']:<20.4f} | {-summary['improvements_delta']['unsupported_rate_reduction']:+<10.4f}")
    print("=" * 88 + "\n")

    output_path = settings.BASE_DIR / output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Strip large raw records before dumping
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation summary saved to: {output_path}")

    return summary


if __name__ == "__main__":
    run_full_evaluation()
