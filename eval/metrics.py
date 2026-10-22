"""
Evaluation Metrics Calculator.
Computes Hit@k, MRR (Mean Reciprocal Rank), Answer Correctness,
and Unsupported Answer (Hallucination) Rate.
"""

import re
from typing import List, Dict, Any, Tuple
from src.retrieval.vector_store import RetrievedPassage


def is_chunk_relevant(passage: RetrievedPassage, expected_sources: List[Dict[str, Any]]) -> bool:
    """Checks whether a retrieved passage matches any of the expected sources."""
    if not expected_sources:
        return False

    for src in expected_sources:
        expected_file = src.get("filename", "").lower()
        expected_page = src.get("page")

        if passage.filename.lower() == expected_file:
            # If page is specified, match page or nearby page (+/- 1)
            if expected_page is not None:
                if abs(passage.page_number - expected_page) <= 1:
                    return True
            else:
                return True
    return False


def calculate_retrieval_metrics(
    eval_records: List[Dict[str, Any]],
    k_values: Tuple[int, ...] = (1, 3, 5)
) -> Dict[str, float]:
    """
    Computes Hit@k (for each k) and MRR across answerable benchmark queries.
    """
    answerable_records = [r for r in eval_records if r.get("is_answerable", True)]
    if not answerable_records:
        return {f"hit_rate@{k}": 0.0 for k in k_values} | {"mrr": 0.0}

    hit_counts = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for rec in answerable_records:
        passages: List[RetrievedPassage] = rec.get("retrieved_passages", [])
        expected = rec.get("expected_sources", [])

        # Find rank of first relevant passage
        first_rel_rank = None
        for idx, p in enumerate(passages, start=1):
            if is_chunk_relevant(p, expected):
                first_rel_rank = idx
                break

        # Calculate Reciprocal Rank
        if first_rel_rank is not None:
            reciprocal_ranks.append(1.0 / first_rel_rank)
        else:
            reciprocal_ranks.append(0.0)

        # Calculate Hit@k
        for k in k_values:
            if any(is_chunk_relevant(p, expected) for p in passages[:k]):
                hit_counts[k] += 1

    total = len(answerable_records)
    results = {
        f"hit_rate@{k}": round(hit_counts[k] / total, 4) for k in k_values
    }
    results["mrr"] = round(sum(reciprocal_ranks) / total, 4)
    results["sample_count"] = total

    return results


def token_f1_score(prediction: str, ground_truth: str) -> float:
    """Computes character/token overlap F1 score between prediction and ground truth."""
    pred_tokens = re.findall(r"\w+", prediction.lower())
    gt_tokens = re.findall(r"\w+", ground_truth.lower())

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0

    precision = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common) / len(pred_tokens)
    recall = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common) / len(gt_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def calculate_generation_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Computes:
    - Answer Correctness (token F1 / entity coverage against ground truth)
    - Rate of Unsupported Answers (hallucinations or failed refusals on unanswerable queries)
    """
    if not eval_records:
        return {"answer_correctness": 0.0, "unsupported_rate": 0.0}

    f1_scores = []
    unsupported_count = 0

    for rec in eval_records:
        prediction = rec.get("answer", "")
        ground_truth = rec.get("ground_truth_answer", "")
        is_answerable = rec.get("is_answerable", True)
        is_refusal = rec.get("is_refusal", False)

        # F1 score
        f1 = token_f1_score(prediction, ground_truth)
        f1_scores.append(f1)

        # Check unsupported answer logic:
        # Case A: Query is unanswerable, but system failed to refuse (hallucinated an answer)
        if not is_answerable:
            if not is_refusal:
                unsupported_count += 1
        # Case B: Query is answerable, but key entities are missing and answer hallucinated wrong facts
        else:
            key_entities = rec.get("key_entities", [])
            if key_entities:
                # Entity recall
                present = sum(1 for e in key_entities if e.lower() in prediction.lower())
                entity_recall = present / len(key_entities)
                # If entity recall is very low and it didn't refuse, likely hallucinated or irrelevant
                if entity_recall < 0.20 and not is_refusal:
                    unsupported_count += 1

    total = len(eval_records)
    avg_correctness = sum(f1_scores) / total if total > 0 else 0.0
    unsupported_rate = unsupported_count / total if total > 0 else 0.0

    return {
        "answer_correctness": round(avg_correctness, 4),
        "unsupported_rate": round(unsupported_rate, 4),
        "total_evaluated": total,
    }