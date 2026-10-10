"""Evaluation module for Bilingual Document Q&A."""
from eval.metrics import calculate_retrieval_metrics, calculate_generation_metrics
from eval.evaluate import run_full_evaluation

__all__ = ["calculate_retrieval_metrics", "calculate_generation_metrics", "run_full_evaluation"]
