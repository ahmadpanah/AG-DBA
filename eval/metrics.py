"""
eval/metrics.py
~~~~~~~~~~~~~~~
Evaluation metrics for AG-DBA benchmarks.

Paper Section 4.5: Evaluation Metrics
  - NIAH: Recall (binary hit/miss)
  - LongBench-E: F1, ROUGE, EM (task-specific, Table 2)
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


def compute_recall(generated: str, reference: str) -> float:
    """Compute recall for NIAH benchmark.
    
    Paper Section 4.5: "binary hit/miss that shows whether the exact 
    needle string was successfully generated"
    
    Args:
        generated: Generated text
        reference: Reference answer (needle)
    
    Returns:
        Recall: 1.0 if needle found, else 0.0
    """
    # Normalize whitespace and check for substring match
    return 1.0 if reference.lower() in generated.lower() else 0.0


def compute_f1(generated: str, reference: str) -> float:
    """Compute F1-score for QA tasks.
    
    Paper Section 4.5: Used for single_qa and multi_qa in LongBench-E.
    
    Simple implementation: word-level overlap.
    For production, use official SQuAD/QA evaluation scripts.
    
    Args:
        generated: Generated answer
        reference: Reference answer
    
    Returns:
        F1-score in [0, 1]
    """
    # Tokenize into words
    gen_tokens = set(re.findall(r"\w+", generated.lower()))
    ref_tokens = set(re.findall(r"\w+", reference.lower()))
    
    if len(gen_tokens) == 0 or len(ref_tokens) == 0:
        return 1.0 if gen_tokens == ref_tokens else 0.0
    
    # Precision and recall
    common = gen_tokens & ref_tokens
    precision = len(common) / len(gen_tokens) if gen_tokens else 0.0
    recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
    
    # F1
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return f1


def compute_rouge(generated: str, reference: str) -> float:
    """Compute ROUGE-L score for summarization tasks.
    
    Paper Section 4.5: Used for summarization in LongBench-E.
    
    Args:
        generated: Generated summary
        reference: Reference summary
    
    Returns:
        ROUGE-L F1 score in [0, 1]
    """
    if rouge_scorer is None:
        # Fallback: simple word overlap
        return compute_f1(generated, reference)
    
    try:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return scores["rougeL"].fmeasure
    except Exception:
        # Fallback on error
        return compute_f1(generated, reference)


def compute_exact_match(generated: str, reference: str) -> float:
    """Compute exact match for synthetic/coding tasks.
    
    Paper Section 4.5: Used for few_shot, synthetic, code in LongBench-E.
    
    Args:
        generated: Generated output
        reference: Reference output
    
    Returns:
        EM: 1.0 if exact match, else 0.0
    """
    # Normalize whitespace
    gen_norm = " ".join(generated.split()).strip()
    ref_norm = " ".join(reference.split()).strip()
    
    return 1.0 if gen_norm == ref_norm else 0.0


# Batch metric computation (for efficiency)

def compute_recall_batch(generated_list: list[str], reference_list: list[str]) -> list[float]:
    """Batch recall computation."""
    return [compute_recall(g, r) for g, r in zip(generated_list, reference_list)]


def compute_f1_batch(generated_list: list[str], reference_list: list[str]) -> list[float]:
    """Batch F1 computation."""
    return [compute_f1(g, r) for g, r in zip(generated_list, reference_list)]


def compute_rouge_batch(generated_list: list[str], reference_list: list[str]) -> list[float]:
    """Batch ROUGE computation."""
    return [compute_rouge(g, r) for g, r in zip(generated_list, reference_list)]


def compute_em_batch(generated_list: list[str], reference_list: list[str]) -> list[float]:
    """Batch EM computation."""
    return [compute_exact_match(g, r) for g, r in zip(generated_list, reference_list)]
