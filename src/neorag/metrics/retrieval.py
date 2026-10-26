"""Metric retrieval tính trên MỘT câu hỏi (relevance nhị phân).

`retrieved`: danh sách id đã xếp hạng (phần tử đầu là liên quan nhất).
`relevant`: tập id ground truth (khác rỗng).
"""

from __future__ import annotations

import math


def hit_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return float(any(d in relevant for d in retrieved[:k]))


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return sum(d in relevant for d in retrieved[:k]) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return sum(d in relevant for d in retrieved[:k]) / len(relevant)


def f1_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    p, r = precision_at_k(retrieved, relevant, k), recall_at_k(retrieved, relevant, k)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def average_precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """AP@k = (1 / min(|R|, k)) * sum_{i<=k} P@i * rel_i."""
    hits, total = 0, 0.0
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / min(len(relevant), k)


def reciprocal_rank_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    for i, d in enumerate(retrieved[:k], 1):
        if d in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(retrieved[:k], 1) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


RETRIEVAL_METRICS = {
    "hit": hit_at_k,
    "precision": precision_at_k,
    "recall": recall_at_k,
    "f1": f1_at_k,
    "map": average_precision_at_k,
    "mrr": reciprocal_rank_at_k,
    "ndcg": ndcg_at_k,
}


def score_query(retrieved: list[int], relevant: set[int], k_values: list[int]) -> dict[str, float]:
    """Trả về dict dạng {"recall@5": 0.5, ...} cho một câu hỏi."""
    return {
        f"{name}@{k}": fn(retrieved, relevant, k)
        for k in k_values
        for name, fn in RETRIEVAL_METRICS.items()
    }
