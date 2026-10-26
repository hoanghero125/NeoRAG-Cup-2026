import math

import pytest

from neorag.metrics.retrieval import (
    average_precision_at_k,
    f1_at_k,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)

RETRIEVED = [5, 1, 9, 2, 7]
RELEVANT = {1, 2, 3}


def test_hit():
    assert hit_at_k(RETRIEVED, RELEVANT, 1) == 0.0
    assert hit_at_k(RETRIEVED, RELEVANT, 2) == 1.0


def test_precision_recall_f1():
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 3)
    assert f1_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 * 0.4 * (2 / 3) / (0.4 + 2 / 3))
    assert f1_at_k([9], RELEVANT, 1) == 0.0


def test_precision_counts_short_lists_against_k():
    # Nộp ít hơn k doc vẫn chia cho k, không được lợi khi nộp thiếu.
    assert precision_at_k([1], RELEVANT, 5) == pytest.approx(1 / 5)


def test_average_precision():
    # hits tại vị trí 2 và 4: (1/2 + 2/4) / min(3, 5)
    assert average_precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx((0.5 + 0.5) / 3)
    assert average_precision_at_k([1, 2, 3], RELEVANT, 3) == pytest.approx(1.0)


def test_mrr():
    assert reciprocal_rank_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(0.5)
    assert reciprocal_rank_at_k([9, 8], RELEVANT, 2) == 0.0


def test_ndcg():
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([1, 2, 3], RELEVANT, 3) == pytest.approx(1.0)
