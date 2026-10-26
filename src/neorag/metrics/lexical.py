"""Metric generation không cần LLM: String Presence, ROUGE-L, BLEU-4 (trên một câu hỏi)."""

from __future__ import annotations

import math
import re
from collections import Counter

from neorag.text import normalize, tokenize


def _contains(text: str, phrase: str) -> bool:
    """`phrase` xuất hiện trong `text` và không dính liền chữ/số hai bên ("3 tháng" ≠ "13 tháng")."""
    return re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) is not None


def string_presence(answer: str, key_facts: list[str]) -> float:
    """Tỷ lệ key fact xuất hiện trong câu trả lời (so khớp chuỗi sau khi chuẩn hóa).

    Một key fact có thể có nhiều cách viết, phân tách bằng " | ", ví dụ "09/10/2011 | 9/10/2011".
    """
    if not key_facts:
        return 0.0
    text = normalize(answer)
    found = sum(
        any(_contains(text, normalize(alt)) for alt in fact.split(" | ") if alt.strip())
        for fact in key_facts
    )
    return found / len(key_facts)


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def rouge_l(answer: str, reference: str) -> float:
    """ROUGE-L F1 trên token âm tiết."""
    cand, ref = tokenize(answer), tokenize(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_length(cand, ref)
    if lcs == 0:
        return 0.0
    p, r = lcs / len(cand), lcs / len(ref)
    return 2 * p * r / (p + r)


def bleu4(answer: str, reference: str, epsilon: float = 0.1) -> float:
    """Sentence BLEU-4 (trọng số đều 1-4 gram), làm trơn bằng epsilon cho n-gram bằng 0."""
    cand, ref = tokenize(answer), tokenize(reference)
    if not cand or not ref:
        return 0.0
    log_precision = 0.0
    for n in range(1, 5):
        cand_ngrams = Counter(tuple(cand[i : i + n]) for i in range(len(cand) - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        total = max(sum(cand_ngrams.values()), 1)
        match = sum(min(c, ref_ngrams[g]) for g, c in cand_ngrams.items())
        log_precision += math.log((match or epsilon) / total) / 4
    brevity = 1.0 if len(cand) > len(ref) else math.exp(1 - len(ref) / len(cand))
    return brevity * math.exp(log_precision)
