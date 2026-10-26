"""Chạy pipeline trên một bộ câu hỏi và ghi file predictions đúng định dạng nộp bài."""

from __future__ import annotations

import time
from collections.abc import Callable

from neorag.data import Prediction, Query
from neorag.pipelines.base import RAGPipeline


def predict(
    pipeline: RAGPipeline,
    queries: list[Query],
    max_retrieved: int,
    on_step: Callable[[Prediction], None] | None = None,
) -> list[Prediction]:
    """Trả lời từng câu hỏi. Pipeline phải được `index` trước khi gọi hàm này."""
    predictions = []
    for q in queries:
        start = time.perf_counter()
        out = pipeline.run(q.query)
        latency_ms = (time.perf_counter() - start) * 1000
        pred = Prediction(
            id=q.id,
            retrieved_ids=list(dict.fromkeys(int(i) for i in out.retrieved_ids))[:max_retrieved],
            answer=out.answer,
            latency_ms=round(latency_ms, 1),
        )
        predictions.append(pred)
        if on_step:
            on_step(pred)
    return predictions
