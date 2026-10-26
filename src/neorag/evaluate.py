"""Bộ chấm chính thức: so file predictions với ground truth và tính toàn bộ metric.

Bộ chấm chỉ đọc file. Nó không gọi code pipeline của thí sinh, nên kết quả không phụ thuộc
vào việc thí sinh sửa gì trong repo của mình.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING

from neorag.data import Document, Prediction, Query, validate_predictions, write_jsonl
from neorag.metrics import lexical
from neorag.metrics.judge import GatewayEmbedder, Judge
from neorag.metrics.retrieval import RETRIEVAL_METRICS, score_query

if TYPE_CHECKING:  # bộ chấm chỉ cần thư viện chuẩn của Python (chạy được trên Codabench)
    from neorag.llm import LLMClient

LEXICAL = ["string_presence", "rouge_l", "bleu_4"]
JUDGED = ["groundedness", "response_relevancy", "noise_sensitivity"]


@dataclass
class EvalReport:
    n_queries: int
    n_answerable: int
    k_values: list[int]
    metrics: dict[str, float]
    retrieval_score: float | None
    generation_score: float | None
    score: float | None
    judge_model: str | None
    per_query: list[dict] = field(repr=False, default_factory=list)

    def summary(self) -> dict:
        data = {k: v for k, v in self.__dict__.items() if k != "per_query"}
        data["metrics"] = {k: round(v, 4) for k, v in self.metrics.items()}
        for key in ("retrieval_score", "generation_score", "score"):
            if data[key] is not None:
                data[key] = round(data[key], 4)
        return data


def _score_one(
    q: Query,
    p: Prediction,
    corpus: dict[int, Document],
    k_values: list[int],
    judge: Judge | None,
) -> dict:
    row: dict = {"id": q.id, "answerable": q.answerable}

    if not q.answerable:
        if q.relevant_ids:  # đoạn chứa thông tin liên quan mà đáp án chuẩn nhắc tới
            row.update(score_query(p.retrieved_ids, set(q.relevant_ids), k_values))
        if judge:
            row["abstention"] = judge.abstention(q.query, p.answer)
        return row

    row.update(score_query(p.retrieved_ids, set(q.relevant_ids), k_values))
    row["string_presence"] = lexical.string_presence(p.answer, q.key_facts)
    row["rouge_l"] = lexical.rouge_l(p.answer, q.answer)
    row["bleu_4"] = lexical.bleu4(p.answer, q.answer)

    if judge:
        contexts = [corpus[i].text for i in p.retrieved_ids]
        gold = [corpus[i].text for i in q.relevant_ids]
        row["groundedness"] = judge.groundedness(q.query, p.answer, contexts)
        row["noise_sensitivity"] = judge.noise_sensitivity(q.query, p.answer, q.answer, gold)
        row["response_relevancy"] = judge.response_relevancy(q.query, p.answer)
    return row


def _check_gold(gold: list[Query], corpus_ids: set[int]) -> None:
    problems = []
    for q in gold:
        if not q.has_ground_truth:
            problems.append(f"{q.id}: thiếu answer (bạn có đang dùng file chỉ có câu hỏi không?)")
        elif q.answerable and not q.relevant_ids:
            problems.append(f"{q.id}: answerable nhưng relevant_ids rỗng")
        elif not set(q.relevant_ids) <= corpus_ids:
            problems.append(f"{q.id}: relevant_ids không có trong corpus")
    if problems:
        raise ValueError("File gold (ground truth) không hợp lệ:\n- " + "\n- ".join(problems))


def _aggregate(rows: list[dict]) -> dict[str, float]:
    keys = sorted({k for r in rows for k in r} - {"id", "answerable"})
    return {k: mean(r[k] for r in rows if k in r) for k in keys}


def _composite(metrics: dict[str, float], cfg: dict, judged: bool):
    s = cfg["score"]
    retrieval_parts = [metrics.get(f"{m}@{s['rank_k']}") for m in s["retrieval_metrics"]]
    if None in retrieval_parts:
        raise ValueError("Bộ gold không có câu hỏi nào có đáp án để tính retrieval.")
    retrieval_score = mean(retrieval_parts)
    if not judged:
        return retrieval_score, None, None
    parts = [
        1 - metrics[m] if m == "noise_sensitivity" else metrics[m] for m in s["generation_metrics"]
    ]
    if s["abstention_in_generation"] and "abstention" in metrics:
        parts.append(metrics["abstention"])
    generation_score = mean(parts)
    score = s["retrieval_weight"] * retrieval_score + s["generation_weight"] * generation_score
    return retrieval_score, generation_score, score


def evaluate(
    gold: list[Query],
    predictions: list[Prediction],
    corpus: list[Document],
    cfg: dict,
    llm: LLMClient | None = None,
    embedder: GatewayEmbedder | None = None,
    on_step: Callable[[], None] | None = None,
) -> EvalReport:
    """Chấm điểm. `llm=None` bỏ qua các metric cần judge (nhanh, miễn phí, điểm tổng = None)."""
    corpus_ids = {d.id for d in corpus}
    _check_gold(gold, corpus_ids)
    errors = validate_predictions(predictions, gold, corpus_ids, **cfg["submission"])
    if errors:
        raise ValueError("File predictions không hợp lệ:\n- " + "\n- ".join(errors))

    jcfg = cfg["judge"]
    judge = None
    if llm:
        judge = Judge(
            llm,
            chat_model=jcfg["chat_model"],
            embedder=embedder or GatewayEmbedder(llm, jcfg["embedding_model"]),
            temperature=jcfg.get("temperature"),
            relevancy_questions=jcfg["relevancy_questions"],
        )
    by_id = {p.id: p for p in predictions}
    corpus_by_id = {d.id: d for d in corpus}
    k_values = cfg["retrieval"]["k_values"]
    with ThreadPoolExecutor(max_workers=jcfg["max_workers"] if judge else 1) as pool:
        jobs = [
            pool.submit(_score_one, q, by_id[q.id], corpus_by_id, k_values, judge) for q in gold
        ]
        if on_step:
            for job in jobs:
                job.add_done_callback(lambda _: on_step())
        rows = [j.result() for j in jobs]

    metrics = _aggregate(rows)
    retrieval_score, generation_score, score = _composite(metrics, cfg, judged=llm is not None)
    return EvalReport(
        n_queries=len(gold),
        n_answerable=sum(q.answerable for q in gold),
        k_values=cfg["retrieval"]["k_values"],
        metrics=metrics,
        retrieval_score=retrieval_score,
        generation_score=generation_score,
        score=score,
        judge_model=cfg["judge"]["chat_model"] if llm else None,
        per_query=rows,
    )


def to_markdown(report: EvalReport) -> str:
    m = report.metrics
    names = list(RETRIEVAL_METRICS)
    lines = [
        f"**{report.n_queries} câu hỏi** ({report.n_answerable} có đáp án) · "
        f"judge: `{report.judge_model or 'không dùng'}`",
        "",
        "### Retrieval",
        "",
        "| K | " + " | ".join(f"{n}@k" for n in names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for k in report.k_values:
        lines.append(f"| {k} | " + " | ".join(f"{m[f'{n}@{k}']:.4f}" for n in names) + " |")

    gen = [n for n in [*LEXICAL, *JUDGED, "abstention"] if n in m]
    lines += [
        "",
        "### Generation",
        "",
        "| " + " | ".join(gen) + " |",
        "|" + "---|" * len(gen),
        "| " + " | ".join(f"{m[n]:.4f}" for n in gen) + " |",
        "",
        "### Điểm tổng hợp",
        "",
        f"- Retrieval score: **{report.retrieval_score:.4f}**",
    ]
    if report.score is None:
        lines.append("- Generation score / Benchmark score: chưa tính (chạy không có judge)")
    else:
        lines.append(f"- Generation score: **{report.generation_score:.4f}**")
        lines.append(f"- Benchmark score: **{report.score:.4f}**")
    return "\n".join(lines) + "\n"


def save_report(report: EvalReport, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(
        json.dumps(report.summary(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    write_jsonl(out / "per_query.jsonl", report.per_query)
