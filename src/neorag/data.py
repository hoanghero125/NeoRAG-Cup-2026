"""Định dạng dữ liệu của cuộc thi và các hàm đọc/ghi JSONL."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Document:
    """Một đoạn văn trong corpus. `id` là định danh dùng trong mọi file ground truth."""

    id: int
    club: str
    topic: str
    text: str


@dataclass(frozen=True)
class Query:
    """Một câu hỏi. Các trường ground truth chỉ có ở tập train và ở file đáp án BTC giữ."""

    id: str
    query: str
    relevant_ids: list[int] = field(default_factory=list)
    answer: str = ""
    key_facts: list[str] = field(default_factory=list)
    answerable: bool = True
    difficulty: int | None = None
    club: str = ""

    @property
    def has_ground_truth(self) -> bool:
        return bool(self.answer)


@dataclass
class Prediction:
    """Một dòng trong file nộp bài."""

    id: str
    retrieved_ids: list[int]
    answer: str
    latency_ms: float | None = None


def read_jsonl(path: str | Path) -> list[dict]:
    """Đọc JSONL (chấp nhận file có BOM). Mỗi dòng phải là một object JSON."""
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: JSON không hợp lệ ({e.msg})") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: mỗi dòng phải là một object JSON")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _known_fields(cls, row: dict) -> dict:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in row.items() if k in names}


def load_corpus(path: str | Path = DATA_DIR / "corpus.jsonl") -> list[Document]:
    return [Document(**_known_fields(Document, row)) for row in read_jsonl(path)]


def load_queries(path: str | Path) -> list[Query]:
    return [Query(**_known_fields(Query, row)) for row in read_jsonl(path)]


def load_predictions(path: str | Path) -> list[Prediction]:
    """Đọc file nộp bài. Bỏ qua trường thừa; báo lỗi rõ ràng nếu thiếu trường hoặc sai kiểu."""
    predictions, errors = [], []
    for n, row in enumerate(read_jsonl(path), 1):
        problem = None
        if not isinstance(row.get("id"), str):
            problem = "thiếu 'id' hoặc 'id' không phải chuỗi"
        elif not isinstance(row.get("retrieved_ids"), list):
            problem = "thiếu 'retrieved_ids' hoặc không phải list"
        elif not isinstance(row.get("answer"), str):
            problem = "thiếu 'answer' hoặc 'answer' không phải chuỗi"
        if problem:
            errors.append(f"dòng {n}: {problem}")
            continue
        predictions.append(Prediction(**_known_fields(Prediction, row)))
    if errors:
        raise ValueError(f"{path} sai định dạng:\n- " + "\n- ".join(errors[:20]))
    return predictions


def save_predictions(path: str | Path, predictions: Iterable[Prediction]) -> None:
    write_jsonl(path, (asdict(p) for p in predictions))


def validate_predictions(
    predictions: list[Prediction],
    queries: list[Query],
    corpus_ids: set[int],
    max_retrieved: int,
    max_answer_chars: int,
) -> list[str]:
    """Kiểm tra nội dung file nộp bài. Trả về danh sách lỗi (rỗng nghĩa là hợp lệ)."""
    errors: list[str] = []
    expected = {q.id for q in queries}
    seen: set[str] = set()

    for p in predictions:
        where = f"[id={p.id}]"
        if p.id in seen:
            errors.append(f"{where} bị trùng.")
        seen.add(p.id)
        if p.id not in expected:
            errors.append(f"{where} không có trong bộ câu hỏi.")
        if not all(isinstance(i, int) and not isinstance(i, bool) for i in p.retrieved_ids):
            errors.append(f"{where} retrieved_ids phải là list số nguyên.")
            continue
        if not 1 <= len(p.retrieved_ids) <= max_retrieved:
            errors.append(f"{where} retrieved_ids phải có từ 1 đến {max_retrieved} phần tử.")
        if len(set(p.retrieved_ids)) != len(p.retrieved_ids):
            errors.append(f"{where} retrieved_ids có phần tử trùng.")
        unknown = [i for i in p.retrieved_ids if i not in corpus_ids]
        if unknown:
            errors.append(f"{where} retrieved_ids không có trong corpus: {unknown}.")
        if not p.answer.strip():
            errors.append(f"{where} answer rỗng.")
        elif len(p.answer) > max_answer_chars:
            errors.append(f"{where} answer dài {len(p.answer)} ký tự, tối đa {max_answer_chars}.")

    missing = sorted(expected - seen)
    if missing:
        preview = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
        errors.append(f"Thiếu {len(missing)} câu hỏi: {preview}")
    return errors
