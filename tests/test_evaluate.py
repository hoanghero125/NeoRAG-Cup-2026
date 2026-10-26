import re
from dataclasses import replace

import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.config import load_config
from neorag.data import Prediction, validate_predictions
from neorag.evaluate import evaluate, to_markdown


def _judge(messages):
    system = messages[0]["content"]
    if "groundedness" in system or "tính đúng đắn" in system:
        answer_part = messages[1]["content"].split("CÂU TRẢ LỜI:")[-1]
        n = len(re.findall(r"^\d+\. ", answer_part, flags=re.MULTILINE))
        label = "supported" if "groundedness" in system else "correct"
        return "\n".join(f"{i}: {label}" for i in range(1, n + 1))
    if "câu hỏi tiếng Việt" in system:
        return "ProPTIT thành lập khi nào?\nNgày thành lập ProPTIT?\nProPTIT ra đời năm nào?"
    return "1"  # abstention


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def preds():
    return [
        Prediction("q1", [1, 2], "ProPTIT thành lập ngày 09/10/2011."),
        Prediction("q2", [1, 3], "Mình không chắc."),
        Prediction("q3", [3], "Tài liệu không có thông tin này."),
    ]


def test_validate_reports_problems(gold, corpus):
    bad = [Prediction("q1", [1, 1, 99], ""), Prediction("zzz", [1], "x")]
    errors = validate_predictions(
        bad, gold, {d.id for d in corpus}, max_retrieved=10, max_answer_chars=2000
    )
    text = "\n".join(errors)
    for expected in ["trùng", "không có trong corpus", "answer rỗng", "zzz", "Thiếu 2"]:
        assert expected in text


def test_evaluate_without_judge(gold, corpus, preds, cfg):
    report = evaluate(gold, preds, corpus, cfg, llm=None)
    assert report.n_answerable == 2
    assert report.metrics["hit@3"] == pytest.approx(0.5)  # q1 trúng, q2 trượt
    assert report.metrics["string_presence"] == pytest.approx(0.5)
    assert report.score is None and report.generation_score is None
    assert "abstention" not in report.metrics
    assert "Benchmark score: chưa tính" in to_markdown(report)


def test_unanswerable_with_related_docs_counts_for_retrieval(gold, corpus, preds, cfg):
    gold[2] = replace(gold[2], relevant_ids=[3])  # q3 không có đáp án, đáp án chuẩn nhắc tới đoạn 3
    report = evaluate(gold, preds, corpus, cfg, llm=None)
    assert report.metrics["hit@3"] == pytest.approx(2 / 3)  # q1 và q3 trúng, q2 trượt
    assert report.metrics["string_presence"] == pytest.approx(0.5)  # lexical vẫn chỉ tính q1, q2


def test_evaluate_with_fake_judge(gold, corpus, preds, cfg):
    report = evaluate(gold, preds, corpus, cfg, llm=FakeLLM(_judge), embedder=FakeEmbedder())
    m = report.metrics
    assert m["groundedness"] == pytest.approx(1.0)
    assert m["noise_sensitivity"] == pytest.approx(0.0)
    assert m["abstention"] == pytest.approx(1.0)
    assert 0.0 < m["response_relevancy"] <= 1.0
    assert report.score is not None and 0.0 < report.score <= 1.0


def test_evaluate_rejects_question_file_as_gold(gold, corpus, preds, cfg):
    no_answer = [type(q)(q.id, q.query) for q in gold]
    with pytest.raises(ValueError, match="ground truth"):
        evaluate(no_answer, preds, corpus, cfg)
