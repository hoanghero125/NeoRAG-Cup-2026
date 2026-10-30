import pytest

from conftest import FakeEmbedder, FakeLLM
from neorag.data import DATA_DIR, Query, load_corpus, load_queries
from neorag.pipelines import RAGPipeline, load_pipeline
from neorag.pipelines.baseline import BaselinePipeline
from neorag.predict import predict


def test_baseline_runs_end_to_end(corpus):
    pipeline = BaselinePipeline(
        top_k=2, llm=FakeLLM(lambda m: "Ngày 09/10/2011."), embedder=FakeEmbedder()
    )
    pipeline.prepare()
    pipeline.index(corpus)
    preds = predict(pipeline, [Query("q1", "ProPTIT thành lập ngày nào?")], max_retrieved=10)
    assert preds[0].answer == "Ngày 09/10/2011."
    assert len(preds[0].retrieved_ids) == 3  # trả về tối đa n_retrieved (corpus chỉ có 3 doc)
    assert preds[0].retrieved_ids[0] == 1  # doc ProPTIT gần câu hỏi nhất
    assert preds[0].latency_ms is not None


def test_load_pipeline_validates_spec():
    with pytest.raises(ValueError):
        load_pipeline("neorag.pipelines.baseline")
    with pytest.raises(TypeError):
        load_pipeline("neorag.data:Document")
    assert issubclass(BaselinePipeline, RAGPipeline)


def test_shipped_data_is_consistent():
    corpus = load_corpus()
    ids = {d.id for d in corpus}
    assert len(ids) == len(corpus)
    for q in load_queries(DATA_DIR / "train" / "questions.jsonl"):
        assert set(q.relevant_ids) <= ids
        if q.answerable:
            assert q.relevant_ids, q.id


def test_load_pipeline_finds_modules_in_current_directory(tmp_path, monkeypatch):
    pkg = tmp_path / "my_solution_cwd"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "pipeline.py").write_text(
        "from neorag.pipelines import RAGOutput, RAGPipeline\n"
        "class P(RAGPipeline):\n"
        "    def index(self, corpus): pass\n"
        "    def run(self, query): return RAGOutput('a', [1])\n"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.path", [p for p in __import__("sys").path if p not in ("", ".")])
    assert isinstance(load_pipeline("my_solution_cwd.pipeline:P"), RAGPipeline)


def test_predict_with_workers_keeps_question_order(gold):
    from neorag.pipelines import RAGOutput
    from neorag.predict import predict

    class Echo(RAGPipeline):
        def index(self, corpus):
            pass

        def run(self, query):
            return RAGOutput(answer=query, retrieved_ids=[1])

    preds = predict(Echo(), gold, max_retrieved=10, workers=3)
    assert [p.id for p in preds] == [q.id for q in gold]
    assert [p.answer for p in preds] == [q.query for q in gold]
