import json

import pytest

from conftest import FakeEmbedder
from neorag import cli, embeddings
from neorag.data import DATA_DIR
from neorag.llm import LLMClient


def _fake_embed(self, texts, model):
    return FakeEmbedder().embed(texts).tolist()


class _FakeModel:
    device = "cpu"

    def encode(self, texts, **kwargs):
        return FakeEmbedder().embed(texts)


def _fake_chat(self, messages, model, temperature=None):
    system = messages[0]["content"]
    if "<số thứ tự>: <nhãn>" in system:
        return "1: supported"
    if "câu hỏi tiếng Việt" in system:
        return "Câu hỏi?"
    return "1"


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(embeddings, "_load", lambda name: _FakeModel())
    monkeypatch.setattr(embeddings, "is_downloaded", lambda name: True)
    monkeypatch.setattr(LLMClient, "chat", _fake_chat)
    monkeypatch.setattr(LLMClient, "embed", _fake_embed)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_predict_then_evaluate(offline, capsys):
    assert cli.main(["predict", "train"]) == 0
    out = capsys.readouterr().out
    assert "Trả lời 126 câu hỏi" in out
    assert "neorag evaluate train" in out  # gợi ý lệnh ngắn
    assert (offline / "outputs/train_pred.jsonl").exists()

    assert cli.main(["validate"]) == 0  # mặc định là train
    assert cli.main(["evaluate", "train"]) == 0
    assert "Benchmark" in capsys.readouterr().out
    report_path = offline / "outputs/eval_train_pred/report.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["score"] is not None


def test_long_form_still_works(offline):
    train = str(DATA_DIR / "train" / "questions.jsonl")
    assert cli.main(["predict", "--queries", train, "--out", "pred.jsonl"]) == 0
    assert cli.main(["evaluate", "--gold", train, "--predictions", "pred.jsonl", "--no-judge"]) == 0


def test_errors_are_reported_not_raised(offline, capsys):
    code = cli.main(["validate", "missing.jsonl"])
    assert code == 1
    assert "Lỗi" in capsys.readouterr().out


def test_install_script_picks_cuda_build_not_newer_than_driver():
    import importlib.util

    from neorag.data import DATA_DIR as data_dir

    path = data_dir.parent / "scripts" / "install.py"
    spec = importlib.util.spec_from_file_location("install", path)
    install = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(install)
    assert install.compatible_tags((12, 8))[0] == "cu128"
    assert install.compatible_tags((12, 4)) == ["cu124", "cu121", "cu118"]
    assert install.compatible_tags((11, 0)) == []


def test_warns_when_gpu_present_but_torch_is_cpu(offline, capsys, monkeypatch):
    from neorag.pipelines import baseline

    # pipeline dùng embedding chạy trên máy (LocalEmbedder) thay cho gateway
    monkeypatch.setattr(
        baseline, "GatewayEmbedder", lambda llm, name: embeddings.LocalEmbedder(name)
    )
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    assert cli.main(["predict", "train"]) == 0
    assert "python scripts/install.py" in capsys.readouterr().out


def test_split_name_resolves_split_folder(offline, monkeypatch):
    folder = offline / "data" / "public_test"
    folder.mkdir(parents=True)
    (offline / "data" / "corpus.jsonl").write_bytes((DATA_DIR / "corpus.jsonl").read_bytes())
    rows = [
        json.loads(line) for line in (DATA_DIR / "train" / "questions.jsonl").open(encoding="utf-8")
    ][:3]
    gold = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (folder / "gold.jsonl").write_text(gold, encoding="utf-8")
    questions = "".join(json.dumps({"id": r["id"], "query": r["query"]}) + "\n" for r in rows)
    (folder / "questions.jsonl").write_text(questions, encoding="utf-8")
    monkeypatch.setattr(cli, "DATA_DIR", offline / "data")

    assert cli.main(["predict", "public_test"]) == 0
    assert (offline / "outputs/public_test_pred.jsonl").exists()
    assert cli.main(["evaluate", "public_test", "--no-judge"]) == 0
    assert (offline / "outputs/eval_public_test_pred/report.json").exists()


def test_run_does_predict_and_evaluate(offline, capsys):
    assert cli.main(["run", "train", "--no-judge"]) == 0
    assert (offline / "outputs/train_pred.jsonl").exists()
    assert (offline / "outputs/eval_train_pred/report.json").exists()
    assert "Retrieval" in capsys.readouterr().out


def test_predict_writes_submission_zip(offline):
    import zipfile

    assert cli.main(["predict", "train"]) == 0
    with zipfile.ZipFile(offline / "outputs/train_submission.zip") as z:
        assert z.namelist() == ["train_pred.jsonl"]
