"""Dòng lệnh `neorag`: predict, validate, evaluate."""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
import zipfile
from pathlib import Path
from statistics import median

from rich import box
from rich.columns import Columns
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from neorag.config import CONFIG_DIR, load_config
from neorag.data import (
    DATA_DIR,
    load_corpus,
    load_predictions,
    load_queries,
    save_predictions,
    validate_predictions,
)
from neorag.ui import (
    STEP_WIDTH,
    console,
    error,
    fmt_duration,
    header,
    note,
    progress,
    step,
    warn,
)

MAX_ERRORS_SHOWN = 15


def _path(p: str | Path) -> str:
    """Đường dẫn ngắn gọn: tương đối với thư mục hiện tại nếu được."""
    path = Path(p)
    with contextlib.suppress(ValueError):
        path = path.resolve().relative_to(Path.cwd().resolve())
    return escape(str(path))


def _gateway() -> str:
    return escape(os.getenv("OPENAI_BASE_URL") or "api.openai.com (mặc định)")


def _show_errors(errors: list[str]) -> None:
    for e in errors[:MAX_ERRORS_SHOWN]:
        console.print(f"   [red]•[/] {escape(e)}")
    if len(errors) > MAX_ERRORS_SHOWN:
        note(f"... và {len(errors) - MAX_ERRORS_SHOWN} lỗi khác")


def _check_gpu(device: str) -> None:
    from neorag.embeddings import gpu_unused

    if gpu_unused(device):
        warn("Máy có GPU NVIDIA nhưng PyTorch đang là bản CPU nên model chạy trên CPU.")
        warn("Cài lại để dùng GPU: python scripts/install.py --local")


# ---------------------------------------------------------------- predict


def _cmd_predict(args: argparse.Namespace) -> int:
    from neorag.embeddings import LocalEmbedder, is_downloaded
    from neorag.pipelines.base import load_pipeline
    from neorag.predict import predict

    started = time.perf_counter()
    cfg = load_config(args.config)
    corpus = load_corpus(args.corpus)
    queries = load_queries(args.queries)

    header(
        "Predict",
        [
            ("Câu hỏi", f"{_path(args.queries)}  [dim]({len(queries)} câu)[/]"),
            ("Corpus", f"{_path(args.corpus)}  [dim]({len(corpus)} passages)[/]"),
            ("Pipeline", escape(args.pipeline)),
            ("Gateway", _gateway()),
            ("Output", _path(args.out)),
        ],
    )

    with step("Khởi tạo pipeline") as s:
        pipeline = load_pipeline(args.pipeline)
        if model := getattr(pipeline, "llm_model", None):
            s.detail = f"LLM: {model}"

    embedder = getattr(pipeline, "embedder", None)
    if isinstance(embedder, LocalEmbedder):
        downloading = not is_downloaded(embedder.model_name)
        label = f"Nạp embedding model {embedder.model_name}"
        if downloading:
            label = f"Tải embedding model {embedder.model_name} (lần đầu)"
        with step(label, spinner=not downloading) as s:
            pipeline.prepare()
            s.detail = f"device: {embedder.device}"
        _check_gpu(embedder.device)
    else:
        with step("Chuẩn bị pipeline"):
            pipeline.prepare()

    with step(f"Index {len(corpus)} passages"):
        pipeline.index(corpus)

    t0 = time.perf_counter()
    with progress() as bar:
        task = bar.add_task("Trả lời câu hỏi", total=len(queries))
        preds = predict(
            pipeline,
            queries,
            cfg["submission"]["max_retrieved"],
            on_step=lambda _: bar.advance(task),
        )
    _done_line(f"Trả lời {len(queries)} câu hỏi", time.perf_counter() - t0)

    with step("Ghi file và kiểm tra định dạng") as s:
        save_predictions(args.out, preds)
        errors = validate_predictions(
            load_predictions(args.out), queries, {d.id for d in corpus}, **cfg["submission"]
        )
        s.detail = "hợp lệ" if not errors else f"{len(errors)} lỗi"
        if not errors:
            _write_submission_zip(Path(args.out))

    latencies = sorted(p.latency_ms for p in preds)
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    summary = Table.grid(padding=(0, 2))
    summary.add_column(style="dim")
    summary.add_column()
    summary.add_row("Đã ghi", f"{len(preds)} predictions → [bold]{_path(args.out)}[/]")
    summary.add_row(
        "Latency",
        f"median {fmt_duration(median(latencies) / 1000)} · "
        f"p95 {fmt_duration(p95 / 1000)} · max {fmt_duration(latencies[-1] / 1000)}",
    )
    summary.add_row("Tổng thời gian", fmt_duration(time.perf_counter() - started))
    console.print()
    console.print(summary)

    if errors:
        console.print()
        console.print("[red]File chưa hợp lệ để nộp:[/]")
        _show_errors(errors)
        return 1

    console.print()
    if getattr(args, "then_evaluate", False):
        return 0  # `neorag run` tự chuyển sang bước chấm điểm
    has_gold = (DATA_DIR / args.split / "gold.jsonl").exists()
    if has_gold or all(q.has_ground_truth for q in queries):
        console.print("[dim]Bước tiếp theo, chấm điểm:[/]")
        default_out = Path("outputs") / f"{_split_name(args.split)}_pred.jsonl"
        command = f"neorag evaluate {escape(args.split)}"
        if Path(args.out) != default_out:
            command += f" --predictions {_path(args.out)}"
        console.print(f"  {command}")
    else:
        console.print(
            f"[dim]File hợp lệ. Nộp[/] {_path(_submission_zip(Path(args.out)))} "
            "[dim]lên trang thi (docs/submission.md)[/]"
        )
    console.print()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """predict rồi evaluate ngay nếu có đáp án."""
    predict_args = argparse.Namespace(**vars(args), then_evaluate=True)
    if (code := _cmd_predict(predict_args)) != 0:
        return code
    gold_path = Path(args.gold)
    if not gold_path.exists() or not all(q.has_ground_truth for q in load_queries(gold_path)):
        console.print(
            f"[dim]Không có đáp án cho {escape(args.split)} nên bỏ qua bước chấm điểm. "
            f"Nộp file[/] {_path(_submission_zip(Path(args.out)))} [dim]lên trang thi.[/]"
        )
        console.print()
        return 0
    eval_args = argparse.Namespace(**vars(args))
    eval_args.predictions, eval_args.out = args.out, None  # --out của run là file predictions
    return _cmd_evaluate(eval_args)


def _submission_zip(predictions: Path) -> Path:
    """outputs/public_test_pred.jsonl -> outputs/public_test_submission.zip"""
    stem = predictions.stem.removesuffix("_pred")
    return predictions.with_name(f"{stem}_submission.zip")


def _write_submission_zip(predictions: Path) -> None:
    """Codabench nhận bài nộp dạng .zip: nén sẵn file predictions để thí sinh chỉ việc upload."""
    with zipfile.ZipFile(_submission_zip(predictions), "w", zipfile.ZIP_DEFLATED) as z:
        z.write(predictions, predictions.name)


def _done_line(label: str, seconds: float) -> None:
    console.print(f" [green]✓[/] {label:<{STEP_WIDTH}} [dim]{fmt_duration(seconds):>7}[/]")


# ---------------------------------------------------------------- validate


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    queries = load_queries(args.queries)
    preds = load_predictions(args.predictions)
    errors = validate_predictions(
        preds, queries, {d.id for d in load_corpus(args.corpus)}, **cfg["submission"]
    )
    console.print()
    if errors:
        console.print(f" [red]✗[/] {_path(args.predictions)} [red]không hợp lệ[/]")
        _show_errors(errors)
        console.print()
        return 1
    console.print(
        f" [green]✓[/] {_path(args.predictions)} hợp lệ "
        f"[dim]({len(preds)} predictions, khớp {len(queries)} câu hỏi)[/]. Có thể nộp."
    )
    console.print()
    return 0


# ---------------------------------------------------------------- evaluate

_RETRIEVAL_COLS = [
    ("hit", "Hit"),
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("map", "MAP"),
    ("mrr", "MRR"),
    ("ndcg", "NDCG"),
]
_GENERATION_ROWS = [
    ("string_presence", "String Presence", "↑", "key facts có trong câu trả lời"),
    ("rouge_l", "ROUGE-L", "↑", "giống đáp án chuẩn"),
    ("bleu_4", "BLEU-4", "↑", "chỉ để tham khảo"),
    ("groundedness", "Groundedness", "↑", "bám vào tài liệu đã truy xuất"),
    ("response_relevancy", "Response Relevancy", "↑", "đúng trọng tâm câu hỏi"),
    ("noise_sensitivity", "Noise Sensitivity", "↓", "tỉ lệ câu sai so với đáp án"),
    ("abstention", "Abstention", "↑", "từ chối đúng câu không có đáp án"),
]


def _show_report(report, cfg: dict) -> None:
    m = report.metrics
    s = cfg["score"]
    rank_k = s["rank_k"]
    used_retrieval = set(s["retrieval_metrics"])
    used_generation = set(s["generation_metrics"]) | (
        {"abstention"} if s["abstention_in_generation"] else set()
    )

    retrieval = Table(
        title="Retrieval",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        caption=f"[dim]* dùng tính điểm, tại k = {rank_k}[/]",
        caption_justify="left",
    )
    retrieval.add_column("k", justify="right", style="dim")
    for key, name in _RETRIEVAL_COLS:
        retrieval.add_column(f"{name}{' *' if key in used_retrieval else ''}", justify="right")
    for k in report.k_values:
        style = "bold" if k == rank_k else None
        retrieval.add_row(
            str(k), *(f"{m[f'{key}@{k}']:.4f}" for key, _ in _RETRIEVAL_COLS), style=style
        )

    generation = Table(
        title="Generation",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAD,
        caption="[dim]* dùng tính điểm · ↑ càng cao càng tốt, ↓ càng thấp càng tốt[/]",
        caption_justify="left",
    )
    generation.add_column("Metric")
    generation.add_column("Score", justify="right")
    generation.add_column("", justify="center")
    generation.add_column("Ý nghĩa", style="dim")
    n_unanswerable = report.n_queries - report.n_answerable
    for key, name, direction, meaning in _GENERATION_ROWS:
        if key == "abstention":
            meaning = f"{meaning} ({n_unanswerable} câu)"
        if key in m:
            mark = " *" if key in used_generation else ""
            generation.add_row(f"{name}{mark}", f"{m[key]:.4f}", direction, meaning)

    console.print()
    console.print(retrieval)
    console.print()
    console.print(generation)
    console.print()

    def card(title: str, value: float | None, highlight: bool = False) -> Panel:
        text = f"[bold]{value:.4f}[/]" if value is not None else "[dim]—[/]"
        return Panel(
            text,
            title=title,
            title_align="left",
            border_style="green" if highlight else "dim",
            width=24,
            padding=(0, 2),
        )

    cards = [
        card("Retrieval", report.retrieval_score),
        card("Generation", report.generation_score),
        card("Benchmark", report.score, highlight=True),
    ]
    console.print(Columns(cards))
    if report.score is None:
        note("Chưa có điểm Benchmark vì chạy với --no-judge. Bỏ cờ này để chấm đầy đủ.")
    else:
        note(
            f"Benchmark = {s['retrieval_weight']} × Retrieval + "
            f"{s['generation_weight']} × Generation"
        )


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from neorag.evaluate import evaluate, save_report
    from neorag.llm import LLMClient

    cfg = load_config(args.config)
    jcfg = cfg["judge"]
    gold = load_queries(args.gold)
    n_answerable = sum(q.answerable for q in gold)
    judge_info = (
        "[dim]tắt (--no-judge)[/]"
        if args.no_judge
        else f"{escape(jcfg['chat_model'])} [dim]qua[/] "
        f"{escape(os.getenv('JUDGE_BASE_URL') or '') or _gateway()}"
    )
    header(
        "Evaluate",
        [
            (
                "Ground truth",
                f"{_path(args.gold)}  [dim]({len(gold)} câu, "
                f"{len(gold) - n_answerable} không có đáp án)[/]",
            ),
            ("Predictions", _path(args.predictions)),
            ("Judge", judge_info),
        ]
        + ([] if args.no_judge else [("Embedding", escape(jcfg["embedding_model"]))]),
    )

    with step("Đọc file và kiểm tra định dạng"):
        predictions = load_predictions(args.predictions)
        corpus = load_corpus(args.corpus)

    llm = None if args.no_judge else LLMClient.for_judge(cache_path=jcfg["cache_path"])

    t0 = time.perf_counter()
    with progress() as bar:
        task = bar.add_task("Chấm điểm", total=len(gold))
        report = evaluate(
            gold,
            predictions,
            corpus,
            cfg,
            llm,
            on_step=lambda: bar.advance(task),
        )
    _done_line(f"Chấm {len(gold)} câu hỏi", time.perf_counter() - t0)

    out_dir = Path(args.out or Path("outputs") / f"eval_{Path(args.predictions).stem}")
    save_report(report, out_dir)

    _show_report(report, cfg)
    console.print()
    console.print(
        f"[dim]Báo cáo chi tiết:[/] {_path(out_dir)}  "
        "[dim](report.md · report.json · per_query.jsonl)[/]"
    )
    console.print()
    return 0


# ---------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neorag", description="NeoRAG Cup 2026 toolkit")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default=DATA_DIR / "corpus.jsonl", help="file corpus")
    common.add_argument("--config", default=CONFIG_DIR / "eval.toml", help="file cấu hình")
    sub = parser.add_subparsers(dest="command", required=True)

    split = argparse.ArgumentParser(add_help=False)
    split.add_argument(
        "split",
        nargs="?",
        default="train",
        help="bộ câu hỏi: train, public_test, private_test (hoặc đường dẫn .jsonl); mặc định train",
    )

    p = sub.add_parser("predict", parents=[common, split], help="chạy pipeline, ghi file nộp bài")
    p.add_argument("--queries", help="file câu hỏi (mặc định data/<split>/questions.jsonl)")
    p.add_argument(
        "--pipeline",
        default="neorag.pipelines.baseline:BaselinePipeline",
        help="pipeline dạng module.path:ClassName",
    )
    p.add_argument("--out", help="file predictions (mặc định outputs/<split>_pred.jsonl)")
    p.set_defaults(func=_cmd_predict)

    v = sub.add_parser("validate", parents=[common, split], help="kiểm tra định dạng file nộp bài")
    v.add_argument("--queries", help="file câu hỏi (mặc định data/<split>/questions.jsonl)")
    v.add_argument("--predictions", help="file nộp (mặc định outputs/<split>_pred.jsonl)")
    v.set_defaults(func=_cmd_validate)

    e = sub.add_parser("evaluate", parents=[common, split], help="chấm điểm file predictions")
    e.add_argument(
        "--gold",
        help="file có ground truth (mặc định data/<split>/gold.jsonl nếu có, không thì questions.jsonl)",
    )
    e.add_argument("--predictions", help="file predictions (mặc định outputs/<split>_pred.jsonl)")
    e.add_argument("--out", default=None, help="thư mục lưu báo cáo")
    e.add_argument(
        "--no-judge", action="store_true", help="bỏ qua metric cần LLM judge (nhanh, không tốn API)"
    )
    e.set_defaults(func=_cmd_evaluate)

    r = sub.add_parser(
        "run", parents=[common, split], help="predict rồi evaluate luôn (nếu có đáp án)"
    )
    r.add_argument("--queries", help="file câu hỏi (mặc định data/<split>/questions.jsonl)")
    r.add_argument("--gold", help="file đáp án (mặc định tự tìm theo <split>)")
    r.add_argument(
        "--pipeline",
        default="neorag.pipelines.baseline:BaselinePipeline",
        help="pipeline dạng module.path:ClassName",
    )
    r.add_argument("--out", help="file predictions (mặc định outputs/<split>_pred.jsonl)")
    r.add_argument(
        "--no-judge", action="store_true", help="bỏ qua metric cần LLM judge (nhanh, không tốn API)"
    )
    r.set_defaults(func=_cmd_run)
    return parser


def _first_existing(*paths: Path) -> Path:
    return next((p for p in paths if p.exists()), paths[0])


def _split_name(split: str) -> str:
    """public_test -> public_test; data/public_test/questions.jsonl -> public_test; my.jsonl -> my"""
    path = Path(split)
    if path.suffix != ".jsonl":
        return split
    return path.parent.name if path.name in ("questions.jsonl", "gold.jsonl") else path.stem


def _fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Điền đường dẫn mặc định từ tên bộ câu hỏi: `neorag evaluate train` thay cho lệnh dài."""
    split = args.split
    if split.endswith(".jsonl"):
        questions = gold = Path(split)
    else:
        # data/<bộ>/questions.jsonl: câu hỏi; data/<bộ>/gold.jsonl: đáp án Public/Private (chỉ BTC có).
        # Tập train có đáp án ngay trong questions.jsonl.
        questions = DATA_DIR / split / "questions.jsonl"
        gold = _first_existing(DATA_DIR / split / "gold.jsonl", questions)
    predictions = Path("outputs") / f"{_split_name(split)}_pred.jsonl"
    for name, default in (
        ("queries", questions),
        ("gold", gold),
        ("out", predictions),
        ("predictions", predictions),
    ):
        if name == "out" and args.command == "evaluate":
            continue  # với evaluate, --out là thư mục báo cáo
        if hasattr(args, name) and getattr(args, name) is None:
            setattr(args, name, default)
    return args


def _api_error(e: Exception) -> tuple[str, str, str | None]:
    """Đổi lỗi từ gateway thành thông báo dễ hiểu: (tiêu đề, nội dung, gợi ý)."""
    import openai

    body = getattr(e, "body", None)
    detail = body.get("message", str(e)) if isinstance(body, dict) else str(e)
    detail = escape(detail.split("\n\n")[0][:400])
    if isinstance(e, openai.APIConnectionError):
        return (
            "Không kết nối được tới gateway",
            f"Không gọi được {_gateway()}.",
            "Kiểm tra kết nối mạng và OPENAI_BASE_URL trong file .env.",
        )
    if isinstance(e, openai.AuthenticationError):
        return (
            "Key không hợp lệ",
            detail,
            "Kiểm tra OPENAI_API_KEY trong file .env (dán đúng key BTC gửi, không có dấu cách).",
        )
    if "budget" in detail.lower():
        return ("Hết budget", detail, "Key của bạn đã dùng hết budget $5. Liên hệ BTC.")
    if isinstance(e, openai.RateLimitError):
        return (
            "Vượt rate limit (429)",
            detail,
            "Đợi khoảng một phút rồi chạy lại. Kết quả đã gọi được lưu trong cache.",
        )
    status = getattr(e, "status_code", None)
    if status is None:  # lỗi phía client, thường là thiếu key
        return (
            "Chưa cấu hình API key",
            "Không tìm thấy OPENAI_API_KEY.",
            "Tạo file .env từ .env.example (Windows: copy .env.example .env) rồi điền key BTC gửi.",
        )
    return (f"Gateway trả về lỗi {status}", detail, None)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # tiếng Việt trên console Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv()
    args = _fill_defaults(build_parser().parse_args(argv))
    try:
        return args.func(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Đã dừng.[/] Kết quả gọi LLM đã có vẫn nằm trong cache.")
        return 130
    except (ValueError, FileNotFoundError) as e:
        error("Lỗi", escape(str(e)))
        return 1
    except Exception as e:
        import openai

        if isinstance(e, openai.OpenAIError):
            error(*_api_error(e))
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())
