"""Hiển thị trên terminal cho lệnh `neorag`: tiêu đề, các bước có spinner, thanh tiến độ, bảng."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

console = Console(highlight=False)

STEP_WIDTH = 44  # độ rộng cột tên bước, để thời gian các bước thẳng hàng


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}m{secs:02d}s"


def header(title: str, rows: list[tuple[str, str]]) -> None:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column()
    for key, value in rows:
        grid.add_row(key, value)
    console.print()
    console.print(
        Panel(
            grid,
            title=f"[bold]NeoRAG Cup 2026[/] · {title}",
            title_align="left",
            border_style="cyan",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=False,
        )
    )
    console.print()


class Step:
    """Một bước đang chạy. Gán `detail` để ghi thêm thông tin vào dòng kết quả."""

    def __init__(self, label: str):
        self.label = label
        self.detail = ""
        self.start = time.perf_counter()

    def done(self) -> None:
        elapsed = fmt_duration(time.perf_counter() - self.start)
        detail = f"  [dim]{self.detail}[/]" if self.detail else ""
        console.print(f" [green]✓[/] {self.label:<{STEP_WIDTH}} [dim]{elapsed:>7}[/]{detail}")

    def fail(self) -> None:
        console.print(f" [red]✗[/] {self.label}")


@contextmanager
def step(label: str, spinner: bool = True) -> Iterator[Step]:
    """Chạy một bước: hiện spinner khi đang chạy, in ✓ kèm thời gian khi xong, ✗ khi lỗi.

    `spinner=False` khi bên trong tự in ra màn hình (ví dụ thanh tải model của Hugging Face).
    """
    s = Step(label)
    try:
        if spinner:
            with console.status(f"[cyan]{label}...[/]", spinner="dots"):
                yield s
        else:
            console.print(f" [cyan]↓[/] {label}...")
            yield s
    except BaseException:
        s.fail()
        raise
    s.done()


def progress() -> Progress:
    """Thanh tiến độ dùng chung; tự biến mất khi xong để chỉ còn dòng ✓."""
    return Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("{task.description}"),
        BarColumn(bar_width=30, complete_style="cyan", finished_style="green"),
        MofNCompleteColumn(),
        TextColumn("[dim]•[/]"),
        TimeElapsedColumn(),
        TextColumn("[dim]• ETA[/]"),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    )


def note(text: str) -> None:
    console.print(f"   [dim]{text}[/]")


def warn(text: str) -> None:
    console.print(f"   [yellow]![/] [yellow]{text}[/]")


def error(title: str, message: str, hint: str | None = None) -> None:
    body = message if not hint else f"{message}\n\n[dim]{hint}[/]"
    console.print()
    console.print(
        Panel(body, title=f"[bold red]{title}[/]", title_align="left", border_style="red")
    )
