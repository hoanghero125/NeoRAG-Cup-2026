"""Đọc file cấu hình TOML."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(path: str | Path = CONFIG_DIR / "eval.toml") -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)
