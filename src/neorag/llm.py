"""Client LLM dùng chuẩn OpenAI-compatible, kèm cache SQLite tùy chọn.

Trong cuộc thi, client trỏ tới gateway của BTC (https://proptit.dekthe.dev/v1),
nơi phục vụ model `claude-haiku-5-5` (chat) và `bge-m3` (embedding, dùng cho judge).
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


class _Cache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (key, value))
            self._conn.commit()


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


class LLMClient:
    """Bọc `openai.OpenAI` cho thao tác chat và embedding."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_path: str | Path | None = None,
        max_retries: int = 5,
        timeout: float = 60.0,
    ):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL") or None,
            max_retries=max_retries,
            timeout=timeout,
        )
        self._cache = _Cache(cache_path) if cache_path else None
        self._endpoint = str(
            self._client.base_url
        )  # đưa vào khóa cache: đổi provider thì không dùng lại

    @classmethod
    def for_judge(cls, cache_path: str | Path | None = None) -> LLMClient:
        """Client cho bộ chấm: ưu tiên JUDGE_*, nếu trống thì dùng OPENAI_*."""
        return cls(
            api_key=os.getenv("JUDGE_API_KEY") or None,
            base_url=os.getenv("JUDGE_BASE_URL") or None,
            cache_path=cache_path,
        )

    def chat(self, messages: list[dict], model: str, temperature: float | None = None) -> str:
        """`temperature=None` thì không gửi tham số này (claude-haiku-5-5 chỉ nhận giá trị mặc định)."""
        key = _key("chat", self._endpoint, model, temperature, messages)
        if self._cache and (hit := self._cache.get(key)) is not None:
            return hit
        extra = {} if temperature is None else {"temperature": temperature}
        resp = self._client.chat.completions.create(model=model, messages=messages, **extra)
        text = (resp.choices[0].message.content or "").strip()
        if self._cache:
            self._cache.set(key, text)
        return text

    def embed(self, texts: list[str], model: str, batch_size: int = 32) -> list[list[float]]:
        if len(texts) > batch_size:
            return [
                v
                for i in range(0, len(texts), batch_size)
                for v in self.embed(texts[i : i + batch_size], model, batch_size)
            ]
        key = _key("embed", self._endpoint, model, texts)
        if self._cache and (hit := self._cache.get(key)) is not None:
            return json.loads(hit)
        resp = self._client.embeddings.create(model=model, input=texts, encoding_format="float")
        vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
        if self._cache:
            self._cache.set(key, json.dumps(vectors))
        return vectors
