from __future__ import annotations

import numpy as np
import pytest

from neorag.data import Document, Query


class FakeLLM:
    """LLM giả cho test: trả lời theo hàm `responder`."""

    def __init__(self, responder=None):
        self.responder = responder or (lambda messages: "")
        self.calls: list[list[dict]] = []

    def chat(self, messages, model, temperature=None):
        self.calls.append(messages)
        return self.responder(messages)


class FakeEmbedder:
    """Embedding giả cho test: bag-of-characters, chuẩn hóa L2."""

    device = "cpu"

    def load(self):
        return None

    def embed(self, texts):
        vecs = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                vecs[i, ord(ch) % 64] += 1
        vecs += 1e-3
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


@pytest.fixture
def corpus() -> list[Document]:
    return [
        Document(
            1, "CLB Lập Trình PTIT", "Giới thiệu", "CLB Lập Trình PTIT thành lập ngày 09/10/2011."
        ),
        Document(
            2, "CLB Multimedia PTIT", "Giới thiệu", "CLB Multimedia PTIT thành lập ngày 10/11/2012."
        ),
        Document(3, "CLB Nhà Sáng Tạo Game PTIT (GCC)", "Giới thiệu", "GCC ra đời năm 2023."),
    ]


@pytest.fixture
def gold() -> list[Query]:
    return [
        Query("q1", "ProPTIT thành lập khi nào?", [1], "Ngày 09/10/2011.", ["09/10/2011"]),
        Query("q2", "Multimedia thành lập khi nào?", [2], "Ngày 10/11/2012.", ["10/11/2012"]),
        Query("q3", "S2B viết tắt của gì?", [], "Không có thông tin.", [], answerable=False),
    ]
