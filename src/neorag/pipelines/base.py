"""Interface chung cho mọi pipeline. Pipeline của bạn chỉ cần kế thừa `RAGPipeline`.

Runner (`neorag predict`) và chatbot (`app/chat.py`) đều gọi pipeline qua interface này,
nên bạn tự do thay đổi mọi thứ bên trong: chunking, index, hybrid search, rerank, prompt...
"""

from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from neorag.data import Document


@dataclass
class RAGOutput:
    answer: str
    retrieved_ids: list[int]  # id doc trong corpus, xếp hạng từ liên quan nhất, tối đa 10


class RAGPipeline(ABC):
    def prepare(self) -> None:
        """Tải model hoặc tài nguyên nặng (không bắt buộc). Được gọi một lần trước `index`."""
        return None  # mặc định không cần chuẩn bị gì

    @abstractmethod
    def index(self, corpus: list[Document]) -> None:
        """Xây index từ corpus. Được gọi một lần trước khi trả lời câu hỏi."""

    @abstractmethod
    def run(self, query: str) -> RAGOutput:
        """Trả lời một câu hỏi và trả về các doc đã truy xuất."""


def load_pipeline(spec: str, **kwargs) -> RAGPipeline:
    """Khởi tạo pipeline từ chuỗi "module.path:ClassName"."""
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"Pipeline phải có dạng 'module.path:ClassName', nhận được: {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    if not issubclass(cls, RAGPipeline):
        raise TypeError(f"{spec} phải kế thừa neorag.pipelines.RAGPipeline")
    return cls(**kwargs)
