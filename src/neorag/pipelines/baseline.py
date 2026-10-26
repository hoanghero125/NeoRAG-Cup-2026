"""Baseline chính thức: naive RAG đơn giản nhất có thể.

1. Embed nguyên từng đoạn văn của corpus (không chunking) bằng bge-m3 qua gateway của BTC.
2. Với mỗi câu hỏi, lấy các đoạn có cosine similarity cao nhất.
3. Đưa top-k đoạn vào một prompt đơn giản và gọi LLM một lần.

Baseline cố ý không tối ưu gì để bạn có nhiều chỗ cải thiện, ví dụ: chunking, hybrid search
(BM25 + dense), reranker, query rewriting, lọc theo CLB, prompt tốt hơn, xử lý câu hỏi
không có đáp án, ...
"""

from __future__ import annotations

import os

import numpy as np

from neorag.data import Document
from neorag.llm import LLMClient
from neorag.pipelines.base import RAGOutput, RAGPipeline

PROMPT = """Dựa vào các tài liệu dưới đây, hãy trả lời câu hỏi.

TÀI LIỆU:
{context}

CÂU HỎI: {query}"""


class GatewayEmbedder:
    """Embedding qua gateway (cùng key với LLM). Muốn chạy model local thì dùng
    neorag.embeddings.LocalEmbedder (cài thêm: python scripts/install.py --local)."""

    def __init__(self, llm: LLMClient, model_name: str):
        self.llm, self.model_name = llm, model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        """Trả về ma trận (len(texts), dim) đã chuẩn hóa L2."""
        vectors = np.asarray(self.llm.embed(texts, model=self.model_name), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class BaselinePipeline(RAGPipeline):
    def __init__(
        self,
        top_k: int = 5,
        n_retrieved: int = 10,
        llm_model: str | None = None,
        embedding_model: str | None = None,
        llm: LLMClient | None = None,
        embedder=None,
    ):
        self.top_k = top_k  # số doc đưa vào prompt
        self.n_retrieved = n_retrieved  # số doc trả về để chấm retrieval
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "claude-haiku-5-5")
        self.llm = llm or LLMClient(cache_path=".cache/pipeline.sqlite")
        self.embedder = embedder or GatewayEmbedder(
            self.llm, embedding_model or os.getenv("EMBEDDING_MODEL", "bge-m3")
        )

    def index(self, corpus: list[Document]) -> None:
        self.docs = corpus
        self.doc_vectors = self.embedder.embed([d.text for d in corpus])

    def run(self, query: str) -> RAGOutput:
        q = self.embedder.embed([query])[0]
        ranking = np.argsort(-(self.doc_vectors @ q))[: self.n_retrieved]
        docs = [self.docs[i] for i in ranking]
        context = "\n\n".join(d.text for d in docs[: self.top_k])
        answer = self.llm.chat(
            [{"role": "user", "content": PROMPT.format(context=context, query=query)}],
            model=self.llm_model,
        )
        return RAGOutput(answer=answer, retrieved_ids=[d.id for d in docs])
