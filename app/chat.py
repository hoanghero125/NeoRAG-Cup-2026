"""Chatbot mẫu cho phần Demo Product. Chạy: python -m streamlit run app/chat.py

Đổi pipeline bằng biến môi trường NEORAG_PIPELINE="module.path:ClassName".
Đây chỉ là điểm khởi đầu: giao diện, streaming, lịch sử hội thoại... là phần bạn tự cải thiện.
"""

from __future__ import annotations

import os
import time

import streamlit as st

from neorag.data import load_corpus
from neorag.pipelines import load_pipeline

PIPELINE = os.getenv("NEORAG_PIPELINE", "neorag.pipelines.baseline:BaselinePipeline")

st.set_page_config(page_title="NeoRAG Chatbot - S2B", page_icon="💬")
st.title("💬 Hỏi đáp cộng đồng S2B")
st.caption("CLB Lập Trình PTIT · CLB Multimedia PTIT · CLB Nhà Sáng Tạo Game PTIT")


@st.cache_resource(show_spinner="Đang nạp model và xây index (lần đầu có thể mất vài phút)...")
def get_pipeline():
    corpus = load_corpus()
    pipeline = load_pipeline(PIPELINE)
    pipeline.prepare()
    pipeline.index(corpus)
    return pipeline, {d.id: d for d in corpus}


pipeline, docs = get_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Hỏi gì đó về ProPTIT, Multimedia hoặc GCC..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        start = time.perf_counter()
        with st.spinner("Đang tìm câu trả lời..."):
            out = pipeline.run(question)
        st.markdown(out.answer)
        with st.expander(f"Nguồn tham khảo · {(time.perf_counter() - start):.1f}s"):
            for doc_id in out.retrieved_ids:
                d = docs[doc_id]
                st.markdown(f"**[{d.id}] {d.club}** · _{d.topic}_\n\n{d.text}")
    st.session_state.messages.append({"role": "assistant", "content": out.answer})
