"""Chuẩn hóa và tách từ tiếng Việt dùng chung cho metric và baseline."""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•+])\s+")
# Kết thúc câu: . ! ? … rồi khoảng trắng; không tách sau các viết tắt thường gặp.
_SENTENCE_END = re.compile(
    r"(?<=[.!?…])(?<!\bTP\.)(?<!\bTS\.)(?<!\bThS\.)(?<!\bGS\.)(?<!\bPGS\.)(?<!\bQ\.)(?<!\bP\.)\s+"
)

# Kiểu bỏ dấu "mới" (oà, uỷ) -> kiểu "cũ" (òa, ủy) để hai cách viết so khớp được với nhau.
_TONE_VARIANTS = {
    "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
    "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
    "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
}  # fmt: skip
_TONE_PATTERN = re.compile("|".join(_TONE_VARIANTS))


def normalize(text: str) -> str:
    """NFC, chữ thường, thống nhất vị trí dấu thanh, gộp khoảng trắng. Giữ dấu câu."""
    text = unicodedata.normalize("NFC", text).lower()
    text = _TONE_PATTERN.sub(lambda m: _TONE_VARIANTS[m.group()], text)
    return _SPACE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tách theo âm tiết (khoảng trắng) sau khi bỏ dấu câu."""
    return _PUNCT.sub(" ", normalize(text)).split()


def split_sentences(text: str) -> list[str]:
    """Tách câu theo dòng và dấu kết thúc câu; bỏ ký hiệu đầu dòng (1., -, *, •)."""
    sentences = []
    for line in text.splitlines():
        line = _LIST_MARKER.sub("", line).strip()
        sentences += [s.strip() for s in _SENTENCE_END.split(line)]
    return [s for s in sentences if len(s) > 1]
