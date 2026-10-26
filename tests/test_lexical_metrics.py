import pytest

from neorag.metrics.lexical import bleu4, rouge_l, string_presence
from neorag.text import split_sentences, tokenize


def test_tokenize_keeps_vietnamese_and_drops_punctuation():
    assert tokenize("CLB Lập Trình PTIT, thành lập!") == [
        "clb",
        "lập",
        "trình",
        "ptit",
        "thành",
        "lập",
    ]


def test_split_sentences():
    text = "Câu một. Câu hai!\n- Ý ba"
    assert split_sentences(text) == ["Câu một.", "Câu hai!", "Ý ba"]


def test_string_presence_with_alternatives():
    answer = "CLB thành lập ngày 9/10/2011, phương châm Chia sẻ để cùng nhau phát triển."
    facts = ["09/10/2011 | 9/10/2011", "chia sẻ để cùng nhau phát triển", "Lập trình từ trái tim"]
    assert string_presence(answer, facts) == pytest.approx(2 / 3)
    assert string_presence(answer, []) == 0.0


def test_rouge_l():
    assert rouge_l("a b c d", "a b c d") == pytest.approx(1.0)
    # LCS("a x c", "a b c") = 2 -> P = R = 2/3
    assert rouge_l("a x c", "a b c") == pytest.approx(2 / 3)
    assert rouge_l("", "a b") == 0.0


def test_bleu4():
    ref = "câu lạc bộ lập trình ptit được thành lập năm 2011"
    assert bleu4(ref, ref) == pytest.approx(1.0)
    assert 0.0 < bleu4("câu lạc bộ lập trình ptit", ref) < 1.0
    assert bleu4("xyz", ref) < 0.05
