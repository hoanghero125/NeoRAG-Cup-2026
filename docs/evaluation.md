# Đánh giá

Toàn bộ metric được cài đặt trong [`src/neorag/metrics/`](../src/neorag/metrics) và cấu hình trong [`configs/eval.toml`](../configs/eval.toml). Ban tổ chức (BTC) chấm Public Test và Private Test bằng đúng mã nguồn và cấu hình này.

## 1. Nguyên tắc

* **Bộ chấm chỉ đọc file.** Đầu vào là file predictions (`.jsonl`) của thí sinh và file ground truth của BTC. Bộ chấm không chạy mã nguồn của thí sinh, nên các thay đổi trong repository của thí sinh không ảnh hưởng đến cách chấm.
* **Judge cố định.** Các metric cần LLM dùng `claude-haiku-5-5` (`[judge].chat_model`) và embedding `bge-m3` (`[judge].embedding_model`), cả hai gọi qua gateway của BTC, với prompt cố định trong [`judge.py`](../src/neorag/metrics/judge.py). `claude-haiku-5-5` không cho chỉnh `temperature`, nên chấm lại cùng một câu trả lời có thể lệch khoảng ±1 điểm trên thang 100. Khi chấm trên máy cá nhân, kết quả judge được lưu cache nên chạy lại cho ra cùng điểm; trên hệ thống chấm, mỗi bài nộp được chấm độc lập.
* **Một câu trả lời cho mỗi câu hỏi.** Với mỗi câu hỏi, thí sinh nộp một danh sách đoạn văn đã xếp hạng (`retrieved_ids`, tối đa 10) và một câu trả lời (`answer`, tối đa 2000 ký tự). Metric retrieval được tính tại k ∈ {3, 5, 7} trên cùng danh sách đó.
* **Câu hỏi không có đáp án.** Một số câu hỏi hỏi thông tin không có trong corpus (`answerable: false`). Các câu này không tính metric lexical và judge, mà chấm **Abstention** (hệ thống có từ chối đúng cách không). Nếu đáp án chuẩn nhắc tới thông tin liên quan có trong corpus (ví dụ hỏi đợt tuyển năm 2025 trong khi corpus chỉ có đợt 2026), `relevant_ids` ghi các đoạn đó và câu hỏi vẫn được tính metric retrieval.

## 2. Metric retrieval

Ký hiệu: `R` là tập doc đúng của câu hỏi, `D_k` là k doc đầu trong `retrieved_ids`, `rel_i = 1` nếu doc ở vị trí i thuộc `R`. Mỗi metric tính trên từng câu hỏi rồi lấy trung bình.

| Metric | Công thức | Ý nghĩa |
|---|---|---|
| Hit@k | `1 nếu \|D_k ∩ R\| > 0` | Có tìm được ít nhất một doc đúng không |
| Precision@k | `\|D_k ∩ R\| / k` | Bao nhiêu phần top-k là doc đúng. Nộp ít hơn k doc vẫn chia cho k |
| Recall@k | `\|D_k ∩ R\| / \|R\|` | Tìm được bao nhiêu phần doc đúng |
| F1@k | `2PR / (P + R)` | Cân bằng Precision và Recall |
| MAP@k | `(1 / min(\|R\|, k)) Σ_{i≤k} Precision@i · rel_i` | Doc đúng có được xếp lên đầu không (xét mọi doc đúng) |
| MRR@k | `1 / vị trí doc đúng đầu tiên` (0 nếu không có) | Doc đúng đầu tiên xuất hiện sớm đến đâu |
| NDCG@k | `DCG@k / IDCG@k`, `DCG@k = Σ_{i≤k} rel_i / log2(i + 1)` | Chất lượng thứ hạng, relevance nhị phân |

## 3. Metric generation không cần LLM

Mọi văn bản được chuẩn hóa trước khi so ([`text.py`](../src/neorag/text.py)): Unicode NFC, chữ thường, thống nhất vị trí dấu thanh (`hoà` = `hòa`, `thuỷ` = `thủy`), gộp khoảng trắng. ROUGE-L và BLEU-4 còn bỏ dấu câu rồi tách theo âm tiết.

| Metric | Định nghĩa |
|---|---|
| **String Presence** | Tỷ lệ *key facts* của câu hỏi xuất hiện nguyên cụm trong câu trả lời (sau chuẩn hóa, không dính liền chữ/số hai bên: `3 tháng` không khớp `13 tháng`). Key facts là các chuỗi BTC gán sẵn cho từng câu hỏi (tên, ngày, số liệu...), có thể có nhiều cách viết, ví dụ `09/10/2011 \| 9/10/2011`. Không dùng LLM nên kết quả hoàn toàn xác định. |
| **ROUGE-L** | F1 dựa trên dãy con chung dài nhất (LCS) giữa câu trả lời và đáp án chuẩn. |
| **BLEU-4** | Sentence BLEU với n-gram 1-4 trọng số đều, brevity penalty, làm trơn ε = 0.1 cho n-gram không khớp. Chỉ để tham khảo, không tính vào điểm tổng. |

> Đáp án chuẩn được viết ngắn gọn, đi thẳng vào ý. ROUGE-L và BLEU-4 phạt câu trả lời dài dòng; String Presence, Groundedness và Noise Sensitivity thì không. Vì vậy chép nguyên văn tài liệu vào câu trả lời có thể giữ được một số metric nhưng sẽ mất điểm ROUGE-L, Response Relevancy và điểm trải nghiệm ở phần Demo.

## 4. Metric generation chấm bằng LLM judge

Câu trả lời được tách thành câu ([`split_sentences`](../src/neorag/text.py): tách theo dòng và dấu kết thúc câu, bỏ ký hiệu đầu dòng như `1.`, `-`) rồi đưa cho judge gắn nhãn trong một lần gọi. Câu trả lời được bọc trong thẻ riêng và judge được dặn coi đó là dữ liệu, không phải chỉ dẫn.

| Metric | Định nghĩa | Hướng |
|---|---|---|
| **Groundedness** | Judge gắn nhãn từng câu so với **ngữ cảnh đã truy xuất** (văn bản của các doc trong `retrieved_ids`): `supported`, `unsupported`, `contradictory` hoặc `no_rad` (câu dẫn dắt, chào hỏi). Điểm = `supported / (số câu không phải no_rad)`. Đo việc câu trả lời có bám vào ngữ cảnh hay bịa thêm. | ↑ |
| **Noise Sensitivity** | Judge gắn nhãn từng câu so với **tham chiếu chuẩn** (đáp án chuẩn + các doc đúng): `correct`, `incorrect` hoặc `no_rad`. Điểm = `incorrect / (số câu không phải no_rad)`. Đo mức độ câu trả lời bị nhiễu bởi doc không liên quan hoặc bịa thông tin. | ↓ |
| **Response Relevancy** | Judge sinh ngược 3 câu hỏi từ câu trả lời; điểm là trung bình cosine similarity (cắt về [0, 1]) giữa embedding câu hỏi gốc và các câu hỏi sinh ra. Đo việc câu trả lời có đúng trọng tâm câu hỏi không. | ↑ |
| **Abstention** | Chỉ áp dụng cho câu hỏi không có đáp án. Điểm 1 nếu câu trả lời nói rõ là không có hoặc không đủ thông tin và không đưa ra thông tin cụ thể nào như thể đó là câu trả lời. | ↑ |

Trường hợp đặc biệt:

* Câu trả lời không có câu nào cần kiểm chứng được tính Groundedness = 0 và Noise Sensitivity = 1.
* Câu nào judge trả nhãn không đọc được sẽ được chấm lại riêng; nếu vẫn không đọc được thì tính là `unsupported` (Groundedness) và `incorrect` (Noise Sensitivity).

## 5. Điểm Benchmark

Với mỗi tập (Public, Private):

$$
\begin{aligned}
\text{retrieval} &= \operatorname{mean}\left(\text{Recall@5},\ \text{MRR@5},\ \text{NDCG@5}\right) \\[4pt]
\text{generation} &= \operatorname{mean}\big(\text{String Presence},\ \text{ROUGE-L},\ \text{Groundedness}, \\
&\qquad\quad \text{Response Relevancy},\ 1 - \text{Noise Sensitivity},\ \text{Abstention}\big) \\[4pt]
\text{score} &= 100 \times \left(0.5 \times \text{retrieval} + 0.5 \times \text{generation}\right)
\end{aligned}
$$

Trang thi quy đổi `score` (thang 100) sang điểm xếp hạng $R$ theo ngưỡng của từng bộ:

$$
R =
\begin{cases}
100 \times \dfrac{\text{score} - \text{ngưỡng}}{\text{max} - \text{ngưỡng}} & \text{khi } \text{score} > \text{ngưỡng} \\[10pt]
0 & \text{khi } \text{score} \le \text{ngưỡng}
\end{cases}
$$

`ngưỡng` là `score` của pipeline baseline trong repository trên bộ câu hỏi đó, cộng thêm 1 điểm. Một điểm này bằng biên dao động của judge (mục 1), để bài nộp không cải tiến gì so với baseline luôn nhận 0. `max` là `score` cao nhất hiện có trên bảng xếp hạng, nên điểm của mọi người được tính lại mỗi khi `max` thay đổi. Người cao nhất luôn có 100 điểm, không vượt ngưỡng thì 0 điểm. Public Test: baseline 52.40, ngưỡng 53.40.

Điểm Benchmark chung cuộc tính theo điểm xếp hạng Private Test. Public Test chỉ để thử nghiệm, không tính vào kết quả. Điểm Benchmark chiếm 40% tổng điểm cuộc thi; cách quy đổi sang thang 40% do BTC công bố trước buổi pitching.

`neorag evaluate` in `score` trên thang 0–1, trang thi hiện trên thang 100 (ví dụ 0.5240 tương ứng 52.40). Trên máy, thí sinh chỉ chấm được tập train; điểm Public Test và Private Test chỉ có khi nộp bài lên trang thi, vì đáp án do BTC giữ. Mọi trọng số nằm trong mục `[score]` của `configs/eval.toml`.

## 6. Chạy bộ chấm

```bash
# Chấm nhanh, không gọi LLM: chỉ retrieval, String Presence, ROUGE-L, BLEU-4
neorag evaluate train --no-judge

# Chấm đầy đủ với judge chính thức (cần key gateway)
neorag evaluate train

# Chỉ định đường dẫn file
neorag evaluate --gold data/train/questions.jsonl --predictions outputs/my_pred.jsonl
```

Kết quả được ghi vào `outputs/eval_<tên file predictions>/`, ví dụ `outputs/eval_train_pred/`:

* `report.md`: bảng metric
* `report.json`: số liệu tổng hợp
* `per_query.jsonl`: điểm từng câu hỏi, dùng để phân tích lỗi

Mọi lời gọi judge được lưu cache trong `.cache/judge.sqlite`; chấm lại cùng câu trả lời không tiêu tốn thêm ngân sách.

## 7. Giới hạn đã biết

* LLM judge không hoàn hảo; điểm Groundedness và Noise Sensitivity có thể lệch ở các câu mơ hồ. Dùng `per_query.jsonl` để kiểm tra các trường hợp điểm thấp bất thường.
* String Presence so khớp chuỗi, nên viết số liệu, ngày tháng, tên riêng đúng như trong tài liệu sẽ có lợi.
* Tách câu dựa trên dấu câu nên một câu rất dài sẽ được chấm như một câu.
