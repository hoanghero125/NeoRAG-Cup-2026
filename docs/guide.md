# Hướng dẫn sử dụng repo

Tài liệu này hướng dẫn cài đặt repository, chạy baseline, xây dựng pipeline và tạo bài nộp.

## 1. Yêu cầu

- Python 3.10 trở lên.
- Key cá nhân do Ban tổ chức cấp. Cả mô hình ngôn ngữ `claude-haiku-5-5` lẫn embedding `bge-m3` đều được gọi qua gateway bằng key này, nên không cần GPU.

## 2. Cài đặt

Tạo repository riêng từ template này ("Use this template") và clone về máy:

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

Sau đó tạo môi trường Python riêng cho repository theo một trong hai cách dưới đây, và chạy các lệnh trong thư mục gốc của repository.

### Cách 1: venv

Dùng Python đã cài sẵn trên máy, không cần thêm công cụ.

| Bước | Windows (PowerShell) | macOS, Linux |
|---|---|---|
| Tạo môi trường | `python -m venv .venv` | `python3 -m venv .venv` |
| Kích hoạt | `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |
| Cài đặt | `python scripts/install.py` | `python scripts/install.py` |

Nếu PowerShell báo lỗi "running scripts is disabled on this system" khi kích hoạt, chạy `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` một lần rồi kích hoạt lại, hoặc dùng Command Prompt (cmd) với lệnh `.venv\Scripts\activate.bat`.

Nếu máy có Anaconda và dấu nhắc lệnh đang hiện `(base)`, chạy `conda deactivate` trước khi kích hoạt `.venv`.

### Cách 2: conda

Dùng khi máy đã cài Anaconda hoặc Miniconda. Các lệnh giống nhau trên mọi hệ điều hành; trên Windows, chạy trong Anaconda Prompt hoặc PowerShell đã bật conda.

```bash
conda create -n neorag python=3.12 -y
conda activate neorag
python scripts/install.py
```

Không cài vào môi trường `base`: các thư viện của cuộc thi có thể xung đột với gói có sẵn của Anaconda.

### Lưu ý chung

- Mọi lệnh trong tài liệu này chạy trong môi trường đã kích hoạt. Tên môi trường hiện ở đầu dấu nhắc lệnh: `(.venv)` hoặc `(neorag)`. Mỗi lần mở terminal mới cần kích hoạt lại.
- Lỗi `ModuleNotFoundError: No module named 'neorag'` gần như luôn do lệnh đang chạy bằng một Python khác (ví dụ `base` của Anaconda). Kích hoạt đúng môi trường và chạy công cụ qua `python -m`, ví dụ `python -m streamlit`.
- Cách cài thay thế cho `python scripts/install.py`: `pip install -r requirements.txt`.
- Muốn dùng embedding hoặc reranker open-source chạy trên máy, cài thêm bằng `python scripts/install.py --local`. Script tự cài PyTorch bản CUDA nếu máy có GPU NVIDIA; `pip install` thông thường trên Windows luôn cài bản CPU.

## 3. Cấu hình key

```powershell
copy .env.example .env             # Windows
```

```bash
cp .env.example .env               # macOS, Linux
```

Mở `.env` bằng trình soạn thảo bất kỳ và điền `OPENAI_API_KEY` bằng key do Ban tổ chức (BTC) cấp. Các biến còn lại đã có giá trị mặc định. File `.env` đã nằm trong `.gitignore`; không commit key lên GitHub. Cách gọi gateway trực tiếp từ mã nguồn của thí sinh: [Sử dụng gateway](gateway.md).

## 4. Chạy baseline

```bash
neorag run train
```

Lệnh trả lời 126 câu hỏi của tập train, rồi chấm điểm bằng bộ chấm chính thức. Báo cáo được ghi vào `outputs/eval_train_pred/`:

- `report.md`: bảng metric
- `report.json`: số liệu tổng hợp
- `per_query.jsonl`: điểm từng câu hỏi, dùng để phân tích lỗi

### Các lệnh

| Lệnh | Tác dụng |
|---|---|
| `neorag run <bộ>` | Chạy pipeline, tạo file nộp, rồi chấm điểm nếu có đáp án (chỉ tập train) |
| `neorag run <bộ> --no-judge` | Chấm nhanh, không gọi LLM judge: chỉ retrieval, String Presence, ROUGE-L, BLEU-4 |
| `neorag predict <bộ>` | Chỉ chạy pipeline, ghi `outputs/<bộ>_pred.jsonl` và `outputs/<bộ>_submission.zip` |
| `neorag evaluate train` | Chỉ chấm file predictions đã có (chỉ tập train) |
| `neorag validate <bộ>` | Kiểm tra định dạng file predictions |

`<bộ>` là `train`, `public_test` hoặc `private_test`, ứng với thư mục cùng tên trong `data/`. Đáp án Public Test và Private Test do BTC giữ, nên với hai bộ này lệnh chỉ tạo file nộp; điểm có sau khi nộp lên trang thi. Mỗi lần chạy ghi đè kết quả lần trước của cùng bộ; muốn giữ lại thì đổi tên file đầu ra bằng `--out`. Thêm `--help` sau mỗi lệnh để xem đầy đủ tùy chọn.

Kết quả gọi LLM được cache trong `.cache/`, nên chạy lại cùng câu hỏi không tốn thêm ngân sách của key.

## 5. Xây dựng pipeline

Pipeline kế thừa `RAGPipeline` và cài đặt hai phương thức `index` và `run`. `prepare` là tùy chọn, dùng để nạp các model nặng.

```python
from neorag.data import Document
from neorag.pipelines import RAGOutput, RAGPipeline


class MyPipeline(RAGPipeline):
    def prepare(self) -> None:
        ...  # nạp embedding model, reranker, ...

    def index(self, corpus: list[Document]) -> None:
        ...  # chunking, embedding, vector store, BM25, ...

    def run(self, query: str) -> RAGOutput:
        ...  # viết lại câu hỏi, truy xuất, rerank, sinh câu trả lời
        return RAGOutput(answer=answer, retrieved_ids=[12, 5, 33])
```

Chạy pipeline bằng runner có sẵn:

```bash
neorag run train --pipeline my_solution.pipeline:MyPipeline
```

Lưu ý:

- `retrieved_ids` phải là id đoạn văn trong `data/corpus.jsonl`, xếp từ liên quan nhất, tối đa 10 id. Khi chia nhỏ tài liệu, ánh xạ các chunk về id gốc và bỏ trùng.
- Mọi bước cần mô hình ngôn ngữ phải dùng `claude-haiku-5-5` qua gateway ([quy định](rules.md)).
- Baseline ([`src/neorag/pipelines/baseline.py`](../src/neorag/pipelines/baseline.py)) là điểm tham khảo: embed nguyên từng đoạn văn, lấy top-k theo cosine similarity và gọi mô hình một lần với một prompt ngắn.

## 6. Chatbot

Thí sinh tự lựa chọn công nghệ cho giao diện chatbot: Streamlit, Gradio, FastAPI kết hợp React hoặc bất kỳ framework nào khác. Repository kèm một chatbot Streamlit tối giản trong `app/chat.py` để tham khảo cách gọi pipeline:

```bash
python -m streamlit run app/chat.py

# Chạy với pipeline của thí sinh
NEORAG_PIPELINE=my_solution.pipeline:MyPipeline python -m streamlit run app/chat.py      # macOS, Linux
$env:NEORAG_PIPELINE="my_solution.pipeline:MyPipeline"; python -m streamlit run app/chat.py   # Windows PowerShell
```

Dùng `python -m streamlit` thay cho `streamlit` để chatbot chạy bằng đúng Python của môi trường đã kích hoạt. Chatbot dùng trong buổi pitching do thí sinh tự xây dựng và được đánh giá ở tiêu chí Demo Product.

## 7. Tạo file nộp bài

```bash
neorag run public_test
```

Lệnh trả lời 42 câu hỏi, kiểm tra định dạng và tạo `outputs/public_test_submission.zip`. Giới hạn lượt nộp: [Nộp bài](submission.md). Thao tác trên hệ thống chấm: [Nộp bài trên aichallenge](aichallenge.md).

## 8. Cấu trúc repo

```
├── app/chat.py              Chatbot mẫu (Streamlit, tham khảo)
├── configs/eval.toml        Cấu hình chính thức của bộ chấm
├── data/                    Corpus, nguồn dữ liệu và câu hỏi (train/, public_test/, private_test/)
├── docs/                    Tài liệu cuộc thi
├── requirements.txt         Cài đặt bằng pip install -r
├── scripts/install.py       Cài đặt môi trường
├── src/neorag/
│   ├── pipelines/           Interface RAGPipeline và baseline
│   ├── metrics/             Metric retrieval, lexical và LLM judge
│   ├── evaluate.py          Bộ chấm
│   ├── predict.py           Chạy pipeline, đo latency, ghi file nộp
│   ├── llm.py               Client gọi gateway, có cache
│   ├── embeddings.py        Embedding chạy trên máy (cài --local)
│   ├── data.py              Đọc, ghi và kiểm tra dữ liệu
│   └── cli.py               Lệnh neorag
└── tests/                   Unit test
```

## 9. Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `ModuleNotFoundError: No module named 'neorag'` | Chưa kích hoạt môi trường hoặc chưa cài đặt. Kích hoạt môi trường (`.venv` hoặc `neorag`), chạy `python scripts/install.py`, rồi dùng `python -m streamlit` thay cho `streamlit` |
| Không nhận lệnh `neorag` | Chưa kích hoạt môi trường hoặc chưa cài đặt; xem mục 2 |
| Key không hợp lệ | Kiểm tra `OPENAI_API_KEY` trong `.env`: dán đúng key được cấp, không có dấu cách |
| Hết budget | Key đã dùng hết 5 USD; kiểm tra tại [proptit.dekthe.dev/usage](https://proptit.dekthe.dev/usage/). Liên hệ BTC |
| Vượt rate limit (429) | Đợi khoảng một phút rồi chạy lại; các câu đã trả lời được giữ trong cache |
| Không kết nối được tới gateway | Kiểm tra mạng và `OPENAI_BASE_URL` trong `.env` |
| File predictions không hợp lệ | Đọc danh sách lỗi in ra; thường là thiếu câu hỏi, trùng id hoặc id đoạn văn không tồn tại |
