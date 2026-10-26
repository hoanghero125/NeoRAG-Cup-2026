# Sử dụng gateway

Ban tổ chức (BTC) cung cấp một gateway tương thích OpenAI API để thí sinh truy cập mô hình ngôn ngữ và embedding của cuộc thi. Mọi thư viện hoặc framework hỗ trợ OpenAI API (OpenAI SDK, LangChain, LlamaIndex, ...) đều dùng được bằng cách đổi `base_url` và `api_key`.

## 1. Thông tin kết nối

| Hạng mục | Giá trị |
|---|---|
| Base URL | `https://proptit.dekthe.dev/v1` |
| Xác thực | Header `Authorization: Bearer <key>`, với key cá nhân do BTC cấp |
| Mô hình ngôn ngữ | `claude-haiku-5-5` (endpoint `/chat/completions`) |
| Mô hình embedding | `bge-m3` (endpoint `/embeddings`) |
| Tra cứu ngân sách | [proptit.dekthe.dev/usage](https://proptit.dekthe.dev/usage/) |
| Tài liệu API của gateway | [proptit.dekthe.dev](https://proptit.dekthe.dev/) |

Key được đọc từ biến môi trường `OPENAI_API_KEY` trong file `.env` (xem [Hướng dẫn sử dụng](guide.md#3-cấu-hình-key)). Không ghi key trực tiếp vào mã nguồn.

## 2. Gọi mô hình ngôn ngữ

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://proptit.dekthe.dev/v1")

response = client.chat.completions.create(
    model="claude-haiku-5-5",
    messages=[
        {"role": "system", "content": "Trả lời ngắn gọn dựa trên tài liệu được cung cấp."},
        {"role": "user", "content": "CLB Lập Trình PTIT được thành lập ngày nào?"},
    ],
)
print(response.choices[0].message.content)
```

`claude-haiku-5-5` chỉ chạy với giá trị `temperature` mặc định. Không truyền tham số `temperature`; gateway sẽ trả lỗi nếu có.

## 3. Gọi mô hình embedding

```python
response = client.embeddings.create(
    model="bge-m3",
    input=["CLB Lập Trình PTIT", "CLB Multimedia PTIT"],
    encoding_format="float",
)
vectors = [item.embedding for item in response.data]
```

Mỗi request nên chứa tối đa khoảng 32 đoạn văn. Corpus lớn hơn cần chia thành nhiều batch.

## 4. Gọi bằng HTTP

```bash
curl https://proptit.dekthe.dev/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "Xin chào"}]}'
```

Trên Windows PowerShell:

```powershell
$headers = @{ Authorization = "Bearer $env:OPENAI_API_KEY" }
$body = '{"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "Xin chào"}]}'
Invoke-RestMethod -Uri https://proptit.dekthe.dev/v1/chat/completions -Method Post -Headers $headers -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

Các lệnh trên đọc key từ biến môi trường `OPENAI_API_KEY` của terminal (khác với file `.env`). Đặt biến này trước khi chạy: `export OPENAI_API_KEY=<key>` trên macOS và Linux, `$env:OPENAI_API_KEY="<key>"` trên PowerShell.

## 5. Dùng client có sẵn trong repository

`neorag.llm.LLMClient` đã cấu hình sẵn gateway, tự thử lại khi gặp lỗi tạm thời và lưu cache kết quả, nên gọi lại cùng nội dung không tiêu tốn thêm ngân sách:

```python
from neorag.llm import LLMClient

llm = LLMClient(cache_path=".cache/my_pipeline.sqlite")
answer = llm.chat([{"role": "user", "content": "Xin chào"}], model="claude-haiku-5-5")
vectors = llm.embed(["CLB Lập Trình PTIT"], model="bge-m3")
```

## 6. Ngân sách và giới hạn

| Hạng mục | Giới hạn |
|---|---|
| Ngân sách | 5 USD cho toàn bộ cuộc thi, không cấp bổ sung |
| Tốc độ | 100 request/phút, 300.000 token/phút |
| Đồng thời | Tối đa 8 request cùng lúc |

Embedding `bge-m3` do BTC tự vận hành nên chi phí không đáng kể; phần lớn ngân sách dành cho `claude-haiku-5-5`. Nên cache kết quả gọi mô hình trong quá trình thử nghiệm.

### Tra cứu ngân sách còn lại

Trang [Tra cứu hạn mức API](https://proptit.dekthe.dev/usage/) hiển thị ngân sách còn lại của key:

1. Mở [proptit.dekthe.dev/usage](https://proptit.dekthe.dev/usage/).
2. Dán key vào ô nhập ở đầu trang.
3. Bấm **Xem**. Ngân sách còn lại (USD) hiển thị ngay bên dưới.

Key chỉ được lưu trong tab trình duyệt đang mở và chỉ được gửi tới gateway của BTC. Nên kiểm tra ngân sách trước các lần chạy lớn, ví dụ chấm đầy đủ tập train bằng LLM judge.

## 7. Mã lỗi thường gặp

| Mã | Ý nghĩa | Cách xử lý |
|---|---|---|
| 401 | Key không hợp lệ | Kiểm tra key trong `.env`, không có dấu cách thừa |
| 400 | Tham số không được hỗ trợ, hoặc key đã hết ngân sách | Đọc thông báo lỗi; bỏ tham số `temperature`; kiểm tra ngân sách tại [trang tra cứu](https://proptit.dekthe.dev/usage/) |
| 429 | Vượt giới hạn tốc độ | Chờ khoảng một phút rồi gửi lại; giảm số request đồng thời |
| 5xx | Lỗi tạm thời phía gateway | Thử lại sau; báo BTC nếu lỗi kéo dài |
