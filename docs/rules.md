# Quy định

Quy định này áp dụng cho mọi thí sinh tham gia NeoRAG Cup 2026. Ban tổ chức (BTC) có quyền giải thích và bổ sung quy định trong quá trình diễn ra cuộc thi; mọi thay đổi được thông báo tới tất cả thí sinh.

## 1. Công nghệ được phép sử dụng

- **Kiến trúc pipeline:** thí sinh tự lựa chọn kỹ thuật chunking, embedding, vector store, BM25 hoặc hybrid search, reranker, viết lại câu hỏi, agent, GraphRAG và các kỹ thuật khác.
- **Mô hình ngôn ngữ:** duy nhất `claude-haiku-5-5`, truy cập qua gateway của BTC (`https://proptit.dekthe.dev/v1`) bằng key cá nhân. Mọi bước sử dụng mô hình ngôn ngữ (sinh câu trả lời, viết lại câu hỏi, rerank bằng LLM, agent...) đều phải dùng mô hình này.
- **Embedding và reranker:** được dùng `bge-m3` qua gateway bằng cùng key, hoặc các mô hình open-source không phải mô hình sinh văn bản (embedding, cross-encoder reranker...) chạy trên máy của thí sinh.
- **Ngôn ngữ lập trình, framework và giao diện:** thí sinh tự lựa chọn. Chatbot có thể xây dựng bằng bất kỳ công nghệ nào (Streamlit, Gradio, FastAPI kết hợp React, ...). Mã nguồn mẫu trong repository, bao gồm pipeline baseline và chatbot Streamlit, chỉ mang tính tham khảo.
- **Dữ liệu huấn luyện:** được dùng tập train để phát triển, tinh chỉnh prompt, xây dựng few-shot hoặc fine-tune các mô hình được phép.
- **Dữ liệu bổ sung:** được bổ sung thông tin công khai về cộng đồng S2B, với điều kiện ghi rõ nguồn và trình bày tại buổi pitching. Kết quả benchmark vẫn tính theo `id` đoạn văn của `data/corpus.jsonl`.

## 2. Hành vi bị cấm

- Sử dụng mô hình ngôn ngữ khác ngoài `claude-haiku-5-5` qua gateway của BTC, kể cả API bên ngoài hay mô hình chạy cục bộ.
- Chia sẻ, mượn hoặc sử dụng key của người khác.
- Gán nhãn thủ công, tra cứu thủ công hoặc dùng người để trả lời câu hỏi Public Test và Private Test.
- Hard-code câu trả lời hoặc `retrieved_ids` cho các câu hỏi trong bộ test.
- Nhờ người khác viết mã nguồn; chia sẻ file predictions giữa các thí sinh.
- Chỉnh sửa bộ chấm hoặc cấu hình chấm rồi báo cáo kết quả đó như điểm chính thức.
- Chỉnh sửa mã nguồn trong ngày Private Test.
- Đưa vào câu trả lời nội dung nhằm tác động tới LLM judge (chỉ dẫn, nhãn chấm điểm, thẻ định dạng...). Mã nguồn bộ chấm được công khai để thí sinh hiểu cách đánh giá, không phải để khai thác; BTC kiểm tra lại câu trả lời của các bài có điểm cao.

## 3. Xử lý vi phạm

BTC có quyền yêu cầu thí sinh chạy lại pipeline để đối chiếu với bài đã nộp. Thí sinh vi phạm mục 2 bị hủy điểm Benchmark; trường hợp nghiêm trọng bị loại khỏi cuộc thi.

## 4. Key truy cập gateway

Cách kết nối tới gateway: [Sử dụng gateway](gateway.md).


- Mỗi key có ngân sách 5 USD cho toàn bộ cuộc thi và không được cấp bổ sung. Key ngừng hoạt động khi hết ngân sách. Ngân sách còn lại được tra cứu tại [proptit.dekthe.dev/usage](https://proptit.dekthe.dev/usage/).
- Giới hạn tốc độ: 100 request/phút, 300.000 token/phút, tối đa 8 request đồng thời. Khi gặp lỗi 429, chờ rồi chạy lại; client trong repository đã tự động thử lại.
- Kết quả gọi mô hình được lưu cache trong `.cache/`; chạy lại cùng câu hỏi không tiêu tốn thêm ngân sách.

## 5. Bảo mật

- Không commit key lên repository. Đặt key trong file `.env` (đã có trong `.gitignore`) và chỉ commit `.env.example`.
- Nếu key bị lộ, báo ngay cho BTC để thu hồi và cấp key mới. Xóa commit chứa key là chưa đủ.

## 6. Sản phẩm cuối cùng

- Repository mã nguồn trên GitHub, kèm README hướng dẫn chạy pipeline và chatbot.
- Bài nộp Public Test và Private Test trên hệ thống chấm.
- Slide trình bày.
- Chatbot hoạt động được để demo tại buổi pitching.
