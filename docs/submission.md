# Nộp bài

Bài nộp được chấm tự động trên [aichallenge.ptit.edu.vn](https://aichallenge.ptit.edu.vn/). Tài liệu này mô tả định dạng bài nộp và quy trình nộp ở từng giai đoạn. Thao tác chi tiết trên hệ thống chấm (tài khoản, đăng ký, đọc kết quả, xử lý lỗi): [Nộp bài trên aichallenge](aichallenge.md).

## 1. Định dạng bài nộp

Bài nộp là một file `.jsonl` mã hóa UTF-8, mỗi dòng ứng với một câu hỏi:

```json
{"id": "public-0001", "retrieved_ids": [12, 5, 33, 7, 1], "answer": "CLB Lập Trình PTIT được thành lập ngày 09/10/2011.", "latency_ms": 842.3}
```

| Trường | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `id` | string | Có | Trùng với `id` trong file câu hỏi; mỗi câu hỏi đúng một dòng |
| `retrieved_ids` | list[int] | Có | Từ 1 đến 10 id đoạn văn trong `data/corpus.jsonl`, sắp xếp theo mức độ liên quan giảm dần, không trùng lặp |
| `answer` | string | Có | Câu trả lời cuối cùng, không rỗng, tối đa 2.000 ký tự |
| `latency_ms` | number | Không | Thời gian xử lý câu hỏi; BTC dùng để tham khảo khi đánh giá Demo Product |

Các trường khác bị bỏ qua. Bài nộp sai định dạng sẽ bị hệ thống từ chối kèm lý do.

Lệnh `neorag run` và `neorag predict` sinh đúng định dạng trên và tự kiểm tra trước khi ghi file. Thí sinh tự viết runner cần kiểm tra file bằng lệnh:

```bash
neorag validate public_test --predictions <đường dẫn tới file>
```

## 2. Public Test

| Hạng mục | Chi tiết |
|---|---|
| Thời gian | 00:00 26/10 – 23:59 07/11/2026 (giờ Việt Nam) |
| Số câu hỏi | 42 |
| Lượt nộp | Tối đa 10 lần mỗi ngày, 50 lần cho cả giai đoạn; bài bị lỗi (Failed) không tính lượt |
| Kết quả | Hiển thị trên bảng xếp hạng sau khi chấm xong (vài phút); không tính vào kết quả chung cuộc |

1. BTC đăng file câu hỏi `data/public_test/questions.jsonl` lên repository của cuộc thi. Tải file về và đặt vào cùng đường dẫn trong repository của bạn.
2. Chạy `neorag run public_test`. Lệnh tạo file `outputs/public_test_submission.zip`, chứa file `.jsonl` đã được nén vì hệ thống chấm chỉ nhận định dạng `.zip`.
3. Trên trang cuộc thi, mở tab **My Submissions**, chọn giai đoạn Public Test và tải lên file `.zip`.
4. Kết quả có sau vài phút. Bảng xếp hạng lấy kết quả của lần nộp gần nhất.

## 3. Private Test

| Hạng mục | Chi tiết |
|---|---|
| Thời gian | 00:00 – 23:59 08/11/2026 (giờ Việt Nam) |
| Số câu hỏi | 42 |
| Lượt nộp | Tối đa 5 lần; bài bị lỗi (Failed) không tính lượt |
| Kết quả | Thí sinh xem được điểm của mình; bảng xếp hạng ẩn đến khi BTC công bố. Dùng để tính điểm Benchmark chung cuộc |

1. BTC đăng file câu hỏi `data/private_test/questions.jsonl` lên repository của cuộc thi vào đầu ngày 08/11. Tải file về và đặt vào cùng đường dẫn trong repository của bạn.
2. Chạy `neorag run private_test` và tải lên `outputs/private_test_submission.zip` ở giai đoạn Private Test. Thí sinh không chỉnh sửa mã nguồn trong ngày thi.
3. Hệ thống chấm ngay. Thí sinh xem điểm bài nộp của mình trong log và tab **My Submissions**; bảng xếp hạng ẩn đến khi BTC công bố.
4. Lần nộp gần nhất được dùng để tính điểm.

## 4. Đối chiếu kết quả

BTC có thể yêu cầu thí sinh chạy lại pipeline trên một số câu hỏi, ví dụ trong buổi demo, để đối chiếu với bài đã nộp. Chênh lệch lớn không có lời giải thích hợp lý được xử lý theo [Quy định](rules.md#3-xử-lý-vi-phạm).
