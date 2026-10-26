# Nộp bài trên aichallenge

Bài nộp của NeoRAG Cup 2026 được chấm tự động trên [aichallenge.ptit.edu.vn](https://aichallenge.ptit.edu.vn/) (PTIT AI Arena). Tài liệu này hướng dẫn tạo tài khoản, đăng ký cuộc thi, nộp bài, đọc kết quả và xử lý lỗi trên hệ thống chấm. Định dạng file nộp và giới hạn lượt nộp: [Nộp bài](submission.md).

## 1. Tạo tài khoản

1. Mở [aichallenge.ptit.edu.vn](https://aichallenge.ptit.edu.vn/), chọn **Sign up**.
2. Điền tên người dùng (5–15 ký tự gồm chữ thường, số, gạch ngang hoặc gạch dưới), email và mật khẩu, đánh dấu đồng ý điều khoản rồi bấm **Đăng ký**.
3. Mở email kích hoạt để kích hoạt tài khoản. Nếu không nhận được email, chọn **Gửi lại email kích hoạt** ở trang đăng ký.
4. Đặt tên hiển thị trên bảng xếp hạng tại **Edit Profile** → **Display Name**. Dùng họ và tên thật để Ban tổ chức (BTC) đối chiếu kết quả.

## 2. Đăng ký cuộc thi

1. Mở trang cuộc thi [bit.ly/neorag2026](https://bit.ly/neorag2026).
2. Mở tab **My Submissions**, đánh dấu *I accept the terms and conditions of the competition.* rồi bấm **Register**.
3. Trạng thái chuyển sang **Pending** cho đến khi BTC duyệt. Sau khi được duyệt, tab **My Submissions** hiện khung nộp bài.

## 3. Nộp bài

1. Tạo file nộp trong repository:

   | Giai đoạn | Lệnh | File nộp |
   |---|---|---|
   | Public Test | `neorag run public_test` | `outputs/public_test_submission.zip` |
   | Private Test | `neorag run private_test` | `outputs/private_test_submission.zip` |

2. Mở tab **My Submissions**, chọn đúng giai đoạn **PUBLIC TEST** hoặc **PRIVATE TEST**.
3. Bấm biểu tượng kẹp giấy và chọn file `.zip`. Bài nộp được tải lên và chấm tự động.
4. Theo dõi trạng thái trong bảng bài nộp bên dưới khung nộp bài:

| Trạng thái | Ý nghĩa |
|---|---|
| Submitting, Submitted, Running | Đang tải lên hoặc đang chấm. Một bài thường chấm xong sau 2–5 phút |
| Finished | Chấm xong, đã có điểm |
| Failed | Bài nộp bị lỗi; xem mục 5 |

Bài nộp ở trạng thái **Failed** không tính vào số lượt nộp.

## 4. Xem kết quả

### Điểm từng bài nộp

Bấm biểu tượng con mắt ở cột **Actions** của bài nộp, mở tab **LOGS**, chọn **Scoring output**. Log ghi điểm các thành phần và raw score trên thang 100, ví dụ với pipeline baseline:

```
┌───────────────────────────┬─────────┐
│ Recall@5                  │   43.45 │
│ MRR@5                     │   44.21 │
│ NDCG@5                    │   39.29 │
├───────────────────────────┼─────────┤
│ Retrieval                 │   42.31 │
╞═══════════════════════════╪═════════╡
│ String Presence           │   52.14 │
│ ROUGE-L                   │   33.11 │
│ Groundedness              │   83.29 │
│ Response Relevancy        │   60.07 │
│ 1 - Noise Sensitivity     │   46.26 │
│ Abstention                │  100.00 │
├───────────────────────────┼─────────┤
│ Generation                │   62.48 │
╞═══════════════════════════╪═════════╡
│ Raw score                 │   52.40 │
└───────────────────────────┴─────────┘
```

- **Retrieval** là trung bình của 3 dòng phía trên nó; **Generation** là trung bình của 6 dòng phía trên nó.
- **Raw score** = 0.5 × Retrieval + 0.5 × Generation, là điểm gốc của bài nộp.
- Noise Sensitivity càng thấp càng tốt, nên được tính vào điểm dưới dạng 1 − Noise Sensitivity.

Định nghĩa từng metric: [Đánh giá](evaluation.md). Ghi chú dạng `scoring_hostname-...` trong danh sách bài nộp là tên máy chấm do hệ thống ghi lại, không phải log chấm điểm.

### Bảng xếp hạng

Tab **Results** hiện bảng xếp hạng của từng giai đoạn. Bảng lấy lần nộp cuối cùng của mỗi thí sinh và hiện điểm xếp hạng, là raw score được quy đổi theo baseline: bài không vượt ngưỡng (điểm baseline cộng 1) nhận 0, bài cao nhất nhận 100. Công thức quy đổi: [Đánh giá](evaluation.md#5-điểm-benchmark).

| Giai đoạn | Điểm của mình | Bảng xếp hạng |
|---|---|---|
| Public Test | Xem được | Công khai |
| Private Test | Xem được | Ẩn đến khi BTC công bố |

## 5. Xử lý lỗi

Với bài **Failed**, mở tab **LOGS** → **Scoring errors** để xem nguyên nhân.

| Thông báo | Nguyên nhân | Cách xử lý |
|---|---|---|
| `Không tìm thấy file .jsonl nào trong bài nộp` | File `.zip` không chứa file predictions | Nộp file `_submission.zip` do `neorag run` tạo, hoặc nén đúng file `.jsonl` |
| `Bài nộp có ... file .jsonl` | File `.zip` chứa nhiều file predictions | Chỉ nén một file `.jsonl` |
| `File predictions không hợp lệ` | Thiếu câu hỏi, trùng `id`, `retrieved_ids` sai, `answer` rỗng hoặc quá 2.000 ký tự | Đọc danh sách lỗi; kiểm tra lại trên máy bằng `neorag validate <bộ> --predictions <file>` |
| `Có vẻ bạn nộp nhầm file của bộ khác` | Nộp file Public vào giai đoạn Private hoặc ngược lại | Chọn đúng giai đoạn rồi nộp lại |
| `Lỗi khi chấm (...). Đây không phải lỗi của bạn` | Lỗi phía hệ thống chấm | Báo BTC; BTC sẽ chấm lại bài |
| `Reached maximum allowed submissions` | Hết lượt nộp trong ngày hoặc trong giai đoạn | Chờ sang ngày tiếp theo (Public Test) |
