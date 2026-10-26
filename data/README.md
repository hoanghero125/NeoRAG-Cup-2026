# Dữ liệu

## 1. Tổng quan

Bộ dữ liệu về cộng đồng S2B tại Học viện Công nghệ Bưu chính Viễn thông, gồm 3 CLB: CLB Lập Trình PTIT (ProPTIT), CLB Multimedia PTIT (MPC) và CLB Nhà Sáng Tạo Game PTIT (GCC). Dữ liệu được tổng hợp và đối chiếu từ website, blog, fanpage và tài liệu nội bộ của các CLB; thời điểm truy cập gần nhất là 24/10/2026.

```
data/
├── corpus.jsonl            118 đoạn văn, mỗi đoạn tự đủ ngữ cảnh và nêu rõ tên CLB
├── sources.csv             Nguồn, ngày nguồn và mức xác minh của từng đoạn
├── train/
│   └── questions.jsonl     126 câu hỏi kèm ground truth, dùng để phát triển
├── public_test/
│   └── questions.jsonl     42 câu hỏi Public Test, không kèm đáp án
└── private_test/
    └── questions.jsonl     42 câu hỏi Private Test, không kèm đáp án
```

Phiên bản hiện tại: corpus v1.0. Đề Public Test và Private Test được phát hành theo lịch của BTC; trước khi phát hành, thư mục tương ứng để trống. Các lệnh `neorag` nhận tên thư mục làm tên bộ câu hỏi, ví dụ `neorag run public_test` đọc `data/public_test/questions.jsonl`.

## 2. Corpus

| CLB | Số đoạn | Id |
|---|---|---|
| CLB Lập Trình PTIT | 78 | 1-78 |
| Cộng đồng S2B (chung) | 6 | 79-84 |
| CLB Multimedia PTIT | 12 | 85-96 |
| CLB Nhà Sáng Tạo Game PTIT (GCC) | 22 | 97-118 |

Schema mỗi dòng:

```json
{"id": 1, "club": "CLB Lập Trình PTIT", "topic": "Giới thiệu chung", "text": "Giới thiệu CLB Lập Trình PTIT: ..."}
```

* `id`: định danh duy nhất, dùng trong `relevant_ids` và `retrieved_ids`
* `club`: CLB mà đoạn văn nói về; được phép dùng làm metadata filter
* `topic`: nhóm chủ đề (Giới thiệu chung, Tuyển thành viên, Đào tạo, Sự kiện nội bộ, Giải thưởng, Ban chủ nhiệm, Nội quy & nghĩa vụ, Liên hệ, ...)
* `text`: nội dung

## 3. Bộ câu hỏi

```json
{
  "id": "train-0001",
  "query": "CLB Lập Trình PTIT được thành lập ngày nào và phương châm hoạt động là gì?",
  "relevant_ids": [1],
  "answer": "CLB Lập Trình PTIT (PROPTIT) được thành lập ngày 09/10/2011, với phương châm hoạt động \"Chia sẻ để cùng nhau phát triển\".",
  "key_facts": ["09/10/2011", "Chia sẻ để cùng nhau phát triển"],
  "answerable": true,
  "difficulty": 1,
  "club": "CLB Lập Trình PTIT",
  "type": "trực tiếp"
}
```

| Trường | Ý nghĩa |
|---|---|
| `relevant_ids` | Các đoạn văn cần để trả lời. Với câu hỏi không có đáp án: các đoạn chứa thông tin liên quan mà `answer` nhắc tới, hoặc rỗng |
| `answer` | Đáp án chuẩn, chỉ dùng thông tin trong `relevant_ids` |
| `key_facts` | Các chuỗi bắt buộc phải có trong câu trả lời đúng (dùng cho String Presence); `a \| b` nghĩa là chấp nhận một trong hai cách viết |
| `answerable` | `false` với câu hỏi nằm ngoài tài liệu, dùng để chấm khả năng từ chối |
| `difficulty` | 1: tra cứu trực tiếp một đoạn; 2: diễn đạt khác hoặc cần 2 ý; 3: nhiều đoạn, so sánh, bẫy hoặc suy luận |
| `club` | CLB mà câu hỏi hướng tới: tên một trong ba CLB, `Cộng đồng S2B`, hoặc `Nhiều CLB` với câu hỏi so sánh |
| `type` | Kiểu câu hỏi: `trực tiếp`, `diễn đạt khác`, `không dấu/viết tắt`, `nhiều đoạn`, `so sánh`, `thời gian`, `mâu thuẫn nguồn`, `thiếu dữ liệu`, `không có đáp án`. Dùng để phân tích lỗi theo nhóm |

Các bộ câu hỏi được xây theo tỷ lệ ProPTIT : Multimedia : GCC xấp xỉ 4 : 2 : 2, cộng thêm câu hỏi liên CLB và khoảng 7–9% câu hỏi không có đáp án (train 11/126, mỗi bộ test 3/42). Public và Private Test có tỷ lệ câu khó cao hơn train. Các bộ không trùng câu hỏi với nhau; Private Test chỉ dùng các đoạn đã đối chiếu nguồn công khai.

## 4. Mức xác minh

Cột `verification` trong `sources.csv`:

* **Đã đối chiếu nguồn công khai** (93 đoạn): có URL nguồn và ngày nguồn cụ thể.
* **Nguồn nội bộ** (25 đoạn ProPTIT, chủ yếu là hỏi đáp tuyển thành viên và nội quy): lấy từ tài liệu gốc của CLB, chưa có nguồn công khai để đối chiếu.

## 5. Lưu ý về thời điểm hiệu lực

Nhiều thông tin chỉ đúng tại một thời điểm, và các đoạn văn đều ghi rõ mốc thời gian đó:

* Lịch tuyển thành viên chỉ áp dụng cho đợt được nêu (ví dụ đợt 2026).
* Ban chủ nhiệm gắn với nhiệm kỳ cụ thể; với Multimedia và GCC, thông tin chủ nhiệm lấy từ trang danh sách CLB của Học viện (không ghi ngày cập nhật).
* Một số kênh liên hệ (email, số điện thoại) khác nhau giữa các nguồn; corpus ghi rõ nguồn của từng thông tin.

Một hệ thống RAG tốt nên giữ nguyên các mốc thời gian này trong câu trả lời thay vì trình bày như quy định cố định.

## 6. Giấy phép

Dữ liệu chỉ được sử dụng cho mục đích học tập trong khuôn khổ NeoRAG Cup 2026. Bản quyền nội dung gốc thuộc về các CLB và nguồn được ghi trong `sources.csv`.
