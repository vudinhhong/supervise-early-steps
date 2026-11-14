# Nguồn của dữ liệu dẫn xuất

`code/artifacts/eval_suite_3471.jsonl` là tập hợp dẫn xuất từ ba bộ dữ liệu công
khai, mỗi bộ phát hành theo giấy phép MIT:

| Thành phần | Số bài | Nguồn |
|---|---|---|
| `gsm8k-heldout` | 1200 | `openai/gsm8k`, phần held-out của tập train |
| `gsm8k-test` | 1319 | `openai/gsm8k`, tập test |
| `math-heldout` | 627 | Hendrycks MATH, phần held-out của tập train |
| `math500` | 325 | `HuggingFaceH4/MATH-500` |

File chỉ chứa câu hỏi, đáp án đã chuẩn hoá và nhãn nguồn. Dòng đầu là một bản ghi
`_meta` ghi lại hạt giống chia tập và kết quả phép kiểm rời nhau với tập huấn luyện.

Trọng số mô hình không được phát hành ở đây; chúng thuộc họ Qwen2.5 và theo giấy
phép của nhà phát hành.
