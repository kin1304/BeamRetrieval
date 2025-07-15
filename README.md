# BeamRetrieval

Dự án mô phỏng lại ý tưởng **Beam Retrieval** cho bài toán Multi-Hop Question Answering, dựa trên ý tưởng của bài báo và repository [canghongjian/beam_retriever](https://github.com/canghongjian/beam_retriever).

## Mục tiêu

- Triển khai thuật toán beam retrieval đơn giản cho bài toán truy hồi thông tin nhiều bước (multi-hop).
- Tổ chức code rõ ràng, dễ mở rộng cho các nghiên cứu tiếp theo.

## Cấu trúc thư mục

```
BeamRetrieval/
├── retrieval/
│   ├── __init__.py
│   └── beam_retriever.py
├── results/
├── fullwiki/
├── prompts/
├── qa/
├── utils/
├── README.md
├── .gitignore
```

- `retrieval/`: Chứa code chính cho beam retrieval.
- `results/`: Lưu kết quả truy hồi.
- `fullwiki/`, `prompts/`, `qa/`, `utils/`: Các thư mục mở rộng cho dữ liệu, prompt, QA, hàm phụ trợ.

## Hướng dẫn sử dụng

### 1. Cài đặt môi trường

Bạn nên sử dụng Python 3.7+.

Cài đặt các thư viện cần thiết (nếu có):

```bash
pip install -r requirements.txt  # nếu có file requirements.txt
```

### 2. Chạy thử beam retriever

Chạy file ví dụ:

```bash
python retrieval/beam_retriever.py
```

Kết quả sẽ in ra các đoạn văn bản liên quan nhất tới câu hỏi theo thuật toán beam search đơn giản.

### 3. Tùy chỉnh

Bạn có thể thay đổi tham số `beam_width`, `max_steps` hoặc thay đổi hàm `score` trong `beam_retriever.py` để phù hợp với bài toán của mình.

## Đóng góp

Nếu bạn muốn đóng góp code, ý tưởng hoặc mở rộng project, hãy tạo pull request hoặc liên hệ trực tiếp.

## Tham khảo

- [End-to-End Beam Retrieval for Multi-Hop Question Answering (NAACL 2024)](https://arxiv.org/abs/2308.08973)
- [Repo gốc: canghongjian/beam_retriever](https://github.com/canghongjian/beam_retriever)

---

**Chúc bạn nghiên cứu và phát triển thành công!** 