# Văn bản Quy phạm Pháp luật Gốc (Statutory Legal Corpora)

Thư mục: `data/raw_laws` | LexiGuard Official Gazette Scans

Thư mục này lưu trữ các văn bản luật chính thức dưới định dạng PDF công báo phục vụ kiểm thử và nạp dữ liệu (Ingestion Pipeline) cho LexiGuard.

---

## Danh mục Văn bản

1. **`bo-luat-dan-su-2015.pdf`**:
   - Tên đầy đủ: Bộ luật Dân sự số 91/2015/QH13
   - Ngày ban hành: 24/11/2015 | Ngày có hiệu lực: 01/01/2017
   - Kích thước: ~6.68 MB | Gồm 6 phần, 27 chương, 689 điều.
   - Vai trò: Khung pháp lý nền tảng điều chỉnh giao dịch dân sự, hợp đồng, trách nhiệm bồi thường thiệt hại và lãi suất.

2. **`bo-luat-lao-dong-2019.pdf`**:
   - Tên đầy đủ: Bộ luật Lao động số 45/2019/QH14
   - Ngày ban hành: 20/11/2019 | Ngày có hiệu lực: 01/01/2021
   - Kích thước: ~5.54 MB | Gồm 17 chương, 220 điều.
   - Vai trò: Điều chỉnh quan hệ lao động, tiền lương, thử việc, chấm dứt hợp đồng lao động và kỷ luật lao động.

> [!NOTE]
> Đường dẫn tệp trong thư mục này được sử dụng cố định bởi các test suite (`tests/test_docling_parser.py`, `tests/test_ocr_pdfs.py`) và pipeline nạp (`scripts/ingest_laws.py`).
