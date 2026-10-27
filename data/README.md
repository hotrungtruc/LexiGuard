# LexiGuard Data Repository Architecture

Thư mục: `data/` | Bản đồ Cấu trúc & Điều hướng Toàn bộ Dữ liệu Hệ thống

Toàn bộ dữ liệu của LexiGuard được phân cấp chuẩn hóa thành 5 phân hệ chuyên biệt, phục vụ trọn vẹn vòng đời từ Ingestion pháp luật, Tiền xử lý quang học hợp đồng, Đối chiếu bản án thực tế đến Đánh giá chất lượng tự động.

---

## 1. Sơ đồ Cấu trúc Tổng thể

```plaintext
data/
│
├── raw_laws/                  # [1] Văn bản quy phạm pháp luật gốc (PDF Công báo)
│   ├── bo-luat-dan-su-2015.pdf
│   ├── bo-luat-lao-dong-2019.pdf
│   └── README.md
│
├── extracted_laws/            # [2] Dữ liệu tri thức pháp luật đã bóc tách & nạp DB
│   ├── bo-luat-dan-su-2015.md / _ast.json / _chunks.json / _graph.json
│   ├── bo-luat-lao-dong-2019.md / _ast.json / _chunks.json / _graph.json
│   ├── combined_qdrant_chunks.json
│   ├── neo4j_import.cypher
│   └── README.md
│
├── court_judgments/           # [3] Bản án TAND, Án lệ & Quyết định Trọng tài thực tế
│   ├── lao_dong/              # 5 bản án tranh chấp HĐ Lao động (thử việc, sa thải, NCA...)
│   ├── kinh_doanh_thuong_mai/ # 8 bản án tranh chấp Kinh doanh Thương mại (phạt 8%, lãi chậm trả...)
│   ├── an_le_trong_tai/       # 4 Án lệ 69/2023/AL, GĐT 09/2020, 05/2021, PQTT 111/2023
│   ├── extracted_judgments_ocr.json
│   └── README.md
│
├── sample_contracts/          # [4] Hợp đồng mẫu & tài liệu thực nghiệm OCR
│   ├── raw_scans/             # Ảnh scan quang học thực tế (JPG, 300+ DPI, có méo/nhiễu)
│   ├── extracted_pdfs/        # PDF vector tái cấu trúc chất lượng cao
│   ├── extracted_texts/       # Phân tích layout theo Nghị định 30 (JSON & Markdown)
│   └── README.md
│
└── eval_dataset/              # [5] Bộ dữ liệu Ground Truth & Ma trận Đánh giá (RAGAS)
    ├── golden_qa_contracts.json              # 13 Test cases thực chứng từ bản án
    ├── golden_contracts_manifest.json        # Ma trận kiểm thử 4 tầng
    ├── master_stealth_contract_manifest.json # Danh mục bẫy pháp lý ngụy trang tinh vi
    └── README.md
```

---

## 2. Chi tiết Chức năng Các Phân hệ

| Phân hệ | Mục đích Sử dụng | Thành phần Chính | Tần suất Thay đổi |
| :--- | :--- | :--- | :--- |
| **`raw_laws/`** | Cung cấp tài liệu luật gốc chính quy cho quá trình nạp (Ingestion). | 2 bộ luật dạng PDF công báo. | Cố định (chỉ bổ sung khi nạp thêm luật). |
| **`extracted_laws/`** | Lưu trữ AST phân cấp, Vector Chunks và Triples quan hệ phục vụ FastMCP & Hybrid GraphRAG. | 10 files Markdown, JSON, Cypher. | Tự động cập nhật khi chạy `scripts/ingest_laws.py`. |
| **`court_judgments/`** | Lưu trữ các bản án có thật phục vụ bóc tách lỗ hổng hợp đồng (Contract Reverse Dissection). | 17 file PDF bản án phân loại thành 3 nhóm (Lao động, KDTM, Án lệ/Trọng tài). | Bổ sung khi mở rộng bộ án lệ. |
| **`sample_contracts/`** | Thử nghiệm pipeline quang học OCR, tiền xử lý Deskew và phân vùng theo Nghị định 30. | 6 ảnh scan thô, 2 PDF dựng lại, 4 file trích xuất cấu trúc. | Thường xuyên mở rộng khi thu thập mẫu hợp đồng mới. |
| **`eval_dataset/`** | Đánh giá độc lập mô hình RAGAS (Faithfulness, Relevancy, Precision) và phát hiện bẫy ngụy trang. | 3 file manifest JSON chứa ground truth chuẩn. | Ổn định, làm mốc benchmark chuẩn hóa. |

---

## 3. Quy tắc An toàn Dữ liệu (Path Invariants)

1. **Tuyệt đối không đổi tên đường dẫn**:
   - `data/raw_laws/bo-luat-dan-su-2015.pdf`
   - `data/extracted_laws/combined_qdrant_chunks.json`
   - `data/sample_contracts/extracted_pdfs/`
   Các đường dẫn này được gắn chặt vào các unit test và các module fallback trong `src/mcp_server/legal_tools.py`.
2. **Khi thêm bản án mới vào `data/court_judgments/`**:
   - Luôn đặt file PDF vào đúng thư mục con (`lao_dong/`, `kinh_doanh_thuong_mai/`, hoặc `an_le_trong_tai/`).
   - Cập nhật thông tin vào bảng tổng hợp tại [data/court_judgments/README.md](file:///c:/PROJECTS/LexiGuard/data/court_judgments/README.md).
