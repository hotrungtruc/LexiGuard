# Thư mục Hợp đồng & Văn bản Mẫu (Sample Contracts & Real Scans)

Thư mục: `data/sample_contracts` | LexiGuard OCR & Document Reconstruction Pipeline

Thư mục này phục vụ quy trình thử nghiệm và kiểm chứng pipeline OCR xử lý hợp đồng/công văn thực tế, bao gồm từ ảnh chụp/scan quang học thô đến văn bản tái cấu trúc PDF và trích xuất Markdown/JSON theo Nghị định 30/2020/NĐ-CP.

---

## 1. Cấu trúc Phân nhánh Thư mục

```plaintext
data/sample_contracts/
├── raw_scans/                 # Ảnh chụp / Quét quang học thực tế (JPG, 300+ DPI)
│   ├── SCAN CV GUI BO VV TU VAN PHAP LUAT.jpg
│   ├── SCAN CV GUI BO VV TU VAN PHAP LUAT 2.jpg
│   ├── SCAN CV GUI BO VV TU VAN PHAP LUAT 3.jpg
│   ├── SCAN CV GUI BO VV TU VAN PHAP LUAT 4.jpg
│   ├── SCAN CV GUI BO VV TU VAN PHAP LUAT 5.jpg
│   └── luu-ban-nhap-tu-dong-5.jpg
│
├── extracted_pdfs/            # PDF dựng lại chuẩn xác từ dữ liệu trích xuất
│   ├── cong_van_498_dskt.pdf  # Công văn 498/ĐSKT (PDF chuẩn dạng vector, layout NĐ 30)
│   └── don_kien_nghi_120_dls.pdf # Đơn kiến nghị 120/DLS
│
├── extracted_texts/           # Văn bản trích xuất phân tích layout theo Nghị định 30
│   ├── cong_van_498_dskt.json # Metadata vùng không gian, bounding boxes & thực thể
│   ├── cong_van_498_dskt.md   # Nội dung văn bản định dạng Markdown
│   ├── don_kien_nghi_120_dls.json
│   └── don_kien_nghi_120_dls.md
│
└── README.md                  # Hướng dẫn cấu trúc thư mục
```

---

## 2. Luồng Xử lý Dữ liệu

1. **Đầu vào ảnh thô (`raw_scans/`)**:
   - Chứa các ảnh scan công văn, hợp đồng thực tế với hiện tượng méo góc (skewing), nhiễu nền hoặc bóng mờ.
   - Được xử lý tiền xử lý quang học qua `src/ingestion/image_preprocessor.py` (Deskew, CLAHE, Lanczos4).
2. **Nhận dạng & Phân vùng theo Nghị định 30 (`extracted_texts/`)**:
   - Sử dụng `Decree30XYCut` phân tích 8 vùng không gian hành chính chuẩn: Quốc hiệu, Số hiệu văn bản, Cơ quan ban hành, Trích yếu, Nội dung, Nơi nhận, Chữ ký.
   - Xuất dữ liệu cấu trúc dạng JSON và Markdown (`.json`, `.md`).
3. **Dựng lại PDF vector độ phân giải cao (`extracted_pdfs/`)**:
   - Chạy script `scripts/generate_ocr_pdfs.py` đọc từ `extracted_texts/` để sinh PDF vector chuẩn hóa phục vụ kiểm thử end-to-end cho `DoclingParser`.
