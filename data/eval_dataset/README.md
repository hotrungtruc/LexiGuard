# Bộ Dữ liệu Đánh giá Chuẩn (Evaluation & Benchmark Datasets)

Thư mục: `data/eval_dataset` | LexiGuard Evaluation Framework

Thư mục này lưu trữ các bộ dữ liệu đối chuẩn (Ground Truth Datasets & Manifests) phục vụ việc đánh giá chất lượng tự động của hệ thống LexiGuard (RAGAS framework, F1-Score phát hiện lỗ hổng, kiểm thử Multi-Agent LangGraph).

---

## Danh mục Tệp Dữ liệu

```plaintext
data/eval_dataset/
├── golden_qa_contracts.json              # 13 Test cases đối chứng bóc tách từ 17 Bản án TAND & Phán quyết VIAC
├── golden_contracts_manifest.json        # Ma trận kiểm thử 4 tầng (Tier 1 - Tier 4)
├── master_stealth_contract_manifest.json # Danh mục bẫy pháp lý ngụy trang tinh vi (Stealth Traps)
└── README.md                             # Bản tài liệu tra cứu này
```

---

## Chi tiết Từng Tập Dữ liệu

### 1. `golden_qa_contracts.json`
- **Quy mô**: 13 ca kiểm thử chuyên sâu (BENCH-CASE-001 đến BENCH-CASE-013).
- **Nguồn gốc**: Bóc tách trực tiếp từ các Bản án có thật tại `data/court_judgments/` (TAND TP.HCM, TAND Tối cao, VIAC).
- **Cấu trúc mỗi case**:
  - `case_reference`: Số bản án / quyết định cơ quan tài phán.
  - `contract_file`: File PDF bản án gốc tương ứng tại `data/court_judgments/`.
  - `input_contract_clause`: Điều khoản hợp đồng bị tranh chấp thực tế.
  - `audit_question`: Câu hỏi rà soát pháp lý gửi tới Agent.
  - `expected_articles`: Các điều luật quy định bắt buộc phải viện dẫn đúng.
  - `ground_truth_reasoning`: Lập luận phán quyết chuẩn của Thẩm phán/Trọng tài viên.
  - `court_verdict_summary`: Phán quyết cuối cùng của Tòa án (vô hiệu, cắt giảm trần 8%, bác yêu cầu...).

### 2. `golden_contracts_manifest.json`
- **Mục tiêu**: Đánh giá ma trận 4 tầng đối với toàn bộ chu trình xử lý (Full Matrix E2E):
  - **Tier 1**: Hợp đồng lao động số hoá + Bẫy tính toán bảng lương (Arithmetic Traps).
  - **Tier 2**: Hợp đồng dịch vụ phần mềm bẫy ngụy trang (Stealth Digital).
  - **Tier 3**: Hợp đồng scan méo góc có vết nhiễu & bảng biểu phức tạp.
  - **Tier 4**: Hợp đồng FDI có yếu tố đa ngôn ngữ/án lệ chéo.

### 3. `master_stealth_contract_manifest.json`
- **Mục tiêu**: Đo lường khả năng bắt bẫy vi mô (Micro-trap Sensitivity) bao gồm:
  - Bẫy nhập gốc lãi chậm trả (Lãi kép - Anatocism).
  - Bẫy phạt vi phạm vượt trần 8% ngụy trang bằng nhiều loại tên gọi phụ phí SLA.
  - Bẫy dịch thuật "Liquidated Damages" trái pháp luật Việt Nam.
