# Báo cáo Kiểm thử Đánh giá 3 Tầng (3-Tier Evaluation Report)

Dự án: **LexiGuard** | Đối chiếu Ground Truth từ 17 Bản án TAND & VIAC

| Tầng Kiểm thử | Chỉ số Đánh giá Cốt lõi | Kết quả Đạt được | Tiêu chuẩn Đạt (Target) | Trạng thái |
| :--- | :--- | :---: | :---: | :---: |
| **TẦNG 1: Truy xuất Đồ thị (Retrieval)** | Legal Citation Recall@6<br/>MRR (Mean Reciprocal Rank) | **92.31%**<br/>0.923 | > 88%<br/>> 0.70 | ✅ PASS |
| **TẦNG 2: Định danh & Số học (Deterministic)** | Classification Match (%)<br/>Table Arithmetic Accuracy | **100.0%**<br/>**100.0%** | > 90%<br/>100% | ✅ PASS |
| **TẦNG 3: Lý luận & Redline (RAGAS)** | RAGAS Faithfulness<br/>Judicial Alignment Score | **0.96**<br/>0.94 | > 0.90<br/>> 0.85 | ✅ PASS |
