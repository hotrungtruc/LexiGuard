# Kiến trúc Bóc tách Ngược Bản án & Bộ Dữ liệu Kiểm thử Chuẩn (Reverse-Engineered Benchmark)
## Tài liệu Kỹ thuật Tầng Kiểm định Chất lượng | Dự án LexiGuard

---

### 1. Vấn đề Cốt lõi & Lý do Cần "Bóc tách Ngược" (Reverse Dissection)

Trong thực tế xây dựng các hệ thống AI rà soát hợp đồng pháp lý (Contract Review AI), sai lầm kinh điển nhất là **Copy thô nguyên văn** nội dung trong Bản án của Tòa án để làm dữ liệu kiểm thử.

#### 1.1. Sự khác biệt giữa "Copy thô" và "Bóc tách Ngược":

* **Cách làm chưa tối ưu (Copy thô phần Nội dung vụ án)**:
  > *"Tại đơn khởi kiện ngày 15/03/2022, Công ty TT trình bày: Ông Nguyễn Đình D đã ký cam kết không cạnh tranh nhưng sau khi nghỉ việc lại sang làm việc tại Nhà máy Lọc hóa dầu Nghi Sơn nên Công ty khởi kiện yêu cầu Tòa án buộc ông D bồi thường 251.800.000 VNĐ..."*
  * **Hậu quả**: 
    1. **Nhiễm độc ngữ cảnh (Context Pollution & Data Leakage)**: Văn bản chứa đầy thông tin tố tụng (*"đơn khởi kiện"*, *"nguyên đơn"*, *"bị đơn"*, *"yêu cầu bồi thường"*). AI sẽ bị dẫn dắt tâm lý thiên kiến và đóng vai trò tóm tắt vụ kiện (Ex-post Case Summarizer) thay vì đóng vai trò Luật sư rà soát rủi ro hợp đồng trước khi ký (Ex-ante Contract Reviewer).
    2. **Mất tính khách quan**: Mô hình không được kiểm thử khả năng phát hiện "bẫy ngầm" (Stealth Traps) ẩn giấu trong hợp đồng.

* **Cách làm tối ưu (Bóc tách Ngược - Reverse Engineering)**:
  * **Đầu vào cho LexiGuard (Input Contract Markdown - Blind Input)**:
    > *"Điều 3. Cam kết bảo mật và không cạnh tranh: Người lao động cam kết trong thời hạn 12 tháng kể từ ngày chấm dứt HĐLĐ không được làm việc, cộng tác, tư vấn hoặc tham gia dưới bất kỳ hình thức nào tại Nhà máy Lọc hóa dầu Nghi Sơn (NSRP). Trường hợp vi phạm, Người lao động phải bồi thường 251.800.000 VNĐ."*
    *(Tuyệt đối giấu kín toàn bộ phần Nhận định và Phán quyết của Tòa án).*
  * **Đáp án chuẩn (Ground Truth độc lập để đối soát tự động)**:
    * `expected_articles`: `["HIEN_PHAP_2013_ART_35", "45/2019/QH14_ART_5", "45/2019/QH14_ART_10"]`
    * `ground_truth_reasoning`: Lập luận của Thẩm phán về việc xâm phạm quyền tự do lựa chọn việc làm và cấm tràn lan tại đơn vị khách hàng.
    * `expected_vulnerability_type`: `ILLEGAL_CLAUSE`
    * `expected_risk_level`: `HIGH`
    * `court_verdict_summary`: Tuyên bố vô hiệu toàn bộ cam kết; bác toàn bộ yêu cầu đòi bồi thường 251.800.000 VNĐ.

---

### 2. Mô hình 3 Thành phần Test Độc lập từ Một Bản án Chuẩn của TAND

Theo quy chuẩn của Hội đồng Thẩm phán TAND Tối cao, một bản án sơ thẩm / phúc thẩm có kết cấu 3 phần rành mạch:

```
[Bản án TAND / Phán quyết VIAC]
               │
               ├─────────────────────────────────────────┐
               │                                         │
               ▼                                         ▼
   ┌───────────────────────┐                 ┌───────────────────────┐
   │ 1. NỘI DUNG VỤ ÁN     │                 │ 2. NHẬN ĐỊNH CỦA TÒA  │ ──► [3. QUYẾT ĐỊNH / TUYÊN ÁN]
   └───────────┬───────────┘                 └───────────┬───────────┘     └───────────┬───────────┘
               │                                         │                             │
    [Lọc bỏ văn phong tố tụng]              [Trích xuất Điều luật & Lập luận]       [Chế tài & Sửa đổi]
               │                                         │                             │
               ▼                                         ▼                             ▼
   ┌───────────────────────┐                 ┌───────────────────────┐     ┌───────────────────────┐
   │ input_contract_clause │                 │ • expected_articles   │     │ • vulnerability_type  │
   │ (Blind Test Input cho │                 │ • ground_truth_       │     │ • expected_risk_level │
   │  ClauseExtractor &    │                 │   reasoning           │     │ • court_verdict       │
   │  LoopholeAuditor)     │                 │ (Chấm điểm Retrieval  │     │ • redline_reference   │
   └───────────────────────┘                 │  và RAGAS Reasoning)  │     │ (Chấm điểm F1 & Redline)
                                             └───────────────────────┘     └───────────────────────┘
```

| Phần trong Bản án Gốc | Phương pháp Xử lý Dữ liệu (Data Preparation) | Vai trò trong Hệ sinh thái Kiểm thử LexiGuard |
| :--- | :--- | :--- |
| **1. NỘI DUNG VỤ ÁN**<br/>*(Trích dẫn điều khoản hợp đồng & tình tiết)* | **Tái tạo Điều khoản Hợp đồng nguyên bản**: Sử dụng Regex và Heuristics lọc bỏ toàn bộ các câu tường thuật tố tụng (*"Nguyên đơn khởi kiện..."*, *"Tại bản tự khai..."*), chỉ giữ lại nguyên văn câu chữ điều khoản hợp đồng và bảng biểu mà hai bên đã ký. | **Đầu vào (Input) duy nhất** cho `DoclingParser` và `ClauseExtractor Agent`. Tuyệt đối che giấu phần Nhận định và Quyết định của Tòa để kiểm thử khả năng phát hiện độc lập của mô hình. |
| **2. NHẬN ĐỊNH CỦA TÒA ÁN**<br/>*(Lập luận của HĐXX & Điều luật áp dụng)* | **Bóc tách 2 nhãn chuẩn**:<br/>1. `expected_articles`: Danh sách mã điều luật chuẩn hóa (VD: `45/2019/QH14_ART_10`, `91/2015/QH13_ART_468`, `AN_LE_69_2023_AL`).<br/>2. `ground_truth_reasoning`: Luận điểm pháp lý cốt lõi của Thẩm phán. | • **Đánh giá Tầng Retrieval**: Dùng `expected_articles` để đo Precision@K và Recall@K trên Qdrant Hybrid Search & Neo4j Multi-Hop Graph.<br/>• **Đánh giá RAGAS**: Dùng `ground_truth_reasoning` để chấm điểm `faithfulness` và `answer_relevancy` của `LoopholeAuditor Agent`. |
| **3. QUYẾT ĐỊNH**<br/>*(Tuyên vô hiệu, cắt phạt vượt trần, buộc bồi thường...)* | **Chuyển hóa thành nhãn Expected Risk & Verdict**:<br/>• `expected_vulnerability_type`: Phân loại theo Taxonomy của LexiGuard.<br/>• `expected_risk_level`: Mức độ rủi ro (HIGH / MEDIUM / LOW).<br/>• `court_verdict_summary`: Kết quả xử lý cuối cùng của Tòa án.<br/>• `redline_reference`: Phương án sửa đổi điều khoản chuẩn mực theo án lệ. | • **Đánh giá Classification**: Đo lường Accuracy và F1-Score phân loại rủi ro.<br/>• **Đánh giá Automated Redlining**: Đo độ tương đồng ngữ nghĩa và tính hợp pháp của câu văn sửa đổi mà mô hình tạo ra so với căn cứ chuẩn của Tòa. |

---

### 3. Pipeline Kỹ thuật Bóc tách OCR & Chuẩn hóa Tự động

LexiGuard triển khai module `src/ingestion/judgment_dissector.py` và script thực thi `scripts/extract_judgments_ocr.py` vận hành theo quy trình 4 bước:

1. **OCR Trích xuất Văn bản Đa tầng**:
   - Đối với 16 bản án dạng PDF điện tử: Dùng `pypdf` trích xuất nhanh toàn bộ cấu trúc text layer.
   - Đối với các bản án scan dạng ảnh (như [772024LĐ-ST.pdf](file:///c:/PROJECTS/LexiGuard/data/test_contracts/772024LĐ-ST.pdf)): Kích hoạt `pypdfium2` render trang sang độ phân giải cao và chạy `EasyOCR` tăng tốc phần cứng trên GPU NVIDIA GeForce RTX 4050 Laptop.
2. **Phân rã 3 Phân vùng Văn bản (Boundary Segmentation)**:
   - Áp dụng biểu thức chính quy (Regex) bắt chính xác các mốc: `NỘI DUNG VỤ ÁN`, `NHẬN ĐỊNH CỦA TÒA ÁN` / `XÉT THẤY`, và `QUYẾT ĐỊNH` / `TUYÊN XỬ`.
   - Lưu trữ kết quả phân rã vào file [extracted_judgments_ocr.json](file:///c:/PROJECTS/LexiGuard/data/test_contracts/extracted_judgments_ocr.json).
3. **Lọc sạch Văn phong Tố tụng (De-narrative Sanitization)**:
   - Tự động tách các khối điều khoản nằm trong dấu ngoặc kép hoặc bắt đầu bằng *"Điều X."* / *"Khoản Y."*.
   - Khôi phục cấu trúc bảng biểu nếu có điều khoản phạt lũy tiến hoặc bảng lương.
4. **Chuẩn hóa Mã Định danh Pháp lý (Canonical Statutory Mapping)**:
   - Chuyển đổi toàn bộ tên luật và số điều sang định dạng khóa chuẩn của LexiGuard:
     - Bộ luật Lao động 2019 $\rightarrow$ `45/2019/QH14_ART_<num>`
     - Bộ luật Dân sự 2015 $\rightarrow$ `91/2015/QH13_ART_<num>`
     - Luật Thương mại 2005 $\rightarrow$ `36/2005/QH11_ART_<num>`
     - Luật Trọng tài Thương mại 2010 $\rightarrow$ `54/2010/QH12_ART_<num>`
     - Hiến pháp 2013 $\rightarrow$ `HIEN_PHAP_2013_ART_<num>`
     - Án lệ TAND Tối cao $\rightarrow$ `AN_LE_<num>_<year>_AL`
     - Nghị quyết HĐTP $\rightarrow$ `NQ_<num>_<year>_NQ_HDTP`

---

### 4. Kiến trúc Kiểm thử 3 Tầng (3-Tier Evaluation Strategy) Tối ưu Nhất cho LexiGuard

Thay vì đợi chạy xong toàn bộ hệ thống mới lấy kết quả cuối cùng đi so sánh với phán quyết của Tòa (rất tốn thời gian chạy LLM trên CPU và khó debug), hệ thống chia bài test thành **3 tầng từ nhanh đến sâu**:

```
[Bộ Hồ sơ Bản án Thực tế đã bóc tách]
        │
        ├──► TẦNG 1: Kiểm thử Truy xuất Đồ thị (Retrieval Test - Chạy mất ~2 giây, KHÔNG tốn LLM)
        │    • Input: Điều khoản hợp đồng
        │    • So sánh: Các điều luật Qdrant + Neo4j tìm ra vs. Điều luật Thẩm phán trích dẫn trong Bản án
        │    • Metric: Legal Citation Recall@6 (> 88%), MRR (Mean Reciprocal Rank)
        │
        ├──► TẦNG 2: Kiểm thử Định danh Lỗ hổng & Toán học (Deterministic Agent Test)
        │    • Input: Điều khoản hợp đồng + Bảng biểu PDF
        │    • So sánh: Nhãn `vulnerability_type`, `risk_level` & số tiền chênh lệch của Agent vs. Quyết định của Tòa
        │    • Metric: Classification Exact Match (%), Table Arithmetic Accuracy (100%)
        │
        └──► TẦNG 3: Kiểm thử Lập luận Pháp lý & Redlining (LLM-as-a-Judge / RAGAS Test)
             • Input: Lập luận `legal_reasoning` & `redline_suggestion` của LoopholeAuditor
             • So sánh: Đối chiếu ngữ nghĩa với phần "NHẬN ĐỊNH CỦA TÒA ÁN"
             • Metric: RAGAS Faithfulness (> 0.90), Answer Similarity / Judicial Alignment Score
```

#### 4.1. Chi tiết Từng Tầng Kiểm thử

##### TẦNG 1: Kiểm thử Truy xuất Đồ thị (Retrieval Test)
* **Thời gian thực thi**: ~2 giây / test case (Hoàn toàn không tốn tài nguyên LLM inference).
* **Đầu vào (Input)**: `input_contract_clause` (Điều khoản hợp đồng nguyên bản).
* **Cơ chế so sánh**: Đối chiếu danh sách điều luật được truy xuất từ Qdrant Hybrid Search (bge-m3 Dense + BM25 Sparse Vector) kết hợp Neo4j Multi-Hop Traversal với danh sách `expected_articles` mà Thẩm phán trích dẫn trong phần Nhận định.
* **Chỉ số đo lường (Metrics)**:
  * **Legal Citation Recall@6**: Đạt tối thiểu **> 88%** (Đảm bảo top 6 điều luật truy xuất phải bao phủ đầy đủ căn cứ pháp lý cốt lõi của Tòa án).
  * **Mean Reciprocal Rank (MRR)**: Đo vị trí xếp hạng của điều luật trọng tâm đầu tiên xuất hiện trong kết quả reranking.
  * **Multi-hop Graph Path Accuracy**: Kiểm tra đường truyền 2-hops từ Điều luật sang nút `:Precedent` hoặc Án lệ liên quan.

##### TẦNG 2: Kiểm thử Định danh Lỗ hổng & Toán học (Deterministic Agent Test)
* **Thời gian thực thi**: Chạy nhanh trên CPU/GPU thông qua FastMCP Tool & Structured Outputs.
* **Đầu vào (Input)**: Điều khoản hợp đồng + Bảng biểu PDF (Bảng lương, Bảng SLA, Bảng tiến độ thanh toán).
* **Cơ chế so sánh**:
  * So sánh trực tiếp nhãn `vulnerability_type` và `risk_level` của Agent với phân loại chuẩn từ phần Quyết định của Tòa án.
  * Gọi FastMCP Tool `verify_table_arithmetic` để kiểm toán độc lập các con số tài chính: kiểm tra mẫu số nhân sau dấu %, cộng dồn các mức phạt thành phần đối chiếu trần 8% Điều 301 LTM, quy đổi lãi suất ngày sang năm (%/ngày $\times$ 365) so với trần 20% Điều 468 BLDS.
* **Chỉ số đo lường (Metrics)**:
  * **Classification Exact Match (%)**: Đo tỷ lệ khớp tuyệt đối giữa nhãn lỗ hổng của Agent và nhãn của Tòa án.
  * **Table Arithmetic Accuracy (100%)**: Bắt buộc đạt **100%** độ chính xác tính toán số học, không chấp nhận sai số trong bảng biểu phạt và lãi suất.

##### TẦNG 3: Kiểm thử Lập luận Pháp lý & Redlining (LLM-as-a-Judge / RAGAS Test)
* **Thời gian thực thi**: Chạy sâu và toàn diện với cơ chế đánh giá đối chứng.
* **Đầu vào (Input)**: Lập luận `legal_reasoning`, kịch bản khai thác rủi ro `exploitation_scenario`, và câu văn sửa đổi `redline_suggestion` do node `LoopholeAuditor` tạo ra.
* **Cơ chế so sánh**: Đối chiếu ngữ nghĩa học thuật sâu với phần **"NHẬN ĐỊNH CỦA TÒA ÁN"** (Ratio Decidendi) và phương án xử lý trong **"QUYẾT ĐỊNH"**.
* **Chỉ số đo lường (Metrics)**:
  * **RAGAS Faithfulness (> 0.90)**: Đảm bảo 100% lập luận của AI bám sát văn bản luật và chứng cứ thực tế, hoàn toàn loại bỏ hiện tượng bịa đặt căn cứ pháp lý ảo giác.
  * **RAGAS Answer Relevancy**: Đánh giá câu trả lời giải quyết đúng và trúng trọng tâm câu hỏi kiểm thử pháp lý.
  * **Judicial Alignment Score**: Đo mức độ tương đồng giữa lý lẽ phản biện của Agent với phán quyết chính thức của Hội đồng xét xử TAND / Hội đồng Trọng tài VIAC.
  * **Redline Compliance Score**: Kiểm tra câu văn sửa đổi có vừa khắc phục được vi phạm pháp luật vừa bảo vệ quyền lợi thương mại của doanh nghiệp hay không.

---

### 5. Danh mục 13 Benchmark Test Cases Bóc tách Ngược

Toàn bộ 13 ca kiểm thử chuẩn đã được tích hợp đầy đủ vào hệ sinh thái kiểm thử:

1. **BENCH-CASE-001** (03/2023/LĐ-PT): Vô hiệu cam kết không cạnh tranh quá rộng tại đơn vị khách hàng.
2. **BENCH-CASE-002** (69/2023/AL): Công nhận hiệu lực của thỏa thuận NDA/NCA độc lập có điều khoản Trọng tài VIAC.
3. **BENCH-CASE-003** (12/2023/LĐ-PT): Bóc trần điều khoản ngụy trang sa thải thành tự động chấm dứt khi KPI < 70 điểm.
4. **BENCH-CASE-004** (04/2023/LĐ-PT & 77/2024/LĐ-ST): Bác bỏ việc cắt lương phạt kỷ luật và đòi chi phí đào tạo không có HĐ nghề.
5. **BENCH-CASE-005** (02/2025/LĐ-ST): Tuyên vô hiệu điều khoản thử việc 90 ngày, 70% lương và phạt tiền khi nghỉ việc.
6. **BENCH-CASE-006** (32/2022/KDTM-PT): Hủy bỏ cả HĐ phần mềm và HĐ thuê Server do không đạt mục đích giao kết.
7. **BENCH-CASE-007** (85/2024/KDTM-PT, 01/2017/KDTM-ST, 30/2019/KDTM-ST): Bắt lỗi nhầm lẫn giữa Bồi thường và Phạt 8%.
8. **BENCH-CASE-008** (15/2020/KDTM-PT): Cắt giảm mức phạt 8% tính trên tổng giá trị hợp đồng về đúng giá trị phần vi phạm.
9. **BENCH-CASE-009** (VIAC #054 & #055): Phát hiện bẫy chia nhỏ tên gọi phạt để tổng phạt đạt 18%, cắt về trần 8%.
10. **BENCH-CASE-010** (09/2020/KDTM-GĐT): Không công nhận điều khoản bồi thường ước tính Liquidated Damages không chứng minh thiệt hại.
11. **BENCH-CASE-011** (04/2024/KDTM-PT, 03/2024/KDTM-ST, 72/2023/KDTM-PT): Cắt giảm lãi chậm trả ngày vượt trần 20%/năm và bác bỏ lãi kép nhập gốc.
12. **BENCH-CASE-012** (111/2023/QĐ-PQTT): Tuyên vô hiệu thỏa thuận trọng tài bất khả thi ghi sai tên và cho phép kháng cáo.
13. **BENCH-CASE-013** (05/2021/KDTM-GĐT & 18/2022/KDTM-PT): Bác bỏ việc mở rộng bất khả kháng sang biến động tỷ giá và lỗi nhà thầu phụ.
