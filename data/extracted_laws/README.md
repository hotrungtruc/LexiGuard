# Dữ liệu Pháp luật Trích xuất (Extracted Legal Knowledge Base)

Thư mục: `data/extracted_laws` | LexiGuard Ingestion Artifacts

Thư mục này chứa các sản phẩm cấu trúc (structured artifacts) được sinh ra tự động bởi pipeline `scripts/ingest_laws.py` từ 2 bộ luật nguồn (`data/raw_laws/`). Đây là nguồn tri thức nền tảng cung cấp cho Qdrant Vector DB, Neo4j Graph Database và các MCP Tools (`legal_tools.py`).

> [!WARNING]
> Không được đổi tên hay di chuyển các file trong thư mục này vì các service (FastMCP server `legal_tools.py`, script `reset_databases.py`) đọc trực tiếp từ đường dẫn `data/extracted_laws/`.

---

## 1. Cấu trúc Các Tệp Dữ liệu

```plaintext
data/extracted_laws/
├── bo-luat-dan-su-2015.md          # Markdown trích xuất từ Docling (Bộ luật Dân sự 2015)
├── bo-luat-dan-su-2015_ast.json    # Cây cú pháp AST (Phần -> Chương -> Mục -> Điều -> Khoản -> Điểm)
├── bo-luat-dan-su-2015_chunks.json # Các chunk phân cấp có breadcrumbs cú pháp
├── bo-luat-dan-su-2015_graph.json  # Bộ 3 thực thể đồ thị (Triples: Cites, Modifies, Exceptions)
│
├── bo-luat-lao-dong-2019.md        # Markdown trích xuất từ Docling (Bộ luật Lao động 2019)
├── bo-luat-lao-dong-2019_ast.json  # Cây cú pháp AST
├── bo-luat-lao-dong-2019_chunks.json # Các chunk phân cấp
├── bo-luat-lao-dong-2019_graph.json  # Bộ 3 thực thể đồ thị
│
├── combined_qdrant_chunks.json     # File tổng hợp 1.340+ chunks chuẩn hóa nạp vào Qdrant
├── neo4j_import.cypher             # Script Cypher chứa truy vấn nạp node & quan hệ vào Neo4j
└── README.md                       # Bản tài liệu tra cứu này
```

---

## 2. Quy trình & Các Dịch vụ Tiêu thụ

| Tệp Dữ liệu | Cơ chế Sinh | Dịch vụ Tiêu thụ | Chức năng Phục vụ |
| :--- | :--- | :--- | :--- |
| `*.md` | IBM Docling OCR (`DoclingParser`) | Ingestion pipeline | Lưu trữ biểu diễn Markdown nguyên bản của văn bản luật |
| `*_ast.json` | `LegalChunker.build_ast()` | Ingestion pipeline | Mô hình hóa toàn bộ phân cấp chương, điều, khoản của đạo luật |
| `*_chunks.json` | `LegalChunker.chunk_ast()` | Qdrant Ingestion | Chia chunk ngữ nghĩa bảo toàn ngữ cảnh điều luật cha |
| `*_graph.json` | `GraphExtractor.extract()` | Neo4j Ingestion | Bóc tách quan hệ viện dẫn chéo (CITES, EXCEPTIONS, DEFINES) |
| `combined_qdrant_chunks.json` | `ingest_laws.py` | `QdrantLegalStore` & FastMCP `legal_tools` | Nạp vector hybrid và làm fallback tra cứu offline khi không có server |
| `neo4j_import.cypher` | `GraphExtractor.to_cypher()` | Neo4j Bolt DB | Thực thi nạp trực tiếp toàn bộ đồ thị tri thức pháp luật vào Neo4j |
