# LexiGuard Hierarchical Statutory Parsing, Data Schemas & Knowledge Graph Extraction

## Enterprise-Grade Vietnamese Legislative AST, Zero-Loss Clause Segmentation, and Deterministic Graph Ingestion
Powered by Pydantic v2, Deterministic Regex-First Graph Extraction, and Neo4j Cypher Batching.

---

## 1. Architectural Motivation

### 1.1 Limitations of Naive Chunking on Vietnamese Legal Codes
Standard retrieval-augmented generation (RAG) architectures that rely on naive text splitters (such as fixed character or token sliding windows) fail catastrophically when applied to civil law legislative corpora:

1. **Semantic Boundary Fracture**: Dividing text by arbitrary token counts splits legal articles and clauses mid-sentence. When statutory conditions (*"If..."*) are separated from legislative sanctions (*"...shall be penalized"*), semantic integrity is permanently broken.
2. **Context Loss**: A statutory clause such as *"2. The employer shall not unilaterally terminate an employment contract..."* completely loses its governing scope when isolated from its parent Article, Chapter, and governing Code.
3. **Cross-Reference Blindness**: Vietnamese statutes frequently employ cross-statutory citations (e.g., *"Except as stipulated in Article 125 of this Code"* or *"pursuant to Clause 2, Article 150"*). Semantic vector similarity (cosine distance) cannot bridge cross-references when the vocabulary between the referencing and target clauses is orthogonal.

### 1.2 The LexiGuard Hierarchical Graph-RAG Architecture
LexiGuard addresses these limitations through a layered statutory processing architecture:
* **Hierarchical Statutory Chunker**: Deconstructs legal codes following Vietnam's formal legislative drafting hierarchy: `Part (Phần)` $\rightarrow$ `Chapter (Chương)` $\rightarrow$ `Section (Mục)` $\rightarrow$ `Article (Điều)` $\rightarrow$ `Clause (Khoản)` $\rightarrow$ `Point (Điểm)`.
* **Lineage & Context Enrichment**: Prepends full statutory breadcrumbs to each chunk prior to dense embedding.
* **Deterministic Graph Extraction**: Extracts structural containment (`CONTAINS`) and cross-statutory citations (`REFERENCES`, `EXCEPTION_TO`, `CITES_CLAUSE`, `APPLIES_STATUTE`) into a Neo4j Knowledge Graph for 1-to-2 hop traversals.

---

## 2. Standardized Data Schemas (Pydantic v2)

All data structures are defined in [src/core/schemas.py](file:///c:/PROJECTS/LexiGuard/src/core/schemas.py) using **Pydantic v2**, providing runtime type validation, serialization performance, and tight schema synchronization across Qdrant and Neo4j.

```
                              [LegalDocument]
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
            [LegalPart]                             [LegalChapter]
                 │                                       │
                 ▼                                       ▼
           [LegalChapter]                          [LegalSection]
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     ▼
                               [LegalArticle]
                                     │
                                     ▼
                                [LegalClause]
                                     │
                                     ▼
                                [LegalPoint]
```

### 2.1 Statutory Abstract Syntax Tree (AST)

| Schema Model | Hierarchy Level | Primary Fields | Canonical Citation Example |
| :--- | :--- | :--- | :--- |
| **`LegalPoint`** | Lettered point (`Điểm`) | `point_id` ("a", "b"), `content` | Điểm a Khoản 1 Điều 115 BLLĐ 2019 |
| **`LegalClause`** | Numbered clause (`Khoản`) | `clause_num` (1, 2), `content`, `points` | Khoản 1 Điều 115 BLLĐ 2019 |
| **`LegalArticle`** | Statutory article (`Điều`) | `article_num` (115), `title`, `clauses` | Điều 115 BLLĐ 2019 |
| **`LegalChapter`** | Legislative chapter (`Chương`) | `chapter_num` ("I"), `title`, `articles` | Chương I: Những quy định chung |
| **`LegalDocument`** | Complete statutory code | `law_id`, `law_title`, `chapters`, `metadata` | Bộ luật Lao động 2019 |

---

### 2.2 Vector Storage Unit: `LegalChunk`

Designed specifically as the atomic payload unit (Points) in Qdrant Vector Database:

| Field Name | Type | Description | Production Example |
| :--- | :--- | :--- | :--- |
| `chunk_id` | String | Deterministic unique identifier | `VN_BLDS_2015_D004_K02` |
| `law_id` | String | Normalized code slug | `bo-luat-dan-su-2015` |
| `canonical_citation` | String | Formal legislative citation | Khoản 2 Điều 4 Bộ luật Dân sự 2015 |
| `context_enriched_text` | String | Breadcrumb-enriched string for embedding | Prepend `[Law] > [Chapter] > [Article]` |
| `cross_references` | List[String] | Extracted citations | `["Điều 3"]` |
| `page_start` / `page_end` | Integer | Source PDF page boundaries | `1`, `1` |
| `bbox` | List[Float] | Docling spatial bounding box | `[l, t, r, b]` |
| `is_active` | Boolean | Statutory validity state | `True` |

#### Context Enrichment Mechanism
Prior to vectorization with `BAAI/bge-m3`, `LegalChunker` synthesizes `context_enriched_text`:
```plaintext
[Bộ luật Dân sự 2015] > [Chương I: Những quy định chung] > [Điều 4: Áp dụng Bộ luật dân sự]
Khoản 2: Luật khác có liên quan điều chỉnh quan hệ dân sự trong các lĩnh vực cụ thể không được trái với các nguyên tắc cơ bản của pháp luật dân sự quy định tại Điều 3 của Bộ luật này.
```
*Engineering Impact*: Dense 1024-dimensional embeddings retain the global legislative context of the parent Law, Chapter, and Article title, preventing semantic drift during vector search.

---

### 2.3 Neo4j Knowledge Graph Schema

Entities and relationships conform to standardized graph models in [src/core/schemas.py](file:///c:/PROJECTS/LexiGuard/src/core/schemas.py):

| Relationship Type | Semantic Definition | Statutory Context Example |
| :--- | :--- | :--- |
| `CONTAINS` | Hierarchical containment | Law contains Chapter; Chapter contains Article; Article contains Clause. |
| `REFERENCES` | General statutory cross-reference | *"Pursuant to Article 3...", "As stipulated in Article X..."* |
| `EXCEPTION_TO` | Exclusion or exception clause | *"Except as provided in Article X of this Code..."* |
| `CITES_CLAUSE` | Specific clause cross-reference | *"In accordance with Clause 2, Article 150..."* |
| `APPLIES_STATUTE` | Precedent applicability | Judicial Precedent applying a specific statutory article |
| `DEFINED_IN` | Statutory definition anchor | Legal definition pinned to a foundational article |
| `MANDATES_GUIDANCE`| Regulatory delegation | Legislative delegation to subordinate governmental decrees |
![Neo4j Statutory Knowledge Subgraph - Single Chapter Exemplar (Chapter I: General Provisions)](assets/neoj4_viz.png)
*Figure: Focused Neo4j Knowledge Subgraph illustrating the hierarchical tree (`CONTAINS`) and cross-referencing structure (`REFERENCES`) for a single statutory chapter (Chapter I: General Provisions).*

---

## 3. Hierarchical Statutory Chunker (`LegalChunker`)

Implemented in [src/ingestion/legal_chunker.py](file:///c:/PROJECTS/LexiGuard/src/ingestion/legal_chunker.py).

### 3.1 Zero-Loss Clause & Bullet Segmentation
Scanned Vietnamese legal texts exhibit varied formatting: standard numbers (`1.`, `2.`), bulleted numbers (`- 1.`, `- 2.`), lettered lists (`a)`, `b)`), and dash bullets (`-`).

`LegalChunker` implements specialized regular expressions designed to prevent clause fragmentation:

| Structural Target | Extraction Pattern (Regex) | Matched Format Example |
| :--- | :--- | :--- |
| **Part** | `^(?:#+\s*)?(?:PHẦN\|PHAN)\s+(?:THỨ\s+([A-ZÀ-Ỹa-zà-ỹ]+)\|([IVXLCDM]+\|\d+))` | "PHẦN THỨ NHẤT" |
| **Chapter** | `^(?:#+\s*)?(?:CHƯƠNG\|CHUONG)\s+([IVXLCDM]+\|\d+)` | "CHƯƠNG I" |
| **Section** | `^(?:#+\s*)?(?:MỤC\|MUC)\s+(\d+\|[IVXLCDM]+)` | "MỤC 1" |
| **Article** | `^(?:#+\s*)?(?:ĐIỀU\|DIEU)\s+(\d+)\s*[:.]\s*(.*)$` | "Điều 4. Áp dụng Bộ luật dân sự" |
| **Clause** | `^(?:#+\s*)?(?:[-*]\s*)?(\d+)\.\s+(.*)$` | "1.", "- 1.", "* 2." |
| **Point** | `^(?:#+\s*)?(?:[-*]\s*)?([a-zđA-ZĐ])\)\s+(.*)$` | "a)", "- b)", "* c)" |

### 3.2 Deterministic Identifier Strategy
To enable $O(1)$ zero-lookup alignment between Qdrant points and Neo4j graph nodes:
* **Civil Code 2015**: Prefix `VN_BLDS_2015`
  * Article 4: `VN_BLDS_2015_D004`
  * Clause 2, Article 4: `VN_BLDS_2015_D004_K02`
* **Labour Code 2019**: Prefix `VN_BLLD_2019`
  * Article 115: `VN_BLLD_2019_D115`
  * Clause 1, Article 115: `VN_BLLD_2019_D115_K01`

---

## 4. Deterministic Knowledge Graph Extraction (`LegalGraphExtractor`)

Implemented in [src/ingestion/graph_extractor.py](file:///c:/PROJECTS/LexiGuard/src/ingestion/graph_extractor.py).

### 4.1 Deterministic Regex-First Methodology
Rather than relying on unconstrained LLM information extraction—which frequently hallucinates non-existent statutory citations or misattributes cross-code references—`LegalGraphExtractor` operates deterministically:
1. **Rule-Based Parsing**: Regex engines extract explicit statutory citations (`Điều \d+`, `Khoản \d+ Điều \d+`) with 100% precision.
2. **Semantic Classification**:
   - Negative/exclusion modifiers (*"trừ trường hợp"*, *"ngoại trừ"*) $\rightarrow$ `EXCEPTION_TO`.
   - Granular clause indicators (*"khoản \d+ điều \d+"*) $\rightarrow$ `CITES_CLAUSE`.
   - Explicit cross-code references (*"theo quy định của Bộ luật Dân sự"*) $\rightarrow$ Cross-Statute Reference edges.
3. **Referential Integrity**: When a cited target article appears later in the document, placeholder nodes are automatically allocated to prevent dangling edges.

### 4.2 High-Throughput Batch Graph Ingestion
Graph mutations are executed in transactions using Cypher `UNWIND $batch AS item MERGE ...` batch statements:
* Groups node creations and relationship insertions into transactions of 100 items.
* Reduces database write execution time from ~60 seconds down to under 5 seconds for complete statutes.

---

## 5. Full Statutory Ingestion Execution & Corpus Metrics

Statutes ingested from original scanned gazette PDFs located in [data/raw_laws](file:///c:/PROJECTS/LexiGuard/data/raw_laws):
* `bo-luat-dan-su-2015.pdf` (94 scanned pages)
* `bo-luat-lao-dong-2019.pdf` (83 scanned pages)

### 5.1 Ingestion Pipeline Execution Metrics

| Statutory Code | Total Pages | Slicing Batches | Raw Elements | Qdrant Chunks | Neo4j Nodes | Neo4j Edges | Execution Time (CUDA) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Civil Code 2015** (`VN_BLDS_2015`) | 94 | 7 batches | 1,835 | 866 | 967 | 1,014 | ~16 min |
| **Labour Code 2019** (`VN_BLLD_2019`) | 83 | 6 batches | 1,426 | 684 | 621 | 658 | ~14 min |
| **Total Corpus** | **177 pages** | **13 batches** | **3,261** | **1,550** | **1,588** | **1,672** | **~30 min** |

### 5.2 Neo4j Knowledge Graph Composition
* **Total Nodes**: **1,588**
  - `Clause`: 1,087 nodes
  - `Article`: 474 nodes
  - `Chapter`: 24 nodes
  - `Law`: 3 nodes
  - `Precedent`: 3 nodes
* **Total Relationships**: **1,672**
  - `CONTAINS`: 1,568 edges
  - `REFERENCES`: 72 edges
  - `CITES_CLAUSE`: 16 edges
  - `APPLIES_STATUTE`: 8 edges
  - `EXCEPTION_TO`: 2 edges

### 5.3 Extracted Artifacts Directory
Extracted data artifacts reside in [data/extracted_laws/](file:///c:/PROJECTS/LexiGuard/data/extracted_laws):
* `combined_qdrant_chunks.json`: 1,550 standardized chunks with dense/sparse embeddings.
* `neo4j_import.cypher`: Complete, idempotent Cypher import script.
* Individual AST and chunk files for each statute.

---

## 6. CLI Operations & Test Suite Verification

### 6.1 Ingestion Commands
```bash
# Full production ingestion (all 177 pages)
python scripts/ingest_laws.py

# Ingest specific page limits for development
python scripts/ingest_laws.py --max-pages 10

# Database reset and verification
python scripts/reset_databases.py
```

### 6.2 Automated Test Execution
Run the unit and integration test suite:
```bash
python -m pytest tests/test_chunker.py tests/test_schema_sync.py -v
```

The focused deterministic benchmark suite currently passes **11/11 tests**.
Full graph-ingestion validation still requires the configured Neo4j/Qdrant
services and should be reported separately from the evaluator unit tests.
