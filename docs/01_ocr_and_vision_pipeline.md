# LexiGuard Multimodal Document AI & OCR Pipeline

## High-Precision Vietnamese Legal Document Parsing, Spatial Provenance, and Structural Layout Extraction
Powered by IBM Docling (DocLayNet, TableFormer) with optional OCR acceleration and spatial provenance.

---

## 1. System Overview

The `src/ingestion` module provides an enterprise-grade document ingestion pipeline specifically tailored for Vietnamese statutory codes, administrative decrees, and complex economic contracts. 

The pipeline provides high accuracy, consistent diacritic fidelity, and architectural simplicity:
- **Layout Segmentation & Document Structure**: IBM Docling with the DocLayNet deep learning model parses paragraphs, section headers, statutory titles, and footnotes into a clean semantic hierarchy.
- **Optical Character Recognition (OCR)**: Docling OCR is configured for Vietnamese and English when OCR is enabled. Accelerator selection is runtime-dependent; CUDA is preferred when available, with CPU fallback.
- **Table Matrix Extraction**: IBM Docling TableFormer extracts complex administrative and compensation tables into normalized Markdown and coordinate-anchored cell matrices.
- **Incremental Batch Slicing**: High-capacity multi-page statutes (up to 100+ scanned pages) can be sliced into configurable batches with cleanup between slices. This reduces memory pressure but does not guarantee zero native-extension failures.

![Multimodal Legal OCR Pipeline Architecture](assets/single_branch_ocr_pipeline.png)

---

## 2. Core Processing Capabilities

### 2.1 Bilingual Optical Character Recognition (EasyOCR `vi` + `en`)
Vietnamese legal documents contain complex tone marks and diacritic combinations (e.g., `ộ`, `ể`, `ẵ`, `ự`). The OCR engine is configured with:
- Combined language models (`['vi', 'en']`) to handle Vietnamese statutes alongside English technical, economic, and commercial clauses.
- Optimized image scaling (`scale=2.0`), balancing fine character resolution with high processing throughput (~10.1 seconds per page on CUDA).
- Complete preservation of Unicode diacritics, ensuring downstream dense embedding models (`BAAI/bge-m3`) and sparse models (`BM25`) perform exact statutory token matching.

### 2.2 Layout Segmentation & Structural Classification (DocLayNet)
DocLayNet automatically detects and isolates:
- Section headings and article titles (`# CHƯƠNG`, `## Điều`).
- Numbered statutory clauses and lettered points.
- Header lines, footer notes, and gazette pagination markers.

### 2.3 Table Structure Recognition (TableFormer)
Legal statutes frequently incorporate multi-column administrative schedules, compensation brackets, and fee matrices. The pipeline detects tabular boundaries and reconstructs cell relationships into standard Markdown tables while retaining cell coordinate provenance.

### 2.4 Coordinate Normalization & Visual Provenance
Every extracted element preserves its spatial bounding box:
- Coordinate format: `[l, t, r, b]` with `BOTTOMLEFT` origin.
- Clause-level visual provenance enabling side-by-side UI audit verification against the original scanned PDF page.

---

## 3. Engineering Implementation Details

### 3.1 DoclingParser Configuration
Located in [src/ingestion/docling_parser.py](../src/ingestion/docling_parser.py), `DoclingParser` lazy-loads the IBM Docling `DocumentConverter` only when a document is parsed. This keeps deterministic unit tests independent from native OCR extensions:

```python
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = self.do_ocr

if self.do_ocr:
    pipeline_options.ocr_options = EasyOcrOptions(
        lang=["vi", "en"],
        scale=2.0,  # Optimized scale: 10.39s/page with high diacritic fidelity
    )

pipeline_options.do_table_structure = self.do_table_structure
pipeline_options.accelerator_options = AcceleratorOptions(
    device=AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU,
    num_threads=self.num_threads
)
```

### 3.2 Incremental Batch Slicing Engine
Large scanned legal statutes such as the Vietnamese Civil Code 2015 (94 scanned pages) can saturate GPU memory if loaded as an unsegmented document. `DoclingParser.parse()` incorporates memory-safe batch slicing:
- Documents exceeding `batch_size` (default: 15 pages) are temporarily isolated into sub-PDFs using `pypdfium2`.
- Each slice is converted independently with explicit progress logging.
- Spatial provenance (`ItemProvenance.page_no`) is calibrated to reflect original document page indices:
  ```python
  prov = self._extract_item_provenance(item)
  if page_range is not None:
      prov.page_no += page_range[0]
  ```
- CUDA cache is explicitly purged between slices via `torch.cuda.empty_cache()`.

---

## 4. Empirical Performance & Benchmark Results

The pipeline was benchmarked and validated on the official scanned statutory corpora in [data/raw_laws](file:///c:/PROJECTS/LexiGuard/data/raw_laws):

### 4.1 Corpus Extraction Metrics

| Statutory Code | Total Scanned Pages | OCR Batches | Characters Extracted | Layout Elements | Execution Throughput | Peak VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Bộ luật Dân sự 2015** (`bo-luat-dan-su-2015.pdf`) | 94 pages | 7 batches | 195,396 chars | 1,835 elements | ~10.3 s / page | 2,235 MB |
| **Bộ luật Lao động 2019** (`bo-luat-lao-dong-2019.pdf`) | 83 pages | 6 batches | 168,356 chars | 1,426 elements | ~9.8 s / page | 2,210 MB |
| **Combined Corpus** | **177 pages** | **13 batches** | **363,752 chars** | **3,261 elements** | **10.1 s / page (avg)** | **< 2.3 GB** |

### 4.2 Historical Corpus Run

The following output is a historical corpus run and should not be interpreted as a current production SLA. Re-run it in the validated native integration environment before publishing performance claims.
```
2026-10-19 11:36:20.856 | INFO | Splitting bo-luat-dan-su-2015.pdf (94 pages) into batches of 15 pages...
2026-10-19 11:36:20.856 | INFO | [bo-luat-dan-su-2015.pdf] Processing OCR batch 1/7 (pages 1-15)...
...
2026-10-19 11:45:33.967 | INFO | Extracted 195396 characters, 1835 layout elements across 94 pages.
2026-10-19 11:45:34.090 | INFO | [OK] Ingestion completed for Bộ luật Dân sự 2015: 866 chunks, 967 nodes, 1014 edges.
2026-10-19 11:52:26.365 | INFO | Extracted 168356 characters, 1426 layout elements across 83 pages.
2026-10-19 11:52:26.525 | INFO | [OK] Ingestion completed for Bộ luật Lao động 2019: 684 chunks, 621 nodes, 658 edges.
2026-10-19 11:52:26.614 | INFO | ALL LEGAL CORPORA INGESTED SUCCESSFULLY!
2026-10-19 11:52:26.614 | INFO | Total Unified Chunks for Qdrant: 1550
2026-10-19 11:52:26.614 | INFO | Total Unified Graph Nodes:       1588
2026-10-19 11:52:26.614 | INFO | Total Unified Graph Edges:       1672
```

---

## 5. Usage & CLI Reference

### 5.1 Programmatic API
```python
from pathlib import Path
from src.ingestion.docling_parser import DoclingParser

parser = DoclingParser(
    do_ocr=True,
    do_table_structure=False,
    ocr_engine="easyocr",
    device="cuda"
)

# Parse full statute with 15-page batch slicing
doc = parser.parse(
    "data/raw_laws/bo-luat-dan-su-2015.pdf",
    batch_size=15
)

print(f"Extracted {len(doc.full_markdown)} characters across {doc.total_pages} pages.")
for elem in doc.elements[:5]:
    print(f"[{elem.element_type}] Page {elem.provenance.page_no}: {elem.text[:80]}")
```

### 5.2 Command Line Ingestion
To run the full legal corpus ingestion pipeline:
```bash
python scripts/ingest_laws.py --all-pages --engine easyocr --device cuda --batch-size 15
```

Optional CLI flags:
- `--max-pages N`: Limit parsing to first $N$ pages per code for rapid verification.
- `--all-pages`: Parse all pages across both statutory codes.
- `--device {cuda,cpu,auto}`: Accelerator selection (default: `auto`).
- `--batch-size N`: Batch slice size in pages (default: `15`).
- `--no-neo4j-sync`: Skip direct Neo4j synchronization.
- `--no-qdrant-sync`: Skip direct Qdrant vector upsert.

---

## 6. Testing & Quality Assurance
The pipeline is verified with unit tests in [tests/test_ocr_pdfs.py](file:///c:/PROJECTS/LexiGuard/tests/test_ocr_pdfs.py) and [tests/test_docling_parser.py](file:///c:/PROJECTS/LexiGuard/tests/test_docling_parser.py):
```bash
pytest tests/test_ocr_pdfs.py tests/test_docling_parser.py -v
```
The deterministic ingestion/table benchmark is safe for the default test run. Native Docling/OCR tests must be run explicitly:

```powershell
$env:LEXIGUARD_RUN_DOCLING_INTEGRATION = "1"
python -m pytest tests/test_docling_parser.py tests/test_ocr_pdfs.py -q
```

On Windows, validate the pandas/pyarrow environment before enabling this mode..
