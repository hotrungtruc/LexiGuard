"""
Docling Parser: Advanced OCR & Document Layout Extraction using IBM Docling
=============================================================================
Extract PDFs, including scanned pages and financial tables, into Markdown
with spatial metadata (page_no, bbox, char_offsets).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Field

from src.core.hardware import hardware_profile

# Prevent pandas/pyarrow timezone initialization from loading an incompatible
# native path on Windows before Docling starts its conversion pipeline.
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")

# Docling is loaded lazily because its pandas/pyarrow native dependencies can
# crash the interpreter during test discovery on some Windows environments.
DOCLING_AVAILABLE = False
_DOCLING_LOAD_ATTEMPTED = False


def _load_docling_components() -> bool:
    """Load Docling only when a real document conversion is requested."""
    global DOCLING_AVAILABLE, _DOCLING_LOAD_ATTEMPTED
    if _DOCLING_LOAD_ATTEMPTED:
        return DOCLING_AVAILABLE
    _DOCLING_LOAD_ATTEMPTED = True
    try:
        global DocumentConverter, PdfFormatOption, InputFormat
        global PdfPipelineOptions, TableStructureOptions, AcceleratorOptions
        global AcceleratorDevice, RapidOcrOptions, EasyOcrOptions, OcrMode
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            EasyOcrOptions,
            OcrMode,
            PdfPipelineOptions,
            RapidOcrOptions,
            TableStructureOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except (ImportError, OSError) as exc:
        logger.warning("Docling native components unavailable; fallback mode enabled: {}", exc)
        return False
    DOCLING_AVAILABLE = True
    return True


class BoundingBox(BaseModel):
    """Spatial bounding box representation for document layout elements."""

    l: float = Field(description="Left / X0 coordinate")
    t: float = Field(description="Top / Y0 coordinate")
    r: float = Field(description="Right / X1 coordinate")
    b: float = Field(description="Bottom / Y1 coordinate")
    coord_origin: str = Field(
        default="BOTTOMLEFT", description="Coordinate system origin (e.g. BOTTOMLEFT or TOPLEFT)"
    )

    @property
    def as_list(self) -> List[float]:
        return [self.l, self.t, self.r, self.b]

    @property
    def width(self) -> float:
        return abs(self.r - self.l)

    @property
    def height(self) -> float:
        return abs(self.b - self.t)


class ItemProvenance(BaseModel):
    """Document spatial provenance tracing back to exact page and coordinates."""

    page_no: int = Field(ge=1, description="1-indexed page number")
    bbox: Optional[BoundingBox] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class ParsedElement(BaseModel):
    """An individual structural element (Paragraph, Heading, Table, List Item, etc.)."""

    element_id: str
    element_type: Literal["heading", "paragraph", "table", "list_item", "footnote", "code", "other"]
    text: str
    markdown: str
    provenance: ItemProvenance
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedTable(BaseModel):
    """Detailed tabular structure (e.g. salary grades, liquidated damages brackets)."""

    table_id: str
    page_no: int
    num_rows: int
    num_cols: int
    markdown: str
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    bbox: Optional[BoundingBox] = None


class ParsedDocument(BaseModel):
    """Complete parsed document output containing full markdown, elements, and tables."""

    source_path: str
    filename: str
    total_pages: int
    full_markdown: str
    elements: List[ParsedElement] = Field(default_factory=list)
    tables: List[ParsedTable] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DoclingParser:
    """
    Enterprise Document Layout & OCR Parser utilizing IBM Docling.
    Features:
      - OCR scanning for image-based / scanned PDF contracts.
      - Deep Table Structure Recognition (salary brackets, penalties, payment schedules).
      - Clause-level spatial provenance (page_no, bounding boxes).
      - Markdown conversion with clean syntactic hierarchies.
    """

    def __init__(
        self,
        do_ocr: bool = True,
        do_table_structure: bool = False,
        ocr_engine: str = "easyocr",
        device: str = "auto",
        num_threads: Optional[int] = None,
    ):
        self.do_ocr = do_ocr
        self.do_table_structure = do_table_structure
        self.ocr_engine = ocr_engine.lower() if ocr_engine else "easyocr"
        # Centralized Hardware Profiling: dynamic GPU/CPU & optimal OCR threads
        if not device or device.lower() == "auto":
            self.device = hardware_profile.optimal_device
        else:
            self.device = device.lower()

        if num_threads is None or num_threads <= 0:
            self.num_threads = hardware_profile.optimal_ocr_threads
            logger.info(
                f"Docling using centralized hardware profile: {self.num_threads} worker threads "
                f"({hardware_profile.cpu_physical_cores} physical cores), device: {self.device}"
            )
        else:
            self.num_threads = num_threads

        self._converter = None

        logger.info("DoclingParser initialized; native components load on first valid parse.")

    def _init_docling_converter(self):
        """Initializes IBM Docling DocumentConverter with optimized PDF pipeline options."""
        try:
            if not _load_docling_components():
                self._converter = None
                return
            import torch

            # 1. Device selection: Auto-detect GPU/CUDA or fallback to CPU
            target_device = AcceleratorDevice.CPU
            if self.device in ("cuda", "auto"):
                if torch.cuda.is_available():
                    target_device = AcceleratorDevice.CUDA
                    # Register torch CUDA/cuDNN DLLs on Windows for ONNX Runtime / RapidOCR
                    try:
                        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
                        if hasattr(os, "add_dll_directory") and os.path.exists(torch_lib):
                            os.add_dll_directory(torch_lib)
                        if torch_lib not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
                    except Exception as dll_err:
                        logger.warning(f"Failed to register torch CUDA DLLs: {dll_err}")

                    device_name = torch.cuda.get_device_name(0)
                    logger.info(f"Docling GPU Acceleration ACTIVATED: CUDA ({device_name})")
                elif self.device == "cuda":
                    logger.warning(
                        "CUDA device was requested but torch.cuda.is_available() is False. Falling back to CPU."
                    )

            if target_device == AcceleratorDevice.CPU:
                logger.info(f"Docling running on CPU ({self.num_threads} worker threads).")

            # 2. Configure official IBM Docling pipeline options
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = self.do_ocr
            if self.do_ocr:
                if self.ocr_engine == "easyocr":
                    try:
                        pipeline_options.ocr_options = EasyOcrOptions(
                            lang=["vi", "en"],
                            scale=2.0,
                        )
                        logger.info(
                            "Docling OCR backend configured with EasyOCR (lang=['vi', 'en'], scale=2.0)."
                        )
                    except Exception as easy_err:
                        logger.warning(
                            f"Failed to initialize EasyOCR: {easy_err}. Falling back to RapidOCR."
                        )
                        pipeline_options.ocr_options = RapidOcrOptions(mode=OcrMode.FULL_PAGE)
                else:
                    try:
                        pipeline_options.ocr_options = RapidOcrOptions(mode=OcrMode.FULL_PAGE)
                        logger.info("Docling OCR backend configured with RapidOCR.")
                    except Exception as ocr_err:
                        logger.debug(f"Defaulting OCR options: {ocr_err}")

            pipeline_options.do_table_structure = self.do_table_structure
            if self.do_table_structure and hasattr(pipeline_options, "table_structure_options"):
                pipeline_options.table_structure_options.do_cell_matching = True

            pipeline_options.accelerator_options = AcceleratorOptions(
                device=target_device, num_threads=self.num_threads
            )

            self._converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
            logger.info(
                f"IBM Docling DocumentConverter initialized with device={target_device}, "
                f"threads={self.num_threads}, ocr={self.do_ocr}, table_structure={self.do_table_structure}."
            )
        except Exception as e:
            logger.error(f"Failed to initialize Docling DocumentConverter: {e}. Falling back.")
            self._converter = None

    def parse(
        self,
        file_path: Union[str, Path],
        max_pages: Optional[int] = None,
        page_range: Optional[Tuple[int, int]] = None,
        batch_size: Optional[int] = None,
    ) -> ParsedDocument:
        """
        Parses a PDF/image document into a structured ParsedDocument.
        Extracts full markdown, clause elements with bbox, and parsed tables
        using official IBM Docling pipeline (DocLayNet + EasyOCR + TableFormer).
        """
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        logger.info(
            f"Parsing document layout & OCR via IBM Docling: {path.name} "
            f"(threads={self.num_threads}, max_pages={max_pages}, page_range={page_range}, batch_size={batch_size})"
        )

        if self._converter is None:
            self._init_docling_converter()

        if DOCLING_AVAILABLE and self._converter is not None:
            # Batch slicing for large PDF documents to optimize VRAM and report progress
            if (
                path.suffix.lower() == ".pdf"
                and batch_size
                and batch_size > 0
                and page_range is None
            ):
                try:
                    import pypdfium2 as pdfium

                    src_doc = pdfium.PdfDocument(str(path))
                    total_doc_pages = len(src_doc)
                    limit_pages = (
                        min(total_doc_pages, max_pages)
                        if max_pages and max_pages > 0
                        else total_doc_pages
                    )
                    if limit_pages > batch_size:
                        logger.info(
                            f"Splitting {path.name} ({limit_pages} pages) into batches of {batch_size} pages..."
                        )
                        all_elements: List[ParsedElement] = []
                        all_tables: List[ParsedTable] = []
                        markdown_parts: List[str] = []
                        total_batches = (limit_pages + batch_size - 1) // batch_size

                        for b_idx, start_idx in enumerate(
                            range(0, limit_pages, batch_size), start=1
                        ):
                            end_idx = min(start_idx + batch_size, limit_pages)
                            logger.info(
                                f"[{path.name}] Processing OCR batch {b_idx}/{total_batches} (pages {start_idx + 1}-{end_idx})..."
                            )
                            sub_doc = self._parse_with_docling(
                                path, page_range=(start_idx, end_idx)
                            )
                            markdown_parts.append(sub_doc.full_markdown)
                            all_elements.extend(sub_doc.elements)
                            all_tables.extend(sub_doc.tables)
                            try:
                                import torch

                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:
                                pass

                        return ParsedDocument(
                            source_path=str(path),
                            filename=path.name,
                            total_pages=limit_pages,
                            full_markdown="\n\n".join(markdown_parts),
                            elements=all_elements,
                            tables=all_tables,
                            metadata={
                                "engine": "ibm_docling",
                                "status": "success",
                                "batched": True,
                            },
                        )
                except Exception as batch_err:
                    logger.warning(
                        f"Batch processing encountered note: {batch_err}. Falling back to standard conversion."
                    )

            return self._parse_with_docling(path, max_pages=max_pages, page_range=page_range)
        else:
            return self._fallback_parse(path, max_pages=max_pages)

    def _parse_with_docling(
        self,
        path: Path,
        max_pages: Optional[int] = None,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> ParsedDocument:
        """Executes conversion using IBM Docling engine."""
        temp_slice_path = None
        target_path = path
        try:
            if path.suffix.lower() == ".pdf":
                if page_range is not None or max_pages is not None:
                    try:
                        import pypdfium2 as pdfium

                        src_doc = pdfium.PdfDocument(str(path))
                        total_doc_pages = len(src_doc)

                        if page_range is not None:
                            start_idx, end_idx = page_range
                            pages_to_slice = list(
                                range(max(0, start_idx), min(total_doc_pages, end_idx))
                            )
                        else:
                            pages_to_slice = list(range(min(total_doc_pages, max_pages)))

                        if len(pages_to_slice) < total_doc_pages:
                            import tempfile

                            slice_doc = pdfium.PdfDocument.new()
                            slice_doc.import_pages(src_doc, pages_to_slice)
                            fd, temp_str = tempfile.mkstemp(
                                suffix=".pdf",
                                prefix=f"lexiguard_slice_{path.stem}_",
                            )
                            os.close(fd)
                            temp_slice_path = Path(temp_str)
                            slice_doc.save(str(temp_slice_path))
                            target_path = temp_slice_path
                    except Exception as slice_err:
                        logger.warning(
                            f"Could not slice PDF: {slice_err}. Proceeding with full file."
                        )

            conversion_result = self._converter.convert(str(target_path))
            doc = conversion_result.document

            full_markdown = doc.export_to_markdown()
            total_pages = (
                doc.pages_count
                if hasattr(doc, "pages_count")
                else len(doc.pages)
                if hasattr(doc, "pages")
                else 1
            )

            elements: List[ParsedElement] = []
            tables: List[ParsedTable] = []

            # Iterate through Docling items to extract provenance and layout
            if hasattr(doc, "iterate_items"):
                for idx, (item, _level) in enumerate(doc.iterate_items()):
                    elem_type = self._map_docling_label(item)
                    text = getattr(item, "text", "")

                    # For table items, get table markdown
                    if elem_type == "table" and hasattr(item, "export_to_markdown"):
                        elem_markdown = item.export_to_markdown()
                    else:
                        elem_markdown = text

                    prov = self._extract_item_provenance(item)
                    if page_range is not None:
                        prov.page_no += page_range[0]

                    parsed_elem = ParsedElement(
                        element_id=f"elem_{idx:04d}",
                        element_type=elem_type,
                        text=text,
                        markdown=elem_markdown,
                        provenance=prov,
                        metadata={"docling_label": str(getattr(item, "label", ""))},
                    )
                    elements.append(parsed_elem)

                    # If table, also store in tables collection
                    if elem_type == "table":
                        parsed_table = self._build_parsed_table(item, idx, prov.page_no, prov.bbox)
                        if parsed_table:
                            tables.append(parsed_table)

            return ParsedDocument(
                source_path=str(path),
                filename=path.name,
                total_pages=total_pages,
                full_markdown=full_markdown,
                elements=elements,
                tables=tables,
                metadata={"engine": "ibm_docling", "status": "success"},
            )

        except Exception as e:
            logger.error(
                f"Docling conversion encountered an error: {e}. Executing fallback parser."
            )
            return self._fallback_parse(path)
        finally:
            if temp_slice_path and temp_slice_path.exists():
                try:
                    temp_slice_path.unlink()
                except Exception:
                    pass

    def _map_docling_label(
        self, item: Any
    ) -> Literal["heading", "paragraph", "table", "list_item", "footnote", "code", "other"]:
        """Maps Docling item labels to standardized element types."""
        label_str = str(getattr(item, "label", "")).lower()
        if "section_header" in label_str or "title" in label_str or "heading" in label_str:
            return "heading"
        elif "table" in label_str:
            return "table"
        elif "list_item" in label_str:
            return "list_item"
        elif "footnote" in label_str:
            return "footnote"
        elif "code" in label_str:
            return "code"
        elif "paragraph" in label_str or "text" in label_str:
            return "paragraph"
        return "other"

    def _extract_item_provenance(self, item: Any) -> ItemProvenance:
        """Extracts page number and bounding box coordinates from Docling provenance."""
        page_no = 1
        bbox = None

        if hasattr(item, "prov") and item.prov:
            first_prov = item.prov[0]
            page_no = getattr(first_prov, "page_no", 1)
            raw_bbox = getattr(first_prov, "bbox", None)

            if raw_bbox:
                coord_origin = str(getattr(raw_bbox, "coord_origin", "BOTTOMLEFT"))
                bbox = BoundingBox(
                    l=float(getattr(raw_bbox, "l", 0.0)),
                    t=float(getattr(raw_bbox, "t", 0.0)),
                    r=float(getattr(raw_bbox, "r", 0.0)),
                    b=float(getattr(raw_bbox, "b", 0.0)),
                    coord_origin=coord_origin,
                )

        return ItemProvenance(page_no=page_no, bbox=bbox)

    def _build_parsed_table(
        self, item: Any, idx: int, page_no: int, bbox: Optional[BoundingBox]
    ) -> Optional[ParsedTable]:
        """Constructs a ParsedTable model from Docling TableItem."""
        try:
            markdown = item.export_to_markdown() if hasattr(item, "export_to_markdown") else ""

            # Extract data grid if accessible
            data_grid: List[List[str]] = []
            if hasattr(item, "data") and hasattr(item.data, "grid"):
                for row in item.data.grid:
                    data_grid.append(
                        [str(cell.text) if hasattr(cell, "text") else str(cell) for cell in row]
                    )

            headers = data_grid[0] if data_grid else []
            rows = data_grid[1:] if len(data_grid) > 1 else []

            return ParsedTable(
                table_id=f"table_{idx:03d}",
                page_no=page_no,
                num_rows=len(data_grid),
                num_cols=len(headers) if headers else 0,
                markdown=markdown,
                headers=headers,
                rows=rows,
                bbox=bbox,
            )
        except Exception as err:
            logger.warning(f"Error parsing table item {idx}: {err}")
            return None

    def _fallback_parse(self, path: Path, max_pages: Optional[int] = None) -> ParsedDocument:
        """
        Graceful fallback parser when Docling is unavailable or crashes.
        Extracts raw text pages, builds simulated bounding boxes, and detects tables.
        """
        logger.info(f"Running fallback text/layout extractor for {path.name}...")
        elements: List[ParsedElement] = []
        tables: List[ParsedTable] = []
        full_text_chunks: List[str] = []
        total_pages = 1

        is_pdf = path.suffix.lower() == ".pdf"
        if is_pdf:
            try:
                import pypdf

                reader = pypdf.PdfReader(str(path))
                total_pages = len(reader.pages)
                elem_idx = 0

                pages_to_process = reader.pages[:max_pages] if max_pages else reader.pages
                for page_idx, page in enumerate(pages_to_process, start=1):
                    page_text = page.extract_text() or ""
                    full_text_chunks.append(f"\n\n<!-- Page {page_idx} -->\n\n" + page_text)

                    lines = [line.strip() for line in page_text.splitlines() if line.strip()]
                    if not lines:
                        # Scanned page without embedded text layer (requires OCR model)
                        prov = ItemProvenance(
                            page_no=page_idx,
                            bbox=BoundingBox(
                                l=0.0, t=0.0, r=595.0, b=842.0, coord_origin="TOPLEFT"
                            ),
                        )
                        elements.append(
                            ParsedElement(
                                element_id=f"elem_{elem_idx:04d}",
                                element_type="paragraph",
                                text=f"[Scanned page {page_idx} - pending OCR]",
                                markdown=f"*[Scanned page {page_idx} - pending OCR]*",
                                provenance=prov,
                                metadata={
                                    "fallback": True,
                                    "is_scanned": True,
                                    "page_no": page_idx,
                                },
                            )
                        )
                        elem_idx += 1
                    else:
                        for line_idx, line in enumerate(lines):
                            elem_type: Literal[
                                "heading", "paragraph", "table", "list_item", "other"
                            ] = "paragraph"

                            # Simple heuristic detection for legal headings
                            if line.startswith(
                                (
                                    "Điều ",
                                    "Chương ",
                                    "Mục ",
                                    "Section ",
                                    "Article ",
                                    "ARTICLE ",
                                    "SECTION ",
                                )
                            ):
                                elem_type = "heading"
                            elif line.startswith(("- ", "* ", "1. ", "2. ", "a) ", "b) ")):
                                elem_type = "list_item"

                            # Approximated fallback bounding box (top to bottom on standard A4: 595 x 842 pt)
                            page_height = 842.0
                            page_width = 595.0
                            line_height = 20.0
                            top_offset = min(50.0 + (line_idx * line_height), page_height - 30.0)

                            prov = ItemProvenance(
                                page_no=page_idx,
                                bbox=BoundingBox(
                                    l=50.0,
                                    t=top_offset,
                                    r=page_width - 50.0,
                                    b=top_offset + line_height,
                                    coord_origin="TOPLEFT",
                                ),
                            )

                            elements.append(
                                ParsedElement(
                                    element_id=f"elem_{elem_idx:04d}",
                                    element_type=elem_type,
                                    text=line,
                                    markdown=f"### {line}" if elem_type == "heading" else line,
                                    provenance=prov,
                                    metadata={"fallback": True, "line_no": line_idx + 1},
                                )
                            )
                            elem_idx += 1
            except Exception as e:
                logger.warning(f"Fallback PDF reader could not extract text directly: {e}")
                full_text_chunks.append(
                    f"# {path.stem}\n\nDocument ingested for OCR layout processing."
                )
        else:
            # Image input fallback
            try:
                import cv2
                import numpy as np

                img_arr = np.fromfile(str(path), dtype=np.uint8)
                img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                h, w = (img.shape[:2]) if img is not None else (1000, 700)
                prov = ItemProvenance(
                    page_no=1,
                    bbox=BoundingBox(l=0.0, t=0.0, r=float(w), b=float(h), coord_origin="TOPLEFT"),
                )
                elements.append(
                    ParsedElement(
                        element_id="elem_0000",
                        element_type="paragraph",
                        text=f"[Image Document {path.name}]",
                        markdown=f"*[Image Document {path.name}]*",
                        provenance=prov,
                        metadata={"fallback": True, "is_image": True},
                    )
                )
                full_text_chunks.append(f"# {path.stem}\n\n[Image Document {path.name}]")
            except Exception as e:
                logger.warning(f"Fallback image reader could not parse: {e}")
                full_text_chunks.append(
                    f"# {path.stem}\n\nDocument ingested for OCR layout processing."
                )

        full_markdown = "\n".join(full_text_chunks)

        return ParsedDocument(
            source_path=str(path),
            filename=path.name,
            total_pages=total_pages,
            full_markdown=full_markdown,
            elements=elements,
            tables=tables,
            metadata={"engine": "fallback_parser", "status": "extracted"},
        )
