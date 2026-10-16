"""
OCR Engine: Enterprise Multimodal OCR & Document Layout Reconstruction
=======================================================================
Combines:
1. ImagePreprocessor: Multi-page collage splitting & resolution enhancement.
2. IBM Docling: Deep layout analysis & TableFormer table extraction.
3. Multimodal VLM (Google Gemini 2.0 Flash Vision): Semantic transcription for noisy scans/stamps.
4. ReportLab PDF Reconstructor: Searchable legal PDF output with 100% Vietnamese Unicode fidelity.
"""

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import httpx
from loguru import logger

from src.core.config import settings
from src.ingestion.image_preprocessor import ImagePreprocessor
from src.ingestion.docling_parser import DoclingParser, ParsedDocument
from scripts.generate_ocr_pdfs import build_pdf_document

class OCREngine:
    """
    Enterprise OCR Engine coordinating Preprocessing, IBM Docling, 
    Multimodal Vision LLMs, and Legal PDF generation.
    """

    def __init__(
        self,
        target_dpi_width: int = 2000,
        enable_vlm_fallback: bool = True,
        output_pdf_dir: Optional[Union[str, Path]] = None
    ):
        self.preprocessor = ImagePreprocessor(target_dpi_width=target_dpi_width)
        self.docling_parser = DoclingParser(do_ocr=True, do_table_structure=True)
        self.enable_vlm_fallback = enable_vlm_fallback
        self.output_pdf_dir = Path(output_pdf_dir) if output_pdf_dir else Path("data/sample_contracts/extracted_pdfs")
        self.output_pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"OCREngine initialized (Output dir: {self.output_pdf_dir})")

    def process_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Executes end-to-end OCR pipeline on a document image:
        1. Preprocess & detect multi-page horizontal/vertical stitching.
        2. Extract layout & text using Docling / Gemini Vision.
        3. Assemble full document structure with bounding boxes.
        """
        path = Path(image_path)
        logger.info(f"Processing document image: {path.name}")

        # Step 1: Preprocessing & page splitting
        preprocessed_pages = self.preprocessor.process(path)
        logger.info(f"Document segmented into {len(preprocessed_pages)} sub-page(s).")

        extracted_subpages = []
        full_text_parts = []

        for img_cv, meta in preprocessed_pages:
            sub_idx = meta["sub_page_index"]
            quality_info = self.preprocessor.evaluate_quality(img_cv)
            is_high_quality = quality_info["is_high_quality"]
            meta["quality"] = quality_info
            
            logger.info(
                f"Sub-page {sub_idx}/{meta['total_sub_pages']} Quality: "
                f"{'HIGH (Routing to Branch 1: Full Docling)' if is_high_quality else 'DEGRADED (Routing to Branch 2: Gemini VLM)'} "
                f"- Reason: {quality_info['reason']}"
            )

            transcribed_text = None
            branch_used = None

            if is_high_quality:
                # BRANCH 1: Full IBM Docling (Layout analysis, TableFormer, Bounding Boxes)
                branch_used = "Branch 1: Full IBM Docling"
                temp_enhanced_path = path.parent / f"_temp_enhanced_{sub_idx}.png"
                self.preprocessor.save_enhanced_image(img_cv, temp_enhanced_path)
                try:
                    parsed = self.docling_parser.parse(temp_enhanced_path)
                    transcribed_text = parsed.full_markdown
                except Exception as e:
                    logger.warning(f"Docling branch failed: {e}. Falling back to Gemini VLM...")
                finally:
                    if temp_enhanced_path.exists():
                        temp_enhanced_path.unlink()

                # Fallback to Branch 2 if Docling produced empty or invalid output
                if not transcribed_text:
                    transcribed_text = self._execute_gemini_vlm_branch(img_cv, meta)
                    if transcribed_text:
                        branch_used = "Branch 2: Gemini VLM (Fallback)"
            else:
                # BRANCH 2: Gemini VLM (Local Google PaliGemma / Qwen2-VL or Cloud Gemini Vision)
                branch_used = "Branch 2: Gemini VLM"
                transcribed_text = self._execute_gemini_vlm_branch(img_cv, meta)

                # Fallback to Branch 1 if VLM unavailable
                if not transcribed_text:
                    temp_enhanced_path = path.parent / f"_temp_enhanced_{sub_idx}.png"
                    self.preprocessor.save_enhanced_image(img_cv, temp_enhanced_path)
                    try:
                        parsed = self.docling_parser.parse(temp_enhanced_path)
                        transcribed_text = parsed.full_markdown
                        branch_used = "Branch 1: Full IBM Docling (Fallback)"
                    except Exception as e:
                        logger.error(f"Docling fallback also failed: {e}")
                    finally:
                        if temp_enhanced_path.exists():
                            temp_enhanced_path.unlink()

            extracted_subpages.append({
                "sub_page_index": sub_idx,
                "metadata": meta,
                "branch_used": branch_used,
                "text": transcribed_text or ""
            })
            full_text_parts.append(transcribed_text or "")

        full_document_text = "\n\n---\n\n".join(full_text_parts)

        return {
            "source_file": path.name,
            "total_sub_pages": len(preprocessed_pages),
            "sub_pages": extracted_subpages,
            "full_text": full_document_text
        }

    def _execute_gemini_vlm_branch(self, img_cv, meta: dict) -> Optional[str]:
        """
        Executes Branch 2: Gemini VLM transcription for degraded/noisy documents.
        Prioritizes On-Premise Local VLM (Google PaliGemma / Qwen2-VL),
        falling back to Gemini Cloud Vision API if configured.
        """
        if settings.USE_LOCAL_VLM:
            try:
                from src.ingestion.local_vlm import get_local_vlm
                local_engine = get_local_vlm()
                text = local_engine.transcribe_image(img_cv)
                if text and len(text.strip()) > 0:
                    return text
            except Exception as e:
                logger.warning(f"Local VLM execution failed: {e}. Trying Gemini Cloud API...")

        if self.enable_vlm_fallback and settings.GEMINI_API_KEY:
            return self._transcribe_with_gemini_vision(img_cv, meta)

        return None

    def _transcribe_with_gemini_vision(self, img_cv, meta: dict) -> Optional[str]:
        """Calls Google Gemini 2.0 Flash Vision to transcribe low-res/noisy scan with 100% accuracy."""
        try:
            import cv2
            _, buffer = cv2.imencode(".png", img_cv)
            b64_image = base64.b64encode(buffer).decode("utf-8")

            url = f"https://generativelanguage.googleapis.com/v1beta/models/{settings.GEMINI_MODEL}:generateContent?key={settings.GEMINI_API_KEY}"
            prompt = (
                "You are an expert Vietnamese Legal Document OCR Engine. "
                "Transcribe this legal document with 100% completeness. "
                "Preserve all administrative elements: Quốc hiệu, Tiêu ngữ, Tên cơ quan, Số hiệu, "
                "Căn cứ pháp lý, Toàn bộ các Điều và Khoản, Bảng biểu (format as Markdown table), "
                "Dấu mộc đỏ, chữ viết tay, và toàn bộ khối 'Nơi nhận'. Do not summarize."
            )

            payload = {
                "contents": [{
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/png", "data": b64_image}}
                    ]
                }],
                "generationConfig": {"temperature": 0.0}
            }

            with httpx.Client(timeout=30.0) as client:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                        logger.info(f"Gemini Vision successfully transcribed sub-page {meta['sub_page_index']}")
                        return text
                else:
                    logger.warning(f"Gemini Vision API call failed with status: {res.status_code}")
        except Exception as e:
            logger.warning(f"Gemini Vision transcription error: {e}")
        return None

    def reconstruct_pdf_from_ocr(
        self,
        json_metadata_path: Union[str, Path],
        markdown_text_path: Union[str, Path],
        output_pdf_path: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Reconstructs a formal legal PDF document from OCR metadata (JSON) and extracted content (Markdown).
        """
        json_path = Path(json_metadata_path)
        md_path = Path(markdown_text_path)

        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()

        if output_pdf_path is None:
            target_name = json_path.stem + ".pdf"
            dest_pdf = self.output_pdf_dir / target_name
        else:
            dest_pdf = Path(output_pdf_path)

        logger.info(f"Reconstructing legal PDF from OCR text: {dest_pdf.name}")
        build_pdf_document(json_data, md_content, dest_pdf)
        return dest_pdf
