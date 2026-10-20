"""
Judgment Dissector: OCR & Reverse Engineering of Vietnamese Court Judgments
=============================================================================
Module bóc tách ngược các Bản án Tòa án Nhân dân (TAND) và Quyết định Trọng tài (VIAC)
thành 3 thành phần kiểm thử độc lập cho hệ sinh thái LexiGuard:
  1. Input Contract Clause (Pristine Markdown): Tái tạo câu chữ điều khoản hợp đồng gốc,
     lọc bỏ toàn bộ câu chữ tường thuật tố tụng.
  2. Ground Truth Judicial Reasoning & Expected Legal Articles: Trích xuất lập luận của
     Thẩm phán và mã điều luật chuẩn để chấm điểm tầng Retrieval (Qdrant/Neo4j) & RAGAS.
  3. Ground Truth Verdict & Remedy: Phán quyết của Tòa, phân loại lỗ hổng (VulnerabilityCategory),
     mức độ rủi ro (risk_level) và đề xuất sửa đổi chuẩn (redline_reference).
"""

import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, ConfigDict
from loguru import logger

from src.agents.state import VulnerabilityCategory, DisputeForum


# ============================================================================
# 1. DATA MODELS FOR DISSECTED JUDGMENT & BENCHMARK TEST CASE
# ============================================================================

class JudgmentMetadata(BaseModel):
    """Metadata extracted from court judgment header."""
    model_config = ConfigDict(extra="ignore")

    case_number: str = Field(description="Số bản án / quyết định (VD: 03/2023/LĐ-PT)")
    case_date: Optional[str] = Field(default=None, description="Ngày xét xử / ban hành")
    court_name: str = Field(description="Tên Tòa án xét xử hoặc Trung tâm Trọng tài")
    case_type: str = Field(description="Lĩnh vực tranh chấp (Lao động, Kinh doanh thương mại)")
    dispute_subject: Optional[str] = Field(default=None, description="V/v tranh chấp hợp đồng")
    source_file: str = Field(description="Tên file PDF gốc")


class DissectedJudgment(BaseModel):
    """Raw 3-part partition of Vietnamese court judgment document."""
    model_config = ConfigDict(extra="ignore")

    metadata: JudgmentMetadata
    raw_noi_dung_vu_an: str = Field(description="Phần 1: NỘI DUNG VỤ ÁN")
    raw_nhan_dinh_toa_an: str = Field(description="Phần 2: NHẬN ĐỊNH CỦA TÒA ÁN")
    raw_quyet_dinh: str = Field(description="Phần 3: QUYẾT ĐỊNH / TUYÊN ÁN")
    ocr_engine: str = Field(default="Docling+EasyOCR", description="Công nghệ OCR đã áp dụng")


class ReverseTestItem(BaseModel):
    """
    Standardized benchmark test item reverse-engineered from court judgment
    compatible with golden_qa_contracts.json and RAGAS evaluation pipelines.
    """
    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Mã định danh test case (VD: BENCH-CASE-001)")
    case_reference: str = Field(description="Số hiệu bản án hoặc án lệ gốc")
    contract_file: str = Field(description="Tên file văn bản / PDF nguồn")
    clause_id: str = Field(description="Tên hoặc vị trí điều khoản tranh chấp")
    
    # Thành phần 1: Input thuần khiết cho Agent (ĐÃ GIẤU NHẬN ĐỊNH & PHÁN QUYẾT)
    input_contract_clause: str = Field(
        description="Điều khoản hợp đồng nguyên bản đã lọc bỏ ngôn ngữ tố tụng"
    )
    audit_question: str = Field(
        description="Câu hỏi kiểm thử đặt cho Multi-Agent Council rà soát"
    )

    # Thành phần 2: Ground Truth từ phần NHẬN ĐỊNH CỦA TÒA ÁN
    expected_articles: List[str] = Field(
        default_factory=list,
        description="Danh sách mã điều luật chuẩn (VD: 45/2019/QH14_ART_10)"
    )
    ground_truth_reasoning: str = Field(
        description="Lập luận pháp lý cốt lõi của Thẩm phán / Trọng tài viên"
    )

    # Thành phần 3: Ground Truth từ phần QUYẾT ĐỊNH CỦA TÒA ÁN
    expected_vulnerability_type: str = Field(
        description="Nhóm lỗ hổng pháp lý (ILLEGAL_CLAUSE, AMBIGUOUS_LOOPHOLE...)"
    )
    expected_risk_level: str = Field(
        default="HIGH",
        description="Mức độ rủi ro pháp lý (HIGH, MEDIUM, LOW)"
    )
    court_verdict_summary: str = Field(
        description="Tóm tắt phán quyết của Tòa án / Trọng tài (Tuyên vô hiệu, cắt giảm...)"
    )
    redline_reference: Optional[str] = Field(
        default=None,
        description="Câu văn hợp đồng sửa đổi chuẩn (Redline Suggestion)"
    )
    target_forum: str = Field(
        default="TAND",
        description="Cơ quan tài phán thụ lý (TAND hoặc VIAC)"
    )


# ============================================================================
# 2. STATUTORY MAPPING DICTIONARY
# ============================================================================

STATUTORY_MAPPINGS = {
    # Bộ luật Lao động 2019 (45/2019/QH14)
    "bộ luật lao động": "45/2019/QH14",
    "bllđ": "45/2019/QH14",
    "luật lao động": "45/2019/QH14",
    # Bộ luật Dân sự 2015 (91/2015/QH13)
    "bộ luật dân sự": "91/2015/QH13",
    "blds": "91/2015/QH13",
    "luật dân sự": "91/2015/QH13",
    # Luật Thương mại 2005 (36/2005/QH11)
    "luật thương mại": "36/2005/QH11",
    "ltm": "36/2005/QH11",
    # Luật Trọng tài Thương mại 2010 (54/2010/QH12)
    "luật trọng tài thương mại": "54/2010/QH12",
    "ltttm": "54/2010/QH12",
    # Hiến pháp 2013
    "hiến pháp": "HIEN_PHAP_2013",
}


# ============================================================================
# 3. CORE JUDGMENT DISSECTOR IMPLEMENTATION
# ============================================================================

class JudgmentDissector:
    """
    Trích xuất OCR và bóc tách cấu trúc 3 phần của Bản án TAND & Phán quyết VIAC.
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._easyocr_reader = None

    def _get_ocr_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._easyocr_reader is None:
            import easyocr
            self._easyocr_reader = easyocr.Reader(["vi", "en"], gpu=self.use_gpu)
        return self._easyocr_reader

    def extract_full_text(self, pdf_path: Union[str, Path]) -> Tuple[str, str]:
        """
        Trích xuất toàn bộ văn bản từ file PDF:
        - Sử dụng pypdf cho PDF có text layer sẵn.
        - Fallback sang pypdfium2 + EasyOCR GPU cho trang quét (scanned).
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        engine_used = "pypdf_digital"
        full_text = ""

        # Bước 1: Thử trích xuất qua pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(path))
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
            full_text = "\n\n".join(pages_text).strip()
        except Exception as e:
            logger.warning(f"pypdf extraction failed for {path.name}: {e}")

        # Bước 2: Kiểm tra nếu là file scan (quá ít ký tự trích xuất được)
        if len(full_text) < 100:
            logger.info(f"File {path.name} contains no embedded text. Running EasyOCR on GPU...")
            try:
                import pypdfium2 as pdfium
                import numpy as np

                pdf = pdfium.PdfDocument(str(path))
                ocr = self._get_ocr_reader()
                ocr_pages = []
                for p_idx, page in enumerate(pdf):
                    img = np.array(page.render(scale=1.5).to_pil())
                    lines = ocr.readtext(img, detail=0)
                    ocr_pages.append(f"<!-- Page {p_idx + 1} -->\n" + "\n".join(lines))
                full_text = "\n\n".join(ocr_pages)
                engine_used = "pypdfium2_easyocr_gpu"
            except Exception as ocr_err:
                logger.error(f"OCR execution failed for {path.name}: {ocr_err}")

        return full_text, engine_used

    def partition_judgment(self, full_text: str, source_filename: str) -> DissectedJudgment:
        """
        Phân rã văn bản bản án thành 3 phân vùng tiêu chuẩn:
        - NỘI DUNG VỤ ÁN
        - NHẬN ĐỊNH CỦA TÒA ÁN
        - QUYẾT ĐỊNH
        """
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]

        # Metadata extraction
        case_number = source_filename.replace(".pdf", "")
        court_name = "TÒA ÁN NHÂN DÂN"
        case_date = None
        case_type = "Kinh doanh thương mại" if "KDTM" in source_filename else "Lao động"

        # Regex tìm kiếm các mốc phân chia
        noi_dung_idx = -1
        nhan_dinh_idx = -1
        quyet_dinh_idx = -1

        noi_dung_regex = re.compile(r"^(NỘI DUNG VỤ ÁN|NỘI DUNG VỤ ÁN:)", re.IGNORECASE)
        nhan_dinh_regex = re.compile(
            r"^(NHẬN ĐỊNH CỦA TÒA ÁN|NHẬN ĐỊNH CỦA HỘI ĐỒNG XÉT XỬ|XÉT THẤY|NHẬN ĐỊNH:)",
            re.IGNORECASE
        )
        quyet_dinh_regex = re.compile(
            r"^(QUYẾT ĐỊNH|QUYẾT ĐỊNH:|TUYÊN XỬ|TUYÊN XỬ:)",
            re.IGNORECASE
        )

        for idx, line in enumerate(lines):
            # Check case number in header
            if "bản án số" in line.lower() or "quyết định số" in line.lower():
                case_number = line
            if "tòa án nhân dân" in line.lower():
                court_name = line

            # Match section boundaries
            if noi_dung_regex.search(line) and noi_dung_idx == -1:
                noi_dung_idx = idx
            elif nhan_dinh_regex.search(line) and nhan_dinh_idx == -1 and idx > (noi_dung_idx if noi_dung_idx != -1 else 0):
                nhan_dinh_idx = idx
            elif quyet_dinh_regex.search(line) and idx > (nhan_dinh_idx if nhan_dinh_idx != -1 else 0):
                quyet_dinh_idx = idx

        # Extract text chunks
        if noi_dung_idx != -1 and nhan_dinh_idx != -1:
            raw_noi_dung = "\n".join(lines[noi_dung_idx:nhan_dinh_idx])
        elif noi_dung_idx != -1:
            raw_noi_dung = "\n".join(lines[noi_dung_idx:noi_dung_idx + 80])
        else:
            raw_noi_dung = "\n".join(lines[:len(lines) // 3])

        if nhan_dinh_idx != -1 and quyet_dinh_idx != -1:
            raw_nhan_dinh = "\n".join(lines[nhan_dinh_idx:quyet_dinh_idx])
        elif nhan_dinh_idx != -1:
            raw_nhan_dinh = "\n".join(lines[nhan_dinh_idx:nhan_dinh_idx + 120])
        else:
            raw_nhan_dinh = "\n".join(lines[len(lines) // 3: 2 * len(lines) // 3])

        if quyet_dinh_idx != -1:
            raw_quyet_dinh = "\n".join(lines[quyet_dinh_idx:])
        else:
            raw_quyet_dinh = "\n".join(lines[2 * len(lines) // 3:])

        meta = JudgmentMetadata(
            case_number=case_number,
            case_date=case_date,
            court_name=court_name,
            case_type=case_type,
            dispute_subject="Tranh chấp hợp đồng",
            source_file=source_filename
        )

        return DissectedJudgment(
            metadata=meta,
            raw_noi_dung_vu_an=raw_noi_dung,
            raw_nhan_dinh_toa_an=raw_nhan_dinh,
            raw_quyet_dinh=raw_quyet_dinh
        )

    def extract_statutory_articles(self, nhan_dinh_text: str) -> List[str]:
        """
        Bóc tách các điều luật được Thẩm phán trích dẫn trong phần Nhận định
        và chuyển thành mã định danh chuẩn (VD: 45/2019/QH14_ART_10, 91/2015/QH13_ART_468).
        """
        articles: List[str] = []

        # Sort keys by length descending to match longer statutory names first
        keys_sorted = sorted(STATUTORY_MAPPINGS.keys(), key=len, reverse=True)
        law_regex = "|".join(re.escape(k) for k in keys_sorted)
        full_pattern = re.compile(rf"({law_regex})", re.IGNORECASE)

        last_end = 0
        for m in full_pattern.finditer(nhan_dinh_text):
            matched_key = m.group(1).lower()
            law_id = STATUTORY_MAPPINGS.get(matched_key)
            if not law_id:
                continue

            # Look back in text segment before this statutory name
            preceding = nhan_dinh_text[last_end:m.start()]
            last_end = m.end()

            # Find all numbers (articles) in the preceding text segment
            nums = re.findall(r"(\d+)", preceding)
            for n in nums:
                # Filter out year numbers (> 1000) like 2015, 2019, 2023
                if int(n) < 1000:
                    formatted_art = f"{law_id}_ART_{n}"
                    if formatted_art not in articles:
                        articles.append(formatted_art)

        # Trích xuất Án lệ (nếu có)
        an_le_match = re.findall(r"Án lệ số\s+(\d+/\d+/AL|\d+)", nhan_dinh_text, re.IGNORECASE)
        for al in an_le_match:
            al_clean = al.replace("/", "_").replace("-", "_")
            al_id = f"AN_LE_{al_clean}"
            if al_id not in articles:
                articles.append(al_id)

        return articles

    def sanitize_clause_input(self, raw_noi_dung: str) -> str:
        """
        Lọc bỏ ngôn ngữ tố tụng ("Nguyên đơn trình bày...", "Theo bản án sơ thẩm..."),
        giữ lại nguyên văn điều khoản hợp đồng.
        """
        # Loại bỏ các đoạn tường thuật tố tụng phổ biến
        narrative_patterns = [
            r"Tại\s+(?:đơn khởi kiện|phiên tòa|bản tự khai).*?(?:trình bày|yêu cầu):?",
            r"Theo\s+(?:lời khai của|trình bày của).*?:?",
            r"Bị đơn\s+(?:trình bày|cho rằng|không đồng ý).*?:?",
            r"Người có quyền lợi.*?cho rằng:?",
            r"Đại diện Viện kiểm sát phát biểu:?",
            r"Về quan điểm của Tòa án.*?xét thấy:?",
        ]

        cleaned = raw_noi_dung
        for p in narrative_patterns:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)

        # Tìm các khối trích dẫn nằm trong dấu ngoặc kép hoặc bắt đầu bằng "Điều ..."
        quoted_clauses = re.findall(r'["“](.*?)["”]', cleaned, flags=re.DOTALL)
        candidate_clauses = [q.strip() for q in quoted_clauses if len(q.strip()) > 40 and "Điều" in q]

        if candidate_clauses:
            return candidate_clauses[0]

        # Nếu không có dấu ngoặc kép, lấy đoạn chứa từ khóa "Điều ... thỏa thuận"
        clause_match = re.search(r"(Điều\s+\d+[\.:].{40,500})", cleaned, flags=re.DOTALL)
        if clause_match:
            return clause_match.group(1).strip()

        # Fallback: Trả về phần đầu đã làm sạch
        return cleaned[:600].strip()
