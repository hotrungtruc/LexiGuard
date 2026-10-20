"""
Tests for Judgment Dissector & Reverse-Dissected Golden Benchmark Dataset
"""

import json
from pathlib import Path
import pytest
from src.ingestion.judgment_dissector import JudgmentDissector, DissectedJudgment, ReverseTestItem
from src.agents.state import VulnerabilityCategory


def test_dissector_partition_logic():
    """Validates that JudgmentDissector correctly partitions sample judgment text into 3 parts."""
    sample_text = """
    TÒA ÁN NHÂN DÂN TỈNH THANH HÓA
    Bản án số: 03/2023/LĐ-PT
    
    NỘI DUNG VỤ ÁN:
    Nguyên đơn khởi kiện yêu cầu bồi thường do vi phạm cam kết không cạnh tranh tại Điều 3 Hợp đồng.
    Theo thỏa thuận tại Điều 3: "Người lao động không được làm việc tại Lọc hóa dầu Nghi Sơn trong 12 tháng".
    
    NHẬN ĐỊNH CỦA TÒA ÁN:
    [1] Về thẩm quyền: Vụ án thuộc thẩm quyền của Tòa án.
    [2] Về nội dung: Căn cứ Điều 10 Bộ luật Lao động và Điều 35 Hiến pháp 2013, thỏa thuận xâm phạm quyền tự do việc làm.
    
    QUYẾT ĐỊNH:
    1. Tuyên bố Điều 3 của Hợp đồng vô hiệu.
    2. Bác yêu cầu khởi kiện của Nguyên đơn.
    """
    dissector = JudgmentDissector(use_gpu=False)
    dissected = dissector.partition_judgment(sample_text, "032023LĐ-PT.pdf")

    assert isinstance(dissected, DissectedJudgment)
    assert "Điều 3" in dissected.raw_noi_dung_vu_an
    assert "Điều 10" in dissected.raw_nhan_dinh_toa_an
    assert "vô hiệu" in dissected.raw_quyet_dinh.lower()


def test_dissector_statutory_articles_extraction():
    """Validates extraction of canonical statutory IDs from court reasoning."""
    sample_nhan_dinh = """
    Căn cứ vào Điều 357, Điều 468 của Bộ luật Dân sự và Điều 301 Luật Thương mại,
    đồng thời áp dụng Án lệ số 09/2016/AL để giải quyết mức lãi suất chậm trả.
    """
    dissector = JudgmentDissector(use_gpu=False)
    articles = dissector.extract_statutory_articles(sample_nhan_dinh)

    assert "91/2015/QH13_ART_357" in articles or "91/2015/QH13_ART_468" in articles
    assert "36/2005/QH11_ART_301" in articles
    assert "AN_LE_09_2016_AL" in articles


def test_golden_qa_benchmark_dataset_integrity():
    """Validates that golden_qa_contracts.json contains all 13 canonical reverse-dissected cases."""
    golden_path = Path("data/eval_dataset/golden_qa_contracts.json")
    assert golden_path.exists(), "golden_qa_contracts.json not found"

    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["total_cases"] == 13
    assert len(data["test_cases"]) == 13

    for item in data["test_cases"]:
        # Verify Reverse Engineering 3-part components
        assert "input_contract_clause" in item, f"Missing input_contract_clause in {item['id']}"
        assert len(item["input_contract_clause"]) > 50, f"Clause too short in {item['id']}"

        assert "expected_articles" in item, f"Missing expected_articles in {item['id']}"
        assert len(item["expected_articles"]) >= 1, f"No expected articles in {item['id']}"

        assert "ground_truth_reasoning" in item, f"Missing ground_truth_reasoning in {item['id']}"
        assert len(item["ground_truth_reasoning"]) > 50, f"Reasoning too short in {item['id']}"

        assert "expected_vulnerability_type" in item, f"Missing expected_vulnerability_type in {item['id']}"
        assert "expected_risk_level" in item, f"Missing expected_risk_level in {item['id']}"
        assert "court_verdict_summary" in item, f"Missing court_verdict_summary in {item['id']}"
        assert "redline_reference" in item, f"Missing redline_reference in {item['id']}"


def test_extracted_ocr_judgments_summary():
    """Validates that extracted_judgments_ocr.json contains all 17 PDFs including scanned OCR."""
    ocr_json_path = Path("data/test_contracts/extracted_judgments_ocr.json")
    assert ocr_json_path.exists(), "extracted_judgments_ocr.json not found"

    with open(ocr_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    assert len(records) == 17

    scanned_record = next((r for r in records if r["source_file"] == "772024LĐ-ST.pdf"), None)
    assert scanned_record is not None, "772024LĐ-ST.pdf missing from OCR records"
    assert scanned_record["ocr_engine"] == "pypdfium2_easyocr_gpu"
    assert scanned_record["text_length"] > 1000, "Scanned PDF has insufficient OCR text"
