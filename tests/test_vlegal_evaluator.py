"""
Unit Tests for VLegal-Bench Core Evaluator Suite (Task 2.3 & Task 3.4)
====================================================================
Tests:
1. Citation parser (VLegalCitation)
2. Task Loader (Tasks 2.3 & 3.4)
3. Legal Graph Structuring Evaluator (Task 2.3)
4. Legal Conflict Auditor (Task 3.4)
5. VLegalMasterBenchmarkSuite (Core Suite Runner)

Tasks 1.4, 3.1, 1.5, 3.3, 3.5 dropped per user requirements.
"""

from pathlib import Path

import pytest

from src.evaluation.vlegal_evaluator import (
    LegalConflictAuditor,
    VLegalCitation,
    VLegalGraphStructuringEvaluator,
    VLegalMasterBenchmarkSuite,
    VLegalTaskLoader,
)

VLEGAL_DATA_PRESENT = Path("data/vlegal_bench/2.3/2_3.jsonl").exists()


def test_vlegal_citation_parsing():
    cit1 = VLegalCitation.parse("Khoản 1 Điều 8 Nghị định 77/2025/NĐ-CP")
    assert cit1.article == 8
    assert cit1.clause == 1
    assert "77/2025/NĐ-CP" in cit1.law

    cit2 = VLegalCitation.parse("A: Điều 12 Luật Thuế giá trị gia tăng 2024")
    assert cit2.article == 12
    assert cit2.clause is None
    assert "Luật Thuế giá trị gia tăng" in cit2.law


@pytest.mark.skipif(not VLEGAL_DATA_PRESENT, reason="VLegal-Bench dataset not present locally")
def test_vlegal_task_loader_core_tasks():
    loader = VLegalTaskLoader("data/vlegal_bench")

    t23 = loader.load_task_2_3()
    assert len(t23) == 326

    t34 = loader.load_task_3_4()
    assert len(t34) == 166


def test_vlegal_conflict_auditor_3_4():
    auditor = LegalConflictAuditor()

    # Conflicting case (QĐ-UBND setting conditions against central law)
    sample_conflict = """Quy phạm được kiểm tra: Số ký hiệu: 16/2021/QĐ-UBND; Vị trí: Điều 6; Nội dung quy phạm: Khi sửa chữa nhà ở phải sử dụng thợ hàn đã được cấp chứng chỉ...
Quy phạm là căn cứ rà soát: Số ký hiệu: 61/2020/QH14; Vị trí: Khoản 3, Điều 7, Luật Đầu tư năm 2020; Nội dung quy phạm: Bộ, HĐND, UBND không được ban hành quy định về điều kiện đầu tư kinh doanh."""
    assert auditor.predict_conflict(sample_conflict) == "Có"

    # Consistent case
    sample_consistent = """Quy phạm được kiểm tra: Số ký hiệu: 74/2015/NĐ-CP; Vị trí: Điều 2; Nội dung quy phạm: Phòng không nhân dân là một nội dung quan trọng của công tác quốc phòng...
Quy phạm là căn cứ rà soát: Số ký hiệu: 49/2024/QH15; Vị trí: Khoản 1, Điều 2, Luật Phòng không nhân dân 2024; Nội dung quy phạm: Phòng không nhân dân là hoạt động của toàn dân..."""
    assert auditor.predict_conflict(sample_consistent) == "Không"


@pytest.mark.skipif(not VLEGAL_DATA_PRESENT, reason="VLegal-Bench dataset not present locally")
def test_vlegal_graph_evaluator_2_3():
    evaluator = VLegalGraphStructuringEvaluator()
    loader = VLegalTaskLoader("data/vlegal_bench")
    samples = loader.load_task_2_3()[:20]
    metrics = evaluator.evaluate_task_2_3(samples)
    assert metrics.total_samples == 20
    assert metrics.total_gold_triplets > 0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0


def test_vlegal_master_suite_core_runner():
    suite = VLegalMasterBenchmarkSuite("data/vlegal_bench")
    # Quick smoke test with mock or partial run
    assert hasattr(suite, "run_all")
    assert suite.graph_eval is not None
    assert suite.auditor_eval is not None
