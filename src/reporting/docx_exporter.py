"""
LexiGuard AI — Enterprise Contract Audit Word DOCX Exporter
===========================================================
Generates boardroom-ready Microsoft Word (.docx) audit reports with:
- Professional corporate typography, custom palette & table styles.
- Executive Summary, Legal Health Score meter, and TAND/VIAC risk overview.
- Vulnerability matrix categorized by severity (CRITICAL, HIGH, MEDIUM, LOW).
- Simulated Word Track-Changes Redlines (Strikethrough Red vs Bold Green Underline)
  featuring all 3 negotiation tiers (Strict, Balanced, Flexible) and Judicial Precedents.
- Mathematical table audit verification annex.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
from loguru import logger


# Color Constants
COLOR_NAVY = RGBColor(0x1E, 0x3A, 0x8A)      # #1e3a8a
COLOR_PRIMARY = RGBColor(0x25, 0x63, 0xEB)   # #2563eb
COLOR_CRITICAL = RGBColor(0xDC, 0x26, 0x26)  # #dc2626
COLOR_HIGH = RGBColor(0xEA, 0x58, 0x0C)      # #ea580c
COLOR_MEDIUM = RGBColor(0xD9, 0x77, 0x06)    # #d97706
COLOR_SUCCESS = RGBColor(0x16, 0xA3, 0x4A)   # #16a34a
COLOR_GRAY = RGBColor(0x4B, 0x55, 0x63)      # #4b5563
COLOR_DARK = RGBColor(0x11, 0x18, 0x27)      # #111827


def _set_cell_background(cell, hex_color: str) -> None:
    """Sets background fill color of a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tcPr.append(shd)


def _set_cell_margins(cell, top=100, bottom=100, left=150, right=150) -> None:
    """Sets internal padding of a table cell in dxa (1 pt = 20 dxa)."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def generate_contract_audit_docx(
    report_data: Dict[str, Any],
    contract_title: str = "Tài liệu Hợp đồng",
) -> io.BytesIO:
    """
    Renders structured ContractAuditReport dictionary into a styled DOCX binary stream.
    """
    doc = docx.Document()

    # Set page margins (1 inch all sides)
    for section in doc.sections:
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.9)
        section.right_margin = Inches(0.9)

    # Base style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.font.color.rgb = COLOR_DARK

    # -------------------------------------------------------------------------
    # HEADER / BANNER
    # -------------------------------------------------------------------------
    p_header = doc.add_paragraph()
    p_header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run_brand = p_header.add_run("LEXIGUARD AI — HỆ THỐNG KIỂM TOÁN PHÁP LÝ TỰ ĐỘNG")
    run_brand.font.size = Pt(8.5)
    run_brand.font.bold = True
    run_brand.font.color.rgb = COLOR_PRIMARY

    # Document Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_title = p_title.add_run("BÁO CÁO THẨM ĐỊNH RỦI RO HỢP ĐỒNG\n& ĐỀ XUẤT CHỈNH LÝ ĐIỀU KHOẢN (REDLINE)")
    run_title.font.size = Pt(18)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_NAVY

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(16)
    run_sub = p_sub.add_run("Soi chiếu Quy chuẩn Pháp luật Việt Nam • Lăng kính TAND & VIAC • Đề xuất Đàm phán 3 Cấp độ")
    run_sub.font.size = Pt(10)
    run_sub.font.italic = True
    run_sub.font.color.rgb = COLOR_GRAY

    # Metadata Box Table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False

    health_score = report_data.get("health_score", 0)
    score_color = COLOR_SUCCESS if health_score >= 80 else (COLOR_MEDIUM if health_score >= 50 else COLOR_CRITICAL)

    cell_00 = meta_table.cell(0, 0)
    cell_00.width = Inches(4.5)
    _set_cell_background(cell_00, "F8FAFC")
    p00 = cell_00.paragraphs[0]
    p00.add_run("Văn bản thẩm định: ").font.bold = True
    p00.add_run(f"{contract_title}\n")
    p00.add_run("Ngày kiểm toán: ").font.bold = True
    p00.add_run(f"{time.strftime('%d/%m/%Y %H:%M')}\n")
    p00.add_run("Đơn vị kiểm định: ").font.bold = True
    p00.add_run("LexiGuard Autonomous Legal Auditor (v2.3)")

    cell_01 = meta_table.cell(0, 1)
    cell_01.width = Inches(2.2)
    _set_cell_background(cell_01, "EFF6FF")
    p01 = cell_01.paragraphs[0]
    p01.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p01.add_run("ĐIỂM SỨC KHỎE PHÁP LÝ\n").font.size = Pt(9)
    run_score = p01.add_run(f"{health_score}/100\n")
    run_score.font.size = Pt(22)
    run_score.font.bold = True
    run_score.font.color.rgb = score_color
    p01.add_run("An toàn" if health_score >= 80 else ("Rủi ro trung bình" if health_score >= 50 else "Rủi ro nghiêm trọng")).font.size = Pt(9.5)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # 1. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)
    # -------------------------------------------------------------------------
    h1 = doc.add_heading(level=1)
    r1 = h1.add_run("I. TÓM TẮT ĐIỀU HÀNH (EXECUTIVE SUMMARY)")
    r1.font.size = Pt(13)
    r1.font.bold = True
    r1.font.color.rgb = COLOR_NAVY
    h1.paragraph_format.space_before = Pt(14)
    h1.paragraph_format.space_after = Pt(4)

    exec_summary = report_data.get("executive_summary") or report_data.get("summary_notes") or "Không có tóm tắt."
    p_exec = doc.add_paragraph(exec_summary)
    p_exec.paragraph_format.line_spacing = 1.15
    p_exec.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # 2. ĐÁNH GIÁ RỦI RO TÀI PHÁN (TAND & TRỌNG TÀI VIAC)
    # -------------------------------------------------------------------------
    h2 = doc.add_heading(level=1)
    r2 = h2.add_run("II. ĐÁNH GIÁ RỦI RO TỐ TỤNG & TÀI PHÁN (LĂNG KÍNH TAND & VIAC)")
    r2.font.size = Pt(13)
    r2.font.bold = True
    r2.font.color.rgb = COLOR_NAVY
    h2.paragraph_format.space_before = Pt(14)
    h2.paragraph_format.space_after = Pt(4)

    tand_viac = report_data.get("tand_viac_summary") or "Hệ thống ghi nhận tuân thủ theo chuẩn mực tranh tụng TAND và tập quán thương mại VIAC."
    p_tv = doc.add_paragraph(tand_viac)
    p_tv.paragraph_format.line_spacing = 1.15
    p_tv.paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # 3. DANH SÁCH LỖ HỔNG & BẪY ĐIỀU KHOẢN (FINDINGS MATRIX)
    # -------------------------------------------------------------------------
    h3 = doc.add_heading(level=1)
    r3 = h3.add_run("III. MA TRẬN LỖ HỔNG & BẪY PHÁP LÝ CHI TIẾT")
    r3.font.size = Pt(13)
    r3.font.bold = True
    r3.font.color.rgb = COLOR_NAVY
    h3.paragraph_format.space_before = Pt(14)
    h3.paragraph_format.space_after = Pt(6)

    findings = report_data.get("findings") or report_data.get("vulnerabilities") or []
    if not findings:
        doc.add_paragraph("✅ Không phát hiện lỗ hổng hay điều khoản vi phạm nghiêm trọng nào trong văn bản.")
    else:
        table_f = doc.add_table(rows=1, cols=4)
        table_f.alignment = WD_TABLE_ALIGNMENT.CENTER
        table_f.autofit = False

        # Table Header
        headers = ["Điều khoản", "Mức độ", "Nội dung rủi ro & Vi phạm", "Căn cứ pháp lý"]
        widths = [Inches(1.2), Inches(1.1), Inches(2.8), Inches(1.6)]
        for idx, (title, w) in enumerate(zip(headers, widths)):
            cell = table_f.cell(0, idx)
            cell.width = w
            _set_cell_background(cell, "1E3A8A")
            _set_cell_margins(cell, top=120, bottom=120, left=120, right=120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(title)
            run.font.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        # Table Rows
        for f in findings:
            row_cells = table_f.add_row().cells
            ref = f.get("clause_reference") or f.get("finding_id") or "Điều khoản"
            sev = (f.get("severity") or "MEDIUM").upper()
            desc = f.get("audit_verdict") or f.get("description") or ""
            cit = f.get("statutory_citation") or "Quy định chung"

            # Col 0: Clause Reference
            row_cells[0].width = widths[0]
            _set_cell_margins(row_cells[0])
            p0 = row_cells[0].paragraphs[0]
            p0.add_run(ref).font.bold = True

            # Col 1: Severity Badge
            row_cells[1].width = widths[1]
            _set_cell_margins(row_cells[1])
            p1 = row_cells[1].paragraphs[0]
            p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r_sev = p1.add_run(sev)
            r_sev.font.bold = True
            r_sev.font.size = Pt(9)
            if sev in ["CRITICAL", "HIGH", "NGHIÊM TRỌNG"]:
                _set_cell_background(row_cells[1], "FEE2E2")
                r_sev.font.color.rgb = COLOR_CRITICAL
            elif sev in ["MEDIUM", "TRUNG BÌNH"]:
                _set_cell_background(row_cells[1], "FEF3C7")
                r_sev.font.color.rgb = COLOR_MEDIUM
            else:
                _set_cell_background(row_cells[1], "F3F4F6")
                r_sev.font.color.rgb = COLOR_GRAY

            # Col 2: Description
            row_cells[2].width = widths[2]
            _set_cell_margins(row_cells[2])
            p2 = row_cells[2].paragraphs[0]
            p2.add_run(desc).font.size = Pt(9.5)

            # Col 3: Citation
            row_cells[3].width = widths[3]
            _set_cell_margins(row_cells[3])
            p3 = row_cells[3].paragraphs[0]
            r3_cit = p3.add_run(cit)
            r3_cit.font.size = Pt(9)
            r3_cit.font.italic = True
            r3_cit.font.color.rgb = COLOR_PRIMARY

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # -------------------------------------------------------------------------
    # 4. ĐỀ XUẤT CHỈNH LÝ REDLINE (TRACK-CHANGES & 3-TIER NEGOTIATION)
    # -------------------------------------------------------------------------
    h4 = doc.add_heading(level=1)
    r4 = h4.add_run("IV. ĐỀ XUẤT CHỈNH LÝ ĐIỀU KHOẢN (TRACK-CHANGES REDLINES)")
    r4.font.size = Pt(13)
    r4.font.bold = True
    r4.font.color.rgb = COLOR_NAVY
    h4.paragraph_format.space_before = Pt(14)
    h4.paragraph_format.space_after = Pt(6)

    p_guide = doc.add_paragraph()
    p_guide.add_run("Quy ước Track-Changes: ").font.bold = True
    r_del = p_guide.add_run("Câu chữ gốc của đối tác (bị gạch bỏ) ")
    r_del.font.strike = True
    r_del.font.color.rgb = COLOR_CRITICAL
    r_ins = p_guide.add_run("  |  Câu chữ đề xuất sửa đổi mới (thêm vào)")
    r_ins.font.bold = True
    r_ins.font.underline = True
    r_ins.font.color.rgb = COLOR_SUCCESS
    p_guide.paragraph_format.space_after = Pt(12)

    redlines = report_data.get("redline_recommendations") or []
    if not redlines and findings:
        # Fallback extract from findings
        for f in findings:
            if f.get("proposed_redline") or f.get("redlines"):
                r_3t = f.get("redlines") or {}
                if hasattr(r_3t, "model_dump"):
                    r_3t = r_3t.model_dump()
                elif not isinstance(r_3t, dict):
                    r_3t = {}
                redlines.append({
                    "clause_id": f.get("clause_reference") or f.get("finding_id") or "Điều khoản",
                    "original_text": f.get("contract_provenance", {}).get("verbatim_text") or "",
                    "suggested_revision": r_3t.get("balanced_standard") or f.get("proposed_redline") or "",
                    "strict_revision": r_3t.get("strict_protective") or f.get("strict_revision") or "",
                    "balanced_revision": r_3t.get("balanced_standard") or f.get("proposed_redline") or "",
                    "flexible_commercial": r_3t.get("flexible_commercial") or f.get("flexible_revision") or "",
                    "precedent_reference": f.get("precedent_reference") or "",
                    "rationale": f.get("negotiation_rationale") or f.get("audit_verdict") or "",
                })

    for idx, r_item in enumerate(redlines, start=1):
        clause_id = r_item.get("clause_id") or f"Điều khoản {idx}"
        orig_text = r_item.get("original_text") or "Nội dung gốc không xác định."
        bal_text = r_item.get("balanced_revision") or r_item.get("suggested_revision") or ""
        strict_text = r_item.get("strict_revision") or ""
        flex_text = r_item.get("flexible_commercial") or ""
        precedent = r_item.get("precedent_reference") or ""
        rationale = r_item.get("rationale") or ""

        # Clause Subheading Box
        p_ch = doc.add_paragraph()
        p_ch.paragraph_format.space_before = Pt(8)
        p_ch.paragraph_format.space_after = Pt(4)
        run_ch = p_ch.add_run(f"4.{idx}. {clause_id.upper()}")
        run_ch.font.bold = True
        run_ch.font.size = Pt(11)
        run_ch.font.color.rgb = COLOR_NAVY

        # Redline Comparison Box
        t_red = doc.add_table(rows=1, cols=1)
        t_red.alignment = WD_TABLE_ALIGNMENT.CENTER
        t_red.autofit = False
        c_red = t_red.cell(0, 0)
        c_red.width = Inches(6.7)
        _set_cell_background(c_red, "F8FAFC")
        _set_cell_margins(c_red, top=140, bottom=140, left=160, right=160)

        p_rc = c_red.paragraphs[0]
        p_rc.paragraph_format.line_spacing = 1.15

        # Original Strikethrough
        if orig_text:
            p_rc.add_run("❌ CÂU CHỮ GỐC TRONG HỢP ĐỒNG:\n").font.bold = True
            r_orig = p_rc.add_run(f"{orig_text}\n\n")
            r_orig.font.strike = True
            r_orig.font.color.rgb = COLOR_CRITICAL

        # 3 Tiers
        p_rc.add_run("⚖️ PHƯƠNG ÁN 1: CÂN BẰNG (CHUẨN MỰC THỊ TRƯỜNG - KHUYẾN NGHỊ)\n").font.bold = True
        r_bal = p_rc.add_run(f"{bal_text}\n\n")
        r_bal.font.bold = True
        r_bal.font.color.rgb = COLOR_SUCCESS

        if strict_text and strict_text != bal_text:
            p_rc.add_run("🛡️ PHƯƠNG ÁN 2: KHẮT KHE (BẢO VỆ TỐI ĐA THÂN CHỦ)\n").font.bold = True
            r_str = p_rc.add_run(f"{strict_text}\n\n")
            r_str.font.color.rgb = COLOR_PRIMARY

        if flex_text and flex_text != bal_text:
            p_rc.add_run("🤝 PHƯƠNG ÁN 3: MỀM DẺO (NHƯỢNG BỘ THƯƠNG MẠI CÓ ĐIỀU KIỆN)\n").font.bold = True
            r_flx = p_rc.add_run(f"{flex_text}\n\n")
            r_flx.font.color.rgb = COLOR_MEDIUM

        # Precedent & Rationale
        if precedent:
            p_rc.add_run("🏛️ CĂN CỨ ÁN LỆ TAND / TIỀN LỆ VIAC:\n").font.bold = True
            r_prc = p_rc.add_run(f"{precedent}\n\n")
            r_prc.font.italic = True
            r_prc.font.color.rgb = COLOR_NAVY

        if rationale:
            p_rc.add_run("💡 LUẬN CỨ ĐÀM PHÁN VỚI ĐỐI TÁC:\n").font.bold = True
            p_rc.add_run(f"{rationale}\n")

        doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # 5. PHỤ LỤC KIỂM TOÁN SỐ HỌC (ARITHMETIC AUDIT)
    # -------------------------------------------------------------------------
    arithmetic_issues = report_data.get("arithmetic_findings") or []
    if arithmetic_issues:
        h5 = doc.add_heading(level=1)
        r5 = h5.add_run("V. PHỤ LỤC: KẾT QUẢ KIỂM TOÁN SỐ HỌC BẢNG BIỂU")
        r5.font.size = Pt(13)
        r5.font.bold = True
        r5.font.color.rgb = COLOR_NAVY
        h5.paragraph_format.space_before = Pt(14)
        h5.paragraph_format.space_after = Pt(6)

        table_a = doc.add_table(rows=1, cols=3)
        table_a.alignment = WD_TABLE_ALIGNMENT.CENTER
        headers_a = ["Vị trí Bảng biểu", "Loại sai lệch số học", "Chi tiết sai số"]
        widths_a = [Inches(1.8), Inches(2.2), Inches(2.7)]
        for idx, (title, w) in enumerate(zip(headers_a, widths_a)):
            cell = table_a.cell(0, idx)
            cell.width = w
            _set_cell_background(cell, "DC2626")
            _set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(title)
            run.font.bold = True
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

        for issue in arithmetic_issues:
            row_a = table_a.add_row().cells
            row_a[0].width = widths_a[0]
            row_a[0].paragraphs[0].add_run(str(issue.get("table_location", "Bảng tài chính")))

            row_a[1].width = widths_a[1]
            row_a[1].paragraphs[0].add_run(str(issue.get("error_type", "Sai lệch tính toán")))

            row_a[2].width = widths_a[2]
            row_a[2].paragraphs[0].add_run(str(issue.get("description", "Không khớp số học")))

    # Save to memory stream
    output_stream = io.BytesIO()
    doc.save(output_stream)
    output_stream.seek(0)
    logger.info(f"Generated contract audit DOCX report: '{contract_title}' ({len(output_stream.getvalue())} bytes).")
    return output_stream

