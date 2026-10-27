"""
Script: generate_ocr_pdfs.py
Converts OCR extracted Vietnamese legal texts into official formatted PDF documents.
Uses ReportLab with native Windows Times New Roman TTF for 100% Vietnamese Unicode support.
"""

import os
import json
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Vietnamese TrueType fonts from Windows Fonts
FONTS_DIR = "C:/Windows/Fonts"
pdfmetrics.registerFont(TTFont("Times", os.path.join(FONTS_DIR, "times.ttf")))
pdfmetrics.registerFont(TTFont("Times-Bold", os.path.join(FONTS_DIR, "timesbd.ttf")))
pdfmetrics.registerFont(TTFont("Times-Italic", os.path.join(FONTS_DIR, "timesi.ttf")))
pdfmetrics.registerFont(TTFont("Times-BoldItalic", os.path.join(FONTS_DIR, "timesbi.ttf")))
pdfmetrics.registerFontFamily(
    "Times",
    normal="Times",
    bold="Times-Bold",
    italic="Times-Italic",
    boldItalic="Times-BoldItalic"
)

def md_to_xml(text: str) -> str:
    """Safely converts markdown bold and italic markers into valid ReportLab XML."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    # Convert bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Convert italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    text = text.replace("`", "")
    return text

def build_pdf_document(json_data: dict, md_content: str, output_pdf_path: Path):
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.8 * cm,
        bottomMargin=1.8 * cm,
        title=json_data.get("metadata", {}).get("title", "Văn bản pháp lý"),
        author="LexiGuard OCR Engine"
    )

    # Define custom legal typography styles
    style_normal = ParagraphStyle(
        "VN_Normal",
        fontName="Times",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=4
    )
    style_italic = ParagraphStyle(
        "VN_Italic",
        fontName="Times-Italic",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#2C3E50"),
        spaceAfter=3
    )
    style_bold = ParagraphStyle(
        "VN_Bold",
        fontName="Times-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#000000"),
        spaceBefore=6,
        spaceAfter=4
    )
    style_header_agency = ParagraphStyle(
        "VN_Agency",
        fontName="Times-Bold",
        fontSize=10,
        leading=13,
        alignment=1, # Center
        textColor=colors.HexColor("#000000")
    )
    style_header_national = ParagraphStyle(
        "VN_National",
        fontName="Times-Bold",
        fontSize=10,
        leading=13,
        alignment=1, # Center
        textColor=colors.HexColor("#000000")
    )
    style_header_motto = ParagraphStyle(
        "VN_Motto",
        fontName="Times-Italic",
        fontSize=10,
        leading=13,
        alignment=1, # Center
        textColor=colors.HexColor("#000000")
    )
    style_title = ParagraphStyle(
        "VN_DocTitle",
        fontName="Times-Bold",
        fontSize=15,
        leading=19,
        alignment=1, # Center
        textColor=colors.HexColor("#8B0000"), # Dark Red
        spaceAfter=6
    )
    style_subtitle = ParagraphStyle(
        "VN_SubTitle",
        fontName="Times-Bold",
        fontSize=12,
        leading=16,
        alignment=1,
        textColor=colors.HexColor("#1A1A1A"),
        spaceAfter=10
    )
    style_article_header = ParagraphStyle(
        "VN_ArticleHeader",
        fontName="Times-Bold",
        fontSize=12,
        leading=16,
        spaceBefore=8,
        spaceAfter=3,
        textColor=colors.HexColor("#003366")
    )
    style_callout = ParagraphStyle(
        "VN_Callout",
        fontName="Times",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#7B241C"),
        leftIndent=15,
        rightIndent=15,
        spaceBefore=5,
        spaceAfter=5
    )
    style_provenance = ParagraphStyle(
        "VN_Provenance",
        fontName="Times-Italic",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#7F8C8D")
    )

    story = []

    # 1. Top Provenance Banner
    meta = json_data.get("metadata", {})
    source_file = json_data.get("file_name", "")
    ocr_engine = json_data.get("ocr_engine", "PaddleOCR + VietOCR")
    prov_text = f"<b>LexiGuard OCR Pipeline</b> | Nguồn ảnh: <i>{source_file}</i> ({json_data.get('width', 0)}x{json_data.get('height', 0)}px) | Engine: <i>{ocr_engine}</i>"
    story.append(Paragraph(prov_text, style_provenance))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#BDC3C7"), spaceBefore=3, spaceAfter=12))

    # 2. National Motto & Agency Header (2-column layout)
    agency = meta.get("issuing_authority", "CƠ QUAN BAN HÀNH").upper()
    doc_number = meta.get("doc_number", "")
    
    agency_cell = [
        Paragraph(md_to_xml(agency), style_header_agency),
        Paragraph(f"Số: <b>{doc_number}</b>" if doc_number else "", ParagraphStyle("CenterNum", fontName="Times", fontSize=10, leading=13, alignment=1))
    ]
    national_cell = [
        Paragraph("CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", style_header_national),
        Paragraph("Độc lập - Tự do - Hạnh phúc", style_header_motto),
        HRFlowable(width="50%", thickness=0.8, color=colors.black, spaceBefore=2, spaceAfter=4)
    ]
    
    header_table = Table([[agency_cell, national_cell]], colWidths=[8.0 * cm, 9.5 * cm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 14))

    # 3. Document Title
    doc_type = meta.get("doc_type", "VĂN BẢN")
    doc_title = meta.get("title", "")
    story.append(Paragraph(md_to_xml(doc_type.upper()), style_title))
    if doc_title and doc_title.upper() != doc_type.upper():
        story.append(Paragraph(md_to_xml(doc_title), style_subtitle))
    story.append(Spacer(1, 8))

    # 4. Extract content sections from Markdown
    lines = md_content.splitlines()
    in_content = False
    
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        
        # Start processing after "## 📝 Nội dung văn bản trích xuất nguyên văn"
        if "Nội dung văn bản trích xuất" in line:
            in_content = True
            continue
        
        if not in_content:
            continue
            
        # Skip repetitive header lines that are already in header_table
        if any(h in line for h in ["CỘNG HÒA XÃ HỘI", "Độc lập - Tự do", "Số:", "Số :", doc_type.upper()]):
            continue

        # Format headings, bases, articles, callouts
        if line.startswith("### "):
            heading_text = line.replace("### ", "").strip()
            story.append(Spacer(1, 6))
            story.append(Paragraph(md_to_xml(heading_text), style_article_header))
        elif line.startswith("#### "):
            subheading_text = line.replace("#### ", "").strip()
            story.append(Paragraph(f"<b>{md_to_xml(subheading_text)}</b>", style_bold))
        elif line.startswith("- *") or line.startswith("*Căn cứ") or line.startswith("- Căn cứ"):
            clean_text = line.lstrip("- *").rstrip("*").strip()
            story.append(Paragraph(f"• <i>{md_to_xml(clean_text)}</i>", style_italic))
        elif line.startswith("> "):
            callout = line.lstrip("> ").strip()
            story.append(Paragraph(f"⚠️ {md_to_xml(callout)}", style_callout))
        elif line.startswith("| ") and "---" not in line:
            # Table row in markdown
            pass
        elif line.startswith("**") and ("CHÁNH ÁN" in line or "CỤC TRƯỞNG" in line or "GIÁM ĐỐC" in line or "CHỦ TỊCH" in line):
            # Signer title block
            clean_signer = line.replace("**", "").replace("*", "")
            signer_style = ParagraphStyle(
                "SignerTitle",
                fontName="Times-Bold",
                fontSize=11,
                leading=15,
                alignment=2, # Right align
                spaceBefore=15
            )
            story.append(Paragraph(md_to_xml(clean_signer), signer_style))
        elif any(s in line for s in ["Nguyễn Viết Xuân", "Hồ Quang Huy", "Nguyễn Hồng Trường", "Huỳnh Cách Mạng", "Nguyễn Ngọc Tuấn"]):
            clean_name = line.replace("**", "").strip()
            name_style = ParagraphStyle(
                "SignerName",
                fontName="Times-Bold",
                fontSize=12,
                leading=16,
                alignment=2, # Right align
                spaceBefore=25
            )
            story.append(Paragraph(md_to_xml(clean_name), name_style))
        else:
            # Standard paragraph / Article
            story.append(Paragraph(md_to_xml(line), style_normal))

    # 5. Footer Bounding Box & Spatial Grounding Note
    story.append(Spacer(1, 15))
    bboxes = json_data.get("bounding_boxes", [])
    box_count = len(bboxes)
    footer_note = (
        f"<font size='8' color='#7F8C8D'>Văn bản được tạo tự động từ luồng OCR LexiGuard. "
        f"Đã trích xuất {box_count} vùng tọa độ không gian (Bounding Boxes) chuẩn hóa để đối chiếu điều khoản.</font>"
    )
    story.append(Paragraph(footer_note, ParagraphStyle("Footer", fontName="Times-Italic", fontSize=8, leading=11, alignment=1)))

    doc.build(story)
    print(f"[OK] Generated PDF: {output_pdf_path.name}")

def main():
    base_dir = Path("data/sample_contracts")
    extracted_dir = base_dir / "extracted_texts"
    output_dir = base_dir / "extracted_pdfs"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(extracted_dir.glob("*.json")))
    print(f"Found {len(json_files)} extracted OCR data files. Generating PDF documents...")

    for json_file in json_files:
        stem = json_file.stem
        md_file = extracted_dir / f"{stem}.md"
        
        if not md_file.exists():
            print(f"[SKIP] Missing MD file for {json_file.name}, skipping.")
            continue

        with open(json_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        with open(md_file, "r", encoding="utf-8") as f:
            md_content = f.read()

        # Map output filename matching the original image base
        pdf_name = f"{stem}.pdf"
        output_pdf = output_dir / pdf_name

        build_pdf_document(json_data, md_content, output_pdf)

    print(f"\nAll {len(json_files)} PDF documents have been successfully generated in: {output_dir}")

if __name__ == "__main__":
    main()
