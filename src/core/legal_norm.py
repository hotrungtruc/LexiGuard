"""
Universal Vietnamese Legal Instrument Normalizer & Slug Generator
===================================================================
Provides domain-agnostic, canonical identifier resolution for ANY Vietnamese legislation:
- Codes (Bộ luật): Civil, Labor, Penal, Maritime, Civil Procedure, etc.
- Laws (Luật): Commercial, Enterprise, Land, Cyber, IP, Pharmacy, Tax, Real Estate, etc.
- Decrees (Nghị định): e.g. 13/2023/ND-CP, 01/2021/ND-CP
- Circulars (Thông tư): e.g. 10/2020/TT-BLDTBXH
- Resolutions & Decisions (Nghị quyết, Quyết định): e.g. 01/2019/NQ-HDTP
- Internal Regulations & SOPs (Quy chế, Quy định): Corporate Governance, Financial SOPs, etc.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from loguru import logger

# Map of standard Vietnamese diacritic characters to plain ASCII
VIETNAMESE_ACCENTS_MAP = {
    'à': 'a', 'á': 'a', 'ả': 'a', 'ã': 'a', 'ạ': 'a',
    'ă': 'a', 'ằ': 'a', 'ắ': 'a', 'ẳ': 'a', 'ẵ': 'a', 'ặ': 'a',
    'â': 'a', 'ầ': 'a', 'ấ': 'a', 'ẩ': 'a', 'ẫ': 'a', 'ậ': 'a',
    'đ': 'd',
    'è': 'e', 'é': 'e', 'ẻ': 'e', 'ẽ': 'e', 'ẹ': 'e',
    'ê': 'e', 'ề': 'e', 'ế': 'e', 'ể': 'e', 'ễ': 'e', 'ệ': 'e',
    'ì': 'i', 'í': 'i', 'ỉ': 'i', 'ĩ': 'i', 'ị': 'i',
    'ò': 'o', 'ó': 'o', 'ỏ': 'o', 'õ': 'o', 'ọ': 'o',
    'ô': 'o', 'ồ': 'o', 'ố': 'o', 'ổ': 'o', 'ỗ': 'o', 'ộ': 'o',
    'ơ': 'o', 'ờ': 'o', 'ớ': 'o', 'ở': 'o', 'ỡ': 'o', 'ợ': 'o',
    'ù': 'u', 'ú': 'u', 'ủ': 'u', 'ũ': 'u', 'ụ': 'u',
    'ư': 'u', 'ừ': 'u', 'ứ': 'u', 'ử': 'u', 'ữ': 'u', 'ự': 'u',
    'ỳ': 'y', 'ý': 'y', 'ỷ': 'y', 'ỹ': 'y', 'ỵ': 'y',
}


def strip_vietnamese_diacritics(text: str) -> str:
    """Converts Vietnamese characters with accents into plain Latin ASCII."""
    if not text:
        return ""
    text_lower = text.lower()
    res = "".join(VIETNAMESE_ACCENTS_MAP.get(ch, ch) for ch in text_lower)
    return unicodedata.normalize('NFKD', res).encode('ASCII', 'ignore').decode('utf-8')


# Canonical law shortcuts registry
_KNOWN_SHORTCUTS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\b(?:trong[-_]?tai|ltttm)\b', re.IGNORECASE), "VN_LTTTM_2010"),
    (re.compile(r'\b(?:kinh[-_]?doanh[-_]?bat[-_]?dong[-_]?san|lkdbds)\b', re.IGNORECASE), "VN_LKDBDS_2023"),
    (re.compile(r'\b(?:so[-_]?huu[-_]?tri[-_]?tue|lshtt)\b', re.IGNORECASE), "VN_LSHTT_2022"),
    (re.compile(r'\b(?:an[-_]?ninh[-_]?mang|lanm)\b', re.IGNORECASE), "VN_LANM_2018"),
    (re.compile(r'\b(?:an[-_]?toan[-_]?thong[-_]?tin[-_]?mang|latttm)\b', re.IGNORECASE), "VN_LATTTM_2015"),
    (re.compile(r'\b(?:dan[-_]?su|blds)\b', re.IGNORECASE), "VN_BLDS_2015"),
    (re.compile(r'\b(?:lao[-_]?dong|blld)\b', re.IGNORECASE), "VN_BLLD_2019"),
    (re.compile(r'\b(?:thuong[-_]?mai|ltm)\b', re.IGNORECASE), "VN_LTM_2005"),
    (re.compile(r'\b(?:doanh[-_]?nghiep|ldn)\b', re.IGNORECASE), "VN_LDN_2020"),
    (re.compile(r'\b(?:dat[-_]?dai|ldd)\b', re.IGNORECASE), "VN_LDD_2024"),
    (re.compile(r'\b(?:nha[-_]?o|lno)\b', re.IGNORECASE), "VN_LNO_2023"),
    (re.compile(r'\b(?:xay[-_]?dung|lxd)\b', re.IGNORECASE), "VN_LXD_2020"),
    (re.compile(r'\b(?:dau[-_]?thau|ldt)\b', re.IGNORECASE), "VN_LDT_2023"),
    (re.compile(r'\b(?:hinh[-_]?su|blhs)\b', re.IGNORECASE), "VN_BLHS_2015"),
    (re.compile(r'\b(?:to[-_]?tung[-_]?dan[-_]?su|blttds)\b', re.IGNORECASE), "VN_BLTTDS_2015"),
]

# User/runtime registered law mappings
_CUSTOM_LAW_REGISTRY: Dict[str, str] = {}


def register_custom_law(name_or_title: str, canonical_code: str) -> None:
    """Registers an arbitrary custom legal instrument into the dynamic resolver."""
    clean_key = strip_vietnamese_diacritics(name_or_title).strip().lower()
    _CUSTOM_LAW_REGISTRY[clean_key] = canonical_code.strip()
    logger.info(f"Registered custom law: '{name_or_title}' -> '{canonical_code}'")


def canonical_law_slug(law_identifier: str) -> str:
    """
    Dynamically maps ANY Vietnamese law title, slug, or document name into a deterministic
    canonical code (e.g. 'VN_LANM_2018', 'VN_ND13_2023', 'VN_QCTCNB_2024').
    
    Fully generalized: handles any statute, decree, circular, resolution, or corporate rulebook.
    """
    if not law_identifier:
        return "VN_LAW_UNKNOWN"

    raw_str = law_identifier.strip()

    # 1. If already a canonical code (e.g. VN_BLDS_2015, VN_LANM_2018), return normalized
    if re.match(r'^VN_[A-Z0-9_]+$', raw_str, re.IGNORECASE):
        return raw_str.upper()

    norm_str = strip_vietnamese_diacritics(raw_str).lower()

    # 2. Check runtime custom registry
    for custom_name, custom_code in _CUSTOM_LAW_REGISTRY.items():
        if custom_name in norm_str:
            return custom_code

    # 3. Check known shortcuts with year override
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', norm_str)
    year_found = year_match.group(1) if year_match else None

    for pat, code in _KNOWN_SHORTCUTS:
        if pat.search(norm_str):
            if year_found:
                prefix = code.rsplit('_', 1)[0]
                return f"{prefix}_{year_found}"
            return code

    # 4. Universal Dynamic Resolution for arbitrary Vietnamese legislation
    # Determine instrument category prefix
    if re.search(r'\b(?:bo\s*luat|blds|blld|blhs|blttds)\b', norm_str):
        prefix = "VN_BL"
    elif re.search(r'\b(?:nghi\s*dinh|nd-cp|ndcp)\b', norm_str):
        prefix = "VN_ND"
    elif re.search(r'\b(?:thong\s*tu|tt-)\b', norm_str):
        prefix = "VN_TT"
    elif re.search(r'\b(?:quy\s*che|qc-)\b', norm_str):
        prefix = "VN_QC"
    elif re.search(r'\b(?:quy\s*dinh|qd-)\b', norm_str):
        prefix = "VN_QD"
    elif re.search(r'\b(?:quyet\s*dinh)\b', norm_str):
        prefix = "VN_QDD"
    elif re.search(r'\b(?:phap\s*lenh)\b', norm_str):
        prefix = "VN_PL"
    elif re.search(r'\b(?:nghi\s*quyet|nq-)\b', norm_str):
        prefix = "VN_NQ"
    elif re.search(r'\b(?:luat)\b', norm_str):
        prefix = "VN_L"
    else:
        prefix = "VN_STATUTE"

    # Check for official decree / circular / resolution number like 13/2023/ND-CP or 13-2023
    num_match = re.search(r'(\d+)[/_\-](\d{4})', norm_str)
    if num_match:
        doc_num = num_match.group(1)
        doc_year = num_match.group(2)
        return f"{prefix}{doc_num}_{doc_year}"

    # Extract substantive tokens (ignoring instrument type prefixes and digits)
    clean_slug = re.sub(
        r'^(luat[-_\s]+|bo[-_\s]+luat[-_\s]+|nghi[-_\s]+dinh[-_\s]+|thong[-_\s]+tu[-_\s]+|quy[-_\s]+che[-_\s]+|quy[-_\s]+dinh[-_\s]+)',
        '',
        norm_str
    )
    tokens = [w for w in re.split(r'[-_\s/]+', clean_slug) if w and not w.isdigit()]

    year_str = f"_{year_found}" if year_found else ""

    if tokens:
        abbrev = "".join(w[0].upper() for w in tokens)
        if len(abbrev) == 1 and len(tokens[0]) > 1:
            abbrev = tokens[0][:4].upper()
        return f"{prefix}{abbrev}{year_str}"

    return f"{prefix}_{clean_slug[:8].upper()}{year_str}"
