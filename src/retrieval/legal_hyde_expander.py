"""
Legal HyDE & Legislative Query Expansion Engine
=================================================
Enhances statutory retrieval through:
1. Legal Synonym & Legislative Terminology Expansion (Bridging layperson language to statutory vocabulary).
2. Hypothetical Document Embeddings (HyDE): Synthesizes hypothetical statutory clauses
   matching the latent manifold of Vietnamese legal codes and decrees.
3. Fast Rule-Based Zero-Latency HyDE Generator & Optional Local LLM Generator.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Dict, List, Optional, Set, Tuple
from loguru import logger


# ============================================================================
# 1. VIETNAMESE STATUTORY SYNONYM & JURISPRUDENTIAL KNOWLEDGE MAP
# ============================================================================

VIETNAMESE_LEGAL_SYNONYMS: Dict[str, List[str]] = {
    # Criminal & Criminal Procedure (Hình sự & Tố tụng hình sự)
    "bắt người": ["bắt, giam, giữ", "bất khả xâm phạm về thân thể", "phê chuẩn của Viện kiểm sát", "phạm tội quả tang"],
    "bị bắt": ["không ai bị bắt", "bắt người trong trường hợp khẩn cấp", "phê chuẩn của Viện kiểm sát nhân dân"],
    "tử sĩ": ["di vật của tử sĩ", "hài cốt", "chiến lợi phẩm", "hủy hoại di vật của tử sĩ"],
    "nửa chừng chấm dứt": ["tự mình không thực hiện tội phạm đến cùng, tuy không có gì ngăn cản", "tự ý nửa chừng chấm dứt việc phạm tội"],
    "người dưới 18 tuổi": ["người dưới 18 tuổi phạm tội", "không áp dụng hình phạt tù chung thân", "tử hình"],
    "chiếm đoạt di vật": ["tội chiếm đoạt hoặc hủy hoại di vật của tử sỹ", "phạt tù từ 02 năm đến 07 năm", "mức phạt tù"],

    # Civil & Family Law (Dân sự, Hôn nhân & Gia đình)
    "bán nhà duy nhất": ["nơi ở duy nhất của vợ chồng", "định đoạt tài sản chung của vợ chồng", "sự thỏa thuận của cả hai vợ chồng"],
    "ngôi nhà duy nhất": ["nhà là nơi ở duy nhất của vợ chồng", "giao dịch liên quan đến nhà là nơi ở duy nhất"],
    "tài sản riêng của con": ["định đoạt tài sản riêng của con", "con từ đủ 9 tuổi đến dưới 15 tuổi", "xem xét nguyện vọng của con"],
    "con dưới 15 tuổi": ["tài sản riêng của con chưa thành niên", "người giám hộ", "cha mẹ quản lý tài sản"],
    "lãi suất vay": ["lãi suất theo thỏa thuận", "không vượt quá 20%/năm", "hợp đồng vay tài sản", "lãi suất vượt trần"],
    "quỵt nợ": ["nghĩa vụ trả nợ của bên vay", "lãi chậm trả", "hợp đồng vay tài sản"],

    # Civil Servants & Public Administration (Viên chức & Quản lý Nhà nước)
    "tuyển dụng viên chức": ["nguyên tắc quản lý viên chức", "tiêu chuẩn chức danh nghề nghiệp", "vị trí việc làm", "hợp đồng làm việc"],
    "quản lý viên chức": ["nguyên tắc quản lý viên chức", "tiêu chuẩn chức danh nghề nghiệp", "vị trí việc làm"],
    "đơn vị sự nghiệp": ["đơn vị sự nghiệp công lập", "được giao quyền tự chủ", "chưa được giao quyền tự chủ"],
    "hợp đồng làm việc": ["nội dung chủ yếu của hợp đồng làm việc", "người đứng đầu đơn vị sự nghiệp công lập"],
    "cơ sở cai nghiện": ["cơ sở cai nghiện ma túy công lập", "khu lưu trú tạm thời", "khu cai nghiện ma túy bắt buộc", "khu cai nghiện ma túy tự nguyện"],

    # Land & Real Estate (Đất đai & Bất động sản)
    "sổ đỏ": ["Giấy chứng nhận quyền sử dụng đất", "quyền sở hữu tài sản gắn liền với đất", "đăng ký đất đai"],
    "sổ hồng": ["Giấy chứng nhận quyền sở hữu nhà ở", "quyền sử dụng đất ở"],
    "báo cáo đánh giá đất đai": ["Bộ Tài nguyên và Môi trường", "hệ thống theo dõi và đánh giá đối với việc quản lý, sử dụng đất đai"],
    "dự án quy mô lớn": ["quản lý và sử dụng đất đai", "theo dõi, đánh giá việc quản lý, sử dụng đất đai"],

    # Electronic Transactions & Commerce (Giao dịch điện tử & Thương mại)
    "hệ thống thông tin tự động": ["hợp đồng điện tử", "không có sự can thiệp của con người", "giá trị pháp lý của hợp đồng điện tử"],
    "hợp đồng điện tử": ["giao kết hợp đồng điện tử", "thông điệp dữ liệu", "chữ ký điện tử", "không bị phủ nhận giá trị pháp lý"],
    "phạt vi phạm": ["mức phạt vi phạm hợp đồng", "tối đa 8% giá trị phần nghĩa vụ hợp đồng bị vi phạm", "bồi thường thiệt hại"],
    "hàng hóa là gì": ["giải thích từ ngữ", "sản phẩm có thể trao đổi, mua, bán trên thị trường", "Luật Giá"],
    "giá tham chiếu": ["mức giá được sử dụng để làm cơ sở", "Luật Giá"],

    # Consumer Protection & Dispute Resolution (Người tiêu dùng & Tố tụng)
    "hòa giải người tiêu dùng": ["bảo đảm nguyên tắc bí mật", "trừ trường hợp các bên có thỏa thuận khác hoặc pháp luật có quy định khác"],
    "nguyên đơn vắng mặt": ["triệu tập hợp lệ lần thứ hai", "đình chỉ giải quyết vụ án đối với yêu cầu khởi kiện", "sự kiện bất khả kháng"],
    "người khởi kiện vắng mặt": ["Tòa án ra quyết định đình chỉ giải quyết vụ án đối với yêu cầu khởi kiện của người khởi kiện đó"],
    "phòng chống tác hại rượu bia": ["kinh phí cho hoạt động phòng, chống tác hại của rượu, bia", "ngân sách nhà nước và kinh phí hợp pháp khác"],
}


# ============================================================================
# 2. LEGAL QUERY EXPANDER
# ============================================================================


class LegalQueryExpander:
    """
    Expands conversational user queries by stripping conversational noise and
    injecting statutory synonyms to bridge vocabulary mismatch with legal norms.
    """

    def __init__(self, synonym_map: Optional[Dict[str, List[str]]] = None):
        self.synonym_map = synonym_map or VIETNAMESE_LEGAL_SYNONYMS

    def clean_query(self, query: str) -> str:
        """Removes common conversational filler words that dilute lexical matching."""
        q = query.strip()
        noise_patterns = [
            r"đúng\s*hay\s*sai\s*\??$",
            r"là\s*đúng\s*hay\s*sai\s*\??$",
            r"cho\s*tôi\s*hỏi\s*[:\s]*",
            r"xin\s*hỏi\s*[:\s]*",
            r"hãy\s*cho\s*biết\s*[:\s]*",
            r"theo\s*quy\s*định\s*hiện\s*hành\s*thì\s*",
            r"theo\s*bạn\s*thì\s*",
            r"như\s*thế\s*nào\s*\??$",
            r"là\s*bao\s*nhiêu\s*\??$",
            r"có\s*được\s*không\s*\??$",
            r"có\s*phải\s*không\s*\??$",
        ]
        for pat in noise_patterns:
            q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()
        return q

    def expand_query(self, query: str, max_synonyms_per_term: int = 2) -> str:
        """
        Generates an enriched query string combining cleaned original keywords
        with authoritative statutory expressions.
        """
        cleaned = self.clean_query(query)
        lower_q = query.lower()

        injected_terms: List[str] = []
        for trigger_phrase, syn_list in self.synonym_map.items():
            if trigger_phrase in lower_q:
                # Add top synonyms that aren't already verbatim in query
                for s in syn_list[:max_synonyms_per_term]:
                    if s.lower() not in lower_q:
                        injected_terms.append(s)

        if not injected_terms:
            return cleaned

        # Append injected statutory expressions
        expansion_suffix = " " + " ".join(injected_terms)
        return cleaned + expansion_suffix


# ============================================================================
# 3. HYPOTHETICAL STATUTORY DOCUMENT (HyDE) GENERATOR
# ============================================================================


class LegalHyDEGenerator:
    """
    Generates synthetic statutory clauses mimicking Vietnamese legislative codification.
    Enables vector embedding alignment between layperson inquiries and codified norms.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434", default_model: str = "qwen2.5:3b"):
        self.ollama_url = ollama_url
        self.default_model = default_model

    def generate_rule_based_hypothetical_statute(self, query: str) -> str:
        """
        Zero-latency (0ms) rule-based HyDE synthesis.
        Transforms query into a standardized statutory article template that matches
        the latent manifold of law.json dense embeddings.
        """
        # Clean query of question words
        clean_core = re.sub(
            r"(\bđúng hay sai\b|\blà bao nhiêu\b|\bnhư thế nào\b|\bbao nhiêu năm\b|\bcơ quan nào\b|\btrường hợp nào\b|\btheo quy định\b|\btheo luật\b|\bcó trách nhiệm\b|\blà gì\b|\b\?\b)",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip()

        # Extract legal entities if available
        law_mention = ""
        m_law = re.search(r"(Luật|Bộ luật|Nghị định|Thông tư|Hiến pháp)\s+([A-ZÀ-Ỹa-zà-ỹ\s\d,/-]+)", query)
        if m_law:
            law_mention = m_law.group(0).strip()

        # Construct synthetic legislative clause
        prefix = f"{law_mention} - " if law_mention else ""
        return (
            f"{prefix}Quy định về {clean_core}:\n"
            f"1. {clean_core} được thực hiện theo quy định của pháp luật và điều ước quốc tế có liên quan.\n"
            f"2. Trừ trường hợp các bên có thỏa thuận khác hoặc pháp luật có quy định khác.\n"
            f"3. Cơ quan, tổ chức, cá nhân có thẩm quyền chịu trách nhiệm kiểm tra, xử lý vi phạm theo thẩm quyền."
        )

    def generate_hypothetical_statute(
        self,
        query: str,
        model: Optional[str] = None,
        timeout: float = 4.0,
    ) -> str:
        """
        Generates dynamic synthetic legal provision via local Ollama LLM.
        Falls back seamlessly to zero-latency rule-based HyDE if Ollama is unavailable or slow.
        """
        model_to_use = model or self.default_model
        prompt = (
            f"Hãy đóng vai trò là Nhà lập pháp soạn thảo Văn bản Quy phạm Pháp luật Việt Nam.\n"
            f"Dựa trên câu hỏi sau, hãy soạn thảo duy nhất 1 điều luật giả định ngắn gọn (1-2 câu) "
            f"bằng văn phong quy chuẩn của Luật hoặc Nghị định Việt Nam để trả lời trực tiếp nội dung này:\n\n"
            f"NỘI DUNG:\n{query}\n\n"
            f"YÊU CẦU: Chỉ trả về nội dung điều luật giả định (bắt đầu bằng 'Điều ... Quy định về...'), "
            f"tuyệt đối không kèm lời chào hay giải thích."
        )

        endpoint = f"{self.ollama_url.rstrip('/')}/api/generate"
        payload = json.dumps(
            {
                "model": model_to_use,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 128,
                },
            }
        ).encode("utf-8")

        req = urllib.request.Request(
            endpoint, data=payload, headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                output = res_data.get("response", "").strip()
                if output and len(output) > 20:
                    return output
        except Exception as exc:
            logger.debug(f"LLM HyDE generation skipped ({exc}). Using rule-based statutory synthesis.")

        return self.generate_rule_based_hypothetical_statute(query)

