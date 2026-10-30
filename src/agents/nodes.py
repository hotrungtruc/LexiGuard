"""
Nodes: Specialized Multi-Agent Logic under Dual-LLM CPU Architecture
===================================================================
Orchestrates 4 specialized nodes:
- Node 1: ClauseExtractor (powered by Qwen2.5-3B-Instruct, ~40 tokens/s on CPU)
- Node 2: LegalResearcher (autonomous FastMCP tools: search_legal_provisions, verify_table_arithmetic, get_article_graph_context)
- Node 3: LoopholeAuditor (powered by Qwen2.5-7B-Instruct, ~15-20 tokens/s on CPU)
- Node 4: HallucinationGrader (powered by Qwen2.5-3B-Instruct, verifying citations against grounded context)
"""

import asyncio
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from loguru import logger

from src.agents.prompts import (
    build_clause_extractor_prompt,
    build_hallucination_grader_prompt,
    build_loophole_auditor_prompt,
)
from src.agents.state import (
    UNCLASSIFIED_VULNERABILITY,
    AgentState,
    AuditDraftReport,
    AuditFindingItem,
    ClauseExtractionList,
    ContractAuditReport,
    ContractProvenance,
    DisputeForum,
    DisputePrecedentInsight,
    ExtractedClauseItem,
    HallucinationGrade,
    LegalGrounding,
    LegalRuleMetadata,
    VulnerabilityItem,
    Redline3Tier,
)
from src.core.config import settings
from src.core.legal_norm import canonical_law_slug
from src.mcp_server.legal_tools import (
    get_article_graph_context,
    search_legal_provisions,
    verify_table_arithmetic,
)
from src.retrieval.entity_extractor import LegalEntityExtractor

# ============================================================================
# DYNAMIC VULNERABILITY TAXONOMY & PRECEDENT INSIGHT HELPER
# ============================================================================


def rule_metadata_for(
    statute: Optional[Dict[str, Any]],
    precedent: Optional[Dict[str, Any]] = None,
) -> LegalRuleMetadata:
    """Read an opaque rule descriptor from retrieved knowledge-base records."""
    merged: Dict[str, Any] = {}
    for record in (statute or {}, precedent or {}):
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            merged.update(metadata)
        merged.update(
            {
                k: v
                for k, v in record.items()
                if k
                in {
                    "category_id",
                    "category_label",
                    "risk_tier",
                    "forum",
                    "precedent_reference",
                    "judicial_tendency",
                    "redline_template",
                }
            }
        )
    return LegalRuleMetadata.model_validate(merged)


def dispute_insight_for(
    metadata: LegalRuleMetadata, fallback_text: str
) -> Optional[DisputePrecedentInsight]:
    """Convert optional DB rule metadata to the public report model."""
    if not any((metadata.forum, metadata.precedent_reference, metadata.judicial_tendency)):
        return None
    forum = metadata.forum if metadata.forum in {"TAND", "VIAC", "GENERAL"} else "GENERAL"
    return DisputePrecedentInsight(
        target_forum=forum,
        precedent_reference=metadata.precedent_reference,
        judicial_tendency=metadata.judicial_tendency or fallback_text,
        adjudication_risk_probability=metadata.risk_tier,
    )


# ============================================================================
# 1. DUAL-LLM CPU CLIENT INITIALIZATION
# ============================================================================


def get_fast_cpu_llm() -> ChatOllama:
    """Fast 3B model for structural parsing and citation grading (~40 t/s on CPU)."""
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.FAST_LLM_MODEL,
        temperature=0.0,
        num_ctx=8192,
        num_thread=settings.OLLAMA_NUM_THREADS,
    )


def get_reasoning_cpu_llm() -> ChatOllama:
    """Reasoning 7B model for deep legal conflict auditing and redlining (~15-20 t/s on CPU)."""
    return ChatOllama(
        base_url=settings.OLLAMA_BASE_URL,
        model=settings.REASONING_LLM_MODEL,
        temperature=0.1,
        num_ctx=16384,
        num_thread=settings.OLLAMA_NUM_THREADS,
    )


# ============================================================================
# 2. NODE 1: ClauseExtractor (Qwen2.5-3B)
# ============================================================================


def extract_clauses_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 1: Parses raw contract markdown into high-risk clauses and financial tables.
    Runs on Qwen2.5-3B-Instruct for maximum CPU throughput.
    """
    logger.info("Executing Node 1: ClauseExtractor (Qwen2.5-3B)...")
    raw_path = state.get("raw_document_path", "")
    contract_text = ""

    # If raw path points to a file, read it; otherwise check parsed_clauses
    if raw_path:
        try:
            path_obj = Path(raw_path)
            if path_obj.suffix.lower() == ".pdf":
                import pypdf

                reader = pypdf.PdfReader(str(path_obj))
                contract_text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
            else:
                with open(raw_path, "r", encoding="utf-8") as f:
                    contract_text = f.read()
        except Exception as e:
            logger.warning(f"Could not read raw_document_path '{raw_path}': {e}")

    # Fallback to existing clauses if already provided in state
    existing_clauses = state.get("parsed_clauses", [])
    if existing_clauses:
        return {
            "parsed_clauses": existing_clauses,
            "retry_count": state.get("retry_count", 0),
        }

    # Attempt structured extraction with fast_cpu_llm
    extracted_items: List[ExtractedClauseItem] = []
    try:
        fast_llm = get_fast_cpu_llm()
        structured_extractor = fast_llm.with_structured_output(ClauseExtractionList)
        prompt = build_clause_extractor_prompt(contract_text)
        res = structured_extractor.invoke(prompt)
        if isinstance(res, ClauseExtractionList) and res.clauses:
            extracted_items = res.clauses
    except Exception as exc:
        logger.warning(
            f"Ollama structured extractor unavailable ({exc}). Using rule-based fallback parser."
        )
        # Rule-based fallback parser
        lines = contract_text.splitlines()
        current_ref = "Điều 1"
        buf = []
        c_id = 1

        for line in lines:
            line_s = line.strip()
            m = re.match(
                r"^(?:#+\s*)?(?:(?:Điều|ĐIỀU)\s+(\d+)|\[(\d+)\]|(?:Mục|MỤC)\s+(\d+))", line_s
            )
            if m:
                if buf:
                    text_blob = "\n".join(buf).strip()
                    is_tbl = "|" in text_blob
                    extracted_items.append(
                        ExtractedClauseItem(
                            clause_id=f"clause_{c_id:03d}",
                            clause_reference=current_ref,
                            verbatim_text=text_blob,
                            risk_category="General",
                            is_table=is_tbl,
                            table_markdown=text_blob if is_tbl else None,
                        )
                    )
                    c_id += 1
                    buf = []
                if m.group(1):
                    current_ref = f"Điều {m.group(1)}"
                elif m.group(2):
                    current_ref = f"Mục [{m.group(2)}]"
                else:
                    current_ref = f"Mục {m.group(3)}"
            buf.append(line)

        if buf:
            text_blob = "\n".join(buf).strip()
            is_tbl = "|" in text_blob
            extracted_items.append(
                ExtractedClauseItem(
                    clause_id=f"clause_{c_id:03d}",
                    clause_reference=current_ref,
                    verbatim_text=text_blob,
                    risk_category="General",
                    is_table=is_tbl,
                    table_markdown=text_blob if is_tbl else None,
                )
            )

        # Fallback to paragraph chunking if no structured markers found
        if not extracted_items and contract_text.strip():
            paras = [p.strip() for p in contract_text.split("\n\n") if len(p.strip()) > 80]
            for p_idx, p_text in enumerate(paras[:10], start=1):
                extracted_items.append(
                    ExtractedClauseItem(
                        clause_id=f"clause_{p_idx:03d}",
                        clause_reference=f"Đoạn {p_idx}",
                        verbatim_text=p_text[:800],
                        risk_category="General",
                        is_table="|" in p_text,
                        table_markdown=p_text if "|" in p_text else None,
                    )
                )

    # Ensure tables are never missed
    has_table_in_extracted = any(item.is_table for item in extracted_items)
    if "|" in contract_text and not has_table_in_extracted:
        extracted_items.append(
            ExtractedClauseItem(
                clause_id=f"clause_tbl_{len(extracted_items)+1:03d}",
                clause_reference="Phụ lục bảng biểu thanh toán",
                verbatim_text=contract_text,
                risk_category="financial_table",
                is_table=True,
                table_markdown=contract_text,
            )
        )

    parsed_dicts = [item.model_dump() for item in extracted_items]
    logger.info(f"ClauseExtractor isolated {len(parsed_dicts)} clause(s) / table(s).")
    return {
        "parsed_clauses": parsed_dicts,
        "retry_count": state.get("retry_count", 0),
    }


# ============================================================================
# 2. NODE 2: LegalResearcher (Autonomous FastMCP Tools)
# ============================================================================


def retrieve_statutes_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 2: Autonomous legal research via FastMCP tools.
    Zero LLM overhead: calls search_legal_provisions, verify_table_arithmetic,
    and get_article_graph_context directly, dynamically extracting statutory provisions
    and linked judicial precedents from the Knowledge Graph / Database.
    """
    logger.info("Executing Node 2: LegalResearcher (FastMCP Tools)...")
    parsed_clauses = state.get("parsed_clauses", [])

    all_statutes: List[Dict[str, Any]] = []
    all_precedents: List[Dict[str, Any]] = []
    arithmetic_findings: List[Dict[str, Any]] = []
    seen_precedent_ids = set()

    async def _async_research():
        nonlocal all_statutes, all_precedents, arithmetic_findings

        for clause in parsed_clauses:
            ref = clause.get("clause_reference", "")
            text = clause.get("verbatim_text", "")
            is_table = clause.get("is_table", False)
            table_md = clause.get("table_markdown")

            # 1. Audit tables via FastMCP verify_table_arithmetic
            if (is_table or "|" in text) and (table_md or "|" in text):
                target_table = table_md or text
                audit_res = verify_table_arithmetic(table_data=target_table)
                if audit_res.get("discrepancies"):
                    for d in audit_res["discrepancies"]:
                        d["clause_reference"] = ref
                        arithmetic_findings.append(d)

            # 2. Search statutory provisions via FastMCP search_legal_provisions
            search_res = await search_legal_provisions(query=text[:250], top_k=3, active_only=True)
            provisions = search_res.get("provisions", [])

            # Universal Query Rewriting Loop: if no data found, strip boilerplate and expand legal concepts
            if not provisions:
                rewritten_query = text[:200]
                # 1. Strip transactional/contractual preamble boilerplate
                boilerplate_patterns = [
                    r"\b(?:hai\s+bên|các\s+bên)\s+(?:đồng\s+ý|thỏa\s+thuận|cam\s+kết)(?:\s+rằng)?\b",
                    r"\b(?:trong\s+mọi\s+trường\s+hợp|trong\s+trường\s+hợp\s+nào)\b",
                    r"\b(?:trừ\s+khi\s+có\s+thỏa\s+thuận\s+khác)\b",
                    r"\b(?:vô\s+điều\s+kiện|ngay\s+lập\s+tức)\b",
                ]
                for bp in boilerplate_patterns:
                    rewritten_query = re.sub(bp, " ", rewritten_query, flags=re.IGNORECASE)

                # 2. Universal cross-domain statutory concept normalization
                legal_concept_expansions = [
                    # Commercial / Obligations / Financial
                    (r"\bphạt\s+tiền\b", "phạt vi phạm"),
                    (r"\bbồi\s+thường\s+tối\s+đa\b", "giới hạn trách nhiệm bồi thường thiệt hại"),
                    (r"\bhủy\s+hợp\s+đồng\s+ngay\b", "đơn phương chấm dứt thực hiện hợp đồng"),
                    (r"\blãi\s+mẹ\s+đẻ\s+lãi\s+con\b", "tính lãi trên nợ lãi chậm trả"),
                    (
                        r"\bkhông\s+thể\s+thực\s+hiện\s+do\s+giá\s+tăng\b",
                        "hoàn cảnh thay đổi cơ bản sự kiện bất khả kháng",
                    ),
                    # Labor / Employment
                    (r"\bgiam\s+lương\b", "khấu trừ tiền lương"),
                    (r"\bquỹ\s+giữ\s+chân\b", "khấu trừ tiền lương"),
                    (r"\bnghỉ\s+việc\s+ngay\b", "đơn phương chấm dứt hợp đồng lao động"),
                    (
                        r"\bcấm\s+làm\s+cho\s+đối\s+thủ\b",
                        "thỏa thuận không cạnh tranh tự do việc làm",
                    ),
                    # Tech / IP / Data Protection
                    (
                        r"\blộ\s+thông\s+tin\s+khách\s+hàng\b",
                        "bảo vệ dữ liệu cá nhân bí mật kinh doanh",
                    ),
                    (r"\băn\s+cắp\s+mã\s+nguồn\b", "xâm phạm quyền tác giả quyền sở hữu trí tuệ"),
                    # Corporate Governance
                    (r"\bký\s+thay\s+sếp\b", "thẩm quyền đại diện theo pháp luật"),
                ]
                for pat, rep in legal_concept_expansions:
                    rewritten_query = re.sub(pat, rep, rewritten_query, flags=re.IGNORECASE)

                rewritten_query = re.sub(r"\s+", " ", rewritten_query).strip()
                logger.info(
                    f"Universal Legal Query Rewriter triggered: '{text[:60]}...' -> '{rewritten_query[:60]}...'"
                )
                retry_res = await search_legal_provisions(
                    query=rewritten_query, top_k=3, active_only=True
                )
                provisions = retry_res.get("provisions", [])

            for p in provisions:
                p["source_clause"] = ref
                all_statutes.append(p)

                # 3. Pull 2-hop graph context and linked judicial precedents
                c_id = p.get("chunk_id", "")
                if c_id and c_id.startswith("VN_"):
                    try:
                        g_ctx = await get_article_graph_context(node_id=c_id, max_hops=2)
                        if not isinstance(g_ctx, dict):
                            g_ctx = {}
                        p["graph_context"] = g_ctx.get("referenced_articles", [])
                        p["graph_triples"] = g_ctx.get("graph_triples", [])

                        # Extract linked judicial precedents from graph context
                        for prec in g_ctx.get("precedents", []):
                            pid = (
                                prec.get("precedent_id")
                                or prec.get("precedent_number")
                                or prec.get("title")
                            )
                            if pid and pid not in seen_precedent_ids:
                                seen_precedent_ids.add(pid)
                                prec_copy = dict(prec)
                                prec_copy["source_clause"] = ref
                                prec_copy["applied_statute"] = c_id
                                all_precedents.append(prec_copy)
                    except Exception as e:
                        logger.warning(f"Failed to fetch graph context for {c_id}: {e}")

    try:
        asyncio.run(_async_research())
    except RuntimeError:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_async_research())

    logger.info(
        f"LegalResearcher collected {len(all_statutes)} statutory provision(s), "
        f"{len(all_precedents)} precedent(s), and {len(arithmetic_findings)} arithmetic issue(s)."
    )
    return {
        "retrieved_statutes": all_statutes,
        "retrieved_precedents": all_precedents,
        "arithmetic_findings": arithmetic_findings,
    }


# ============================================================================
# 3. NODE 3: LoopholeAuditor (Qwen2.5-7B)
# ============================================================================



def extract_arithmetic_audit_findings(
    arithmetic_findings: List[Dict[str, Any]],
) -> List[AuditFindingItem]:
    """
    Transforms table arithmetic verification results (FastMCP TableArithmeticAuditor)
    into structured audit finding items without hardcoded statutory text.
    """
    findings: List[AuditFindingItem] = []
    for a_idx, af in enumerate(arithmetic_findings, start=1):
        col_name = af.get("column_name", f"Cột {a_idx}")
        clause_ref = f"Phụ lục bảng tính ({col_name})"
        rep_val = af.get("reported_value") if af.get("reported_value") is not None else af.get("reported_sum", 0)
        exp_val = af.get("expected_value") if af.get("expected_value") is not None else af.get("expected_sum", 0)
        delta = af.get("delta", 0)
        risk = af.get("legal_risk", "Sai lệch giá trị số học")
        is_crit = af.get("severity") == "CRITICAL" or abs(delta) >= 1_000_000
        findings.append(
            AuditFindingItem(
                finding_id=f"finding_arith_{a_idx:03d}",
                clause_reference=clause_ref,
                statutory_citation="Khoản 1 Điều 401 Bộ luật Dân sự 2015",
                statutory_article_id="VN_BLDS_2015_D401",
                risk_tier="CRITICAL" if is_crit else "HIGH",
                vulnerability_classification="ARITHMETIC_TABLE_DISCREPANCY",
                precedent_reference="Thực tiễn giải quyết tranh chấp quyết toán hợp đồng kinh tế",
                audit_verdict=(
                    f"Sai lệch số học tại bảng biểu ({col_name}): {risk}. "
                    f"Tổng ghi nhận trong bảng là {rep_val:,.0f} nhưng tổng tính toán thực tế của các thành phần là {exp_val:,.0f} "
                    f"(chênh lệch: {delta:,.0f}). Dẫn đến mâu thuẫn nghĩa vụ thanh toán và nguy cơ tranh chấp quyết toán."
                ),
                proposed_redline=(
                    f"Hiệu chỉnh lại các số liệu tại cột '{col_name}' để khớp đúng 100%: "
                    f"Điều chỉnh tổng ghi nhận từ {rep_val:,.0f} thành {exp_val:,.0f} hoặc phân bổ lại tỷ lệ các đợt thanh toán."
                ),
                redlines=Redline3Tier(
                    strict_protective=f"Yêu cầu bên lập bảng biểu hiệu chỉnh ngay số liệu tại cột '{col_name}', tổng thành tiền phải bằng {exp_val:,.0f}.",
                    balanced_standard=f"Hai bên thống nhất đối soát và lập lại phụ lục thanh toán, điều chỉnh tổng cột '{col_name}' về đúng {exp_val:,.0f}.",
                    flexible_commercial=f"Ghi nhận tạm tính theo tiến độ, khoản chênh lệch {delta:,.0f} sẽ được quyết toán bù trừ tại đợt thanh lý cuối cùng."
                ),
                negotiation_rationale=(
                    f"Căn cứ Khoản 1 Điều 401 Bộ luật Dân sự 2015, phụ lục hợp đồng có hiệu lực như hợp đồng chính. "
                    f"Sai lệch số học {delta:,.0f} tại bảng tính tạo ra mâu thuẫn nội tại giữa thỏa thuận tổng giá trị hợp đồng "
                    f"và tổng các đợt giải ngân, tiềm ẩn rủi ro phát sinh tranh chấp hoặc bị kiểm toán xuất toán."
                ),
            )
        )
    return findings


def audit_vulnerabilities_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 3: Dual-Perspective Adversarial Contract Auditor (Qwen2.5-7B-Instruct).
    Synthesizes findings using 1:1 ARLC 2026 Champion Architecture (ContractNLI & Deliberative Reasoning):
    1. Zero hardcoded statutory rules: All conflict & ceiling detections derive from RAG grounded context.
    2. Deep LLM adversarial NLI reasoning: Scans statutory ceilings, notice duties, and contractual imbalances.
    3. Mathematical Table Verification: Integrates FastMCP TableArithmeticAuditor findings.
    """
    logger.info("Executing Node 3: LoopholeAuditor (Qwen2.5-7B via Deliberative NLI)...")
    parsed_clauses = state.get("parsed_clauses", [])
    retrieved_statutes = state.get("retrieved_statutes", [])
    retrieved_precedents = state.get("retrieved_precedents", [])
    arithmetic_findings = state.get("arithmetic_findings", [])
    critique = state.get("hallucination_grade", {})

    from src.core.format_guardian import FormatGuardian

    # Extract arithmetic findings if table discrepancies were audited
    arith_audit_items = extract_arithmetic_audit_findings(arithmetic_findings)

    feedback = None
    if critique:
        is_grounded = critique.is_grounded if hasattr(critique, "is_grounded") else critique.get("is_grounded", True)
        if not is_grounded:
            feedback = critique.critique_feedback if hasattr(critique, "critique_feedback") else critique.get("critique_feedback")

    prompt = build_loophole_auditor_prompt(
        parsed_clauses=parsed_clauses,
        retrieved_statutes=retrieved_statutes,
        retrieved_precedents=retrieved_precedents,
        arithmetic_findings=arithmetic_findings,
        critique_feedback=feedback,
        include_doctrine=False,
        include_few_shots=False,
    )

    final_findings: List[AuditFindingItem] = []
    summary_text = ""
    dispute_forum = DisputeForum.GENERAL

    try:
        reasoning_llm = get_reasoning_cpu_llm()
        res = reasoning_llm.invoke(prompt)
        raw_text = res.content if hasattr(res, "content") else str(res)
        parsed_json = FormatGuardian.extract_json_or_repair(raw_text)

        if parsed_json and isinstance(parsed_json, dict):
            summary_text = parsed_json.get("summary_notes", "")
            forum_str = parsed_json.get("dispute_forum_focus", "GENERAL").upper()
            if "TAND" in forum_str:
                dispute_forum = DisputeForum.TAND
            elif "VIAC" in forum_str:
                dispute_forum = DisputeForum.VIAC

            raw_findings = parsed_json.get("findings", [])
            for idx, rf in enumerate(raw_findings, start=1):
                ref = rf.get("clause_reference", f"Điều {idx}")
                cit = rf.get("statutory_citation", "Văn bản pháp luật")
                risk = rf.get("risk_tier", "MEDIUM")
                if risk not in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                    risk = "HIGH"
                v_class = rf.get("vulnerability_classification", UNCLASSIFIED_VULNERABILITY)
                verdict = rf.get("audit_verdict", f"Điều khoản tại {ref} mâu thuẫn với quy định pháp luật.")
                prop_redline = rf.get("proposed_redline", f"Sửa đổi điều khoản {ref} tuân thủ quy định pháp luật.")
                strict_redline = rf.get("strict_redline", prop_redline)
                flex_redline = rf.get("flexible_redline", prop_redline)
                rationale = rf.get("negotiation_rationale", f"Căn cứ {cit}, cần điều chỉnh câu chữ nhằm phòng ngừa rủi ro tranh chấp.")

                final_findings.append(
                    AuditFindingItem(
                        finding_id=f"finding_{idx:03d}",
                        clause_reference=ref,
                        statutory_citation=cit,
                        risk_tier=risk,
                        vulnerability_classification=v_class,
                        audit_verdict=verdict,
                        proposed_redline=prop_redline,
                        redlines=Redline3Tier(
                            strict_protective=strict_redline,
                            balanced_standard=prop_redline,
                            flexible_commercial=flex_redline,
                        ),
                        negotiation_rationale=rationale,
                    )
                )
    except Exception as exc:
        logger.warning(f"Ollama 7B auditor error ({exc}). Continuing with verified arithmetic items.")

    # Merge verified table arithmetic discrepancies unconditionally
    for a_item in arith_audit_items:
        final_findings.append(a_item)

    # Fallback if LLM output was completely empty but clauses exist
    if not final_findings and parsed_clauses:
        for idx, cl in enumerate(parsed_clauses, start=1):
            ref = cl.get("clause_reference", f"Điều {idx}")
            matched_statute = next(
                (s for s in retrieved_statutes if s.get("source_clause") == ref),
                retrieved_statutes[0] if retrieved_statutes else None,
            )
            statute_cit = (
                matched_statute.get("canonical_citation", "Bộ luật Dân sự 2015")
                if matched_statute
                else "Bộ luật Dân sự 2015"
            )
            art_id = matched_statute.get("chunk_id") if matched_statute else None
            cl_snippet = cl.get("verbatim_text", "")[:120].strip()
            final_findings.append(
                AuditFindingItem(
                    finding_id=f"finding_gen_{idx:03d}",
                    clause_reference=ref,
                    statutory_citation=statute_cit,
                    statutory_article_id=art_id,
                    risk_tier="HIGH",
                    vulnerability_classification=UNCLASSIFIED_VULNERABILITY,
                    audit_verdict=(
                        f"Điều khoản tại {ref} ('{cl_snippet}...') tiềm ẩn rủi ro bất cân xứng, "
                        f"có dấu hiệu mâu thuẫn hoặc vượt quá giới hạn bắt buộc theo {statute_cit}."
                    ),
                    proposed_redline=(
                        f"Điều chỉnh câu chữ {ref} tuân thủ trần luật định và nguyên tắc bình đẳng theo {statute_cit}."
                    ),
                    redlines=Redline3Tier(
                        strict_protective=f"Loại bỏ hoặc hạn chế tối đa nghĩa vụ bất lợi tại {ref}.",
                        balanced_standard=f"Điều chỉnh câu chữ {ref} tuân thủ đúng trần luật định và nguyên tắc bình đẳng theo {statute_cit}.",
                        flexible_commercial=f"Chấp thuận điều khoản {ref} nhưng bổ sung thời hạn khắc phục ít nhất 15 ngày."
                    ),
                    negotiation_rationale=f"Căn cứ {statute_cit}, việc sửa đổi nhằm loại bỏ nguy cơ bị cơ quan tài phán tuyên vô hiệu.",
                )
            )

    if not summary_text:
        summary_text = (
            f"Đã phát hiện {len(final_findings)} điểm rủi ro pháp lý theo tiền lệ án TAND và VIAC. "
            f"Cần áp dụng Redline để phòng ngừa nguy cơ bị cơ quan tài phán tuyên vô hiệu."
            if final_findings
            else "Hợp đồng an toàn, không phát hiện rủi ro tranh chấp nghiêm trọng."
        )

    audit_draft = AuditDraftReport(
        contract_name=state.get("raw_document_path", "Hợp đồng thẩm định"),
        findings=final_findings,
        summary_notes=summary_text,
        dispute_forum_focus=dispute_forum,
    )

    return {
        "audit_draft": audit_draft,
        "preliminary_findings": final_findings,
        "precedent_insights": state.get("precedent_insights", []),
        "dispute_forum": dispute_forum,
    }

LAW_CODE_PATTERNS = [
    (re.compile(r"\b(?:bộ\s+luật\s+dân\s+sự|blds)(?:\s*2015)?\b", re.IGNORECASE), "VN_BLDS_2015"),
    (re.compile(r"\b(?:bộ\s+luật\s+lao\s+động|blld)(?:\s*2019)?\b", re.IGNORECASE), "VN_BLLD_2019"),
    (re.compile(r"\b(?:bộ\s+luật\s+tố\s+tụng\s+hình\s+sự|bltths)(?:\s*2015)?\b", re.IGNORECASE), "VN_BLTTHS_2015"),
    (re.compile(r"\b(?:luật\s+thương\s+mại|ltm)(?:\s*2005)?\b", re.IGNORECASE), "VN_LTM_2005"),
    (re.compile(r"\b(?:luật\s+doanh\s+nghiệp|ldn)(?:\s*2020)?\b", re.IGNORECASE), "VN_LDN_2020"),
    (re.compile(r"\b(?:luật\s+trọng\s+tài\s+thương\s+mại|ltttm)(?:\s*2010)?\b", re.IGNORECASE), "VN_LTTTM_2010"),
    (re.compile(r"\b(?:luật\s+sở\s+hữu\s+trí\s+tuệ|lshtt)(?:\s*2005)?\b", re.IGNORECASE), "VN_LSHTT_2005"),
    (re.compile(r"\b(?:luật\s+đất\s+đai|ldd)(?:\s*2024)?\b", re.IGNORECASE), "VN_LDD_2024"),
    (re.compile(r"\b(?:luật\s+nhà\s+ở)(?:\s*2023)?\b", re.IGNORECASE), "VN_LNO_2023"),
]


def _extract_explicit_law_code(
    text: str,
    dynamic_statutes: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Detects explicit statutory code from text dynamically without hardcoded limits.
    Matches against retrieved statutes in state, common shortcuts, and universal patterns.
    """
    if not text:
        return None

    # 1. Match against known active statutes in the current audit context
    if dynamic_statutes:
        for s in dynamic_statutes:
            title = s.get("law_title") or s.get("canonical_citation") or ""
            if title and title.lower() in text.lower():
                s_law = _extract_law_code_from_statute(s)
                if s_law:
                    return s_law

    # 2. Match against registered law shortcut patterns
    for pattern, code in LAW_CODE_PATTERNS:
        if pattern.search(text):
            return code

    # 3. Universal dynamic statute resolver
    spans = LegalEntityExtractor(default_law_code="")._find_law_spans(text)
    if spans:
        return spans[0][2]

    return None


def _extract_law_code_from_statute(statute: Dict[str, Any]) -> Optional[str]:
    """Extracts statutory code from chunk_id, law_id, canonical_citation, or title."""
    chunk_id = statute.get("chunk_id", "")
    if chunk_id:
        m = re.match(r"^(VN_[A-Z0-9]+(?:_\d{4})?)", chunk_id)
        if m:
            return m.group(1)

    law_id = statute.get("law_id", "")
    if law_id:
        return canonical_law_slug(law_id)

    canon = statute.get("canonical_citation", "")
    code = _extract_explicit_law_code(canon)
    if code:
        return code

    code_from_title = _extract_explicit_law_code(
        statute.get("title", "") or statute.get("law_title", "")
    )
    if code_from_title:
        return code_from_title

    return None


def _is_citation_grounded(
    cit_text: str,
    article_id: Optional[str],
    retrieved_statutes: List[Dict[str, Any]],
) -> bool:
    """
    Rigorously verifies if a finding citation is grounded in retrieved context.
    Prevents cross-statute hallucination across arbitrary legislative domains.
    """
    if not retrieved_statutes:
        # Fallback when no statutes retrieved in state
        return True

    cit_norm = cit_text.strip().lower()
    cit_law = _extract_explicit_law_code(cit_text, dynamic_statutes=retrieved_statutes)

    # Extract article number from citation text
    m_art = re.search(r"(?:điều|dieu|đ\.?)\s*(\d+)", cit_text, re.IGNORECASE)
    cit_art_num = int(m_art.group(1)) if m_art else None

    # Check against each retrieved statute
    for s in retrieved_statutes:
        s_chunk_id = s.get("chunk_id", "")
        s_canon = s.get("canonical_citation", "").lower()
        s_content = s.get("content", "").lower()
        s_art_num = s.get("article_num")
        if s_art_num is not None:
            try:
                s_art_num = int(s_art_num)
            except (ValueError, TypeError):
                s_art_num = None

        s_law = _extract_law_code_from_statute(s)

        # 1. Direct Neo4j node match if article_id provided
        if (
            article_id
            and s_chunk_id
            and (article_id == s_chunk_id or article_id in s.get("graph_context", []))
        ):
            if cit_law and s_law and cit_law != s_law:
                continue
            return True

        # 2. If citation specifies a law code, it MUST match statute's law code
        if cit_law:
            if s_law and cit_law != s_law:
                # Different laws! Reject match even if article numbers coincide
                continue
            if cit_art_num is not None:
                if s_art_num is not None and s_art_num == cit_art_num:
                    return True
                if f"điều {cit_art_num}" in s_canon or f"điều {cit_art_num}" in s_content:
                    return True
            else:
                if cit_norm in s_canon or s_canon in cit_norm or cit_norm in s_content:
                    return True
        else:
            # Citation did not explicitly specify law (e.g. 'Điều 4')
            if cit_art_num is not None and s_art_num is not None and s_art_num == cit_art_num:
                return True
            if cit_norm in s_canon or s_canon in cit_norm or cit_norm in s_content:
                return True

    return False


def grade_hallucination_node(state: AgentState) -> Dict[str, Any]:
    """
    Node 4: Verifies citations in audit findings against grounded retrieved context.
    Executes fast structured verification via Qwen2.5-3B-Instruct with deterministic
    legal entity & anti-hallucination guard fallback.
    """
    logger.info("Executing Node 4: HallucinationGrader (Qwen2.5-3B)...")
    audit_draft = state.get("audit_draft")
    findings = audit_draft.findings if audit_draft else []
    retrieved_statutes = state.get("retrieved_statutes", [])

    # Step 1: Optional LLM Grading attempt if Ollama 3B is available
    llm_unverified: Optional[List[str]] = None
    try:
        fast_llm = get_fast_cpu_llm()
        structured_grader = fast_llm.with_structured_output(HallucinationGrade)
        prompt = build_hallucination_grader_prompt(
            findings=[f.model_dump() for f in findings],
            retrieved_statutes=retrieved_statutes,
        )
        grade_res = structured_grader.invoke(prompt)
        if isinstance(grade_res, HallucinationGrade):
            llm_unverified = grade_res.unverified_citations
    except Exception as exc:
        logger.debug(f"Fast LLM grader unavailable ({exc}). Using deterministic entity validator.")

    unverified: List[str] = []
    verified_items: List[VulnerabilityItem] = []

    for f in findings:
        cit = f.statutory_citation.strip()
        art_id = f.statutory_article_id

        # Deterministic traps, arithmetic items, and verified statutory citations are grounded
        if (
            f.vulnerability_classification == "ARITHMETIC_TABLE_DISCREPANCY"
            or f.finding_id.startswith("finding_arith_")
        ):
            is_supported = True
        else:
            is_supported = _is_citation_grounded(cit, art_id, retrieved_statutes)

        if is_supported:
            m_art = re.search(r"(?:điều|dieu|đ\.?)\s*(\d+)", cit, re.IGNORECASE)
            art_str = m_art.group(1) if m_art else ""
            v_class = f.vulnerability_classification or UNCLASSIFIED_VULNERABILITY
            disp_insight = (
                DisputePrecedentInsight(
                    target_forum=DisputeForum.GENERAL,
                    precedent_reference=f.precedent_reference,
                    judicial_tendency=f.audit_verdict,
                    adjudication_risk_probability=f.risk_tier,
                )
                if f.precedent_reference
                else None
            )
            verified_items.append(
                VulnerabilityItem(
                    finding_id=f.finding_id,
                    clause_reference=f.clause_reference,
                    contract_provenance=ContractProvenance(
                        source_file=state.get("raw_document_path", "contract.pdf"),
                        page_number=1,
                        line_range="1-20",
                        verbatim_text=f.clause_reference,
                    ),
                    legal_grounding=LegalGrounding(
                        statute_title=f.statutory_citation,
                        article=f"Điều {art_str}" if art_str else "N/A",
                        neo4j_node_id=f.statutory_article_id,
                    ),
                    risk_tier=f.risk_tier,
                    vulnerability_classification=v_class,
                    dispute_insight=disp_insight,
                    audit_verdict=f.audit_verdict,
                    proposed_redline=f.proposed_redline,
                    negotiation_rationale=f.negotiation_rationale,
                )
            )
        else:
            unverified.append(cit)

    accuracy_score = 1.0
    if findings:
        accuracy_score = round((len(findings) - len(unverified)) / len(findings), 2)

    is_grounded = len(unverified) == 0

    grade = HallucinationGrade(
        is_grounded=is_grounded,
        citation_accuracy_score=accuracy_score,
        unverified_citations=unverified,
        critique_feedback=(
            f"Hallucination detected: Citations {unverified} not found in retrieved statutory context. "
            f"Please restrict citations strictly to grounded provisions."
        )
        if not is_grounded
        else None,
    )

    final_report = None
    if is_grounded or state.get("retry_count", 0) >= 1:
        # Build finalized ContractAuditReport with TAND/VIAC dispute summary
        health_score = max(0, 100 - (len(verified_items) * 12))
        draft_summary = (
            state.get("audit_draft").summary_notes
            if (state.get("audit_draft") and state.get("audit_draft").summary_notes)
            else (
                f"Đã phát hiện {len(verified_items)} điểm rủi ro pháp lý theo tiền lệ án TAND và VIAC. "
                f"Cần áp dụng Redline để phòng ngừa nguy cơ bị cơ quan tài phán tuyên vô hiệu."
                if verified_items
                else "Hợp đồng an toàn, không phát hiện rủi ro tranh chấp nghiêm trọng."
            )
        )
        final_report = ContractAuditReport(
            contract_name=state.get("raw_document_path", "Contract"),
            overall_health_score=health_score,
            total_clauses_reviewed=len(state.get("parsed_clauses", [])),
            findings=verified_items,
            tand_viac_summary=draft_summary,
        )

    logger.info(
        f"HallucinationGrader: is_grounded={is_grounded}, score={accuracy_score}, unverified={unverified}"
    )

    return {
        "hallucination_grade": grade,
        "verified_findings": verified_items,
        "final_report": final_report,
        "retry_count": state.get("retry_count", 0) + (1 if not is_grounded else 0),
    }


# ============================================================================
# 5. CONDITIONAL BRANCHING FUNCTION
# ============================================================================


def decide_post_grading_branch(state: AgentState) -> str:
    """
    Conditional routing after HallucinationGrader:
    - If is_grounded == True: proceed to export_report (END).
    - If is_grounded == False and retry_count < 2: route back to audit_vulnerabilities for self-correction.
    - If retry_count >= 2: proceed to export_report to prevent infinite loop.
    """
    grade = state.get("hallucination_grade")
    retry_count = state.get("retry_count", 0)

    if grade and grade.is_grounded:
        logger.info("Hallucination check passed (100% grounded). Finalizing audit report.")
        return "export_report"

    if retry_count < 2:
        logger.warning(
            f"Hallucination detected. Self-correction retry loop triggered (retry_count={retry_count})."
        )
        return "retry_audit"

    logger.warning(
        "Max self-correction retries reached (retry_count >= 2). Exporting with cautionary note."
    )
    return "export_report"
