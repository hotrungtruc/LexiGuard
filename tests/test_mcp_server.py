"""
Unit & Integration Tests: FastMCP Legal Tools & Autonomous Auditing
==================================================================
Tests the 3 core MCP tools:
1. `search_legal_provisions` (Self-Querying & Payload Filtering)
2. `verify_table_arithmetic` (Row product, column sum, and format checks)
3. `get_article_graph_context` (Neo4j Graph Context & Citation Resolution)
"""

import pytest

from src.mcp_server.legal_tools import (
    SelfQueryEngine,
    get_article_graph_context,
    mcp_server,
    search_legal_provisions,
    verify_table_arithmetic,
)

# ============================================================================
# 1. TEST SELF-QUERYING METADATA ENGINE
# ============================================================================


def test_self_query_engine_law_detection():
    # Detect Civil Code 2015
    q1 = "Quy định về hợp đồng vô hiệu theo Bộ luật Dân sự 2015"
    f1 = SelfQueryEngine.extract_filters(q1)
    assert f1["law_id"] == "bo-luat-dan-su-2015"

    # Detect Labour Code 2019
    q2 = "Thời hạn báo trước khi đơn phương chấm dứt theo BLLĐ"
    f2 = SelfQueryEngine.extract_filters(q2)
    assert f2["law_id"] == "bo-luat-lao-dong-2019"


def test_self_query_engine_article_and_clause_detection():
    q = "Theo quy định tại Khoản 2 Điều 4 Bộ luật Dân sự"
    f = SelfQueryEngine.extract_filters(q)
    assert f["law_id"] == "bo-luat-dan-su-2015"
    assert f["article_num"] == 4
    assert f["clause_num"] == 2


def test_self_query_engine_explicit_override():
    q = "Điều 4 Bộ luật Dân sự"
    # Caller explicitly forces Labour Code
    f = SelfQueryEngine.extract_filters(
        q, explicit_law_id="bo-luat-lao-dong-2019", explicit_article_num=115
    )
    assert f["law_id"] == "bo-luat-lao-dong-2019"
    assert f["article_num"] == 115


# ============================================================================
# 2. TEST MCP TOOL: search_legal_provisions
# ============================================================================


@pytest.mark.asyncio
async def test_search_legal_provisions_self_querying():
    res = await search_legal_provisions(
        query="Khoản 2 Điều 4 Bộ luật Dân sự",
        top_k=3,
        active_only=True,
    )
    assert res["status"] == "success"
    assert res["self_querying"]["detected_law_id"] == "bo-luat-dan-su-2015"
    assert res["self_querying"]["detected_article_num"] == 4
    assert res["self_querying"]["detected_clause_num"] == 2
    assert res["total_matches"] > 0
    # First provision should be Article 4
    p0 = res["provisions"][0]
    assert p0["article_num"] == 4
    assert p0["law_id"] == "bo-luat-dan-su-2015"


@pytest.mark.asyncio
async def test_search_legal_provisions_with_explicit_filter():
    res = await search_legal_provisions(
        query="lương làm thêm giờ",
        law_id="bo-luat-lao-dong-2019",
        top_k=5,
    )
    assert res["status"] == "success"
    assert res["self_querying"]["detected_law_id"] == "bo-luat-lao-dong-2019"
    assert all(p["law_id"] == "bo-luat-lao-dong-2019" for p in res["provisions"])


# ============================================================================
# 3. TEST MCP TOOL: verify_table_arithmetic
# ============================================================================


def test_verify_table_arithmetic_column_sum_mismatch():
    """Test detecting vertical addition discrepancy in total row."""
    table_md = """
    | STT | Hạng mục công việc | Thành tiền (VNĐ) |
    | :--- | :--- | :--- |
    | 1 | Rà soát hợp đồng kinh tế | 10.000.000 |
    | 2 | Soạn thảo biên bản thanh lý | 15.000.000 |
    | Tổng cộng | | 30.000.000 |
    """
    res = verify_table_arithmetic(table_md)
    assert res["status"] == "success"
    assert res["is_valid"] is False
    assert res["error_count"] >= 1
    disc = res["discrepancies"][0]
    assert disc["type"] == "COLUMN_SUM_MISMATCH"
    assert disc["reported_value"] == 30000000.0
    assert disc["expected_value"] == 25000000.0
    assert disc["delta"] == 5000000.0


def test_verify_table_arithmetic_row_product_mismatch():
    """Test detecting quantity * unit price != total subtotal."""
    table_md = """
    | STT | Mặt hàng | Số lượng | Đơn giá | Thành tiền |
    | :---: | :--- | :---: | :---: | :---: |
    | 1 | Dịch vụ tư vấn pháp lý chuyên sâu | 2 | 10.000.000 | 25.000.000 |
    | Tổng cộng | | | | 25.000.000 |
    """
    res = verify_table_arithmetic(table_md)
    assert res["status"] == "success"
    assert res["is_valid"] is False
    row_discs = [d for d in res["discrepancies"] if d["type"] == "ROW_PRODUCT_MISMATCH"]
    assert len(row_discs) >= 1
    assert row_discs[0]["expected_value"] == 20000000.0
    assert row_discs[0]["reported_value"] == 25000000.0
    assert row_discs[0]["delta"] == 5000000.0


def test_verify_table_arithmetic_valid_table():
    """Test table with perfectly aligned arithmetic passing cleanly."""
    table_md = """
    | STT | Nội dung | Số lượng | Đơn giá | Thành tiền |
    | 1 | Thẩm tra pháp lý | 3 | 5.000.000 | 15.000.000 |
    | 2 | Phí công chứng hợp đồng | 1 | 2.000.000 | 2.000.000 |
    | Tổng cộng | | | | 17.000.000 |
    """
    res = verify_table_arithmetic(table_md)
    assert res["status"] == "success"
    assert res["is_valid"] is True
    assert res["error_count"] == 0
    assert len(res["discrepancies"]) == 0


def test_verify_table_arithmetic_json_input():
    """Test table audit using JSON list of row dictionaries."""
    data = [
        {"item": "Dịch vụ A", "so_luong": 3, "don_gia": 1000000, "thanh_tien": 3000000},
        {"item": "Dịch vụ B", "so_luong": 2, "don_gia": 2000000, "thanh_tien": 4000000},
        {"item": "Tổng cộng", "so_luong": None, "don_gia": None, "thanh_tien": 7000000},
    ]
    res = verify_table_arithmetic(data)
    assert res["is_valid"] is True
    assert res["error_count"] == 0


# ============================================================================
# 4. TEST MCP TOOL: get_article_graph_context
# ============================================================================


@pytest.mark.asyncio
async def test_get_article_graph_context_by_node_id():
    res = await get_article_graph_context(
        node_id="VN_BLDS_2015_D004",
        max_hops=2,
    )
    assert res["status"] == "success"
    assert res["target_node_id"] == "VN_BLDS_2015_D004"
    assert "referenced_articles" in res
    assert "graph_triples" in res


@pytest.mark.asyncio
async def test_get_article_graph_context_by_natural_citation():
    # Natural language citation resolving to node ID
    res = await get_article_graph_context(
        node_id="Điều 4 Bộ luật Dân sự 2015",
        max_hops=2,
    )
    assert res["status"] == "success"
    assert res["target_node_id"] == "VN_BLDS_2015_D004"


# ============================================================================
# 5. TEST FASTMCP SERVER REGISTRATION & PROTOCOL INVOCATION
# ============================================================================


@pytest.mark.asyncio
async def test_mcp_server_tool_registration():
    tools = await mcp_server.list_tools()
    tool_names = [t.name for t in tools]
    assert "search_legal_provisions" in tool_names
    assert "verify_table_arithmetic" in tool_names
    assert "get_article_graph_context" in tool_names


@pytest.mark.asyncio
async def test_mcp_server_call_tool_verify_table():
    table_md = """
    | Mặt hàng | Số lượng | Đơn giá | Thành tiền |
    | Sản phẩm 1 | 2 | 50.000 | 120.000 |
    """
    res = await mcp_server.call_tool("verify_table_arithmetic", {"table_data": table_md})
    assert res is not None
    structured = getattr(res, "structured_content", None)
    if isinstance(structured, dict):
        result_data = structured.get("result", structured)
    elif hasattr(res, "content") and res.content:
        import json

        result_data = json.loads(res.content[0].text)
    else:
        result_data = structured
    assert result_data["is_valid"] is False
