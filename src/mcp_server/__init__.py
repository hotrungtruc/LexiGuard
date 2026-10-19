"""
Model Context Protocol (MCP) Legal Server Package
=================================================
Exposes autonomous legal verification, self-querying statutory search,
and table arithmetic auditing tools via FastMCP.
"""

from src.mcp_server.legal_tools import (
    mcp_server,
    search_legal_provisions,
    verify_table_arithmetic,
    get_article_graph_context,
    TableArithmeticAuditor,
    SelfQueryEngine,
)

__all__ = [
    "mcp_server",
    "search_legal_provisions",
    "verify_table_arithmetic",
    "get_article_graph_context",
    "TableArithmeticAuditor",
    "SelfQueryEngine",
]
