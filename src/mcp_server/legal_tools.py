"""
FastMCP Server: Exposes legal research tools (search_law, get_related_articles) via Model Context Protocol
"""

from typing import Any, Dict, List
from loguru import logger
from fastmcp import FastMCP

mcp = FastMCP("LexiGuard Legal Tools")

@mcp.tool()
def search_law(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search governing Vietnamese statutory articles (Civil Code, Labor Code, Commercial Law) using Hybrid Search.
    """
    logger.info(f"MCP Tool 'search_law' invoked with query: {query}")
    return []

@mcp.tool()
def get_related_articles(article_id: str, hops: int = 2) -> List[Dict[str, Any]]:
    """
    Retrieve related articles, guiding decrees, and circulars connected to a statutory article in Neo4j Knowledge Graph.
    """
    logger.info(f"MCP Tool 'get_related_articles' invoked for article: {article_id}")
    return []

if __name__ == "__main__":
    mcp.run()
