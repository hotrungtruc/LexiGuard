"""
Workflow: Khởi tạo và biên dịch (compile) StateGraph của LangGraph
"""

from loguru import logger
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import (
    extract_clauses_node,
    retrieve_statutes_node,
    audit_vulnerabilities_node,
    verify_and_redline_node
)

def create_lexiguard_graph():
    """Builds and compiles the multi-agent legal audit workflow graph."""
    logger.info("Constructing LexiGuard StateGraph...")
    
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("extract_clauses", extract_clauses_node)
    workflow.add_node("retrieve_statutes", retrieve_statutes_node)
    workflow.add_node("audit_vulnerabilities", audit_vulnerabilities_node)
    workflow.add_node("verify_and_redline", verify_and_redline_node)
    
    # Define Edges
    workflow.set_entry_point("extract_clauses")
    workflow.add_edge("extract_clauses", "retrieve_statutes")
    workflow.add_edge("retrieve_statutes", "audit_vulnerabilities")
    workflow.add_edge("audit_vulnerabilities", "verify_and_redline")
    workflow.add_edge("verify_and_redline", END)
    
    app = workflow.compile()
    logger.info("LexiGuard StateGraph compiled successfully.")
    return app

lexiguard_agent = create_lexiguard_graph()
