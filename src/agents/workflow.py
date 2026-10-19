"""
Workflow: StateGraph Definition with Dual-LLM and Self-Correction Loop
======================================================================
Compiles the LangGraph cyclic state machine:
ClauseExtractor -> LegalResearcher -> LoopholeAuditor -> HallucinationGrader
                                            ^                       |
                                            |--- (Fail / Retry < 2) -|
                                            v (Pass)
                                           END
"""

from loguru import logger
from langgraph.graph import StateGraph, END
from src.agents.state import AgentState
from src.agents.nodes import (
    extract_clauses_node,
    retrieve_statutes_node,
    audit_vulnerabilities_node,
    grade_hallucination_node,
    decide_post_grading_branch,
)


def create_lexiguard_graph():
    """Builds and compiles the multi-agent legal audit workflow graph."""
    logger.info("Constructing LexiGuard Dual-LLM StateGraph with Self-Correction Loop...")

    workflow = StateGraph(AgentState)

    # 1. Register 4 Autonomous Nodes
    workflow.add_node("extract_clauses", extract_clauses_node)
    workflow.add_node("retrieve_statutes", retrieve_statutes_node)
    workflow.add_node("audit_vulnerabilities", audit_vulnerabilities_node)
    workflow.add_node("grade_hallucination", grade_hallucination_node)

    # 2. Define Deterministic Edges
    workflow.set_entry_point("extract_clauses")
    workflow.add_edge("extract_clauses", "retrieve_statutes")
    workflow.add_edge("retrieve_statutes", "audit_vulnerabilities")
    workflow.add_edge("audit_vulnerabilities", "grade_hallucination")

    # 3. Define Self-Reflection Conditional Routing
    workflow.add_conditional_edges(
        "grade_hallucination",
        decide_post_grading_branch,
        {
            "retry_audit": "audit_vulnerabilities",
            "export_report": END,
        },
    )

    app = workflow.compile()
    logger.info("LexiGuard Dual-LLM StateGraph compiled successfully.")
    return app


lexiguard_agent = create_lexiguard_graph()
