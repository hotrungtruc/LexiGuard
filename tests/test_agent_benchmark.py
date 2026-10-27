from src.evaluation.agent_benchmark import (
    AgentBenchmarkCase,
    evaluate_agent_predictions,
    evaluate_citation_verification,
)


def test_agent_benchmark_scores_macro_f1_critical_recall_and_dynamic_category():
    cases = [
        AgentBenchmarkCase("critical", "positive", "NEW_LAW_RULE", "CRITICAL", critical=True),
        AgentBenchmarkCase("unsupported", "unsupported", None, None, expected_abstention=True),
    ]
    predictions = {
        "critical": {
            "category": "NEW_LAW_RULE",
            "risk": "CRITICAL",
            "verified_citations": [],
            "unsupported_citations": [],
        },
        "unsupported": {
            "abstain": True,
            "verified_citations": [],
            "unsupported_citations": ["FICTIONAL_ART_999"],
        },
    }
    result = evaluate_agent_predictions(cases, predictions)
    assert result["classification"]["macro_f1"] == 1.0
    assert result["classification"]["critical_recall"] == 1.0
    assert result["classification"]["abstention_accuracy"] == 1.0
    assert result["citation_verification"]["hallucination_rate"] == 1.0


def test_citation_verification_is_separate_from_classification():
    cases = [
        AgentBenchmarkCase(
            "q1",
            "positive",
            "RULE_A",
            "HIGH",
            supported_citations={"LAW_ART_1"},
        )
    ]
    metrics = evaluate_citation_verification(
        cases,
        {
            "q1": {
                "category": "WRONG_CATEGORY",
                "verified_citations": ["LAW_ART_1"],
                "unsupported_citations": ["MADE_UP_ART_9"],
            }
        },
    )
    assert metrics["citation_recall"] == 1.0
    assert metrics["hallucination_rate"] == 1.0
