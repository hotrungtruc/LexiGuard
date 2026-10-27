from src.evaluation.retrieval_benchmark import (
    RetrievalQuery,
    aggregate_results,
    evaluate_graph_paths,
    evaluate_ranked_results,
)


def test_retrieval_metrics_use_rank_and_grades():
    query = RetrievalQuery(
        query_id="Q1",
        query="exact",
        query_type="exact",
        relevant_chunks={"a": 3, "b": 1},
    )
    result = evaluate_ranked_results(
        query, [{"chunk_id": "x"}, {"chunk_id": "a"}, {"chunk_id": "b"}], k=2
    )
    assert result.metrics["recall_at_k"] == 0.5
    assert result.metrics["mrr"] == 0.5
    assert 0 < result.metrics["ndcg_at_k"] < 1


def test_graph_path_recall_and_aggregate():
    query = RetrievalQuery(
        query_id="Q2",
        query="graph",
        query_type="stealth",
        relevant_chunks={"a": 1},
        relevant_graph_paths=(("a", "b"),),
    )
    result = evaluate_ranked_results(query, ["a"])
    assert evaluate_graph_paths(query, [("a", "b")])["graph_path_recall"] == 1.0
    assert aggregate_results([result])["mrr"] == 1.0


def test_dynamic_metadata_and_law_filter_are_reported():
    query = RetrievalQuery(
        query_id="Q3",
        query="dynamic",
        query_type="exact",
        relevant_chunks={"new-rule": 3},
        required_law_id="new-law",
        expected_dynamic_metadata={"category_id": "NEW_LAW_RULE"},
    )
    result = evaluate_ranked_results(query, [{"chunk_id": "new-rule", "law_id": "new-law"}])
    assert len(result.failures) == 1
    assert "dynamic metadata" in result.failures[0]
