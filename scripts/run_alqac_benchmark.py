"""
Script: run_alqac_benchmark.py
==============================
Executes end-to-end legal information retrieval benchmark on ALQAC datasets:
1. Ingests full legal corpus (e.g. ALQAC 2023: 17 statutes, 2,249 articles).
2. Embeds & upserts all articles into Qdrant In-Memory Vector Store.
3. Evaluates 4 pipelines:
   - Method 1: Okapi BM25 Lexical Search (Sparse)
   - Method 2: Qdrant Vector Search (Dense)
   - Method 3: Hybrid Search (BM25 + Dense RRF)
   - Method 4: LexiGuard Entity-Aware Hybrid Search (RRF + Citation Grounding)
4. Computes exact metrics across K in {5, 10, 20}:
   - Hit Rate@K
   - MRR@K
   - NDCG@K
   - Multi-item Recall@K, Precision@K, F2@K
5. Outputs comparative table and saves JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

from src.evaluation.alqac_benchmark import (
    ALQACSeasonLoader,
    CorpusBM25Index,
    LexiGuardFullRetriever,
    QdrantVectorBenchmarkIndex,
    evaluate_method_predictions,
)
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Full ALQAC Benchmark Suite")
    parser.add_argument(
        "--season",
        type=str,
        default="2023",
        choices=["2023", "2022", "2024", "2025"],
        help="ALQAC competition season to evaluate (recommended: 2023 or 2022)",
    )
    parser.add_argument(
        "--disable-reranker",
        action="store_true",
        help="Disable Stage-2 Cross-Encoder neural reranker",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/alqac", help="Root path to alqac datasets directory"
    )
    parser.add_argument(
        "--top-k", type=int, default=20, help="Maximum ranking depth K (default: 20)"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Custom output path for report JSON (default: reports/alqac_benchmark_{season}.json)",
    )
    args = parser.parse_args()

    out_file = args.output_report or f"reports/alqac_benchmark_{args.season}.json"

    print("\n" + "=" * 85)
    print(f"🔥 LEXIGUARD LEGAL BENCHMARK: ALQAC {args.season} (FULL CORPUS & GOLD QUESTIONS)")
    print("=" * 85)

    # 1. Load Corpus & Questions
    loader = ALQACSeasonLoader(args.data_dir)
    articles, questions = loader.load_season(season=args.season)

    cache_dir = Path(args.data_dir) / ".cache_embeddings"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build BM25 Index
    t_bm25_build = time.time()
    bm25_index = CorpusBM25Index(articles)
    logger.info(f"[1/4] BM25 Index built in {time.time() - t_bm25_build:.2f}s.")

    # 3. Build & Upsert into Qdrant In-Memory Vector Store
    t_qdrant_build = time.time()
    qdrant_index = QdrantVectorBenchmarkIndex(
        articles=articles,
        cache_dir=cache_dir,
        season=args.season,
    )
    logger.info(
        f"[2/4] Qdrant In-Memory Vector Store ready in {time.time() - t_qdrant_build:.2f}s."
    )

    # 4. Initialize Cross-Encoder Reranker (Phase 2 SOTA)
    reranker = None
    if not args.disable_reranker:
        try:
            t_reranker_build = time.time()
            reranker = CrossEncoderReranker()
            logger.info(
                f"[3/4] Cross-Encoder Neural Reranker ready in {time.time() - t_reranker_build:.2f}s."
            )
        except Exception as e:
            logger.warning(
                f"Cross-Encoder initialization failed: {e}. Running without Stage-2 reranking."
            )

    lexiguard_full = LexiGuardFullRetriever(
        bm25=bm25_index,
        qdrant=qdrant_index,
        reranker=reranker,
    )
    logger.info("[4/4] LexiGuard Full Hybrid & Reranker Retriever initialized.")

    k_list = (5, 10, 20)

    # 5. Execute Retrievals
    total_pipelines = 5 if reranker is not None else 4
    logger.info(
        f"Executing benchmark over {len(questions)} questions across {total_pipelines} pipelines..."
    )

    # Method 1: BM25 Lexical
    logger.info("--> Querying Pipeline 1: Okapi BM25 Lexical (Compound N-grams)...")
    t0 = time.time()
    bm25_preds = {}
    for q in questions:
        res = bm25_index.search(q.text, top_k=args.top_k)
        bm25_preds[q.question_id] = [art.key for score, art in res]
    t_bm25 = time.time() - t0
    m1_metrics = evaluate_method_predictions(
        "BM25 Lexical", questions, bm25_preds, k_list=k_list, total_time_seconds=t_bm25
    )

    # Method 2: Qdrant Dense Vector
    logger.info("--> Querying Pipeline 2: Qdrant Vector Search (Dense)...")
    t0 = time.time()
    dense_preds = {}
    for q in questions:
        res = qdrant_index.search(q.text, top_k=args.top_k)
        dense_preds[q.question_id] = [art.key for score, art in res]
    t_dense = time.time() - t0
    m2_metrics = evaluate_method_predictions(
        "Qdrant Dense Vector", questions, dense_preds, k_list=k_list, total_time_seconds=t_dense
    )

    # Method 3: Hybrid Search (BM25 + Qdrant via RRF)
    logger.info("--> Querying Pipeline 3: Hybrid Search (BM25 + Dense RRF)...")
    t0 = time.time()
    hybrid_preds = {}
    for q in questions:
        res = lexiguard_full.search_hybrid(q.text, top_k=args.top_k)
        hybrid_preds[q.question_id] = [art.key for score, art in res]
    t_hybrid = time.time() - t0
    m3_metrics = evaluate_method_predictions(
        "Hybrid (BM25 + Dense RRF)",
        questions,
        hybrid_preds,
        k_list=k_list,
        total_time_seconds=t_hybrid,
    )

    # Method 4: LexiGuard Entity-Aware Hybrid Search
    logger.info("--> Querying Pipeline 4: LexiGuard Entity-Aware Hybrid...")
    t0 = time.time()
    entity_preds = {}
    for q in questions:
        res = lexiguard_full.search_entity_aware(q.text, top_k=args.top_k)
        entity_preds[q.question_id] = [art.key for score, art in res]
    t_entity = time.time() - t0
    m4_metrics = evaluate_method_predictions(
        "LexiGuard Entity-Aware",
        questions,
        entity_preds,
        k_list=k_list,
        total_time_seconds=t_entity,
    )

    methods = [m1_metrics, m2_metrics, m3_metrics, m4_metrics]

    # Method 5: Two-Stage Cross-Encoder Neural Reranking (Phase 2 SOTA)
    if reranker is not None:
        logger.info(
            "--> Querying Pipeline 5: Two-Stage Neural Cross-Encoder Rerank (Phase 2 SOTA)..."
        )
        t0 = time.time()
        rerank_preds = {}
        for q in questions:
            res = lexiguard_full.search_two_stage_rerank(q.text, top_k=args.top_k, candidates_k=30)
            rerank_preds[q.question_id] = [art.key for score, art in res]
        t_rerank = time.time() - t0
        m5_metrics = evaluate_method_predictions(
            "LexiGuard SOTA Rerank",
            questions,
            rerank_preds,
            k_list=k_list,
            total_time_seconds=t_rerank,
        )
        methods.append(m5_metrics)

    # 6. Format and Print Comparison Table
    col_names = [m.method_name for m in methods]
    col_width = 16
    metric_col_width = 24
    total_width = metric_col_width + 3 + len(methods) * (col_width + 3)

    print("\n" + "=" * total_width)
    print(
        f"BENCHMARK RESULTS: ALQAC {args.season} ({len(articles):,} Articles | {len(questions):,} Questions)"
    )
    print("=" * total_width)
    header = f"{'Evaluation Metric':<{metric_col_width}} | " + " | ".join(
        f"{name[:col_width]:<{col_width}}" for name in col_names
    )
    print(header)
    print("-" * total_width)

    rows = [
        ("Hit Rate@5", lambda m: f"{m.hit_rate_at_k[5] * 100:.2f}%"),
        ("Hit Rate@10", lambda m: f"{m.hit_rate_at_k[10] * 100:.2f}%"),
        ("Hit Rate@20", lambda m: f"{m.hit_rate_at_k[20] * 100:.2f}%"),
        ("MRR@5", lambda m: f"{m.mrr_at_k[5]:.4f}"),
        ("MRR@10", lambda m: f"{m.mrr_at_k[10]:.4f}"),
        ("MRR@20", lambda m: f"{m.mrr_at_k[20]:.4f}"),
        ("NDCG@5", lambda m: f"{m.ndcg_at_k[5]:.4f}"),
        ("NDCG@10", lambda m: f"{m.ndcg_at_k[10]:.4f}"),
        ("NDCG@20", lambda m: f"{m.ndcg_at_k[20]:.4f}"),
        ("Recall@5", lambda m: f"{m.recall_at_k[5] * 100:.2f}%"),
        ("Recall@10", lambda m: f"{m.recall_at_k[10] * 100:.2f}%"),
        ("Precision@5", lambda m: f"{m.precision_at_k[5] * 100:.2f}%"),
        ("F2-Score@5 (ALQAC)", lambda m: f"{m.f2_at_k[5]:.4f}"),
        ("F2-Score@10", lambda m: f"{m.f2_at_k[10]:.4f}"),
        ("Latency / Query (ms)", lambda m: f"{m.latency_ms_per_query:.2f} ms"),
    ]

    for label, fn in rows:
        vals = " | ".join(f"{fn(m):<{col_width}}" for m in methods)
        print(f"{label:<{metric_col_width}} | {vals}")

    print("=" * total_width)

    # 7. Save Structured JSON Report
    report_dict = {
        "season": args.season,
        "total_corpus_articles": len(articles),
        "total_questions_evaluated": len(questions),
        "k_evaluated": list(k_list),
        "pipelines": {
            m.method_name: {
                "hit_rate_at_k": m.hit_rate_at_k,
                "mrr_at_k": m.mrr_at_k,
                "ndcg_at_k": m.ndcg_at_k,
                "recall_at_k": m.recall_at_k,
                "precision_at_k": m.precision_at_k,
                "f2_at_k": m.f2_at_k,
                "latency_ms_per_query": m.latency_ms_per_query,
            }
            for m in methods
        },
    }

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Report JSON saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
