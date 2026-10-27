"""
ALQAC Comprehensive Legal Benchmark Suite (Standard Benchmark Protocol)
========================================================================
Supports: ALQAC 2023 (Recommended), ALQAC 2022, ALQAC 2024, ALQAC 2025.

Core Pipeline Workflow:
1. Ingest & Parse Corpus: Extract (law_id, article_id, text) from law.json / legal_corpus.json.
2. Vector DB Upsert: Embed all articles via FastEmbed ONNX dense model and upsert into in-memory Qdrant.
3. Multi-Pipeline Querying:
   - Method 1: Okapi BM25 Lexical Search (Sparse)
   - Method 2: Qdrant Vector Search (Dense)
   - Method 3: Hybrid Search (BM25 + Qdrant via Reciprocal Rank Fusion)
   - Method 4: LexiGuard Entity-Aware Hybrid (RRF + Citation Grounding Boost)
4. Rigorous Metric Computation across Top K (5, 10, 20):
   - Hit Rate@K (Success Rate)
   - MRR@K (Mean Reciprocal Rank within top K)
   - NDCG@K (Normalized Discounted Cumulative Gain)
   - Recall@K (Multi-item coverage)
   - Precision@K & F2-Score@K
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.core.legal_norm import canonical_law_slug
from src.retrieval.entity_extractor import LegalEntityExtractor


# ============================================================================
# 1. DATA STRUCTURES
# ============================================================================

@dataclass(frozen=True)
class BenchmarkArticle:
    """A statutory article in the legal corpus."""
    corpus_idx: int
    law_id: str
    article_id: str
    title: str
    text: str

    @property
    def key(self) -> Tuple[str, str]:
        return (self.law_id.strip(), self.article_id.strip())

    @property
    def full_text(self) -> str:
        header = f"{self.law_id} Điều {self.article_id}"
        if self.title and self.title.lower() != header.lower():
            return f"{header}. {self.title}\n{self.text}"
        return f"{header}\n{self.text}"


@dataclass(frozen=True)
class BenchmarkQuestion:
    """An evaluation question with ground truth relevant articles."""
    question_id: str
    text: str
    relevant_articles: Tuple[Tuple[str, str], ...]  # ((law_id, article_id), ...)
    answer: Optional[str] = None
    question_type: Optional[str] = None


@dataclass
class MethodMetrics:
    """Evaluated metrics for a specific retrieval method."""
    method_name: str
    hit_rate_at_k: Dict[int, float]
    mrr_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    f2_at_k: Dict[int, float]
    latency_ms_per_query: float


# ============================================================================
# 2. MULTI-SEASON CORPUS & QUERY LOADER
# ============================================================================

class ALQACSeasonLoader:
    """Unified loader for ALQAC 2022, 2023, 2024, and 2025."""

    def __init__(self, base_dir: Union[str, Path] = "data/alqac"):
        self.base_dir = Path(base_dir)

    def load_season(self, season: str = "2023") -> Tuple[List[BenchmarkArticle], List[BenchmarkQuestion]]:
        """
        Loads corpus and questions for the given season.
        For 2023: Loads law.json (2,249 articles) and combined train + public_test + private_test (310 questions).
        For 2022: Loads law.json (1,377 articles) and question.json (520 questions).
        For 2025: Loads law.json (3,363 articles) and train + private_test_GOLD (811 questions).
        """
        if season == "2023":
            s_dir = self.base_dir / "ALQAC_2023"
            law_file = s_dir / "law.json"
            articles = self._parse_laws(law_file)

            questions: List[BenchmarkQuestion] = []
            for q_file in ["train.json", "public_test_GOLD.json", "private_test_GOLD.json"]:
                p = s_dir / q_file
                if p.exists():
                    questions.extend(self._parse_questions(p, id_prefix=p.stem))
            logger.info(f"Loaded ALQAC 2023: {len(articles)} articles, {len(questions)} questions.")
            return articles, questions

        elif season == "2022":
            s_dir = self.base_dir / "additional_data" / "ALQAC_2022_training_data"
            law_file = s_dir / "law.json"
            articles = self._parse_laws(law_file)
            q_file = s_dir / "question.json"
            questions = self._parse_questions(q_file, id_prefix="2022")
            logger.info(f"Loaded ALQAC 2022: {len(articles)} articles, {len(questions)} questions.")
            return articles, questions

        elif season == "2024":
            s_dir = self.base_dir / "ALQAC_2024"
            articles = self._parse_laws(s_dir / "law.json")
            questions = []
            for q_file in ["train.json", "public_test_GOLD.json", "private_test_GOLD.json"]:
                p = s_dir / q_file
                if p.exists():
                    questions.extend(self._parse_questions(p, id_prefix=p.stem))
            return articles, questions

        elif season == "2025":
            s_dir = self.base_dir / "ALQAC_2025"
            articles = self._parse_laws(s_dir / "law.json")
            questions = []
            for q_file in ["train.json", "private_test_GOLD.json"]:
                p = s_dir / q_file
                if p.exists():
                    questions.extend(self._parse_questions(p, id_prefix=p.stem))
            return articles, questions

        else:
            raise ValueError(f"Unsupported ALQAC season: {season}")

    @staticmethod
    def _parse_laws(law_file: Path) -> List[BenchmarkArticle]:
        data = json.loads(law_file.read_text(encoding="utf-8"))
        articles: List[BenchmarkArticle] = []
        idx = 0
        for law in data:
            law_id = law["id"].strip()
            for art in law.get("articles", []):
                art_id = str(art["id"]).strip()
                art_text = art.get("text", "").strip()

                lines = art_text.splitlines()
                title = lines[0].strip() if lines else ""
                body = "\n".join(lines[1:]).strip() if len(lines) > 1 else art_text

                articles.append(BenchmarkArticle(
                    corpus_idx=idx,
                    law_id=law_id,
                    article_id=art_id,
                    title=title,
                    text=body,
                ))
                idx += 1
        return articles

    @staticmethod
    def _parse_questions(q_file: Path, id_prefix: Optional[str] = None) -> List[BenchmarkQuestion]:
        data = json.loads(q_file.read_text(encoding="utf-8"))
        questions: List[BenchmarkQuestion] = []
        for item in data:
            raw_id = str(item.get("question_id") or item.get("id"))
            qid = f"{id_prefix}_{raw_id}" if id_prefix else raw_id
            text = item.get("text", "")
            rels = tuple(
                (r["law_id"].strip(), str(r["article_id"]).strip())
                for r in item.get("relevant_articles", [])
            )
            questions.append(BenchmarkQuestion(
                question_id=qid,
                text=text,
                relevant_articles=rels,
                answer=item.get("answer"),
                question_type=item.get("question_type"),
            ))
        return questions


# ============================================================================
# 3. BM25 RETRIEVAL ENGINE (VIETNAMESE COMPOUND N-GRAMS & DENOISING)
# ============================================================================

def denoise_query(text: str) -> str:
    """Removes common evaluation question tail noise that dilutes retrieval IDF."""
    return re.sub(
        r'[,.]?\s*(đúng hay sai|đúng không|như vậy đúng không|theo bạn việc này đúng hay sai|theo bạn đúng hay sai)\??$',
        '',
        text,
        flags=re.IGNORECASE,
    ).strip()


def tokenize_vietnamese_legal(text: str, is_query: bool = False) -> List[str]:
    """
    Vietnamese legal compound tokenizer:
    - Cleans evaluation tail noise for queries
    - Generates unigram tokens + adjacent bigram compound tokens (e.g. bồi_thường, thiệt_hại, hợp_đồng)
    """
    if is_query:
        text = denoise_query(text)
    words = re.findall(r'\w+', text.lower())
    if not words:
        return []
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
    return words + bigrams


class CorpusBM25Index:
    """In-memory Okapi BM25 engine with Vietnamese compound n-gram tokenization."""

    def __init__(self, articles: List[BenchmarkArticle], k1: float = 1.2, b: float = 0.75):
        self.articles = articles
        self.k1 = k1
        self.b = b
        self.N = len(articles)

        self.doc_tokens = [tokenize_vietnamese_legal(a.full_text, is_query=False) for a in articles]
        self.doc_lengths = [len(t) for t in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(self.N, 1)

        df = Counter()
        for t in self.doc_tokens:
            for w in set(t):
                df[w] += 1

        self.idf = {
            w: math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)
            for w, freq in df.items()
        }
        self.doc_tfs = [Counter(t) for t in self.doc_tokens]

    def search(self, query: str, top_k: int = 20) -> List[Tuple[float, BenchmarkArticle]]:
        q_tokens = tokenize_vietnamese_legal(query, is_query=True)
        scores: List[Tuple[float, int]] = []

        for idx, doc_len in enumerate(self.doc_lengths):
            tf_map = self.doc_tfs[idx]
            score = 0.0
            for term in q_tokens:
                if term in self.idf:
                    f = tf_map.get(term, 0)
                    if f > 0:
                        denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                        score += self.idf[term] * (f * (self.k1 + 1.0)) / denom
            scores.append((score, idx))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [(score, self.articles[idx]) for score, idx in scores[:top_k]]


# ============================================================================
# 4. QDRANT IN-MEMORY VECTOR STORE & DENSE RETRIEVER
# ============================================================================

class QdrantVectorBenchmarkIndex:
    """In-memory Qdrant collection with FastEmbed dense multilingual embeddings."""

    def __init__(
        self,
        articles: List[BenchmarkArticle],
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: Optional[Path] = None,
        season: str = "2023",
    ):
        self.articles = articles
        self.model_name = model_name
        self.season = season
        self.collection_name = f"alqac_{season}_articles"

        from fastembed import TextEmbedding
        logger.info(f"Initializing FastEmbed TextEmbedding: {model_name}...")
        self.embed_model = TextEmbedding(model_name=model_name)

        # 1. Compute or Load Cached Embeddings
        cache_file = cache_dir / f"embeddings_alqac_{season}.npy" if cache_dir else None
        if cache_file and cache_file.exists():
            logger.info(f"Loading cached embeddings from {cache_file}...")
            embeddings = np.load(cache_file)
        else:
            logger.info(f"Generating dense embeddings for {len(articles)} articles (batch_size=64)...")
            t0 = time.time()
            texts = [a.full_text for a in articles]
            embs_gen = self.embed_model.embed(texts, batch_size=64)
            embeddings = np.array(list(embs_gen), dtype=np.float32)
            logger.info(f"Embedded {len(articles)} articles in {time.time()-t0:.2f}s.")
            if cache_file:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_file, embeddings)
                logger.info(f"Cached embeddings to {cache_file}.")

        self.dim = embeddings.shape[1]

        # 2. Setup in-memory Qdrant client & upsert points
        logger.info(f"Initializing in-memory Qdrant with {len(articles)} points (dim={self.dim})...")
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )

        points = [
            PointStruct(
                id=a.corpus_idx,
                vector=embeddings[a.corpus_idx].tolist(),
                payload={"law_id": a.law_id, "article_id": a.article_id},
            )
            for a in articles
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"Qdrant in-memory collection '{self.collection_name}' ready!")

    def search(
        self,
        query: str,
        top_k: int = 20,
        query_vector: Optional[List[float]] = None,
    ) -> List[Tuple[float, BenchmarkArticle]]:
        if query_vector is None:
            query_vector = list(self.embed_model.embed([query]))[0].tolist()
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )
        scored: List[Tuple[float, BenchmarkArticle]] = []
        for p in res.points:
            score = float(p.score)
            article = self.articles[p.id]
            scored.append((score, article))
        return scored


# ============================================================================
# 5. HYBRID FUSION (RECIPROCAL RANK FUSION) & LEXIGUARD ENTITY-AWARE
# ============================================================================

def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[float, BenchmarkArticle]]],
    weights: Optional[List[float]] = None,
    rrf_k: int = 60,
    top_k: int = 20,
) -> List[Tuple[float, BenchmarkArticle]]:
    """Merges multiple ranked lists via Reciprocal Rank Fusion."""
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    fused_scores: Dict[Tuple[str, str], float] = defaultdict(float)
    article_by_key: Dict[Tuple[str, str], BenchmarkArticle] = {}

    for list_idx, r_list in enumerate(ranked_lists):
        w = weights[list_idx]
        for rank_idx, (score, art) in enumerate(r_list):
            key = art.key
            article_by_key[key] = art
            # Standard RRF formula: weight / (rrf_k + rank + 1)
            fused_scores[key] += w / (rrf_k + (rank_idx + 1))

    sorted_keys = sorted(fused_scores.keys(), key=lambda k: fused_scores[k], reverse=True)
    return [(fused_scores[k], article_by_key[k]) for k in sorted_keys[:top_k]]


from collections import defaultdict


class LexiGuardFullRetriever:
    """
    Combines:
    1. Okapi BM25 Lexical (with Vietnamese Compound N-grams)
    2. Dense Vector Qdrant
    3. RRF Hybrid Fusion
    4. Entity Extractor Grounded Citation & Law Boosting (strictly non-leaking)
    """

    def __init__(
        self,
        bm25: CorpusBM25Index,
        qdrant: Optional[QdrantVectorBenchmarkIndex] = None,
        reranker: Optional[Any] = None,
    ):
        self.bm25 = bm25
        self.qdrant = qdrant
        self.extractor = LegalEntityExtractor()
        self.reranker = reranker

    def search(
        self,
        query: str,
        top_k: int = 20,
    ) -> List[Tuple[float, BenchmarkArticle]]:
        """Primary project retrieval: executes high-performance BM25 Lexical (Compound N-grams)."""
        return self.bm25.search(query, top_k=top_k)

    def search_hybrid(
        self,
        query: str,
        top_k: int = 20,
        bm25_res: Optional[List[Tuple[float, BenchmarkArticle]]] = None,
        dense_res: Optional[List[Tuple[float, BenchmarkArticle]]] = None,
    ) -> List[Tuple[float, BenchmarkArticle]]:
        if bm25_res is None:
            bm25_res = self.bm25.search(query, top_k=max(top_k * 3, 60))
        if dense_res is None:
            dense_res = self.qdrant.search(query, top_k=max(top_k * 3, 60))
        return reciprocal_rank_fusion([dense_res, bm25_res], weights=[1.0, 1.0], top_k=top_k)

    def search_entity_aware(
        self,
        query: str,
        top_k: int = 20,
        base_hybrid: Optional[List[Tuple[float, BenchmarkArticle]]] = None,
    ) -> List[Tuple[float, BenchmarkArticle]]:
        if base_hybrid is None:
            base_hybrid = self.search_hybrid(query, top_k=max(top_k * 3, 60))

        detected_citations = self.extractor.extract_citations(query)
        detected_law = self.extractor.detect_law_code(query)

        boosted: List[Tuple[float, BenchmarkArticle]] = []
        for score, art in base_hybrid:
            b_score = score

            # Boost exact article citation
            for cit in detected_citations:
                if cit.article_num is not None and str(cit.article_num) == art.article_id:
                    b_score += 0.05

            # Boost explicit law citation strictly when genuine law is detected
            if detected_law:
                art_slug = canonical_law_slug(art.law_id)
                if art_slug == detected_law:
                    b_score += 0.02

            boosted.append((b_score, art))

        boosted.sort(key=lambda x: x[0], reverse=True)
        return boosted[:top_k]

    def search_two_stage_rerank(
        self,
        query: str,
        top_k: int = 20,
        candidates_k: int = 30,
        base_candidates: Optional[List[Tuple[float, BenchmarkArticle]]] = None,
    ) -> List[Tuple[float, BenchmarkArticle]]:
        """
        Two-stage retrieval pipeline:
        Stage 1: Entity-Aware Hybrid Search (BM25 + Qdrant Dense + Legal Citation Boost) retrieves Top 30-50 candidates.
        Stage 2: Cross-Encoder neural re-ranking (BAAI/bge-reranker-base) computes full token-level cross-attention.
        """
        if self.reranker is None:
            return self.search_entity_aware(query, top_k=top_k)

        if base_candidates is None:
            base_candidates = self.search_entity_aware(query, top_k=candidates_k)

        c_pairs = [(art, art.full_text) for _, art in base_candidates]
        reranked = self.reranker.rerank(query, c_pairs, top_k=top_k)
        return [(score, art) for score, art in reranked]


# ============================================================================
# 6. RIGOROUS METRICS ENGINE (Hit Rate@K, MRR@K, NDCG@K, Recall@K, F2@K)
# ============================================================================

def compute_ndcg_at_k(ranked_keys: List[Tuple[str, str]], gold_set: Set[Tuple[str, str]], k: int) -> float:
    """Calculates Normalized Discounted Cumulative Gain at K for binary relevance."""
    top_k = ranked_keys[:k]
    dcg = 0.0
    for idx, key in enumerate(top_k):
        if key in gold_set:
            dcg += 1.0 / math.log2(idx + 2)

    # Ideal DCG
    ideal_hits = min(len(gold_set), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_method_predictions(
    method_name: str,
    questions: List[BenchmarkQuestion],
    predictions: Dict[str, List[Tuple[str, str]]],
    k_list: Tuple[int, ...] = (5, 10, 20),
    total_time_seconds: float = 0.0,
) -> MethodMetrics:
    """Computes Hit Rate@K, MRR@K, NDCG@K, Recall@K, Precision@K, F2@K."""
    total_q = len(questions)
    if total_q == 0:
        raise ValueError("Cannot evaluate empty question set.")

    hit_rates = {k: [] for k in k_list}
    mrr_scores = {k: [] for k in k_list}
    ndcg_scores = {k: [] for k in k_list}
    recall_scores = {k: [] for k in k_list}
    precision_scores = {k: [] for k in k_list}
    f2_scores = {k: [] for k in k_list}

    for q in questions:
        gold_set = set(q.relevant_articles)
        if not gold_set:
            continue

        ranked_keys = predictions.get(q.question_id, [])

        for k in k_list:
            top_k_keys = ranked_keys[:k]
            hits = sum(1 for key in top_k_keys if key in gold_set)

            # 1. Hit Rate@K: 1 if at least one gold item is in top K, else 0
            hit_rate = 1.0 if hits > 0 else 0.0
            hit_rates[k].append(hit_rate)

            # 2. MRR@K: 1 / rank of first gold item if <= K, else 0
            first_rank = next((idx + 1 for idx, key in enumerate(top_k_keys) if key in gold_set), None)
            mrr_k = (1.0 / first_rank) if first_rank else 0.0
            mrr_scores[k].append(mrr_k)

            # 3. NDCG@K
            ndcg = compute_ndcg_at_k(ranked_keys, gold_set, k)
            ndcg_scores[k].append(ndcg)

            # 4. Multi-item Recall@K and Precision@K
            rec = hits / len(gold_set)
            prec = hits / k if k > 0 else 0.0
            recall_scores[k].append(rec)
            precision_scores[k].append(prec)

            # 5. F2@K (beta = 2)
            if prec + rec > 0:
                beta_sq = 4.0
                f2 = (1.0 + beta_sq) * (prec * rec) / (beta_sq * prec + rec)
            else:
                f2 = 0.0
            f2_scores[k].append(f2)

    return MethodMetrics(
        method_name=method_name,
        hit_rate_at_k={k: round(sum(v) / len(v), 4) for k, v in hit_rates.items()},
        mrr_at_k={k: round(sum(v) / len(v), 4) for k, v in mrr_scores.items()},
        ndcg_at_k={k: round(sum(v) / len(v), 4) for k, v in ndcg_scores.items()},
        recall_at_k={k: round(sum(v) / len(v), 4) for k, v in recall_scores.items()},
        precision_at_k={k: round(sum(v) / len(v), 4) for k, v in precision_scores.items()},
        f2_at_k={k: round(sum(v) / len(v), 4) for k, v in f2_scores.items()},
        latency_ms_per_query=round((total_time_seconds / total_q) * 1000.0, 2),
    )
