"""
Hybrid Legal Retriever & Cross-Encoder Reranker
================================================
Orchestrates the 3-step statutory retrieval and cross-referencing pipeline:
- Step 1 (Qdrant Hybrid Search): Retrieve top 12 candidates using RRF (Dense + BM25).
- Step 2 (Neo4j Multi-Hop Expansion): Extract article keys and traverse 1-2 hops on
  the legal knowledge graph to pull verbatim text of cross-referenced statutes and precedents.
- Step 3 (Cross-Encoder Reranking): Re-score consolidated candidate pool using
  BAAI/bge-reranker-v2-m3 and filter to top_k high-quality context segments.
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from src.core.config import settings
from src.retrieval.entity_extractor import LegalEntityExtractor
from src.retrieval.lifecycle_manager import LegalLifecycleManager
from src.retrieval.neo4j_store import Neo4jStore
from src.retrieval.qdrant_store import QdrantStore


class LegalCrossEncoderReranker:
    """
    Neural Cross-Encoder reranker using BAAI/bge-reranker-v2-m3.
    Re-scores query-candidate relevance with deep cross-attention.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        custom_rerank_fn: Optional[Callable[[str, List[str]], List[float]]] = None,
    ):
        self.model_name = model_name or settings.RERANKER_MODEL or "BAAI/bge-reranker-v2-m3"
        self._model: Any = None
        self._custom_rerank_fn = custom_rerank_fn

    @property
    def model(self) -> Any:
        """Lazy load CrossEncoder model."""
        if self._model is None and self._custom_rerank_fn is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading CrossEncoder reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Re-scores a candidate pool against the original query.
        """
        if not candidates:
            return []

        # Prepare pairs for cross-attention
        candidate_texts = [
            c.get("context_enriched_text")
            or c.get("content")
            or c.get("article_title")
            or c.get("canonical_citation")
            or ""
            for c in candidates
        ]

        if self._custom_rerank_fn:
            scores = self._custom_rerank_fn(query, candidate_texts)
        else:
            pairs = [[query, text] for text in candidate_texts]
            raw_scores = self.model.predict(pairs)
            scores = raw_scores.tolist() if hasattr(raw_scores, "tolist") else list(raw_scores)

        # Attach rerank score and sort
        scored_candidates = []
        for cand, score in zip(candidates, scores):
            cand_copy = dict(cand)
            cand_copy["rerank_score"] = float(score)
            scored_candidates.append(cand_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Slice to top_k and assign final rank
        final_top = scored_candidates[:top_k]
        for rank_idx, item in enumerate(final_top):
            item["final_rank"] = rank_idx + 1

        logger.info(
            f"Cross-Encoder reranked {len(candidates)} candidates down to top {len(final_top)}."
        )
        return final_top


class HybridLegalRetriever:
    """
    Unified multi-channel Legal Retriever for contract auditing:
    Coordinates Qdrant Hybrid RRF Search, Neo4j Graph Multi-Hop Traversal, and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        qdrant_store: Optional[QdrantStore] = None,
        neo4j_store: Optional[Neo4jStore] = None,
        reranker: Optional[LegalCrossEncoderReranker] = None,
        entity_extractor: Optional[LegalEntityExtractor] = None,
        lifecycle_manager: Optional[LegalLifecycleManager] = None,
        use_bm25_only: Optional[bool] = None,
    ):
        logger.info("Initializing HybridLegalRetriever...")
        self.use_bm25_only = (
            use_bm25_only
            if use_bm25_only is not None
            else (qdrant_store is None and neo4j_store is None)
        )
        self._bm25_retriever: Any = None
        self.qdrant_store: QdrantStore = qdrant_store or QdrantStore()
        self.neo4j_store: Neo4jStore = neo4j_store or Neo4jStore()
        self.reranker: LegalCrossEncoderReranker = reranker or LegalCrossEncoderReranker()
        self.entity_extractor: LegalEntityExtractor = entity_extractor or LegalEntityExtractor()
        self.lifecycle_manager: LegalLifecycleManager = lifecycle_manager or LegalLifecycleManager(
            qdrant_store=self.qdrant_store,
            neo4j_store=self.neo4j_store,
        )

    async def deactivate_statute_async(self, law_id: str) -> Dict[str, Any]:
        """Deactivates an expired/repealed statute synchronously across both Qdrant and Neo4j."""
        return await self.lifecycle_manager.deactivate_statute_async(law_id)

    async def activate_statute_async(self, law_id: str) -> Dict[str, Any]:
        """Activates a statute synchronously across both Qdrant and Neo4j."""
        return await self.lifecycle_manager.activate_statute_async(law_id)

    async def sync_statute_lifecycle_async(
        self, law_id: str, is_active: bool = False
    ) -> Dict[str, Any]:
        """Coordinates lifecycle validity status update across both Qdrant and Neo4j."""
        return await self.lifecycle_manager.sync_statute_lifecycle_async(
            law_id, is_active=is_active
        )

    def deactivate_statute(self, law_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for deactivate_statute_async."""
        return self.lifecycle_manager.sync_statute_lifecycle(law_id, is_active=False)

    def activate_statute(self, law_id: str) -> Dict[str, Any]:
        """Synchronous wrapper for activate_statute_async."""
        return self.lifecycle_manager.sync_statute_lifecycle(law_id, is_active=True)

    async def retrieve_async(
        self,
        query: str,
        top_k: int = 5,
        qdrant_limit: int = 12,
        expansion_hops: int = 2,
        include_precedents: bool = True,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieval pipeline. When use_bm25_only=True, directly executes BM25 Lexical
        (Vietnamese Compound N-grams) for sub-millisecond, top-precision statutory retrieval.
        """
        if self.use_bm25_only:
            logger.info(
                f"=== [BM25 Lexical Mode] Executing Compound N-grams Retrieval for: '{query[:60]}' ==="
            )
            if self._bm25_retriever is None:
                from src.retrieval.bm25_retriever import BM25LexicalRetriever

                self._bm25_retriever = BM25LexicalRetriever()
            return self._bm25_retriever.search(
                query=query,
                top_k=top_k,
                active_only=active_only,
            )

        logger.info(
            f"=== Starting 3-Step Hybrid Legal Retrieval for query: '{query[:60]}' (active_only={active_only}) ==="
        )

        # ----------------------------------------------------------------------
        # FAST PATH: Extract statutory article identifiers from the query.
        # ----------------------------------------------------------------------
        fast_path_node_ids = self.entity_extractor.extract_citation_node_ids(query)
        if fast_path_node_ids:
            logger.info(
                f"[Entity Fast-Path] Phát hiện {len(fast_path_node_ids)} node hạt giống điều luật: {fast_path_node_ids}"
            )

        # ----------------------------------------------------------------------
        # STEP 1: Qdrant hybrid search (RRF dense + BM25), taking 12 results.
        # ----------------------------------------------------------------------
        logger.info(
            f"[Bước 1] Querying Qdrant Hybrid Search (limit={qdrant_limit}, active_only={active_only})..."
        )
        qdrant_candidates = await self.qdrant_store.hybrid_search_async(
            query=query,
            top_k=qdrant_limit,
            prefetch_limit=max(25, qdrant_limit * 2),
            active_only=active_only,
        )

        unified_candidates: Dict[str, Dict[str, Any]] = {}
        candidate_node_ids: List[str] = []

        for idx, item in enumerate(qdrant_candidates):
            node_id = item.get("neo4j_node_id") or item.get("chunk_id")
            if not node_id:
                continue

            item["retrieval_source"] = "qdrant_rrf"
            item["rrf_rank"] = idx + 1
            item["hop_distance"] = 0
            unified_candidates[node_id] = item
            candidate_node_ids.append(node_id)

        logger.info(f"[Bước 1 Hoàn tất] Qdrant thu thập {len(unified_candidates)} ứng viên hợp lệ.")

        # ----------------------------------------------------------------------
        # STEP 2: Neo4j multi-hop expansion, prioritizing fast-path seed nodes.
        # ----------------------------------------------------------------------
        # Put fast-path article nodes at the front of the Neo4j expansion seed list.
        seed_nodes = list(dict.fromkeys(fast_path_node_ids + candidate_node_ids))
        nodes_to_expand = seed_nodes[: max(6, len(fast_path_node_ids))]

        logger.info(
            f"[Bước 2] Expanding {len(nodes_to_expand)} candidate seeds via Neo4j Knowledge Graph "
            f"(Fast-Path seeds: {len(fast_path_node_ids)}, hops=1..{expansion_hops}, active_only={active_only})..."
        )
        expanded_count = 0

        if nodes_to_expand:
            try:
                expansion = await self.neo4j_store.expand_referenced_articles_async(
                    article_or_clause_ids=nodes_to_expand,
                    hops=expansion_hops,
                    include_precedents=include_precedents,
                    active_only=active_only,
                )
                referenced_articles = expansion.get("referenced_articles", [])
                root_entities = expansion.get("root_entities", {})

                # Keep fast-path entity nodes directly in unified_candidates when active.
                for fp_id in fast_path_node_ids:
                    if fp_id in unified_candidates:
                        unified_candidates[fp_id]["is_fast_path"] = True
                        if unified_candidates[fp_id].get("retrieval_source") == "qdrant_rrf":
                            unified_candidates[fp_id]["retrieval_source"] = "hybrid"
                    elif fp_id in root_entities:
                        root_info = root_entities[fp_id]
                        # Automatically remove inactive provisions when active_only=True.
                        if active_only and root_info.get("is_active") is False:
                            logger.info(
                                f"[Entity Fast-Path] Bỏ qua điều khoản hết hiệu lực: '{fp_id}'"
                            )
                            continue

                        unified_candidates[fp_id] = {
                            "point_id": "",
                            "score": 1.0,
                            "chunk_id": fp_id,
                            "neo4j_node_id": fp_id,
                            "neo4j_node_type": "Clause" if "_K" in fp_id else "Article",
                            "canonical_citation": root_info.get("name") or fp_id,
                            "article_title": root_info.get("name") or fp_id,
                            "content": root_info.get("content", ""),
                            "context_enriched_text": f"[{root_info.get('law_id', '')}] {root_info.get('name', '')}: {root_info.get('content', '')}",
                            "law_title": root_info.get("law_id", ""),
                            "cross_references": [],
                            "retrieval_source": "entity_fast_path",
                            "is_fast_path": True,
                            "hop_distance": 0,
                            "relationship_types": [],
                            "relationship_citations": [],
                            "precedents": [],
                            "parent_referencing_node": "",
                        }
                    else:
                        unified_candidates[fp_id] = {
                            "point_id": "",
                            "score": 1.0,
                            "chunk_id": fp_id,
                            "neo4j_node_id": fp_id,
                            "neo4j_node_type": "Clause" if "_K" in fp_id else "Article",
                            "canonical_citation": fp_id,
                            "article_title": fp_id,
                            "content": "",
                            "context_enriched_text": fp_id,
                            "law_title": "",
                            "cross_references": [],
                            "retrieval_source": "entity_fast_path",
                            "is_fast_path": True,
                            "hop_distance": 0,
                            "relationship_types": [],
                            "relationship_citations": [],
                            "precedents": [],
                            "parent_referencing_node": "",
                        }

                for ref in referenced_articles:
                    ref_id = ref.get("node_id")
                    if not ref_id:
                        continue

                    if ref_id in unified_candidates:
                        # Existing candidate discovered via both channels: mark as hybrid and attach precedents
                        existing = unified_candidates[ref_id]
                        existing["retrieval_source"] = "hybrid"
                        if "precedents" not in existing or not existing["precedents"]:
                            existing["precedents"] = ref.get("precedents", [])
                    else:
                        # New candidate discovered through graph cross-referencing.
                        ref_candidate = {
                            "point_id": ref.get("qdrant_point_id", ""),
                            "score": 0.5,  # Baseline expansion relevance
                            "chunk_id": ref_id,
                            "neo4j_node_id": ref_id,
                            "neo4j_node_type": ref.get("entity_type", "Article"),
                            "canonical_citation": ref.get("citation") or ref.get("name") or ref_id,
                            "article_title": ref.get("title", ""),
                            "content": ref.get("content", ""),
                            "context_enriched_text": f"[{ref.get('law_id', '')}] {ref.get('name', '')}: {ref.get('content', '')}",
                            "law_title": ref.get("law_id", ""),
                            "cross_references": [],
                            "retrieval_source": "neo4j_expansion",
                            "hop_distance": ref.get("hop_distance", 1),
                            "relationship_types": ref.get("relationship_types", []),
                            "relationship_citations": ref.get("relationship_citations", []),
                            "precedents": ref.get("precedents", []),
                            "parent_referencing_node": ref.get("parent_referencing_node", ""),
                        }
                        unified_candidates[ref_id] = ref_candidate
                        expanded_count += 1
            except Exception as exc:
                logger.warning(
                    f"Neo4j consolidated expansion failed for nodes {nodes_to_expand}: {exc}"
                )

        candidate_pool = list(unified_candidates.values())
        logger.info(
            f"[Bước 2 Hoàn tất] Neo4j bổ sung {expanded_count} điều khoản dẫn chiếu. "
            f"Tổng số ứng viên trong pool: {len(candidate_pool)}."
        )

        # ----------------------------------------------------------------------
        # STEP 3: Cross-encoder reranking (BAAI/bge-reranker-v2-m3).
        # ----------------------------------------------------------------------
        logger.info(f"[Bước 3] Cross-Encoder Reranking candidate pool down to top {top_k}...")
        final_ranked_candidates = self.reranker.rerank(
            query=query,
            candidates=candidate_pool,
            top_k=top_k,
        )

        logger.info(
            f"=== Hoàn thành truy xuất 3 bước: Trả về {len(final_ranked_candidates)} căn cứ chuẩn hóa ==="
        )
        return final_ranked_candidates

    def retrieve(
        self, query: str, top_k: int = 5, active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Synchronous interface for HybridLegalRetriever.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.retrieve_async(query=query, top_k=top_k, active_only=active_only),
                    )
                    return future.result()
            return loop.run_until_complete(
                self.retrieve_async(query=query, top_k=top_k, active_only=active_only)
            )
        except RuntimeError:
            return asyncio.run(
                self.retrieve_async(query=query, top_k=top_k, active_only=active_only)
            )

    def format_context_for_prompt(self, candidates: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved legal articles and attached precedents into a structured Markdown
        grounding context block ready for injection into LLM / Agent prompts.
        """
        if not candidates:
            return "Không tìm thấy căn cứ pháp luật trực tiếp phù hợp."

        sections = [
            "### ⚖️ CĂN CỨ PHÁP LUẬT VÀ ÁN LỆ THAM CHIẾU (GROUNDED STATUTES & PRECEDENTS)",
            "Dưới đây là các điều khoản pháp luật quy chuẩn và án lệ liên đới được trích xuất từ Hệ thống GraphRAG:",
            "",
        ]

        for item in candidates:
            rank = item.get("final_rank", 1)
            citation = (
                item.get("canonical_citation") or item.get("article_title") or item.get("chunk_id")
            )
            content = (item.get("content") or item.get("context_enriched_text") or "").strip()
            source = item.get("retrieval_source", "hybrid")
            score = item.get("rerank_score", 0.0)
            node_id = item.get("neo4j_node_id") or item.get("chunk_id", "")

            source_label = {
                "hybrid": "Qdrant Vector + Neo4j Graph (Đồng thuận)",
                "qdrant_rrf": "Qdrant Hybrid Search (Dense + BM25)",
                "neo4j_expansion": f"Neo4j Dẫn chiếu ({item.get('hop_distance', 1)} hops từ {item.get('parent_referencing_node', '')})",
            }.get(source, source)

            sections.append(f"#### [{rank}] {citation} (Mã: `{node_id}`)")
            sections.append(
                f"- **Nguồn kiểm chứng**: {source_label} | **Rerank Score**: `{score:.4f}`"
            )
            if item.get("relationship_citations"):
                sections.append(
                    f'- **Mệnh đề dẫn chiếu**: *"{"; ".join(item["relationship_citations"])}"*'
                )
            sections.append(f'- **Nội dung nguyên văn**:\n  > "{content}"')

            precedents = item.get("precedents") or []
            if precedents:
                sections.append("- **Án lệ / Bản án thực tế liên quan**:")
                for prec in precedents:
                    p_num = prec.get("precedent_number", "")
                    p_title = prec.get("title", "")
                    p_rule = prec.get("legal_rule", "")
                    sections.append(f"  * **{p_num}: {p_title}**\n    - *Quy tắc án lệ*: {p_rule}")

            sections.append("")

        return "\n".join(sections)


# Backwards compatibility alias
HybridRetriever = HybridLegalRetriever
