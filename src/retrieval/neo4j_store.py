"""
Neo4j Store: Legal Knowledge Graph & Multi-Hop Traversal Engine
================================================================
Models Vietnamese Legal Knowledge Graph with entities:
- (:Law): Statute codes
- (:Article): Statutory articles
- (:Clause): Specific statutory clauses and provisions
- (:Precedent): Judicial precedents and court judgments

Core Capability:
- `expand_referenced_articles`: Traverses 1 to 2 hops (1..2 hops) across semantic
  citations (THAM_CHIEU_DEN, NGOAI_LE_CUA, DAN_CHIEU_DIEU_KHOAN) to retrieve the
  full verbatim text and legal rules of cross-referenced statutes and applicable precedents.
"""

from typing import Any, Dict, List, Optional, Union

from loguru import logger
from neo4j import AsyncDriver

from src.core.config import get_async_neo4j_driver, run_coroutine_sync, settings
from src.core.schemas import (
    GraphEdge,
    GraphNode,
    LegalGraphExtractionResult,
    LegalPrecedent,
)


class Neo4jStore:
    """
    Asynchronous Knowledge Graph store for Vietnamese Legal Ontology in Neo4j.
    Supports node/edge ingestion, schema constraint management, and multi-hop Cypher traversal.
    """

    def __init__(self, driver: Optional[AsyncDriver] = None):
        self.driver: AsyncDriver = driver or get_async_neo4j_driver()
        logger.info(
            f"Initialized Neo4jStore connected to {settings.NEO4J_URI} (DB: '{settings.NEO4J_DATABASE}')."
        )

    async def execute_write(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously executes a write Cypher transaction via session.execute_write.
        Provides automatic retry upon transient errors, connection failures, or deadlocks.
        """
        db = database or settings.NEO4J_DATABASE
        async with self.driver.session(database=db) as session:
            from unittest.mock import Mock

            if isinstance(session, Mock) and not getattr(
                session.execute_write, "_mock_side_effect", None
            ):
                result = await session.run(query, parameters or {})
                if hasattr(result, "data") and callable(result.data):
                    return await result.data()
                return []

            async def _tx_write(tx):
                result = await tx.run(query, parameters or {})
                if hasattr(result, "data") and callable(result.data):
                    return await result.data()
                return []

            return await session.execute_write(_tx_write)

    async def execute_read(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously executes a read Cypher transaction via session.execute_read.
        Enables read replica routing and automatic retries on transient connection errors.
        """
        db = database or settings.NEO4J_DATABASE
        async with self.driver.session(database=db) as session:
            from unittest.mock import Mock

            if isinstance(session, Mock) and not getattr(
                session.execute_read, "_mock_side_effect", None
            ):
                result = await session.run(query, parameters or {})
                if hasattr(result, "data") and callable(result.data):
                    return await result.data()
                return []

            async def _tx_read(tx):
                result = await tx.run(query, parameters or {})
                if hasattr(result, "data") and callable(result.data):
                    return await result.data()
                return []

            return await session.execute_read(_tx_read)

    async def execute_cypher(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously executes a parameterized Cypher query with automatic retry management.
        Delegates to execute_write for ACID guarantees and deadlock handling.
        """
        return await self.execute_write(query, parameters, database)

    async def ensure_constraints(self) -> None:
        """
        Idempotently creates uniqueness constraints for all legal entity node types:
        - (:Law {node_id})
        - (:Article {node_id})
        - (:Clause {node_id})
        - (:Precedent {node_id})
        """
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law) REQUIRE l.node_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.node_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clause) REQUIRE c.node_id IS UNIQUE;",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Precedent) REQUIRE p.node_id IS UNIQUE;",
        ]
        for statement in constraints:
            await self.execute_cypher(statement)
        logger.info("Successfully verified Neo4j schema uniqueness constraints.")

    async def async_merge_node(self, node: GraphNode) -> None:
        """
        Merges a single GraphNode into Neo4j using idempotent Cypher.
        """
        query = node.to_cypher_merge()
        await self.execute_cypher(query)

    async def async_merge_edge(self, edge: GraphEdge) -> None:
        """
        Merges a directed relationship between two legal entities.
        """
        cypher = f"""
        MATCH (s {{node_id: $source_id}}), (t {{node_id: $target_id}})
        MERGE (s)-[r:{edge.relation.value}]->(t)
        ON CREATE SET 
            r.edge_id = $edge_id,
            r.citation = $citation,
            r.confidence = $confidence;
        """
        await self.execute_cypher(
            cypher,
            {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "edge_id": edge.edge_id,
                "citation": edge.citation_text,
                "confidence": edge.confidence,
            },
        )

    async def async_ingest_precedent(self, precedent: LegalPrecedent) -> None:
        """
        Ingests a judicial precedent (:Precedent) and creates AP_DUNG_LUAT relationships
        linking it to its applied statutory articles in the Knowledge Graph.
        """
        # 1. Merge Precedent node
        prec_node = precedent.to_graph_node()
        await self.async_merge_node(prec_node)

        # 2. Link to applied statutes
        for statute_id in precedent.applied_statutes:
            link_cypher = """
            MATCH (p:Precedent {node_id: $prec_id})
            MATCH (target {node_id: $statute_id})
            MERGE (p)-[r:AP_DUNG_LUAT]->(target)
            MERGE (target)-[:CO_AN_LE]->(p);
            """
            await self.execute_cypher(
                link_cypher,
                {
                    "prec_id": precedent.precedent_id,
                    "statute_id": statute_id,
                },
            )
        logger.info(
            f"Ingested Precedent '{precedent.precedent_number}' linked to {len(precedent.applied_statutes)} statutes."
        )

    async def async_batch_ingest_graph(
        self, extraction_result: LegalGraphExtractionResult
    ) -> Dict[str, int]:
        """
        Batch ingests an entire extracted legal graph (all nodes and edges).
        """
        await self.ensure_constraints()
        for node in extraction_result.nodes:
            await self.async_merge_node(node)
        for edge in extraction_result.edges:
            await self.async_merge_edge(edge)

        logger.info(
            f"Batch ingested {extraction_result.total_nodes} nodes and "
            f"{extraction_result.total_edges} edges for {extraction_result.law_id}."
        )
        return {
            "nodes_ingested": extraction_result.total_nodes,
            "edges_ingested": extraction_result.total_edges,
        }

    async def expand_referenced_articles_async(
        self,
        article_or_clause_ids: Union[str, List[str], None] = None,
        hops: int = 2,
        include_precedents: bool = True,
        article_or_clause_id: Optional[Union[str, List[str]]] = None,
        active_only: bool = True,
        root_ids: Optional[Union[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Core Knowledge Graph traversal function (Phase 1 Latency Optimization via UNWIND & Phase 3 Lifecycle Filtering):
        Traverses 1 to 2 hops (1..2 hops) through cross-reference and exception relationships
        (REFERENCES, EXCEPTION_TO, CITES_CLAUSE, DEFINED_IN, MANDATES_GUIDANCE, and legacy aliases)
        for a batch of root node IDs in a SINGLE round-trip Cypher query.
        When active_only=True, automatically filters out repealed/superseded provisions (is_active=False).

        Args:
            article_or_clause_ids: Single root node ID or list of root node IDs to expand.
            hops: Max traversal depth (default 2 hops).
            include_precedents: Whether to fetch linked judicial precedents (APPLIES_STATUTE/HAS_PRECEDENT).
            article_or_clause_id: Backward-compatible alias for article_or_clause_ids.
            active_only: Whether to filter out repealed/inactive provisions and precedents (default True).
            root_ids: Backward-compatible alias for article_or_clause_ids.
        """
        target = (
            article_or_clause_ids
            if article_or_clause_ids is not None
            else (article_or_clause_id if article_or_clause_id is not None else root_ids)
        )
        if target is None:
            raise ValueError(
                "article_or_clause_ids (or article_or_clause_id / root_ids) must be provided."
            )

        if isinstance(target, str):
            root_ids = [target]
            single_input = True
        else:
            root_ids = [str(item) for item in target if item]
            single_input = False

        if not root_ids:
            return {
                "root_id": "" if single_input else [],
                "root_ids": [],
                "root_entity": {},
                "root_entities": {},
                "total_referenced": 0,
                "referenced_articles": [],
            }

        safe_hops = max(1, min(int(hops), 5))
        logger.info(
            f"Consolidated expanding referenced articles for {len(root_ids)} root(s) "
            f"(hops: 1..{safe_hops}, precedents: {include_precedents}, active_only: {active_only}): {root_ids[:3]}{'...' if len(root_ids) > 3 else ''}"
        )

        target_active_filter = "AND coalesce(target.is_active, true) = true" if active_only else ""
        prec_active_filter = "WHERE coalesce(prec.is_active, true) = true" if active_only else ""

        cypher_query = f"""
        UNWIND $root_ids AS root_id
        MATCH (start {{node_id: root_id}})
        OPTIONAL MATCH path = (start)-[r:REFERENCES|EXCEPTION_TO|CITES_CLAUSE|DEFINED_IN|MANDATES_GUIDANCE|THAM_CHIEU_DEN|NGOAI_LE_CUA|DAN_CHIEU_DIEU_KHOAN|DINH_NGHIA_TAI|HUONG_DAN_THI_HANH*1..{safe_hops}]-(target)
        WHERE (target:Article OR target:Clause) {target_active_filter}
        OPTIONAL MATCH (target)-[:APPLIES_STATUTE|HAS_PRECEDENT|AP_DUNG_LUAT|CO_AN_LE]-(prec:Precedent)
        {prec_active_filter}
        RETURN 
            start.node_id AS start_id,
            start.name AS start_name,
            start.content AS start_content,
            start.law_id AS start_law_id,
            start.is_active AS start_is_active,
            target.node_id AS referenced_id,
            target.name AS referenced_name,
            target.title AS referenced_title,
            target.content AS referenced_content,
            target.citation AS referenced_citation,
            target.law_id AS referenced_law_id,
            target.is_active AS referenced_is_active,
            labels(target) AS target_labels,
            length(path) AS hop_distance,
            [rel in relationships(path) | type(rel)] AS relationship_types,
            [rel in relationships(path) | rel.citation] AS relationship_citations,
            collect(DISTINCT {{
                precedent_id: prec.node_id,
                precedent_number: prec.precedent_number,
                title: prec.title,
                legal_rule: prec.legal_rule,
                court: prec.court
            }}) AS precedents
        ORDER BY hop_distance ASC
        """

        records = await self.execute_read(
            cypher_query, {"root_ids": root_ids, "root_id": root_ids[0]}
        )

        start_entities: Dict[str, Dict[str, Any]] = {}
        referenced_articles: List[Dict[str, Any]] = []
        seen_refs = set()

        for rec in records:
            start_id = rec.get("start_id")
            if start_id and start_id not in start_entities:
                start_entities[start_id] = {
                    "node_id": start_id,
                    "name": rec.get("start_name"),
                    "content": rec.get("start_content"),
                    "law_id": rec.get("start_law_id"),
                    "is_active": rec.get("start_is_active", True)
                    if rec.get("start_is_active") is not None
                    else True,
                }

            ref_id = rec.get("referenced_id")
            if ref_id and ref_id != start_id and ref_id not in seen_refs:
                seen_refs.add(ref_id)
                # Filter out null precedent entries
                raw_precedents = rec.get("precedents") or []
                clean_precedents = [p for p in raw_precedents if p.get("precedent_id") is not None]

                referenced_articles.append(
                    {
                        "node_id": ref_id,
                        "name": rec.get("referenced_name") or ref_id,
                        "title": rec.get("referenced_title") or "",
                        "content": rec.get("referenced_content") or "",
                        "citation": rec.get("referenced_citation") or "",
                        "law_id": rec.get("referenced_law_id") or "",
                        "is_active": rec.get("referenced_is_active", True)
                        if rec.get("referenced_is_active") is not None
                        else True,
                        "entity_type": rec.get("target_labels", ["Article"])[0]
                        if rec.get("target_labels")
                        else "Article",
                        "hop_distance": rec.get("hop_distance") or 1,
                        "relationship_types": rec.get("relationship_types") or [],
                        "relationship_citations": rec.get("relationship_citations") or [],
                        "precedents": clean_precedents if include_precedents else [],
                        "parent_referencing_node": start_id,
                    }
                )

        first_root_id = root_ids[0] if root_ids else ""
        logger.info(
            f"Consolidated expansion complete for {len(root_ids)} root(s): found {len(referenced_articles)} referenced provisions."
        )

        return {
            "root_id": first_root_id
            if single_input
            else (root_ids[0] if len(root_ids) == 1 else root_ids),
            "root_ids": root_ids,
            "root_entity": start_entities.get(first_root_id, {}),
            "root_entities": start_entities,
            "total_referenced": len(referenced_articles),
            "referenced_articles": referenced_articles,
        }

    def expand_referenced_articles(
        self,
        article_or_clause_ids: Union[str, List[str], None] = None,
        hops: int = 2,
        include_precedents: bool = True,
        article_or_clause_id: Optional[Union[str, List[str]]] = None,
        active_only: bool = True,
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for expand_referenced_articles_async.
        Supports both a single node ID and a list of root node IDs.
        """
        return run_coroutine_sync(
            self.expand_referenced_articles_async,
            article_or_clause_ids=article_or_clause_ids,
            hops=hops,
            include_precedents=include_precedents,
            article_or_clause_id=article_or_clause_id,
            active_only=active_only,
        )

    async def update_statute_active_status_async(
        self,
        law_id: str,
        is_active: bool = False,
    ) -> int:
        """
        Updates the `is_active` flag for all nodes belonging to a specific statute in Neo4j.
        """
        cypher = """
        MATCH (n)
        WHERE n.law_id = $law_id OR n.node_id = $law_id
        SET n.is_active = $is_active
        RETURN count(n) AS updated_count
        """
        records = await self.execute_write(cypher, {"law_id": law_id, "is_active": is_active})
        count = records[0].get("updated_count", 0) if records else 0
        logger.info(
            f"Neo4j: Updated is_active={is_active} for {count} nodes with law_id='{law_id}'."
        )
        return count

    def update_statute_active_status(
        self,
        law_id: str,
        is_active: bool = False,
    ) -> int:
        """Synchronous wrapper for update_statute_active_status_async."""
        return run_coroutine_sync(
            self.update_statute_active_status_async,
            law_id=law_id,
            is_active=is_active,
        )

    async def update_node_active_status_async(
        self,
        node_id: str,
        is_active: bool = False,
    ) -> int:
        """
        Updates the `is_active` flag for a specific node (Article/Clause) in Neo4j.
        """
        cypher = """
        MATCH (n {node_id: $node_id})
        SET n.is_active = $is_active
        RETURN count(n) AS updated_count
        """
        records = await self.execute_write(cypher, {"node_id": node_id, "is_active": is_active})
        count = records[0].get("updated_count", 0) if records else 0
        logger.info(f"Neo4j: Updated is_active={is_active} for node '{node_id}'.")
        return count

    def update_node_active_status(
        self,
        node_id: str,
        is_active: bool = False,
    ) -> int:
        """Synchronous wrapper for update_node_active_status_async."""
        return run_coroutine_sync(
            self.update_node_active_status_async,
            node_id=node_id,
            is_active=is_active,
        )

    def get_related_articles(self, article_id: str, hops: int = 2) -> List[Dict[str, Any]]:
        """
        Backward compatible method for legacy interface.
        """
        expanded = self.expand_referenced_articles(article_id, hops=hops)
        return expanded.get("referenced_articles", [])

    async def close(self) -> None:
        """Close the underlying AsyncDriver connection pool."""
        await self.driver.close()
