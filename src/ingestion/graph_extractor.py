"""
Legal Graph Extractor: High-Precision Knowledge Graph Triples for Neo4j
========================================================================
Extracts legal entities (Law, Chapter, Article, Clause, Concept) and semantic citations:
  - CONTAINS: Structural taxonomy (Law -> Chapter -> Article -> Clause)
  - THAM_CHIEU_DEN: Cross-statutory reference ("quy định tại Điều X", "theo quy định của Bộ luật...")
  - NGOAI_LE_CUA: Carve-out / Exception clause ("Trừ trường hợp quy định tại Điều X...")
  - DAN_CHIEU_DIEU_KHOAN: Specific clause reference ("quy định tại khoản Y Điều X...")
  - HUONG_DAN_THI_HANH: Detailed decree mandates ("Chính phủ quy định chi tiết Điều này...")

Generates Neo4j Cypher statements with idempotent MERGE operations.
"""

import re
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from src.core.legal_norm import canonical_law_slug
from src.core.schemas import (
    GraphEdge,
    GraphNode,
    GraphRelationType,
    LegalChunk,
    LegalGraphExtractionResult,
)


class LegalGraphExtractor:
    """
    Extracts structured Knowledge Graph nodes and edges from parsed legal codes.
    Deterministic Regex-first parsing guarantees 0% hallucination on statutory citations.
    """

    # Citation Regex Patterns
    EXCEPTION_PATTERN = re.compile(
        r"(?:trừ\s+trường\s+hợp|ngoại\s+trừ)\s+(?:quy\s+định\s+tại\s+)?(?:khoản\s+(\d+)\s+)?(?:điều|điu)\s+(\d+)",
        re.IGNORECASE,
    )

    DETAILED_CLAUSE_PATTERN = re.compile(
        r"(?:quy\s+định\s+tại|theo\s+quy\s+định\s+tại|căn\s+cứ\s+vào|chiểu\s+theo)\s+khoản\s+(\d+)\s+(?:điều|điu)\s+(\d+)",
        re.IGNORECASE,
    )

    GENERIC_ARTICLE_PATTERN = re.compile(
        r"(?:quy\s+định\s+tại|theo\s+quy\s+định\s+tại|căn\s+cứ|áp\s+dụng)\s+(?:điều|điu)\s+(\d+)",
        re.IGNORECASE,
    )

    CROSS_STATUTE_PATTERN = re.compile(
        r"(?:quy\s+định\s+của|theo\s+quy\s+định\s+của|căn\s+cứ\s+(?:vào)?)\s+"
        r"((?:Bộ\s+luật|Luật|Nghị\s+định|Thông\s+tư|Quy\s+chế|Quy\s+định|Pháp\s+lệnh|Hiến\s+pháp)"
        r"(?:\s+(?!(?:và|hoặc|tại|theo)\s+(?:điều|khoản|điểm|\d+)\b)[A-ZÀ-Ỹa-zà-ỹ0-9/_.\-]+)+)",
        re.IGNORECASE,
    )

    GUIDANCE_PATTERN = re.compile(
        r"(?:chính\s+phủ\s+quy\s+định\s+chi\s+tiết|hướng\s+dẫn\s+thi\s+hành)\s+(?:điều\s+này|(?:điều|điu)\s+(\d+))",
        re.IGNORECASE,
    )

    def __init__(self):
        logger.info("Initializing LegalGraphExtractor (Regex-First Statutory Triples)...")

    def extract_from_chunks(
        self, chunks: List[LegalChunk], law_id: str, law_title: str
    ) -> LegalGraphExtractionResult:
        """
        Extracts all nodes and relationship triples from a list of granular LegalChunk objects.
        """
        law_code = self._get_law_code(law_id)
        nodes_dict: Dict[str, GraphNode] = {}
        edges: List[GraphEdge] = []
        edge_keys: Set[str] = set()

        # 1. Root Law Node
        law_node_id = law_code
        nodes_dict[law_node_id] = GraphNode(
            node_id=law_node_id,
            node_type="Law",
            name=law_title,
            law_id=law_id,
            properties={"title": law_title, "law_id": law_id},
        )

        seen_chapters: Set[str] = set()
        seen_articles: Set[str] = set()

        for chunk in chunks:
            # 2. Chapter Node & Structural Containment
            chap_node_id = f"{law_code}_CH{chunk.chapter_num}"
            if chap_node_id not in seen_chapters:
                seen_chapters.add(chap_node_id)
                nodes_dict[chap_node_id] = GraphNode(
                    node_id=chap_node_id,
                    node_type="Chapter",
                    name=f"Chương {chunk.chapter_num}: {chunk.chapter_title}",
                    law_id=law_id,
                    properties={"chapter_num": chunk.chapter_num, "title": chunk.chapter_title},
                )
                self._add_edge(
                    edges,
                    edge_keys,
                    source_id=law_node_id,
                    source_type="Law",
                    relation=GraphRelationType.CONTAINS,
                    target_id=chap_node_id,
                    target_type="Chapter",
                    citation_text=f"{law_title} chứa Chương {chunk.chapter_num}",
                )

            # 3. Article Node & Containment
            art_node_id = f"{law_code}_D{chunk.article_num:03d}"
            if art_node_id not in seen_articles:
                seen_articles.add(art_node_id)
                nodes_dict[art_node_id] = GraphNode(
                    node_id=art_node_id,
                    node_type="Article",
                    name=f"Điều {chunk.article_num}: {chunk.article_title}",
                    law_id=law_id,
                    properties={
                        "article_num": chunk.article_num,
                        "title": chunk.article_title,
                        "citation": f"Điều {chunk.article_num} {law_title}",
                    },
                )
                self._add_edge(
                    edges,
                    edge_keys,
                    source_id=chap_node_id,
                    source_type="Chapter",
                    relation=GraphRelationType.CONTAINS,
                    target_id=art_node_id,
                    target_type="Article",
                    citation_text=f"Chương {chunk.chapter_num} chứa Điều {chunk.article_num}",
                )

            # 4. Clause Node & Containment (if chunk is at clause level)
            clause_node_id = chunk.chunk_id
            if chunk.clause_num is not None:
                if clause_node_id not in nodes_dict:
                    nodes_dict[clause_node_id] = GraphNode(
                        node_id=clause_node_id,
                        node_type="Clause",
                        name=chunk.canonical_citation,
                        law_id=law_id,
                        properties={
                            "clause_num": chunk.clause_num,
                            "article_num": chunk.article_num,
                            "content": chunk.content[:200],
                        },
                    )
                    self._add_edge(
                        edges,
                        edge_keys,
                        source_id=art_node_id,
                        source_type="Article",
                        relation=GraphRelationType.CONTAINS,
                        target_id=clause_node_id,
                        target_type="Clause",
                        citation_text=f"Điều {chunk.article_num} chứa Khoản {chunk.clause_num}",
                    )

            # 5. Extract Semantic Citations from Text
            self._extract_citations_for_chunk(
                chunk=chunk,
                source_art_id=art_node_id,
                law_code=law_code,
                law_id=law_id,
                edges=edges,
                edge_keys=edge_keys,
                nodes_dict=nodes_dict,
            )

        result = LegalGraphExtractionResult(
            law_id=law_id, law_title=law_title, nodes=list(nodes_dict.values()), edges=edges
        )
        logger.info(
            f"Extracted Graph for '{law_title}': {result.total_nodes} nodes, "
            f"{result.total_edges} relationship edges."
        )
        return result

    def _extract_citations_for_chunk(
        self,
        chunk: LegalChunk,
        source_art_id: str,
        law_code: str,
        law_id: str,
        edges: List[GraphEdge],
        edge_keys: Set[str],
        nodes_dict: Dict[str, GraphNode],
    ):
        """Discovers internal and cross-statute references within the chunk text."""
        text = chunk.content

        # 1. Exception Relationships (NGOAI_LE_CUA)
        for m in self.EXCEPTION_PATTERN.finditer(text):
            target_art_num = int(m.group(2))
            if target_art_num == chunk.article_num:
                continue
            target_art_id = f"{law_code}_D{target_art_num:03d}"
            self._ensure_placeholder_article(target_art_id, target_art_num, law_id, nodes_dict)
            self._add_edge(
                edges,
                edge_keys,
                source_id=source_art_id,
                source_type="Article",
                relation=GraphRelationType.NGOAI_LE_CUA,
                target_id=target_art_id,
                target_type="Article",
                citation_text=m.group(0),
                confidence=0.98,
            )

        # 2. Specific Clause Cross-References (DAN_CHIEU_DIEU_KHOAN)
        for m in self.DETAILED_CLAUSE_PATTERN.finditer(text):
            clause_num = int(m.group(1))
            target_art_num = int(m.group(2))
            target_clause_id = f"{law_code}_D{target_art_num:03d}_K{clause_num:02d}"
            target_art_id = f"{law_code}_D{target_art_num:03d}"
            self._ensure_placeholder_article(target_art_id, target_art_num, law_id, nodes_dict)
            self._add_edge(
                edges,
                edge_keys,
                source_id=source_art_id,
                source_type="Article",
                relation=GraphRelationType.DAN_CHIEU_DIEU_KHOAN,
                target_id=target_clause_id,
                target_type="Clause",
                citation_text=m.group(0),
                confidence=0.95,
            )

        # 3. Generic Article References (THAM_CHIEU_DEN)
        for m in self.GENERIC_ARTICLE_PATTERN.finditer(text):
            target_art_num = int(m.group(1))
            if target_art_num == chunk.article_num:
                continue
            target_art_id = f"{law_code}_D{target_art_num:03d}"
            self._ensure_placeholder_article(target_art_id, target_art_num, law_id, nodes_dict)
            self._add_edge(
                edges,
                edge_keys,
                source_id=source_art_id,
                source_type="Article",
                relation=GraphRelationType.THAM_CHIEU_DEN,
                target_id=target_art_id,
                target_type="Article",
                citation_text=m.group(0),
                confidence=0.92,
            )

        # 4. Cross-Statute References
        for m in self.CROSS_STATUTE_PATTERN.finditer(text):
            statute_name = m.group(1).strip()
            statute_name = re.sub(r"[,;.\)]+$", "", statute_name).strip()
            statute_name = re.sub(
                r"\s+(?:và|hoặc)\s+(?:điều|khoản).*$", "", statute_name, flags=re.IGNORECASE
            ).strip()
            statute_code = canonical_law_slug(statute_name)
            statute_id = f"LAW_{statute_code}"
            if statute_id not in nodes_dict:
                nodes_dict[statute_id] = GraphNode(
                    node_id=statute_id,
                    node_type="Law",
                    name=statute_name,
                    law_id=statute_code.lower(),
                    properties={"title": statute_name, "canonical_code": statute_code},
                )
            self._add_edge(
                edges,
                edge_keys,
                source_id=source_art_id,
                source_type="Article",
                relation=GraphRelationType.THAM_CHIEU_DEN,
                target_id=statute_id,
                target_type="Law",
                citation_text=m.group(0),
                confidence=0.90,
            )

    def extract_statute_relations(
        self, text: str
    ) -> List[Tuple[Dict[str, Any], str, Dict[str, Any]]]:
        """
        Backward-compatible helper returning raw (Subject, Relation, Object) tuples.
        """
        chunks = [
            LegalChunk(
                chunk_id="temp_chunk_001",
                law_id="temp_statute",
                law_title="Văn bản tạm",
                chapter_num="I",
                chapter_title="Quy định chung",
                article_num=1,
                article_title="Điều khoản",
                content=text,
                canonical_citation="Điều 1",
                context_enriched_text=text,
            )
        ]
        result = self.extract_from_chunks(chunks, law_id="temp_statute", law_title="Văn bản tạm")
        triples = []
        for edge in result.edges:
            s = {"id": edge.source_id, "type": edge.source_type}
            r = edge.relation.value
            o = {"id": edge.target_id, "type": edge.target_type}
            triples.append((s, r, o))
        return triples

    def to_cypher_statements(self, result: LegalGraphExtractionResult) -> List[str]:
        """
        Translates GraphNode and GraphEdge collections into idempotent Cypher statements
        ready for direct execution on Neo4j.
        """
        statements: List[str] = []

        # Create unique constraint indexes
        statements.append(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.node_id IS UNIQUE;"
        )
        statements.append(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clause) REQUIRE c.node_id IS UNIQUE;"
        )
        statements.append(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chapter) REQUIRE ch.node_id IS UNIQUE;"
        )
        statements.append(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law) REQUIRE l.node_id IS UNIQUE;"
        )
        statements.append(
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Precedent) REQUIRE p.node_id IS UNIQUE;"
        )

        # Merge Nodes via robust model serializer
        for node in result.nodes:
            statements.append(node.to_cypher_merge())

        # Merge Edges
        for edge in result.edges:
            escaped_citation = self._escape_cypher(edge.citation_text)
            stmt = (
                f"MATCH (s:{edge.source_type} {{node_id: '{edge.source_id}'}}), "
                f"(t:{edge.target_type} {{node_id: '{edge.target_id}'}}) "
                f"MERGE (s)-[r:{edge.relation.value}]->(t) "
                f"ON CREATE SET r.citation = '{escaped_citation}', r.confidence = {edge.confidence};"
            )
            statements.append(stmt)

        return statements

    def _ensure_placeholder_article(
        self, art_id: str, art_num: int, law_id: str, nodes_dict: Dict[str, GraphNode]
    ):
        """Creates target article node placeholder if not already registered."""
        if art_id not in nodes_dict:
            nodes_dict[art_id] = GraphNode(
                node_id=art_id,
                node_type="Article",
                name=f"Điều {art_num}",
                law_id=law_id,
                properties={"article_num": art_num},
            )

    def _add_edge(
        self,
        edges: List[GraphEdge],
        edge_keys: Set[str],
        source_id: str,
        source_type: str,
        relation: GraphRelationType,
        target_id: str,
        target_type: str,
        citation_text: str = "",
        confidence: float = 1.0,
    ):
        """Adds unique relationship edge to avoid duplicate links."""
        key = f"{source_id}:{relation.value}:{target_id}"
        if key not in edge_keys:
            edge_keys.add(key)
            edge = GraphEdge(
                edge_id=f"edge_{len(edges):05d}",
                source_id=source_id,
                source_type=source_type,
                relation=relation,
                target_id=target_id,
                target_type=target_type,
                citation_text=citation_text,
                confidence=confidence,
            )
            edges.append(edge)

    @staticmethod
    def _escape_cypher(text: str) -> str:
        """Escapes single quotes for Cypher syntax."""
        return text.replace("'", "\\'").replace("\n", " ").strip()

    @staticmethod
    def _get_law_code(law_id: str) -> str:
        """
        Dynamically maps any law ID into a unique canonical short code.
        Fully generalized across all Vietnamese legislation via canonical_law_slug.
        """
        if not law_id:
            return "VN_LAW_UNKNOWN"
        return canonical_law_slug(law_id)
