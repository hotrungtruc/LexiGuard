"""
LexiGuard Core Data Schemas (Pydantic v2)
=========================================
Standardized models for Vietnamese Legal Documents:
- Hierarchical statutory representation (Law -> Part -> Chapter -> Section -> Article -> Clause -> Point)
- Qdrant Hybrid Search Chunks with metadata enrichment (Dense + Sparse)
- Knowledge Graph Models (Nodes, Edges, Triples, Cypher generation) for Neo4j GraphRAG
"""

import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, computed_field


def generate_deterministic_uuid(identifier: str) -> str:
    """
    Generate a deterministic UUIDv5 from an arbitrary legal identifier string.
    Ensures O(1) repeatable UUID point indexing in Qdrant matching Neo4j node IDs.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, identifier))


class LegalHierarchyLevel(str, Enum):
    """Statutory structural hierarchy levels according to VN Law on Promulgation of Legal Documents."""
    LAW = "LAW"
    PART = "PART"             # Phần
    CHAPTER = "CHAPTER"       # Chương
    SECTION = "SECTION"       # Mục
    SUB_SECTION = "SUB_SECTION" # Tiểu mục
    ARTICLE = "ARTICLE"       # Điều
    CLAUSE = "CLAUSE"         # Khoản
    POINT = "POINT"           # Điểm


class LegalPoint(BaseModel):
    """Represents a specific point (Điểm a, b, c...) within a statutory clause."""
    model_config = ConfigDict(extra="ignore")

    point_id: str = Field(description="Point identifier (e.g. 'a', 'b', 'c')")
    content: str = Field(description="Verbatim text of the point")
    canonical_citation: Optional[str] = Field(default=None, description="Full formal legal citation")


class LegalClause(BaseModel):
    """Represents an individual clause (Khoản 1, 2, 3...) within an article."""
    model_config = ConfigDict(extra="ignore")

    clause_num: int = Field(description="Clause number (1, 2, 3...)")
    content: str = Field(description="Full text of the clause including its points")
    points: List[LegalPoint] = Field(default_factory=list, description="Sub-points (Điểm) if present")
    canonical_citation: str = Field(description="e.g. 'Khoản 1 Điều 115 Bộ luật Lao động 2019'")


class LegalArticle(BaseModel):
    """Represents a statutory Article (Điều) containing clauses and terms."""
    model_config = ConfigDict(extra="ignore")

    article_num: int = Field(description="Article number (e.g. 115, 328)")
    title: str = Field(description="Title / heading of the article (e.g. 'Nghỉ lễ, tết', 'Đặt cọc')")
    full_text: str = Field(description="Verbatim combined text of all clauses in this article")
    clauses: List[LegalClause] = Field(default_factory=list, description="List of structured clauses")
    canonical_citation: str = Field(description="e.g. 'Điều 115 Bộ luật Lao động 2019'")
    page_start: Optional[int] = Field(default=None, description="Starting page in original document")
    page_end: Optional[int] = Field(default=None, description="Ending page in original document")


class LegalSection(BaseModel):
    """Represents a statutory Section (Mục)."""
    model_config = ConfigDict(extra="ignore")

    section_num: Optional[str] = Field(default=None, description="Section numeral (e.g. '1', '2')")
    title: str = Field(description="Section heading")
    articles: List[LegalArticle] = Field(default_factory=list)


class LegalChapter(BaseModel):
    """Represents a statutory Chapter (Chương)."""
    model_config = ConfigDict(extra="ignore")

    chapter_num: str = Field(description="Chapter numeral (e.g. 'I', 'II', 'VII')")
    title: str = Field(description="Chapter heading (e.g. 'HỢP ĐỒNG LAO ĐỘNG')")
    sections: List[LegalSection] = Field(default_factory=list)
    articles: List[LegalArticle] = Field(default_factory=list)


class LegalPart(BaseModel):
    """Represents a statutory Part (Phần)."""
    model_config = ConfigDict(extra="ignore")

    part_num: str = Field(description="Part numeral (e.g. 'THỨ NHẤT', 'I')")
    title: str = Field(description="Part heading (e.g. 'QUY ĐỊNH CHUNG')")
    chapters: List[LegalChapter] = Field(default_factory=list)


class LegalDocument(BaseModel):
    """Hierarchical structured representation of a complete Vietnamese code or statute."""
    model_config = ConfigDict(extra="ignore")

    law_id: str = Field(description="Unique code identifier (e.g. 'bo-luat-dan-su-2015')")
    title: str = Field(description="Official title (e.g. 'Bộ luật Dân sự 2015')")
    official_number: Optional[str] = Field(default=None, description="Official statute number (e.g. '91/2015/QH13')")
    promulgation_date: Optional[str] = Field(default=None, description="Date of enactment")
    effective_date: Optional[str] = Field(default=None, description="Effective date")
    parts: List[LegalPart] = Field(default_factory=list)
    chapters: List[LegalChapter] = Field(default_factory=list)
    total_articles: int = Field(default=0, description="Total count of parsed articles")
    source_file: str = Field(description="Source PDF filename")
    is_active: bool = Field(default=True, description="Statutory validity flag: False if the statute is repealed or expired")


# ==============================================================================
# QDRANT HYBRID RETRIEVAL CHUNK SCHEMA
# ==============================================================================

class LegalChunk(BaseModel):
    """
    Standard ingestion payload model stored in Qdrant Vector Database.
    Contains full parent hierarchy context to prevent semantic drifting when embedded with BGE-M3.
    """
    model_config = ConfigDict(extra="ignore")

    chunk_id: str = Field(
        description="Deterministic unique ID: {LAW_CODE}_D{ARTICLE:03d}_K{CLAUSE:02d}"
    )
    law_id: str = Field(description="Statute identifier (e.g. 'bo-luat-lao-dong-2019')")
    law_title: str = Field(description="Statute name (e.g. 'Bộ luật Lao động 2019')")
    part_num: Optional[str] = None
    part_title: Optional[str] = None
    chapter_num: str = Field(description="Chapter numeral (e.g. 'VII')")
    chapter_title: str = Field(description="Chapter name")
    section_num: Optional[str] = None
    section_title: Optional[str] = None
    article_num: int = Field(description="Article number")
    article_title: str = Field(description="Article heading")
    clause_num: Optional[int] = Field(default=None, description="Clause number if chunk is clause-level")
    point_id: Optional[str] = Field(default=None, description="Point id if point-level")

    content: str = Field(description="Exact verbatim legal text of this chunk")
    canonical_citation: str = Field(description="Canonical formal citation string")
    context_enriched_text: str = Field(
        description="Text prepended with full ancestor breadcrumb for superior dense embedding retrieval"
    )

    cross_references: List[str] = Field(
        default_factory=list,
        description="Extracted citations to other articles (e.g. ['Điều 112', 'Điều 115'])"
    )
    is_active: bool = Field(
        default=True,
        description="Statutory validity flag: False if the statute or provision is repealed or superseded"
    )
    page_start: int = Field(default=1, ge=1)
    page_end: int = Field(default=1, ge=1)
    bbox: Optional[List[float]] = Field(default=None, description="[left, top, right, bottom] coordinates")

    @computed_field
    @property
    def token_estimate(self) -> int:
        """Rough estimation of word/token count."""
        return len(self.content.split())

    @computed_field
    @property
    def qdrant_point_id(self) -> str:
        """Deterministic UUIDv5 compatible with Qdrant PointStruct ID requirements."""
        return generate_deterministic_uuid(self.chunk_id)

    @computed_field
    @property
    def neo4j_node_id(self) -> str:
        """Deterministic Node ID in Neo4j Knowledge Graph (1:1 with chunk_id)."""
        return self.chunk_id

    @computed_field
    @property
    def neo4j_node_type(self) -> str:
        """Target entity label in Neo4j."""
        if self.point_id:
            return "Point"
        elif self.clause_num:
            return "Clause"
        return "Article"

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert chunk into a standardized Qdrant payload dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "neo4j_node_id": self.neo4j_node_id,
            "neo4j_node_type": self.neo4j_node_type,
            "law_id": self.law_id,
            "law_title": self.law_title,
            "part_num": self.part_num,
            "part_title": self.part_title,
            "chapter_num": self.chapter_num,
            "chapter_title": self.chapter_title,
            "section_num": self.section_num,
            "section_title": self.section_title,
            "article_num": self.article_num,
            "article_title": self.article_title,
            "clause_num": self.clause_num,
            "point_id": self.point_id,
            "content": self.content,
            "canonical_citation": self.canonical_citation,
            "context_enriched_text": self.context_enriched_text,
            "cross_references": self.cross_references,
            "is_active": self.is_active,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "bbox": self.bbox,
            "token_estimate": self.token_estimate,
        }

    def to_neo4j_node(self) -> "GraphNode":
        """Convert this chunk into a corresponding Neo4j GraphNode."""
        return GraphNode(
            node_id=self.neo4j_node_id,
            node_type=self.neo4j_node_type,  # type: ignore[arg-type]
            name=self.canonical_citation,
            law_id=self.law_id,
            properties={
                "article_num": self.article_num,
                "clause_num": self.clause_num,
                "point_id": self.point_id,
                "title": self.article_title,
                "citation": self.canonical_citation,
                "content": self.content,
                "is_active": self.is_active,
                "qdrant_point_id": self.qdrant_point_id,
            },
        )


# ==============================================================================
# NEO4J KNOWLEDGE GRAPH SCHEMA
# ==============================================================================

class GraphRelationType(str, Enum):
    """Standardized relationship types for Legal Knowledge Graph."""
    CONTAINS = "CONTAINS"                       # Structural containment (Law -> Chapter -> Article -> Clause)
    REFERENCES = "REFERENCES"                   # Generic cross-reference (Căn cứ theo, Quy định tại...)
    EXCEPTION_TO = "EXCEPTION_TO"               # Exception / carve-out (Trừ trường hợp quy định tại...)
    CITES_CLAUSE = "CITES_CLAUSE"               # Specific clause reference (Chiếu theo Khoản X...)
    DEFINED_IN = "DEFINED_IN"                   # Definitional anchor (Khái niệm được xác định tại...)
    MANDATES_GUIDANCE = "MANDATES_GUIDANCE"     # Detailed specification / decree mandate (Quy định chi tiết...)
    APPLIES_STATUTE = "APPLIES_STATUTE"         # Precedent applies statutory article (Án lệ áp dụng điều luật)
    HAS_PRECEDENT = "HAS_PRECEDENT"             # Article has applicable judicial precedent (Có án lệ liên quan)

    # Backward-compatible aliases for legacy references
    THAM_CHIEU_DEN = "REFERENCES"
    NGOAI_LE_CUA = "EXCEPTION_TO"
    DAN_CHIEU_DIEU_KHOAN = "CITES_CLAUSE"
    DINH_NGHIA_TAI = "DEFINED_IN"
    HUONG_DAN_THI_HANH = "MANDATES_GUIDANCE"
    AP_DUNG_LUAT = "APPLIES_STATUTE"
    CO_AN_LE = "HAS_PRECEDENT"


class GraphNode(BaseModel):
    """Node entity in Neo4j Knowledge Graph."""
    model_config = ConfigDict(extra="ignore")

    node_id: str = Field(description="Unique deterministic identifier (e.g. 'VN_BLLD_2019_D115')")
    node_type: Literal["Law", "Part", "Chapter", "Section", "Article", "Clause", "Point", "Precedent", "LegalConcept"] = Field(
        description="Entity label in Neo4j"
    )
    name: str = Field(description="Human readable display name")
    law_id: str = Field(description="Belonging statute identifier")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary attributes for Cypher indexing")

    @computed_field
    @property
    def qdrant_point_id(self) -> str:
        """Deterministic UUIDv5 corresponding to this node for Qdrant sync."""
        return generate_deterministic_uuid(self.node_id)

    def to_cypher_merge(self) -> str:
        """Generate Cypher MERGE clause for idempotent node creation."""
        props = dict(self.properties)
        props["name"] = self.name
        props["law_id"] = self.law_id
        props["qdrant_point_id"] = self.qdrant_point_id

        # Format properties as Cypher map literal
        set_statements = []
        for k, v in props.items():
            if v is not None:
                if isinstance(v, str):
                    clean_v = v.replace("'", "\\'").replace("\n", " ")
                    set_statements.append(f"n.{k} = '{clean_v}'")
                elif isinstance(v, bool):
                    set_statements.append(f"n.{k} = {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    set_statements.append(f"n.{k} = {v}")

        set_clause = ", ".join(set_statements)
        if set_clause:
            return f"MERGE (n:{self.node_type} {{node_id: '{self.node_id}'}}) ON CREATE SET {set_clause};"
        return f"MERGE (n:{self.node_type} {{node_id: '{self.node_id}'}});"


class GraphEdge(BaseModel):
    """Directed relationship between two legal entities in Neo4j."""
    model_config = ConfigDict(extra="ignore")

    edge_id: str = Field(description="Unique relationship ID: {SOURCE}_{REL}_{TARGET}")
    source_id: str = Field(description="Origin node ID")
    source_type: str = Field(description="Origin node label")
    relation: GraphRelationType = Field(description="Semantic relationship type")
    target_id: str = Field(description="Destination node ID")
    target_type: str = Field(description="Destination node label")
    citation_text: str = Field(default="", description="Verbatim quote or excerpt establishing the reference")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class LegalGraphExtractionResult(BaseModel):
    """Complete extracted graph representation ready for batch ingestion into Neo4j."""
    model_config = ConfigDict(extra="ignore")

    law_id: str
    law_title: str
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)

    @computed_field
    @property
    def total_nodes(self) -> int:
        return len(self.nodes)

    @computed_field
    @property
    def total_edges(self) -> int:
        return len(self.edges)


class LegalPrecedent(BaseModel):
    """
    Represents a judicial precedent or court judgment (Án lệ / Bản án thực tế)
    in the Vietnamese Legal System, linked to statutory articles in Neo4j.
    """
    model_config = ConfigDict(extra="ignore")

    precedent_id: str = Field(description="Deterministic ID (e.g. 'VN_AL_01_2016')")
    precedent_number: str = Field(description="Official precedent identifier (e.g. 'Án lệ số 01/2016/AL')")
    title: str = Field(description="Case title / topic heading")
    court: str = Field(
        default="Hội đồng Thẩm phán Tòa án nhân dân tối cao",
        description="Adjudicating court body"
    )
    promulgation_date: Optional[str] = Field(default=None, description="Date of precedent adoption")
    case_summary: str = Field(description="Tóm tắt tình huống án lệ thực tế")
    legal_rule: str = Field(description="Khái quát nội dung quy tắc án lệ (Ratio Decidendi)")
    applied_statutes: List[str] = Field(
        default_factory=list,
        description="Node IDs of statutory articles applied (e.g. ['VN_BLDS_2015_D004'])"
    )

    def to_graph_node(self) -> GraphNode:
        """Convert into a Neo4j GraphNode labeled :Precedent."""
        return GraphNode(
            node_id=self.precedent_id,
            node_type="Precedent",
            name=f"{self.precedent_number}: {self.title}",
            law_id="an-le-viet-nam",
            properties={
                "precedent_number": self.precedent_number,
                "title": self.title,
                "court": self.court,
                "case_summary": self.case_summary,
                "legal_rule": self.legal_rule,
                "applied_statutes": self.applied_statutes,
                "promulgation_date": self.promulgation_date,
            },
        )


# ==============================================================================
# UNIFIED SYNCHRONIZED SCHEMA (QDRANT VECTOR DB <-> NEO4J KNOWLEDGE GRAPH)
# ==============================================================================

# Qdrant Hybrid Collection Constants
QDRANT_COLLECTION_NAME = "legal_statutes_corpus"
DENSE_VECTOR_NAME = "bge_m3_dense"
DENSE_VECTOR_DIM = 1024  # BAAI/bge-m3 dense dimension
SPARSE_VECTOR_NAME = "bm25_sparse"


class SynchronizedLegalEntity(BaseModel):
    """
    Unified Synchronized Schema bridging Qdrant Vector Payload and Neo4j Graph Node.
    Guarantees deterministic, bi-directional O(1) alignment between semantic embeddings and ontological graph:
    - entity_id: Deterministic statutory identifier (e.g. 'VN_BLDS_2015_D004_K02')
    - point_uuid: Deterministic UUIDv5 for Qdrant PointStruct ID
    - entity_type: Neo4j Node label ("Article", "Clause", "Point", "Precedent", etc.)
    """
    model_config = ConfigDict(extra="ignore")

    entity_id: str = Field(description="Deterministic ID (e.g. 'VN_BLDS_2015_D004_K02')")
    point_uuid: str = Field(description="Deterministic UUIDv5 for Qdrant PointStruct ID")
    entity_type: Literal["Law", "Part", "Chapter", "Section", "Article", "Clause", "Point", "Precedent", "LegalConcept"] = Field(
        default="Clause", description="Entity type / Neo4j Label"
    )

    # Statutory Hierarchy
    law_id: str
    law_title: str
    part_num: Optional[str] = None
    part_title: Optional[str] = None
    chapter_num: Optional[str] = None
    chapter_title: Optional[str] = None
    section_num: Optional[str] = None
    section_title: Optional[str] = None
    article_num: int
    article_title: str
    clause_num: Optional[int] = None
    point_id: Optional[str] = None

    # Text & Embeddings Grounding
    content: str
    canonical_citation: str
    context_enriched_text: str

    # Graph Relations & Cross References
    cross_references: List[str] = Field(default_factory=list)
    page_start: int = 1
    page_end: int = 1
    bbox: Optional[List[float]] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Generate payload for storage in Qdrant PointStruct."""
        return {
            "chunk_id": self.entity_id,
            "neo4j_node_id": self.entity_id,
            "neo4j_node_type": self.entity_type,
            "law_id": self.law_id,
            "law_title": self.law_title,
            "part_num": self.part_num,
            "part_title": self.part_title,
            "chapter_num": self.chapter_num,
            "chapter_title": self.chapter_title,
            "section_num": self.section_num,
            "section_title": self.section_title,
            "article_num": self.article_num,
            "article_title": self.article_title,
            "clause_num": self.clause_num,
            "point_id": self.point_id,
            "content": self.content,
            "canonical_citation": self.canonical_citation,
            "context_enriched_text": self.context_enriched_text,
            "cross_references": self.cross_references,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "bbox": self.bbox,
        }

    def to_graph_node(self) -> GraphNode:
        """Generate corresponding GraphNode for Neo4j Knowledge Graph."""
        props = dict(self.properties)
        props.update({
            "article_num": self.article_num,
            "clause_num": self.clause_num,
            "point_id": self.point_id,
            "title": self.article_title,
            "citation": self.canonical_citation,
            "content": self.content,
            "qdrant_point_id": self.point_uuid,
        })
        return GraphNode(
            node_id=self.entity_id,
            node_type=self.entity_type,
            name=self.canonical_citation,
            law_id=self.law_id,
            properties=props,
        )

    @classmethod
    def from_legal_chunk(cls, chunk: LegalChunk) -> "SynchronizedLegalEntity":
        """Construct synchronized entity from a LegalChunk."""
        return cls(
            entity_id=chunk.chunk_id,
            point_uuid=chunk.qdrant_point_id,
            entity_type=chunk.neo4j_node_type,  # type: ignore[arg-type]
            law_id=chunk.law_id,
            law_title=chunk.law_title,
            part_num=chunk.part_num,
            part_title=chunk.part_title,
            chapter_num=chunk.chapter_num,
            chapter_title=chunk.chapter_title,
            section_num=chunk.section_num,
            section_title=chunk.section_title,
            article_num=chunk.article_num,
            article_title=chunk.article_title,
            clause_num=chunk.clause_num,
            point_id=chunk.point_id,
            content=chunk.content,
            canonical_citation=chunk.canonical_citation,
            context_enriched_text=chunk.context_enriched_text,
            cross_references=chunk.cross_references,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            bbox=chunk.bbox,
        )
