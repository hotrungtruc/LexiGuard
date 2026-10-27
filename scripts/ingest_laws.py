"""
Script: ingest_laws.py
======================
End-to-End Legal Corpus Extraction Pipeline using Branch 1 (IBM Docling):
1. Ingest raw law scanned PDFs strictly from data/raw_laws/ using Branch 1 (IBM Docling OCR).
2. Hierarchical Chunking (Chương -> Mục -> Điều -> Khoản) with context enrichment for Qdrant.
3. Graph Extraction (Triples: CONTAINS, THAM_CHIEU_DEN, NGOAI_LE_CUA, DAN_CHIEU_DIEU_KHOAN) for Neo4j.
4. Export structured artifacts to data/extracted_laws/ ready for Phase 2 (Hybrid GraphRAG).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

# Configure UTF-8 for console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from neo4j import AsyncGraphDatabase
from src.core.config import settings
from src.core.schemas import LegalChunk, LegalDocument, LegalGraphExtractionResult, LegalPrecedent
from src.ingestion.docling_parser import DoclingParser
from src.ingestion.legal_chunker import LegalChunker
from src.ingestion.graph_extractor import LegalGraphExtractor


from src.core.legal_norm import canonical_law_slug


RAW_LAWS_CONFIG = [
    {
        "filename": "bo-luat-dan-su-2015.pdf",
        "law_id": "bo-luat-dan-su-2015",
        "law_title": "Bộ luật Dân sự 2015",
        "code": "VN_BLDS_2015"
    },
    {
        "filename": "bo-luat-lao-dong-2019.pdf",
        "law_id": "bo-luat-lao-dong-2019",
        "law_title": "Bộ luật Lao động 2019",
        "code": "VN_BLLD_2019"
    }
]


def ingest_single_law(
    config: dict,
    parser: DoclingParser,
    chunker: LegalChunker,
    extractor: LegalGraphExtractor,
    output_dir: Path,
    max_pages: Optional[int] = None,
    batch_size: Optional[int] = 15
) -> Dict[str, Any]:
    """
    Processes a single raw law PDF through GPU-accelerated OCR layout extraction.
    Fully domain-agnostic: supports any Vietnamese statute, decree, circular, or rulebook.
    """
    raw_dir = Path("data/raw_laws")
    raw_filename = config.get("filename", "")
    target_path = Path(raw_filename)
    if target_path.exists():
        pdf_path = target_path
    elif (raw_dir / raw_filename).exists():
        pdf_path = raw_dir / raw_filename
    else:
        pdf_path = target_path

    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing required raw law PDF: {pdf_path}")

    law_id = config.get("law_id") or pdf_path.stem
    law_title = config.get("law_title") or law_id.replace("-", " ").title()
    code = config.get("code") or canonical_law_slug(law_id)
    config["law_id"] = law_id
    config["law_title"] = law_title
    config["code"] = code

    page_budget_desc = f"{max_pages} pages" if max_pages and max_pages > 0 else "FULL DOCUMENT (All Pages)"
    logger.info(f"===> Ingesting [{config['law_title']}] (Code: {code})...")
    logger.info(f"Target file: {pdf_path.name} | Page processing budget: {page_budget_desc} | Batch size: {batch_size}")

    # 1. OCR Layout Parsing (GPU/CUDA if available, else CPU)
    pages_arg = max_pages if max_pages and max_pages > 0 else None
    parsed_doc = parser.parse(pdf_path, max_pages=pages_arg, batch_size=batch_size)
    markdown_text = parsed_doc.full_markdown

    logger.info(
        f"Extracted {len(markdown_text)} characters, "
        f"{len(parsed_doc.elements)} layout elements across {parsed_doc.total_pages} pages."
    )

    # 2. Hierarchical Chunking (Chương -> Điều -> Khoản)
    chunks: List[LegalChunk] = chunker.chunk_statute(
        markdown_text,
        law_id=config["law_id"],
        law_title=config["law_title"]
    )

    # Build hierarchical LegalDocument AST
    doc_ast: LegalDocument = chunker.parse_statute(
        markdown_text,
        law_id=config["law_id"],
        law_title=config["law_title"],
        source_file=config["filename"]
    )

    # 3. Knowledge Graph Extraction (Neo4j Triples)
    graph_result: LegalGraphExtractionResult = extractor.extract_from_chunks(
        chunks,
        law_id=config["law_id"],
        law_title=config["law_title"]
    )
    cypher_queries = extractor.to_cypher_statements(graph_result)

    # 4. Export Artifacts
    stem = Path(config["filename"]).stem

    # Raw Markdown
    md_file = output_dir / f"{stem}.md"
    md_file.write_text(markdown_text, encoding="utf-8")

    # Hierarchical AST
    ast_file = output_dir / f"{stem}_ast.json"
    ast_file.write_text(doc_ast.model_dump_json(indent=2), encoding="utf-8")

    # Qdrant Chunks
    chunks_file = output_dir / f"{stem}_chunks.json"
    chunks_data = [c.model_dump() for c in chunks]
    chunks_file.write_text(json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Neo4j Graph
    graph_file = output_dir / f"{stem}_graph.json"
    graph_data = {
        "law_id": config["law_id"],
        "law_title": config["law_title"],
        "nodes": [n.model_dump() for n in graph_result.nodes],
        "edges": [e.model_dump() for e in graph_result.edges],
        "cypher_statements": cypher_queries
    }
    graph_file.write_text(json.dumps(graph_data, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        f"[OK] Ingestion completed for {config['law_title']}: "
        f"{len(chunks)} chunks, {graph_result.total_nodes} nodes, {graph_result.total_edges} edges across {len(doc_ast.chapters)} chapters."
    )

    return {
        "law_id": config["law_id"],
        "law_title": config["law_title"],
        "chunks": chunks,
        "graph_result": graph_result,
        "cypher_queries": cypher_queries
    }


async def sync_to_neo4j(graph_results: List[LegalGraphExtractionResult]):
    """Synchronizes extracted graph entities and relationships into live Neo4j."""
    logger.info("Connecting to Neo4j to synchronize Legal Knowledge Graph...")
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    async with driver.session() as session:
        # 1. Uniqueness constraints
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law) REQUIRE l.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chapter) REQUIRE ch.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clause) REQUIRE c.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Precedent) REQUIRE p.node_id IS UNIQUE",
        ]
        for c in constraints:
            await session.run(c)

        total_nodes = 0
        total_edges = 0

        for gr in graph_results:
            # 1. Insert Nodes by type in batches using UNWIND
            nodes_by_type: Dict[str, List[dict]] = {}
            for node in gr.nodes:
                props = dict(node.properties)
                props["name"] = node.name
                props["law_id"] = node.law_id
                props["node_id"] = node.node_id
                if "is_active" not in props:
                    props["is_active"] = getattr(node, "is_active", True)
                nodes_by_type.setdefault(node.node_type, []).append({
                    "node_id": node.node_id,
                    "props": props
                })

            for node_type, node_list in nodes_by_type.items():
                batch_size = 500
                for i in range(0, len(node_list), batch_size):
                    batch = node_list[i:i + batch_size]
                    cypher = f"""
                    UNWIND $batch AS item
                    MERGE (n:{node_type} {{node_id: item.node_id}})
                    SET n += item.props
                    """
                    await session.run(cypher, {"batch": batch})
                    total_nodes += len(batch)

            # 2. Insert Edges by relation type in batches using UNWIND
            edges_by_rel: Dict[str, List[dict]] = {}
            for edge in gr.edges:
                rel_str = edge.relation.value if hasattr(edge.relation, "value") else str(edge.relation)
                edges_by_rel.setdefault(rel_str, []).append({
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "citation": edge.citation_text,
                    "confidence": edge.confidence,
                })

            for rel_str, edge_list in edges_by_rel.items():
                batch_size = 500
                for i in range(0, len(edge_list), batch_size):
                    batch = edge_list[i:i + batch_size]
                    cypher = f"""
                    UNWIND $batch AS item
                    MATCH (s {{node_id: item.source_id}}), (t {{node_id: item.target_id}})
                    MERGE (s)-[r:{rel_str}]->(t)
                    SET r.citation = item.citation, r.confidence = item.confidence
                    """
                    await session.run(cypher, {"batch": batch})
                    total_edges += len(batch)

        # 3. Insert Judicial Precedents
        precedents = [
            LegalPrecedent(
                precedent_id="VN_AL_01_2016",
                precedent_number="Án lệ số 01/2016/AL",
                title="Vụ án tranh chấp hợp đồng mua bán tài sản và thỏa thuận phạt cọc",
                court="Hội đồng Thẩm phán Tòa án nhân dân tối cao",
                promulgation_date="2016-04-06",
                case_summary="Tranh chấp về hiệu lực thỏa thuận phạt cọc và nghĩa vụ bồi thường thiệt hại khi một bên từ chối thực hiện hợp đồng.",
                legal_rule="Nếu bên đặt cọc từ chối giao kết hoặc thực hiện hợp đồng thì tài sản đặt cọc thuộc về bên nhận đặt cọc theo quy định về đặt cọc.",
                applied_statutes=["VN_BLDS_2015_D003", "VN_BLDS_2015_D004", "VN_BLDS_2015_D002"]
            ),
            LegalPrecedent(
                precedent_id="VN_AL_09_2016",
                precedent_number="Án lệ số 09/2016/AL",
                title="Vụ án tranh chấp hợp đồng vay tài sản và nghĩa vụ trả lãi quá hạn",
                court="Hội đồng Thẩm phán Tòa án nhân dân tối cao",
                promulgation_date="2016-10-17",
                case_summary="Xác định lãi suất nợ quá hạn và phương thức tính lãi trên nợ gốc chưa thanh toán.",
                legal_rule="Bên vay chậm trả tiền vay phải trả lãi trên nợ gốc theo lãi suất thỏa thuận hoặc trần quy định của Bộ luật Dân sự.",
                applied_statutes=["VN_BLDS_2015_D003", "VN_BLDS_2015_D005"]
            ),
            LegalPrecedent(
                precedent_id="VN_AL_46_2021",
                precedent_number="Án lệ số 46/2021/AL",
                title="Vụ án tranh chấp đơn phương chấm dứt hợp đồng lao động trái pháp luật",
                court="Tòa án nhân dân cấp cao tại Hà Nội",
                promulgation_date="2021-11-25",
                case_summary="Xác định tính bất hợp pháp của hành vi đơn phương chấm dứt HĐLĐ mà không tuân thủ thời hạn báo trước hoặc căn cứ theo quy định của BLLD.",
                legal_rule="Người sử dụng lao động đơn phương chấm dứt hợp đồng lao động trái pháp luật phải bồi thường tiền lương và nhận người lao động trở lại làm việc.",
                applied_statutes=["VN_BLLD_2019_D005", "VN_BLLD_2019_D006", "VN_BLLD_2019_D007"]
            )
        ]

        for p in precedents:
            p_cypher = """
            MERGE (prec:Precedent {node_id: $precedent_id})
            SET prec.precedent_number = $precedent_number,
                prec.title = $title,
                prec.court = $court,
                prec.promulgation_date = $promulgation_date,
                prec.case_summary = $case_summary,
                prec.legal_rule = $legal_rule,
                prec.name = $title,
                prec.is_active = true
            """
            await session.run(p_cypher, {
                "precedent_id": p.precedent_id,
                "precedent_number": p.precedent_number,
                "title": p.title,
                "court": p.court,
                "promulgation_date": p.promulgation_date,
                "case_summary": p.case_summary,
                "legal_rule": p.legal_rule
            })

            for stat_id in p.applied_statutes:
                rel_cypher = """
                MATCH (prec:Precedent {node_id: $precedent_id}), (art:Article {node_id: $stat_id})
                MERGE (prec)-[:APPLIES_STATUTE]->(art)
                """
                await session.run(rel_cypher, {
                    "precedent_id": p.precedent_id,
                    "stat_id": stat_id
                })

        # Summary verification
        label_res = await session.run("MATCH (n) RETURN labels(n)[0] as label, count(n) as cnt ORDER BY cnt DESC")
        label_counts = {record["label"]: record["cnt"] async for record in label_res}
        rel_res = await session.run("MATCH ()-[r]->() RETURN type(r) as rel, count(r) as cnt ORDER BY cnt DESC")
        rel_counts = {record["rel"]: record["cnt"] async for record in rel_res}

        logger.info(f"Neo4j Knowledge Graph Ingestion Finished!")
        logger.info(f"Current Neo4j Labels: {label_counts}")
        logger.info(f"Current Neo4j Relationships: {rel_counts}")

    await driver.close()


async def sync_to_qdrant(chunks: List[LegalChunk]):
    """Syncs extracted legal chunks to Qdrant vector database."""
    try:
        from src.retrieval.qdrant_store import QdrantStore
        logger.info(f"Connecting to Qdrant to upsert {len(chunks)} chunks with hybrid embeddings...")
        store = QdrantStore()
        await store.ensure_collection_exists()
        count = await store.async_upsert_chunks(chunks)
        logger.info(f"Successfully upserted {count} chunks into Qdrant collection '{store.collection_name}'.")
        await store.client.close()
    except Exception as q_err:
        logger.warning(f"Qdrant sync encountered an error: {q_err}")


def main():
    parser_args = argparse.ArgumentParser(description="Ingest Vietnamese Legal Codes into Chunks and Graph")
    parser_args.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Number of pages to parse per raw law PDF (0 or omitted = parse ALL pages of the entire code)"
    )
    parser_args.add_argument(
        "--all-pages",
        action="store_true",
        help="Explicitly parse all pages of both legal codes"
    )
    parser_args.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Accelerator device: auto (detect GPU/CUDA), cuda (force GPU), or cpu (force CPU)"
    )
    parser_args.add_argument(
        "--engine",
        type=str,
        default="easyocr",
        choices=["easyocr", "docling", "rapidocr"],
        help="OCR engine: easyocr (standard GPU OCR with full Vietnamese diacritics), docling, or rapidocr"
    )
    parser_args.add_argument(
        "--batch-size",
        type=int,
        default=15,
        help="Page batch size for incremental OCR processing and progress tracking (default: 15 pages)"
    )
    parser_args.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Path to an arbitrary legal statute PDF (e.g. data/raw_laws/luat-an-ninh-mang-2018.pdf)"
    )
    parser_args.add_argument(
        "--law-title",
        type=str,
        default=None,
        help="Human-readable title of the legal instrument (e.g. 'Luật An ninh mạng 2018')"
    )
    parser_args.add_argument(
        "--law-id",
        type=str,
        default=None,
        help="Custom statutory identifier slug (e.g. 'luat-an-ninh-mang-2018')"
    )
    parser_args.add_argument(
        "--scan-all-raw-laws",
        action="store_true",
        help="Scan and ingest ALL PDF statutes found inside data/raw_laws/"
    )
    parser_args.add_argument(
        "--no-neo4j-sync",
        action="store_true",
        help="Skip direct synchronization to Neo4j database"
    )
    parser_args.add_argument(
        "--no-qdrant-sync",
        action="store_true",
        help="Skip direct synchronization to Qdrant database"
    )
    args = parser_args.parse_args()

    max_pages = None if (args.all_pages or args.max_pages == 0) else args.max_pages

    # Determine list of statutory configurations to ingest
    configs_to_ingest: List[Dict[str, Any]] = []
    if args.pdf:
        pdf_path = Path(args.pdf)
        configs_to_ingest.append({
            "filename": str(pdf_path),
            "law_title": args.law_title or pdf_path.stem.replace("-", " ").title(),
            "law_id": args.law_id or pdf_path.stem,
            "code": canonical_law_slug(args.law_id or pdf_path.stem),
        })
    elif args.scan_all_raw_laws:
        raw_dir = Path("data/raw_laws")
        for p in raw_dir.glob("*.pdf"):
            configs_to_ingest.append({
                "filename": p.name,
                "law_title": p.stem.replace("-", " ").title(),
                "law_id": p.stem,
                "code": canonical_law_slug(p.stem),
            })
    else:
        configs_to_ingest = list(RAW_LAWS_CONFIG)

    logger.info("=================================================================")
    logger.info("Starting LexiGuard Full Legal Corpus Ingestion Pipeline...")
    logger.info(f"Accelerator Device: {args.device} | OCR Engine: {args.engine} | Batch Size: {args.batch_size}")
    logger.info(f"Page Scope: {'ALL PAGES (Complete Statutory Codes)' if max_pages is None else f'{max_pages} pages'}")
    logger.info(f"Target Statutes ({len(configs_to_ingest)}): {[c['law_title'] for c in configs_to_ingest]}")
    logger.info("=================================================================")

    output_dir = Path("data/extracted_laws")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Parser with selected device and engine (auto hardware tuned)
    docling_parser = DoclingParser(
        do_ocr=True,
        do_table_structure=False,
        ocr_engine=args.engine,
        device=args.device,
        num_threads=None,
    )
    chunker = LegalChunker()
    extractor = LegalGraphExtractor()

    all_chunks: List[LegalChunk] = []
    all_cypher: List[str] = []
    all_graph_results: List[LegalGraphExtractionResult] = []
    total_nodes = 0
    total_edges = 0

    for cfg in configs_to_ingest:
        res = ingest_single_law(
            config=cfg,
            parser=docling_parser,
            chunker=chunker,
            extractor=extractor,
            output_dir=output_dir,
            max_pages=max_pages,
            batch_size=args.batch_size
        )
        all_chunks.extend(res["chunks"])
        all_cypher.extend(res["cypher_queries"])
        all_graph_results.append(res["graph_result"])
        total_nodes += res["graph_result"].total_nodes
        total_edges += res["graph_result"].total_edges

    # Save Combined Deliverables
    combined_chunks_path = output_dir / "combined_qdrant_chunks.json"
    combined_chunks_data = [c.model_dump() for c in all_chunks]
    combined_chunks_path.write_text(
        json.dumps(combined_chunks_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    combined_cypher_path = output_dir / "neo4j_import.cypher"
    combined_cypher_path.write_text("\n".join(all_cypher), encoding="utf-8")

    logger.info("=================================================================")
    logger.info("ALL LEGAL CORPORA INGESTED SUCCESSFULLY!")
    logger.info(f"Total Unified Chunks for Qdrant: {len(all_chunks)}")
    logger.info(f"Total Unified Graph Nodes:       {total_nodes}")
    logger.info(f"Total Unified Graph Edges:       {total_edges}")
    logger.info(f"Combined Qdrant Payload:         {combined_chunks_path}")
    logger.info(f"Combined Neo4j Cypher Script:    {combined_cypher_path}")
    logger.info("=================================================================")

    # Sync into Neo4j
    if not args.no_neo4j_sync:
        try:
            asyncio.run(sync_to_neo4j(all_graph_results))
        except Exception as neo_err:
            logger.error(f"Failed to synchronize to Neo4j: {neo_err}")

    # Sync into Qdrant
    if not args.no_qdrant_sync:
        try:
            asyncio.run(sync_to_qdrant(all_chunks))
        except Exception as q_err:
            logger.error(f"Failed to synchronize to Qdrant: {q_err}")


if __name__ == "__main__":
    main()
