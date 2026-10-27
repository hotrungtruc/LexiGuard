"""
Script: reset_databases.py
===========================
Wipes and re-initializes both Neo4j Knowledge Graph and Qdrant Vector Store
to a clean, empty state ready for fresh ingestion.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from neo4j import AsyncGraphDatabase
from src.core.config import settings
from src.retrieval.qdrant_store import QdrantStore


async def reset_neo4j():
    """Wipes all nodes and relationships in Neo4j and recreates constraints."""
    logger.info("Connecting to Neo4j to reset graph...")
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
    )
    async with driver.session() as session:
        # Check count before wipe
        res = await session.run("MATCH (n) RETURN count(n) as cnt")
        rec = await res.single()
        cnt_before = rec["cnt"] if rec else 0
        logger.info(f"Neo4j nodes before wipe: {cnt_before}")

        # Delete all nodes & relationships
        await session.run("MATCH (n) DETACH DELETE n")

        # Verify
        res = await session.run("MATCH (n) RETURN count(n) as cnt")
        rec = await res.single()
        cnt_after = rec["cnt"] if rec else 0
        logger.info(f"Neo4j nodes after wipe: {cnt_after}")

        # Re-apply constraints
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Law) REQUIRE l.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Chapter) REQUIRE ch.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Clause) REQUIRE c.node_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Precedent) REQUIRE p.node_id IS UNIQUE",
        ]
        for c in constraints:
            await session.run(c)
        logger.info("Neo4j unique constraints verified.")

    await driver.close()
    logger.info("[OK] Neo4j reset successfully.")


async def reset_qdrant():
    """Wipes and recreates the Qdrant hybrid vector collection with payload indexes."""
    logger.info("Connecting to Qdrant to reset vector collection...")
    store = QdrantStore()

    collections_res = await store.client.get_collections()
    existing_names = [col.name for col in collections_res.collections]
    logger.info(f"Existing Qdrant collections: {existing_names}")

    if store.collection_name in existing_names:
        cnt_before = await store.client.count(store.collection_name)
        logger.info(f"Points before wipe in '{store.collection_name}': {cnt_before.count}")
        await store.client.delete_collection(store.collection_name)
        logger.info(f"Deleted collection '{store.collection_name}'.")

    # Recreate collection with clean hybrid schema & payload indexes
    await store.ensure_collection_exists()
    cnt_after = await store.client.count(store.collection_name)
    logger.info(f"Points after recreation in '{store.collection_name}': {cnt_after.count}")

    await store.client.close()
    logger.info("[OK] Qdrant collection reset successfully.")


def clean_extracted_artifacts():
    """Cleans files in data/extracted_laws to guarantee no stale files remain."""
    out_dir = Path("data/extracted_laws")
    if out_dir.exists():
        cleaned_count = 0
        for f in out_dir.iterdir():
            if f.is_file():
                f.unlink()
                cleaned_count += 1
        logger.info(f"Cleaned {cleaned_count} old artifact file(s) from {out_dir}.")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created clean output directory: {out_dir}")


async def main():
    logger.info("==================================================")
    logger.info("STARTING COMPLETE DATABASE & ARTIFACTS PURGE...")
    logger.info("==================================================")
    await reset_neo4j()
    await reset_qdrant()
    clean_extracted_artifacts()
    logger.info("==================================================")
    logger.info("ALL DATABASES & EXTRACTED ARTIFACTS PURGED CLEANLY!")
    logger.info("==================================================")


if __name__ == "__main__":
    asyncio.run(main())
