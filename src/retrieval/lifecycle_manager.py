"""
Legal Lifecycle Manager: Synchronized Statute Validity & Repeal Engine
======================================================================
Coordinates lifecycle status (is_active=True/False) across both:
1. Qdrant Vector Database (Payload field: is_active)
2. Neo4j Knowledge Graph (Node property: is_active)

Ensures that when a statute or legal code expires or is repealed,
the retirement is executed in lockstep across both channels,
enabling automatic filtering during retrieval.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from src.retrieval.neo4j_store import Neo4jStore
from src.retrieval.qdrant_store import QdrantStore


class LegalLifecycleManager:
    """
    Orchestrates synchronized validity updates across Qdrant and Neo4j.
    """

    def __init__(
        self,
        qdrant_store: Optional[QdrantStore] = None,
        neo4j_store: Optional[Neo4jStore] = None,
    ):
        self.qdrant_store: QdrantStore = qdrant_store or QdrantStore()
        self.neo4j_store: Neo4jStore = neo4j_store or Neo4jStore()

    async def deactivate_statute_async(self, law_id: str) -> Dict[str, Any]:
        """
        Retires a statute (marks is_active=False) synchronously across Qdrant and Neo4j.
        """
        return await self.sync_statute_lifecycle_async(law_id=law_id, is_active=False)

    async def activate_statute_async(self, law_id: str) -> Dict[str, Any]:
        """
        Activates a statute (marks is_active=True) synchronously across Qdrant and Neo4j.
        """
        return await self.sync_statute_lifecycle_async(law_id=law_id, is_active=True)

    async def sync_statute_lifecycle_async(
        self,
        law_id: str,
        is_active: bool = False,
    ) -> Dict[str, Any]:
        """
        Synchronously updates `is_active` status across both Qdrant Payload and Neo4j Knowledge Graph.
        """
        logger.info(f"Synchronizing lifecycle status for statute '{law_id}': is_active={is_active}")
        qdrant_ok = False
        neo4j_count = 0
        errors: List[str] = []

        # 1. Update Qdrant
        try:
            qdrant_ok = await self.qdrant_store.update_statute_active_status_async(
                law_id, is_active
            )
        except Exception as exc:
            logger.error(f"Failed to update Qdrant lifecycle status for '{law_id}': {exc}")
            errors.append(f"Qdrant error: {exc}")

        # 2. Update Neo4j
        try:
            neo4j_count = await self.neo4j_store.update_statute_active_status_async(
                law_id, is_active
            )
        except Exception as exc:
            logger.error(f"Failed to update Neo4j lifecycle status for '{law_id}': {exc}")
            errors.append(f"Neo4j error: {exc}")

        success = len(errors) == 0 and qdrant_ok
        return {
            "law_id": law_id,
            "is_active": is_active,
            "qdrant_updated": qdrant_ok,
            "neo4j_nodes_updated": neo4j_count,
            "success": success,
            "errors": errors,
        }

    def sync_statute_lifecycle(
        self,
        law_id: str,
        is_active: bool = False,
    ) -> Dict[str, Any]:
        """Synchronous wrapper for sync_statute_lifecycle_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(
                        asyncio.run,
                        self.sync_statute_lifecycle_async(law_id=law_id, is_active=is_active),
                    ).result()
            return loop.run_until_complete(
                self.sync_statute_lifecycle_async(law_id=law_id, is_active=is_active)
            )
        except RuntimeError:
            return asyncio.run(
                self.sync_statute_lifecycle_async(law_id=law_id, is_active=is_active)
            )

    async def deactivate_node_async(self, node_id: str) -> Dict[str, Any]:
        """Marks a specific provision/node as inactive (is_active=False) in both Qdrant and Neo4j."""
        return await self.sync_node_lifecycle_async(node_id=node_id, is_active=False)

    async def sync_node_lifecycle_async(
        self,
        node_id: str,
        is_active: bool = False,
    ) -> Dict[str, Any]:
        """Synchronizes single provision lifecycle active state across Qdrant and Neo4j."""
        logger.info(f"Synchronizing lifecycle status for node '{node_id}': is_active={is_active}")
        qdrant_ok = False
        neo4j_count = 0
        errors: List[str] = []

        try:
            qdrant_ok = await self.qdrant_store.update_point_active_status_async(node_id, is_active)
        except Exception as exc:
            errors.append(f"Qdrant error: {exc}")

        try:
            neo4j_count = await self.neo4j_store.update_node_active_status_async(node_id, is_active)
        except Exception as exc:
            errors.append(f"Neo4j error: {exc}")

        return {
            "node_id": node_id,
            "is_active": is_active,
            "qdrant_updated": qdrant_ok,
            "neo4j_nodes_updated": neo4j_count,
            "success": len(errors) == 0 and qdrant_ok,
            "errors": errors,
        }
