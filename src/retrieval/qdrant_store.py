"""
Qdrant Store: Hybrid Vector Store with Core RRF (Reciprocal Rank Fusion)
========================================================================
Utilizes FastEmbed (Qdrant's lightweight ONNX-optimized embedding library):
- Dense Vector (BAAI/bge-m3): Semantic similarity (e.g. "giam lương", "quỹ giữ chân" <-> "khấu trừ tiền lương")
- Sparse Vector (BM25): Exact keyword & statutory citation matching (e.g. "Điều 127", "45/2019/QH14", "8%")
- Core Engine Hybrid Search: Prefetch(dense) + Prefetch(sparse) + FusionQuery(Fusion.RRF)
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest_models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Prefetch,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from src.core.config import get_async_qdrant_client, settings
from src.core.schemas import (
    DENSE_VECTOR_DIM,
    DENSE_VECTOR_NAME,
    QDRANT_COLLECTION_NAME,
    SPARSE_VECTOR_NAME,
    LegalChunk,
    SynchronizedLegalEntity,
)


class FastEmbedHybridPipeline:
    """
    ONNX-optimized embedding service powered by FastEmbed:
    - Sparse: Qdrant/bm25 for lexical/statutory keyword tokens
    - Dense: Semantic representations (BAAI/bge-m3 or fastembed ONNX models)
    """

    def __init__(
        self,
        dense_model_name: Optional[str] = None,
        sparse_model_name: str = "Qdrant/bm25",
    ):
        self.dense_model_name = dense_model_name or settings.DENSE_EMBEDDING_MODEL
        self.sparse_model_name = sparse_model_name
        self._dense_model: Any = None
        self._sparse_model: Any = None

    @property
    def sparse_model(self) -> Any:
        """Lazy load fastembed SparseTextEmbedding for BM25."""
        if self._sparse_model is None:
            from fastembed import SparseTextEmbedding
            logger.info(f"Loading ONNX SparseTextEmbedding: {self.sparse_model_name}")
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse_model

    @property
    def dense_model(self) -> Any:
        """Lazy load dense embedding model."""
        if self._dense_model is None:
            from fastembed import TextEmbedding
            supported = [m["model"] for m in TextEmbedding.list_supported_models()]
            if self.dense_model_name in supported:
                logger.info(f"Loading ONNX TextEmbedding: {self.dense_model_name}")
                self._dense_model = TextEmbedding(model_name=self.dense_model_name)
            else:
                # Try loading via SentenceTransformer with safetensors support
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading SentenceTransformer: {self.dense_model_name} (use_safetensors=True)")
                    self._dense_model = SentenceTransformer(
                        self.dense_model_name,
                        model_kwargs={"use_safetensors": True}
                    )
                except Exception as exc:
                    fallback_model = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
                    logger.warning(
                        f"Model {self.dense_model_name} not available via SentenceTransformer ({exc}). "
                        f"Falling back to FastEmbed {fallback_model}."
                    )
                    self._dense_model = TextEmbedding(model_name=fallback_model)
        return self._dense_model

    def embed_sparse(self, text: str) -> SparseVector:
        """Generate a SparseVector (indices and values) using fastembed BM25."""
        raw_sparse = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=raw_sparse.indices.tolist() if hasattr(raw_sparse.indices, "tolist") else list(raw_sparse.indices),
            values=raw_sparse.values.tolist() if hasattr(raw_sparse.values, "tolist") else list(raw_sparse.values),
        )

    def embed_sparse_batch(self, texts: List[str]) -> List[SparseVector]:
        """Generate a list of SparseVectors for a batch of texts."""
        if not texts:
            return []
        raw_list = list(self.sparse_model.embed(texts))
        sparse_vectors = []
        for raw in raw_list:
            sparse_vectors.append(
                SparseVector(
                    indices=raw.indices.tolist() if hasattr(raw.indices, "tolist") else list(raw.indices),
                    values=raw.values.tolist() if hasattr(raw.values, "tolist") else list(raw.values),
                )
            )
        return sparse_vectors

    def embed_dense(self, text: str) -> List[float]:
        """Generate a dense embedding vector."""
        model = self.dense_model
        if hasattr(model, "encode"):
            vec = model.encode(text)
            return vec.tolist() if hasattr(vec, "tolist") else list(vec)
        raw = list(model.embed([text]))[0]
        return raw.tolist() if hasattr(raw, "tolist") else list(raw)

    def embed_dense_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for a batch of texts."""
        if not texts:
            return []
        model = self.dense_model
        if hasattr(model, "encode"):
            vecs = model.encode(texts)
            return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vecs]
        raw_list = list(model.embed(texts))
        return [r.tolist() if hasattr(r, "tolist") else list(r) for r in raw_list]

    def embed_hybrid(self, text: str) -> Tuple[List[float], SparseVector]:
        """Simultaneously generate both Dense and Sparse representations for a query."""
        dense = self.embed_dense(text)
        sparse = self.embed_sparse(text)
        return dense, sparse


class QdrantStore:
    """
    Production Qdrant Vector Store providing native RRF (Reciprocal Rank Fusion) Hybrid Search:
    - Prefetch Dense (BAAI/bge-m3)
    - Prefetch Sparse (BM25)
    - FusionQuery(Fusion.RRF)
    """

    def __init__(
        self,
        client: Optional[AsyncQdrantClient] = None,
        collection_name: Optional[str] = None,
        pipeline: Optional[FastEmbedHybridPipeline] = None,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION or QDRANT_COLLECTION_NAME
        self.client: AsyncQdrantClient = client or get_async_qdrant_client()
        self.pipeline: FastEmbedHybridPipeline = pipeline or FastEmbedHybridPipeline()
        logger.info(f"Initialized QdrantStore for collection '{self.collection_name}' with AsyncQdrantClient.")

    async def ensure_collection_exists(self, dense_dim: int = DENSE_VECTOR_DIM) -> bool:
        """
        Creates or verifies the hybrid collection with dense and sparse vector configurations,
        and initializes payload indexes for O(1) filtering.
        """
        try:
            collections_response = await self.client.get_collections()
            existing_names = [col.name for col in collections_response.collections]
            if self.collection_name in existing_names:
                logger.info(f"Qdrant collection '{self.collection_name}' already exists.")
                await self.ensure_payload_indices()
                return True

            logger.info(f"Creating hybrid collection '{self.collection_name}' (Dense: {dense_dim}d, Sparse: BM25)...")
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    DENSE_VECTOR_NAME: VectorParams(
                        size=dense_dim,
                        distance=Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info(f"Successfully configured hybrid collection '{self.collection_name}'.")
            await self.ensure_payload_indices()
            return True
        except Exception as exc:
            logger.error(f"Failed to ensure Qdrant collection '{self.collection_name}': {exc}")
            raise

    async def ensure_payload_indices(self) -> None:
        """
        Idempotently creates payload indexes on key statutory filtering fields.
        Transforms payload filtering complexity from O(N) linear search to O(1) indexed lookup.
        Indexed fields:
        - law_id (Keyword)
        - article_num (Integer)
        - chapter_num (Keyword)
        - clause_num (Integer)
        - neo4j_node_id (Keyword)
        - neo4j_node_type (Keyword)
        """
        index_specs = [
            ("law_id", PayloadSchemaType.KEYWORD),
            ("article_num", PayloadSchemaType.INTEGER),
            ("chapter_num", PayloadSchemaType.KEYWORD),
            ("clause_num", PayloadSchemaType.INTEGER),
            ("neo4j_node_id", PayloadSchemaType.KEYWORD),
            ("neo4j_node_type", PayloadSchemaType.KEYWORD),
            ("is_active", PayloadSchemaType.BOOL),
        ]
        for field_name, schema_type in index_specs:
            try:
                if hasattr(self.client, "create_payload_index") and callable(self.client.create_payload_index):
                    await self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema_type,
                        wait=True,
                    )
                    logger.debug(f"Payload index ensured: {field_name} ({schema_type.value})")
            except Exception as exc:
                # Silently bypass if already exists, unsupported in memory, or client is a mock
                logger.debug(f"Payload index note for '{field_name}': {exc}")


    async def async_upsert_chunks(
        self,
        chunks: List[LegalChunk],
        dense_vectors: Optional[List[List[float]]] = None,
        sparse_vectors: Optional[List[SparseVector]] = None,
    ) -> int:
        """
        Batch upserts LegalChunks into Qdrant with deterministic UUIDs and hybrid vectors.
        """
        if not chunks:
            return 0

        # Compute vectors if not explicitly passed
        if dense_vectors is None or sparse_vectors is None:
            texts = [c.context_enriched_text for c in chunks]
            if dense_vectors is None:
                dense_vectors = self.pipeline.embed_dense_batch(texts)
            if sparse_vectors is None:
                sparse_vectors = self.pipeline.embed_sparse_batch(texts)

        points = []
        for idx, chunk in enumerate(chunks):
            vector_dict: Dict[str, Any] = {}
            if dense_vectors and idx < len(dense_vectors):
                vector_dict[DENSE_VECTOR_NAME] = dense_vectors[idx]
            if sparse_vectors and idx < len(sparse_vectors):
                vector_dict[SPARSE_VECTOR_NAME] = sparse_vectors[idx]

            points.append(
                PointStruct(
                    id=chunk.qdrant_point_id,
                    vector=vector_dict,
                    payload=chunk.to_qdrant_payload(),
                )
            )

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            await self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
            )
        logger.info(f"Successfully upserted {len(points)} chunks in {len(points)//batch_size + 1} batches into Qdrant collection '{self.collection_name}'.")
        return len(points)

    async def async_upsert_entities(
        self,
        entities: List[SynchronizedLegalEntity],
        dense_vectors: Optional[List[List[float]]] = None,
        sparse_vectors: Optional[List[SparseVector]] = None,
    ) -> int:
        """
        Batch upserts SynchronizedLegalEntities into Qdrant with deterministic UUIDs.
        """
        if not entities:
            return 0

        points = []
        for idx, entity in enumerate(entities):
            vector_dict: Dict[str, Any] = {}
            if dense_vectors and idx < len(dense_vectors):
                vector_dict[DENSE_VECTOR_NAME] = dense_vectors[idx]
            if sparse_vectors and idx < len(sparse_vectors):
                vector_dict[SPARSE_VECTOR_NAME] = sparse_vectors[idx]

            points.append(
                PointStruct(
                    id=entity.point_uuid,
                    vector=vector_dict if vector_dict else None,
                    payload=entity.to_qdrant_payload(),
                )
            )

        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Upserted {len(points)} entities into Qdrant collection '{self.collection_name}'.")
        return len(points)

    async def hybrid_search_async(
        self,
        query: str,
        top_k: int = 10,
        prefetch_limit: int = 25,
        dense_vector: Optional[List[float]] = None,
        sparse_vector: Optional[SparseVector] = None,
        active_only: bool = True,
        query_filter: Optional[Filter] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes native hybrid search in Qdrant's core engine:
        1. Prefetch Dense (BAAI/bge-m3 semantic capture: 'giam lương', 'quỹ giữ chân' <-> 'khấu trừ tiền lương')
        2. Prefetch Sparse (BM25 exact token capture: 'Điều 127', '45/2019/QH14', '8%')
        3. Reciprocal Rank Fusion: FusionQuery(Fusion.RRF) combines both ranking spaces.
        4. When active_only=True, automatically filters out repealed statutory provisions (is_active=False).
        """
        logger.info(f"Executing Qdrant hybrid search for query: '{query[:60]}...' (top_k={top_k}, active_only={active_only})")

        # Compute embeddings if not supplied
        if dense_vector is None:
            dense_vector = self.pipeline.embed_dense(query)
        if sparse_vector is None:
            sparse_vector = self.pipeline.embed_sparse(query)

        # Build active lifecycle filter
        effective_filter = query_filter
        if active_only:
            inactive_condition = FieldCondition(key="is_active", match=MatchValue(value=False))
            if effective_filter is None:
                effective_filter = Filter(must_not=[inactive_condition])
            else:
                existing_must_not = list(effective_filter.must_not or [])
                existing_must_not.append(inactive_condition)
                effective_filter = effective_filter.model_copy(update={"must_not": existing_must_not})

        # Build dual prefetch branches
        prefetch = [
            Prefetch(
                query=dense_vector,
                using=DENSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=effective_filter,
            ),
            Prefetch(
                query=sparse_vector,
                using=SPARSE_VECTOR_NAME,
                limit=prefetch_limit,
                filter=effective_filter,
            ),
        ]

        # Execute in Qdrant core with RRF
        response = await self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=effective_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results: List[Dict[str, Any]] = []
        for point in response.points:
            payload = point.payload or {}
            results.append({
                "point_id": str(point.id),
                "score": point.score,
                "chunk_id": payload.get("chunk_id", ""),
                "neo4j_node_id": payload.get("neo4j_node_id", payload.get("chunk_id", "")),
                "neo4j_node_type": payload.get("neo4j_node_type", "Article"),
                "canonical_citation": payload.get("canonical_citation", ""),
                "article_title": payload.get("article_title", ""),
                "content": payload.get("content", ""),
                "context_enriched_text": payload.get("context_enriched_text", ""),
                "law_title": payload.get("law_title", ""),
                "cross_references": payload.get("cross_references", []),
                "payload": payload,
            })

        logger.info(f"Qdrant RRF hybrid search returned {len(results)} ranked matches.")
        return results

    def hybrid_search(self, query: str, top_k: int = 10, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Synchronous wrapper for hybrid search.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # In active loop (e.g. Jupyter or nested loop)
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.hybrid_search_async(query, top_k=top_k, active_only=active_only)
                    )
                    return future.result()
            return loop.run_until_complete(self.hybrid_search_async(query, top_k=top_k, active_only=active_only))
        except RuntimeError:
            return asyncio.run(self.hybrid_search_async(query, top_k=top_k, active_only=active_only))

    async def update_statute_active_status_async(
        self,
        law_id: str,
        is_active: bool = False,
    ) -> bool:
        """
        Updates the `is_active` payload flag for all points belonging to a specific statute in Qdrant.
        """
        filter_spec = Filter(
            must=[
                FieldCondition(
                    key="law_id",
                    match=MatchValue(value=law_id),
                )
            ]
        )
        try:
            await self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_active": is_active},
                points=filter_spec,
                wait=True,
            )
            logger.info(f"Qdrant: Successfully updated is_active={is_active} for law_id='{law_id}'.")
            return True
        except Exception as exc:
            logger.error(f"Failed to update Qdrant payload is_active for law_id='{law_id}': {exc}")
            raise

    def update_statute_active_status(
        self,
        law_id: str,
        is_active: bool = False,
    ) -> bool:
        """Synchronous wrapper for update_statute_active_status_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(
                        asyncio.run,
                        self.update_statute_active_status_async(law_id=law_id, is_active=is_active)
                    ).result()
            return loop.run_until_complete(
                self.update_statute_active_status_async(law_id=law_id, is_active=is_active)
            )
        except RuntimeError:
            return asyncio.run(
                self.update_statute_active_status_async(law_id=law_id, is_active=is_active)
            )

    async def update_point_active_status_async(
        self,
        chunk_id_or_node_id: str,
        is_active: bool = False,
    ) -> bool:
        """
        Updates the `is_active` payload flag for a specific chunk or neo4j_node_id in Qdrant.
        """
        filter_spec = Filter(
            should=[
                FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id_or_node_id)),
                FieldCondition(key="neo4j_node_id", match=MatchValue(value=chunk_id_or_node_id)),
            ]
        )
        try:
            await self.client.set_payload(
                collection_name=self.collection_name,
                payload={"is_active": is_active},
                points=filter_spec,
                wait=True,
            )
            logger.info(f"Qdrant: Successfully updated is_active={is_active} for node '{chunk_id_or_node_id}'.")
            return True
        except Exception as exc:
            logger.error(f"Failed to update Qdrant payload is_active for node '{chunk_id_or_node_id}': {exc}")
            raise

    def update_point_active_status(
        self,
        chunk_id_or_node_id: str,
        is_active: bool = False,
    ) -> bool:
        """Synchronous wrapper for update_point_active_status_async."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(
                        asyncio.run,
                        self.update_point_active_status_async(chunk_id_or_node_id=chunk_id_or_node_id, is_active=is_active)
                    ).result()
            return loop.run_until_complete(
                self.update_point_active_status_async(chunk_id_or_node_id=chunk_id_or_node_id, is_active=is_active)
            )
        except RuntimeError:
            return asyncio.run(
                self.update_point_active_status_async(chunk_id_or_node_id=chunk_id_or_node_id, is_active=is_active)
            )

    async def close(self) -> None:
        """Release underlying AsyncQdrantClient connection."""
        await self.client.close()


# Backward-compatible alias
QdrantLegalStore = QdrantStore
