"""
Cross-Encoder Reranker Module for Legal IR
===========================================
Stage-2 neural reranker based on Cross-Encoder architecture (BAAI/bge-reranker-base).
Computes full bi-directional cross-attention between Query and retrieved candidate Articles:
- Operates on Top 30-50 candidates from Stage 1 (Hybrid BM25 + Dense).
- Evaluates deep contextual relevance scores P(Relevant | Query, Article).
- Accelerates on NVIDIA CUDA GPU with float16 inference or CPU fallback.
"""

from typing import Any, List, Optional, Tuple

import torch
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    """
    Two-Stage Cross-Encoder Reranker:
    Re-scores candidate legal passages by feeding [Query, Passage] pairs
    jointly through a multi-layer cross-attention Transformer.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        logger.info(f"Loading Cross-Encoder Reranker '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Use fp16 on CUDA for maximum throughput
        if self.device == "cuda":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                dtype=torch.float16,
            ).to(self.device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
            ).to(self.device)

        self.model.eval()
        logger.info(f"Cross-Encoder Reranker ready on {self.device}!")

    def compute_scores(self, query: str, texts: List[str]) -> List[float]:
        """Computes raw cross-encoder similarity logits for [query, text] pairs."""
        if not texts:
            return []

        pairs = [[query, text] for text in texts]
        all_scores: List[float] = []

        with torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
                all_scores.extend(logits.cpu().tolist())

        return all_scores

    def rerank(
        self,
        query: str,
        candidates: List[Tuple[Any, str]],
        top_k: int = 20,
    ) -> List[Tuple[float, Any]]:
        """
        Reranks candidate items based on cross-encoder similarity to query.

        Args:
            query: The user natural language question.
            candidates: List of (item_object, passage_text) tuples.
            top_k: Number of highest-ranking items to return.

        Returns:
            List of (score, item_object) tuples sorted by descending relevance.
        """
        if not candidates:
            return []

        texts = [doc_text for _, doc_text in candidates]
        scores = self.compute_scores(query, texts)

        scored = [(score, candidates[idx][0]) for idx, score in enumerate(scores)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
