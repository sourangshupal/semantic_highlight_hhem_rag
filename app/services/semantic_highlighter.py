"""Semantic highlighting service."""

import time
from typing import Any, Dict, List, Optional

from loguru import logger
from transformers import AutoModel

from app.config import get_settings


class SemanticHighlighter:
    """Semantic highlighting for context pruning."""

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load semantic highlighting model."""
        logger.info("Loading semantic highlighting model...")
        try:
            self.model = AutoModel.from_pretrained(
                self.settings.semantic_highlight_model, trust_remote_code=True
            )
            logger.info("✓ Semantic highlighting model loaded")
        except Exception as e:
            logger.error(f"Failed to load semantic highlighting model: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def highlight(
        self, query: str, context: str, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Apply semantic highlighting to context.

        Returns:
            {
                'highlighted_sentences': List[str],
                'compression_rate': float,
                'sentence_probabilities': List[float],
                'processing_time_ms': float
            }
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        threshold = threshold or self.settings.semantic_threshold

        try:
            result = self.model.process(
                question=query,
                context=context,
                threshold=threshold,
                return_sentence_metrics=True,
            )

            processing_time = (time.time() - start_time) * 1000  # ms

            return {
                "highlighted_sentences": result["highlighted_sentences"],
                "compression_rate": result["compression_rate"],
                "sentence_probabilities": result["sentence_probabilities"],
                "processing_time_ms": processing_time,
            }

        except Exception as e:
            logger.error(f"Semantic highlighting error: {e}")
            # Fallback: return original context
            return {
                "highlighted_sentences": [context],
                "compression_rate": 0.0,
                "sentence_probabilities": [1.0],
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    def highlight_documents(
        self, query: str, documents: List[Dict[str, Any]], threshold: Optional[float] = None
    ) -> tuple[List[str], Dict[str, Any]]:
        """
        Highlight multiple documents.

        Returns:
            (highlighted_texts, aggregated_metrics)
        """

        highlighted_texts = []
        total_compression = 0.0
        total_time = 0.0

        for doc in documents:
            result = self.highlight(query, doc["content"], threshold)

            # Combine highlighted sentences
            highlighted_text = "\n".join(result["highlighted_sentences"])
            highlighted_texts.append(highlighted_text)

            total_compression += result["compression_rate"]
            total_time += result["processing_time_ms"]

        # Aggregate metrics
        metrics = {
            "avg_compression_rate": total_compression / len(documents) if documents else 0.0,
            "total_processing_time_ms": total_time,
            "documents_processed": len(documents),
        }

        return highlighted_texts, metrics
