"""HHEM hallucination validation service."""

import time
from typing import Optional

from loguru import logger
from transformers import AutoModelForSequenceClassification

from app.config import get_settings


class HHEMValidator:
    """HHEM model for hallucination detection."""

    def __init__(self):
        self.settings = get_settings()
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load HHEM model."""
        logger.info("Loading HHEM model...")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.settings.hhem_model, trust_remote_code=True
            )
            logger.info("✓ HHEM model loaded")
        except Exception as e:
            logger.error(f"Failed to load HHEM model: {e}")
            raise

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def validate(
        self, context: str, answer: str, threshold: Optional[float] = None
    ) -> dict:
        """
        Validate answer against context.

        Returns:
            {
                'score': float (0-1),
                'is_hallucinated': bool,
                'validation_time_ms': float
            }
        """
        if not self.model:
            raise RuntimeError("Model not loaded")

        start_time = time.time()

        threshold = threshold or self.settings.hhem_threshold

        try:
            # HHEM expects (premise, hypothesis) pairs
            score = self.model.predict([(context, answer)])[0].item()

            validation_time = (time.time() - start_time) * 1000  # ms

            return {
                "score": score,
                "is_hallucinated": score < threshold,
                "validation_time_ms": validation_time,
                "threshold": threshold,
            }

        except Exception as e:
            logger.error(f"HHEM validation error: {e}")
            return {
                "score": None,
                "is_hallucinated": None,
                "validation_time_ms": (time.time() - start_time) * 1000,
                "threshold": threshold,
                "error": str(e),
            }
