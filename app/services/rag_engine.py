"""RAG engine orchestrating all components."""

import time
from typing import Any, Dict, List

import tiktoken
from loguru import logger
from openai import OpenAI

from app.config import get_settings
from app.models import MetricsData, QueryMode, QueryResponse
from app.services.hhem_validator import HHEMValidator
from app.services.semantic_highlighter import SemanticHighlighter
from app.services.vector_store import VectorStore


class RAGEngine:
    """Main RAG engine with semantic highlighting and HHEM."""

    def __init__(
        self,
        vector_store: VectorStore,
        semantic_highlighter: SemanticHighlighter,
        hhem_validator: HHEMValidator,
    ):
        self.settings = get_settings()
        self.vector_store = vector_store
        self.semantic_highlighter = semantic_highlighter
        self.hhem_validator = hhem_validator
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)

        # Token counter
        self.encoding = tiktoken.encoding_for_model(self.settings.llm_model)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))

    def _estimate_cost(self, tokens: int, is_output: bool = False) -> float:
        """Estimate cost for GPT-4o-mini."""
        # GPT-4o-mini pricing (as of Feb 2025)
        # Input: $0.150 / 1M tokens, Output: $0.600 / 1M tokens
        rate = 0.600 if is_output else 0.150
        return (tokens / 1_000_000) * rate

    def _generate_answer(self, query: str, context: str) -> tuple[str, float]:
        """Generate answer using OpenAI."""

        start_time = time.time()

        prompt = f"""Based on the following context, answer the question. Be concise and accurate.

Context:
{context}

Question: {query}

Answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.settings.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on provided context.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            answer = response.choices[0].message.content or ""
            generation_time = (time.time() - start_time) * 1000  # ms

            return answer, generation_time

        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def query_baseline(self, question: str, top_k: int = 5) -> QueryResponse:
        """
        Baseline RAG without optimizations.
        """
        logger.info(f"[BASELINE] Processing: {question}")

        # 1. Retrieval
        start = time.time()
        documents = self.vector_store.search(question, top_k=top_k)
        retrieval_time = (time.time() - start) * 1000

        # 2. Prepare context (no highlighting)
        context = "\n\n---\n\n".join([doc["content"] for doc in documents])
        original_tokens = self._count_tokens(context)

        # 3. Generate answer
        answer, generation_time = self._generate_answer(question, context)
        output_tokens = self._count_tokens(answer)

        # 4. Calculate costs
        input_cost = self._estimate_cost(original_tokens, is_output=False)
        output_cost = self._estimate_cost(output_tokens, is_output=True)
        total_cost = input_cost + output_cost

        # 5. Build metrics
        metrics = MetricsData(
            retrieval_time_ms=retrieval_time,
            highlighting_time_ms=None,
            generation_time_ms=generation_time,
            hhem_time_ms=None,
            total_time_ms=retrieval_time + generation_time,
            original_tokens=original_tokens,
            pruned_tokens=None,
            token_savings_pct=None,
            estimated_cost_usd=total_cost,
            cost_savings_usd=None,
            hhem_score=None,
            is_hallucinated=None,
            compression_rate=None,
        )

        return QueryResponse(
            question=question,
            answer=answer,
            mode=QueryMode.BASELINE,
            sources=[
                {"content": doc["content"][:200] + "...", "metadata": doc["metadata"]}
                for doc in documents
            ],
            metrics=metrics,
            warning=None,
            highlighted_context=None,
        )

    def query_semantic(self, question: str, top_k: int = 5) -> QueryResponse:
        """
        RAG with semantic highlighting only.
        """
        logger.info(f"[SEMANTIC] Processing: {question}")

        # 1. Retrieval
        start = time.time()
        documents = self.vector_store.search(question, top_k=top_k)
        retrieval_time = (time.time() - start) * 1000

        # 2. Original context (for comparison)
        original_context = "\n\n---\n\n".join([doc["content"] for doc in documents])
        original_tokens = self._count_tokens(original_context)

        # 3. Apply semantic highlighting
        highlighted_texts, highlight_metrics = self.semantic_highlighter.highlight_documents(
            question, documents
        )

        # 4. Prepare pruned context
        pruned_context = "\n\n---\n\n".join(highlighted_texts)
        pruned_tokens = self._count_tokens(pruned_context)
        token_savings_pct = ((original_tokens - pruned_tokens) / original_tokens) * 100

        # 5. Generate answer (with pruned context)
        answer, generation_time = self._generate_answer(question, pruned_context)
        output_tokens = self._count_tokens(answer)

        # 6. Calculate costs
        input_cost = self._estimate_cost(pruned_tokens, is_output=False)
        output_cost = self._estimate_cost(output_tokens, is_output=True)
        total_cost = input_cost + output_cost

        # Cost savings compared to baseline
        baseline_cost = self._estimate_cost(original_tokens, is_output=False) + output_cost
        cost_savings = baseline_cost - total_cost

        # 7. Build metrics
        metrics = MetricsData(
            retrieval_time_ms=retrieval_time,
            highlighting_time_ms=highlight_metrics["total_processing_time_ms"],
            generation_time_ms=generation_time,
            hhem_time_ms=None,
            total_time_ms=retrieval_time
            + highlight_metrics["total_processing_time_ms"]
            + generation_time,
            original_tokens=original_tokens,
            pruned_tokens=pruned_tokens,
            token_savings_pct=token_savings_pct,
            estimated_cost_usd=total_cost,
            cost_savings_usd=cost_savings,
            hhem_score=None,
            is_hallucinated=None,
            compression_rate=highlight_metrics["avg_compression_rate"],
        )

        return QueryResponse(
            question=question,
            answer=answer,
            mode=QueryMode.SEMANTIC,
            sources=[
                {"content": ht[:200] + "...", "metadata": doc["metadata"]}
                for ht, doc in zip(highlighted_texts, documents)
            ],
            metrics=metrics,
            warning=None,
            highlighted_context=pruned_context,
        )

    def query_full(self, question: str, top_k: int = 5) -> QueryResponse:
        """
        Full RAG with semantic highlighting + HHEM validation.
        """
        logger.info(f"[FULL] Processing: {question}")

        # 1. Retrieval
        start = time.time()
        documents = self.vector_store.search(question, top_k=top_k)
        retrieval_time = (time.time() - start) * 1000

        # 2. Original context
        original_context = "\n\n---\n\n".join([doc["content"] for doc in documents])
        original_tokens = self._count_tokens(original_context)

        # 3. Apply semantic highlighting
        highlighted_texts, highlight_metrics = self.semantic_highlighter.highlight_documents(
            question, documents
        )

        # 4. Prepare pruned context
        pruned_context = "\n\n---\n\n".join(highlighted_texts)
        pruned_tokens = self._count_tokens(pruned_context)
        token_savings_pct = ((original_tokens - pruned_tokens) / original_tokens) * 100

        # 5. Generate answer
        answer, generation_time = self._generate_answer(question, pruned_context)
        output_tokens = self._count_tokens(answer)

        # 6. HHEM validation
        hhem_result = self.hhem_validator.validate(pruned_context, answer)

        # 7. Calculate costs
        input_cost = self._estimate_cost(pruned_tokens, is_output=False)
        output_cost = self._estimate_cost(output_tokens, is_output=True)
        total_cost = input_cost + output_cost

        baseline_cost = self._estimate_cost(original_tokens, is_output=False) + output_cost
        cost_savings = baseline_cost - total_cost

        # 8. Build metrics
        metrics = MetricsData(
            retrieval_time_ms=retrieval_time,
            highlighting_time_ms=highlight_metrics["total_processing_time_ms"],
            generation_time_ms=generation_time,
            hhem_time_ms=hhem_result["validation_time_ms"],
            total_time_ms=retrieval_time
            + highlight_metrics["total_processing_time_ms"]
            + generation_time
            + hhem_result["validation_time_ms"],
            original_tokens=original_tokens,
            pruned_tokens=pruned_tokens,
            token_savings_pct=token_savings_pct,
            estimated_cost_usd=total_cost,
            cost_savings_usd=cost_savings,
            hhem_score=hhem_result["score"],
            is_hallucinated=hhem_result["is_hallucinated"],
            compression_rate=highlight_metrics["avg_compression_rate"],
        )

        # 9. Warning if hallucinated
        warning = None
        if hhem_result["is_hallucinated"]:
            warning = f"⚠️ Low confidence answer (HHEM score: {hhem_result['score']:.2f}). Please verify."

        return QueryResponse(
            question=question,
            answer=answer,
            mode=QueryMode.FULL,
            sources=[
                {"content": ht[:200] + "...", "metadata": doc["metadata"]}
                for ht, doc in zip(highlighted_texts, documents)
            ],
            metrics=metrics,
            warning=warning,
            highlighted_context=pruned_context,
        )

    def query(self, question: str, mode: QueryMode, top_k: int = 5) -> QueryResponse:
        """Route query based on mode."""

        if mode == QueryMode.BASELINE:
            return self.query_baseline(question, top_k)
        elif mode == QueryMode.SEMANTIC:
            return self.query_semantic(question, top_k)
        elif mode == QueryMode.FULL:
            return self.query_full(question, top_k)
        else:
            raise ValueError(f"Unknown mode: {mode}")
