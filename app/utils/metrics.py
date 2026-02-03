"""Metrics tracking utilities."""

from typing import Any, Dict


class MetricsTracker:
    """Track metrics across multiple queries."""

    def __init__(self):
        self.queries: list[Dict[str, Any]] = []

    def add_query(self, metrics: Dict[str, Any]) -> None:
        """Add a query's metrics."""
        self.queries.append(metrics)

    def get_summary(self) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        if not self.queries:
            return {}

        total_queries = len(self.queries)

        # Token savings
        token_savings = [
            q.get("token_savings_pct", 0)
            for q in self.queries
            if q.get("token_savings_pct") is not None
        ]
        avg_token_savings = sum(token_savings) / len(token_savings) if token_savings else 0

        # Cost savings
        cost_savings = [
            q.get("cost_savings_usd", 0)
            for q in self.queries
            if q.get("cost_savings_usd") is not None
        ]
        total_cost_savings = sum(cost_savings)

        # HHEM scores
        hhem_scores = [
            q.get("hhem_score") for q in self.queries if q.get("hhem_score") is not None
        ]
        avg_hhem_score = sum(hhem_scores) / len(hhem_scores) if hhem_scores else 0

        # Hallucination rate
        hallucinated_count = sum(
            1 for q in self.queries if q.get("is_hallucinated") is True
        )
        hallucination_rate = (hallucinated_count / total_queries) * 100 if total_queries else 0

        return {
            "total_queries": total_queries,
            "avg_token_savings_pct": avg_token_savings,
            "total_cost_savings_usd": total_cost_savings,
            "avg_hhem_score": avg_hhem_score,
            "hallucination_rate_pct": hallucination_rate,
        }

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.queries = []
