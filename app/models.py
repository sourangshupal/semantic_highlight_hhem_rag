"""Pydantic models for request/response validation."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class FileType(str, Enum):
    """Supported file types."""

    PDF = "pdf"
    MARKDOWN = "md"
    TEXT = "txt"
    JSON = "json"


class UploadResponse(BaseModel):
    """Response for file upload."""

    message: str
    filename: str
    file_type: str
    chunks_created: int
    collection_name: str


class QueryMode(str, Enum):
    """Query modes for comparison."""

    BASELINE = "baseline"  # No optimizations
    SEMANTIC = "semantic"  # Only semantic highlighting
    FULL = "full"  # Semantic + HHEM


class QueryRequest(BaseModel):
    """Request model for queries."""

    question: str = Field(..., min_length=1, max_length=500)
    mode: QueryMode = QueryMode.FULL
    top_k: int = Field(default=5, ge=1, le=20)


class MetricsData(BaseModel):
    """Metrics for a single query execution."""

    retrieval_time_ms: float
    highlighting_time_ms: Optional[float] = None
    generation_time_ms: float
    hhem_time_ms: Optional[float] = None
    total_time_ms: float

    original_tokens: int
    pruned_tokens: Optional[int] = None
    token_savings_pct: Optional[float] = None

    estimated_cost_usd: float
    cost_savings_usd: Optional[float] = None

    hhem_score: Optional[float] = None
    is_hallucinated: Optional[bool] = None

    compression_rate: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for queries."""

    question: str
    answer: str
    mode: QueryMode

    sources: List[Dict[str, Any]]
    metrics: MetricsData

    warning: Optional[str] = None
    highlighted_context: Optional[str] = None


class ComparisonRequest(BaseModel):
    """Request for comparing all three modes."""

    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class ComparisonResponse(BaseModel):
    """Response comparing all three modes."""

    question: str

    baseline: QueryResponse
    semantic: QueryResponse
    full: QueryResponse

    summary: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    qdrant_connected: bool
    models_loaded: Dict[str, bool]
    collection_exists: bool
    document_count: int
