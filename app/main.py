"""Main FastAPI application."""

import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import get_settings
from app.models import (
    ComparisonRequest,
    ComparisonResponse,
    HealthResponse,
    QueryMode,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.hhem_validator import HHEMValidator
from app.services.rag_engine import RAGEngine
from app.services.semantic_highlighter import SemanticHighlighter
from app.services.vector_store import VectorStore

# Global services
settings = get_settings()
document_processor = DocumentProcessor()
vector_store = VectorStore()
semantic_highlighter = SemanticHighlighter()
hhem_validator = HHEMValidator()
rag_engine = RAGEngine(vector_store, semantic_highlighter, hhem_validator)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("🚀 Starting Semantic Highlighting + HHEM RAG API")
    logger.info(f"✓ OpenAI Model: {settings.llm_model}")
    logger.info(f"✓ Embeddings: {settings.embedding_model}")
    logger.info(f"✓ Vector Store: Qdrant @ {settings.qdrant_url}")

    # Ensure upload directory exists
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    logger.info("🛑 Shutting down API")


# Initialize FastAPI
app = FastAPI(
    title="Semantic Highlighting + HHEM RAG API",
    description="Minimal RAG implementation demonstrating cost savings and quality improvements",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Semantic Highlighting + HHEM RAG API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""

    # Check Qdrant connection
    try:
        collection_info = vector_store.get_collection_info()
        qdrant_connected = True
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        qdrant_connected = False
        collection_info = {"exists": False, "points_count": 0}

    return HealthResponse(
        status="healthy" if qdrant_connected else "degraded",
        qdrant_connected=qdrant_connected,
        models_loaded={
            "semantic_highlighter": semantic_highlighter.is_loaded(),
            "hhem_validator": hhem_validator.is_loaded(),
        },
        collection_exists=collection_info.get("exists", False),
        document_count=collection_info.get("points_count", 0),
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a document (PDF, MD, TXT, JSON).
    """

    # Validate file type
    file_ext = Path(file.filename or "").suffix.lower().lstrip(".")
    if file_ext not in ["pdf", "md", "txt", "json"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: pdf, md, txt, json",
        )

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset

    if file_size > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_file_size / 1024 / 1024}MB",
        )

    # Save file temporarily
    file_path = Path(settings.upload_dir) / (file.filename or "unnamed")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"File saved: {file_path}")

        # Process document
        chunks = document_processor.process_file(
            str(file_path), file_ext, file.filename or "unnamed"
        )

        # Add to vector store
        num_chunks = vector_store.add_documents(chunks)

        # Clean up
        file_path.unlink()

        return UploadResponse(
            message="Document processed successfully",
            filename=file.filename or "unnamed",
            file_type=file_ext,
            chunks_created=num_chunks,
            collection_name=settings.qdrant_collection_name,
        )

    except Exception as e:
        logger.error(f"Upload processing error: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents with specified mode:
    - baseline: No optimizations
    - semantic: Semantic highlighting only
    - full: Semantic highlighting + HHEM validation
    """

    try:
        response = rag_engine.query(
            question=request.question, mode=request.mode, top_k=request.top_k
        )
        return response

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compare", response_model=ComparisonResponse)
async def compare_modes(request: ComparisonRequest):
    """
    Compare all three modes for the same question.
    Useful for demonstrating benefits.
    """

    try:
        # Run all three modes
        baseline = rag_engine.query_baseline(request.question, request.top_k)
        semantic = rag_engine.query_semantic(request.question, request.top_k)
        full = rag_engine.query_full(request.question, request.top_k)

        # Calculate summary statistics
        summary = {
            "token_savings_pct": semantic.metrics.token_savings_pct,
            "cost_savings_usd": semantic.metrics.cost_savings_usd,
            "compression_rate_pct": semantic.metrics.compression_rate * 100
            if semantic.metrics.compression_rate
            else 0,
            "baseline_cost": baseline.metrics.estimated_cost_usd,
            "semantic_cost": semantic.metrics.estimated_cost_usd,
            "full_cost": full.metrics.estimated_cost_usd,
            "baseline_time_ms": baseline.metrics.total_time_ms,
            "semantic_time_ms": semantic.metrics.total_time_ms,
            "full_time_ms": full.metrics.total_time_ms,
            "hhem_score": full.metrics.hhem_score,
            "is_reliable": not full.metrics.is_hallucinated
            if full.metrics.is_hallucinated is not None
            else None,
            "recommendation": (
                "Full mode recommended - answer is reliable and cost-effective"
                if full.metrics.hhem_score and full.metrics.hhem_score >= 0.5
                else "⚠️ Consider revising query or reviewing sources - low confidence answer"
            ),
        }

        return ComparisonResponse(
            question=request.question,
            baseline=baseline,
            semantic=semantic,
            full=full,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection")
async def reset_collection():
    """Delete all documents from vector store."""
    try:
        vector_store.delete_collection()
        vector_store._ensure_collection()
        return {"message": "Collection reset successfully"}
    except Exception as e:
        logger.error(f"Reset error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
