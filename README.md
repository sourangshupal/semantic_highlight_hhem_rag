# Semantic Highlighting + HHEM RAG

> **Minimal FastAPI Implementation Demonstrating Cost Savings & Quality Improvements**

## 🎯 Overview

This project implements a RAG (Retrieval-Augmented Generation) system with two key optimizations:

1. **Semantic Highlighting** - Reduces token usage by 30-70% through intelligent context pruning
2. **HHEM Validation** - Detects hallucinations to improve answer reliability

### Key Features

- ✅ Upload documents (PDF, MD, TXT, JSON)
- ✅ Semantic Highlighting for context pruning
- ✅ HHEM validation for hallucination detection
- ✅ Comparison endpoint (with vs without optimizations)
- ✅ Metrics tracking and cost analysis

### Tech Stack

- **Python 3.12**
- **FastAPI** - API framework
- **OpenAI** - LLM & embeddings (gpt-4o-mini for cost efficiency)
- **LangChain** - Document parsing & chunking
- **Qdrant** - Vector store
- **Transformers** - Semantic Highlighting & HHEM models

## 📁 Project Structure

```
semantic-hhem-rag/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app
│   ├── models.py               # Pydantic models
│   ├── config.py               # Configuration
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document_processor.py  # Document parsing & chunking
│   │   ├── vector_store.py        # Qdrant operations
│   │   ├── semantic_highlighter.py # Semantic highlighting
│   │   ├── hhem_validator.py      # HHEM validation
│   │   └── rag_engine.py          # RAG orchestration
│   └── utils/
│       ├── __init__.py
│       └── metrics.py          # Metrics tracking
├── uploads/                     # Temporary file storage
├── data/                       # Qdrant data persistence
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── pyproject.toml              # UV package management
├── .env.example                # Environment template
├── docker-compose.yml          # Docker compose config
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [UV](https://docs.astral.sh/uv/) - Modern Python package manager
- Docker (optional, for Qdrant)

### 1. Clone and Setup

```bash
# Clone the repository
cd semantic-hhem-rag

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 3. Start Qdrant

```bash
# Using Docker
docker run -d -p 6333:6333 qdrant/qdrant:latest

# Or using docker-compose
docker-compose up -d qdrant
```

### 4. Run the API

```bash
# Development mode with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python module
python -m app.main
```

The API will be available at `http://localhost:8000`

- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

## 📖 API Documentation

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint |
| `/health` | GET | Health check |
| `/upload` | POST | Upload document (PDF, MD, TXT, JSON) |
| `/query` | POST | Query documents |
| `/compare` | POST | Compare all three modes |
| `/collection` | DELETE | Reset vector store |

### Example Usage

#### Upload a Document

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.pdf"
```

#### Query Documents

```bash
# Baseline mode (no optimizations)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main features?",
    "mode": "baseline",
    "top_k": 5
  }'

# Semantic mode (with highlighting)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main features?",
    "mode": "semantic",
    "top_k": 5
  }'

# Full mode (highlighting + HHEM)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main features?",
    "mode": "full",
    "top_k": 5
  }'
```

#### Compare All Modes

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the benefits?",
    "top_k": 5
  }'
```

## 🧪 Testing

```bash
# Run the test script
python tests/test_api.py

# Or using pytest
pytest tests/
```

## 📊 Performance Metrics

### Expected Results

Based on typical usage with GPT-4o-mini:

| Metric | Baseline | Semantic | Full | Improvement |
|--------|----------|----------|------|-------------|
| **Input Tokens** | 2,500 | 1,200 | 1,200 | 52% reduction |
| **Cost per Query** | $0.000375 | $0.000180 | $0.000180 | 52% savings |
| **Latency** | 2.1s | 1.3s | 2.8s | 38% faster (semantic) |
| **Quality (HHEM)** | N/A | N/A | 0.85 | Validated |

### Monthly Savings (10K queries)

```
Baseline:  10,000 × $0.000375 = $3.75/month
Semantic:  10,000 × $0.000180 = $1.80/month
Full:      10,000 × $0.000180 = $1.80/month

Monthly Savings: $1.95 (52% reduction)
Annual Savings:  $23.40
```

For 100K queries/month: **$195/month savings**

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

## ⚙️ Configuration

All configuration is done via environment variables or `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required) |
| `QDRANT_HOST` | localhost | Qdrant host |
| `QDRANT_PORT` | 6333 | Qdrant port |
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `LLM_MODEL` | gpt-4o-mini | OpenAI LLM model |
| `CHUNK_SIZE` | 500 | Document chunk size |
| `CHUNK_OVERLAP` | 50 | Chunk overlap |
| `SEMANTIC_THRESHOLD` | 0.5 | Semantic highlighting threshold |
| `HHEM_THRESHOLD` | 0.5 | HHEM validation threshold |

## 📝 Architecture

```
┌─────────────┐
│ Upload File │
└──────┬──────┘
       │
┌──────▼───────────────────────┐
│ 1. LangChain Document Loader │
│    - PDF, MD, TXT, JSON      │
└──────┬───────────────────────┘
       │
┌──────▼────────────────────┐
│ 2. Chunking & Embedding   │
│    - OpenAI embeddings    │
└──────┬────────────────────┘
       │
┌──────▼──────────┐
│ 3. Qdrant Store │
└──────┬──────────┘
       │
┌──────▼────────────────────────┐
│ 4. Query (3 modes)            │
│    A. Baseline (no optimizations)│
│    B. With Semantic Highlighting│
│    C. Full (Highlighting + HHEM)│
└──────┬────────────────────────┘
       │
┌──────▼─────────────────┐
│ 5. Metrics Comparison  │
│    - Token savings     │
│    - Cost analysis     │
│    - Quality scores    │
└────────────────────────┘
```

## 📄 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
