.PHONY: help install dev-install start-qdrant run test clean docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make install      - Install dependencies using UV"
	@echo "  make dev-install  - Install dependencies with dev extras"
	@echo "  make start-qdrant - Start Qdrant using Docker"
	@echo "  make run          - Run the FastAPI application"
	@echo "  make test         - Run tests"
	@echo "  make clean        - Clean up cache and temporary files"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-up    - Start all services with Docker Compose"
	@echo "  make docker-down  - Stop all Docker services"

install:
	uv pip install -e .

dev-install:
	uv pip install -e ".[dev]"

start-qdrant:
	docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest || docker start qdrant

run:
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test:
	cd tests && python test_api.py

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
