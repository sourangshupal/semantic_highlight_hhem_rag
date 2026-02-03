"""Qdrant vector store operations."""

import uuid
from typing import Any, Dict, List, Optional

from loguru import logger
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import get_settings


class VectorStore:
    """Qdrant vector store manager with OpenAI embeddings."""

    def __init__(self):
        self.settings = get_settings()
        self.client = QdrantClient(
            host=self.settings.qdrant_host, port=self.settings.qdrant_port
        )
        self.openai_client = OpenAI(api_key=self.settings.openai_api_key)
        self.collection_name = self.settings.qdrant_collection_name

        # Initialize collection
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure collection exists with proper configuration."""

        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info(f"Creating collection: {self.collection_name}")

            # Get embedding dimension from OpenAI
            sample_embedding = self._get_embeddings(["sample"])[0]
            vector_size = len(sample_embedding)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection created with vector size: {vector_size}")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI."""

        response = self.openai_client.embeddings.create(
            model=self.settings.embedding_model, input=texts
        )

        return [item.embedding for item in response.data]

    def add_documents(self, chunks: List[Dict[str, Any]]) -> int:
        """Add document chunks to vector store."""

        logger.info(f"Adding {len(chunks)} chunks to vector store")

        # Extract texts
        texts = [chunk["content"] for chunk in chunks]

        # Get embeddings
        embeddings = self._get_embeddings(texts)

        # Create points
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"content": chunk["content"], **chunk["metadata"]},
            )
            points.append(point)

        # Upload to Qdrant
        self.client.upsert(collection_name=self.collection_name, points=points)

        logger.info(f"Successfully added {len(points)} points")
        return len(points)

    def search(
        self, query: str, top_k: int = 5, filename_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar documents."""

        logger.info(f"Searching for: '{query}' (top_k={top_k})")

        # Get query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Optional filename filter
        query_filter = None
        if filename_filter:
            query_filter = Filter(
                must=[FieldCondition(key="filename", match=MatchValue(value=filename_filter))]
            )

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=query_filter,
        )

        # Format results
        documents = []
        for result in results:
            documents.append(
                {
                    "content": result.payload.get("content", ""),
                    "metadata": {
                        k: v for k, v in result.payload.items() if k != "content"
                    },
                    "score": result.score,
                }
            )

        logger.info(f"Found {len(documents)} results")
        return documents

    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection statistics."""

        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "exists": True,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
            }
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")
            return {"exists": False, "vectors_count": 0, "points_count": 0}

    def delete_collection(self):
        """Delete the collection."""
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(self.collection_name)
