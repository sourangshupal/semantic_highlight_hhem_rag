"""Document processing using LangChain."""

import json
from pathlib import Path
from typing import Any, Dict, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from loguru import logger

from app.config import get_settings


class DocumentProcessor:
    """Process and chunk documents using LangChain."""

    def __init__(self):
        self.settings = get_settings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_document(self, file_path: str, file_type: str) -> List[Document]:
        """Load document based on file type."""

        logger.info(f"Loading {file_type} document: {file_path}")

        try:
            if file_type == "pdf":
                loader = PyPDFLoader(file_path)
            elif file_type == "txt":
                loader = TextLoader(file_path, encoding="utf-8")
            elif file_type == "md":
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_type == "json":
                # Custom JSON handling
                return self._load_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            documents = loader.load()
            logger.info(f"Loaded {len(documents)} raw documents")
            return documents

        except Exception as e:
            logger.error(f"Error loading document: {e}")
            raise

    def _load_json(self, file_path: str) -> List[Document]:
        """Load JSON file as documents."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both single object and array
        if isinstance(data, list):
            docs = []
            for idx, item in enumerate(data):
                content = json.dumps(item, indent=2)
                docs.append(
                    Document(
                        page_content=content, metadata={"source": file_path, "index": idx}
                    )
                )
            return docs
        else:
            content = json.dumps(data, indent=2)
            return [Document(page_content=content, metadata={"source": file_path})]

    def chunk_documents(
        self, documents: List[Document], filename: str
    ) -> List[Dict[str, Any]]:
        """Chunk documents and prepare for vector store."""

        logger.info(f"Chunking {len(documents)} documents")

        # Split documents
        chunks = self.text_splitter.split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks")

        # Prepare for vector store with metadata
        processed_chunks = []
        for idx, chunk in enumerate(chunks):
            processed_chunks.append(
                {
                    "content": chunk.page_content,
                    "metadata": {
                        **chunk.metadata,
                        "filename": filename,
                        "chunk_id": idx,
                        "chunk_size": len(chunk.page_content),
                    },
                }
            )

        return processed_chunks

    def process_file(
        self, file_path: str, file_type: str, filename: str
    ) -> List[Dict[str, Any]]:
        """Complete processing pipeline."""

        logger.info(f"Processing file: {filename}")

        # Load document
        documents = self.load_document(file_path, file_type)

        # Chunk documents
        chunks = self.chunk_documents(documents, filename)

        logger.info(f"File processed: {len(chunks)} chunks ready")

        return chunks
