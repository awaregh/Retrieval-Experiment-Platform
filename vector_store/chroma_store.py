"""
ChromaDB vector store integration.
"""
from __future__ import annotations
import logging
import uuid
from typing import List, Dict, Any, Optional

from core.retrieval import RetrievalResult

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    Wraps ChromaDB for embedding storage and retrieval.
    """

    def __init__(self, collection_name: str = "retrieval_exp", persist_dir: Optional[str] = None):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                if self.persist_dir:
                    self._client = chromadb.PersistentClient(path=self.persist_dir)
                else:
                    self._client = chromadb.EphemeralClient()
            except ImportError:
                raise ImportError("chromadb is required. Install with: pip install chromadb")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add(
        self,
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add embeddings to the collection."""
        collection = self._get_collection()
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        # Convert metadata values to strings (ChromaDB requirement)
        clean_metadatas = []
        for m in metadatas:
            clean_metadatas.append({k: str(v) for k, v in m.items()})

        batch_size = 100
        for i in range(0, len(embeddings), batch_size):
            collection.add(
                embeddings=embeddings[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=clean_metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )
        logger.info(f"Added {len(embeddings)} vectors to collection '{self.collection_name}'.")

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """Query the collection with a vector."""
        collection = self._get_collection()
        where = filters if filters else None
        collection_count = collection.count()
        if collection_count == 0:
            return []
        try:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, collection_count),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        retrieval_results = []
        for idx, (doc_id, doc_text, metadata, distance) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # Convert cosine distance to similarity
            score = 1.0 - distance
            retrieval_results.append(RetrievalResult(
                document_id=metadata.get("document_id", doc_id),
                chunk_id=doc_id,
                text=doc_text,
                score=score,
                metadata=metadata,
                rank=idx + 1,
            ))

        return retrieval_results

    def count(self) -> int:
        """Return number of vectors in collection."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
        except Exception as e:
            logger.warning(f"Could not delete collection: {e}")

    def reset(self) -> None:
        """Reset (delete and recreate) the collection."""
        self.delete_collection()
        self._collection = None
