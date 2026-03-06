"""
Dataset manager: handles ingestion, chunking, and storage of documents.
"""
from __future__ import annotations
import logging
import uuid
from typing import List, Dict, Any, Optional

from core.chunking import get_chunker, Chunk
from datasets.loader import load_dataset

logger = logging.getLogger(__name__)


class DatasetManager:
    """
    Manages datasets: load, chunk, and prepare for indexing.
    """

    def __init__(self):
        self._datasets: Dict[str, Dict] = {}  # dataset_id -> dataset info

    def load(
        self,
        path: str,
        dataset_id: Optional[str] = None,
        format: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Load a dataset from a file and register it.
        Returns dataset_id.
        """
        dataset_id = dataset_id or str(uuid.uuid4())[:8]
        data = load_dataset(path, format=format, **kwargs)
        self._datasets[dataset_id] = {
            "id": dataset_id,
            "path": path,
            "documents": data.get("documents", []),
            "queries": data.get("queries", []),
            "chunks": [],
        }
        logger.info(
            f"Loaded dataset '{dataset_id}' with {len(data.get('documents', []))} documents "
            f"and {len(data.get('queries', []))} queries."
        )
        return dataset_id

    def load_from_dict(
        self,
        data: Dict[str, Any],
        dataset_id: Optional[str] = None,
    ) -> str:
        """Load a dataset from an in-memory dict."""
        dataset_id = dataset_id or str(uuid.uuid4())[:8]
        self._datasets[dataset_id] = {
            "id": dataset_id,
            "path": None,
            "documents": data.get("documents", []),
            "queries": data.get("queries", []),
            "chunks": [],
        }
        return dataset_id

    def chunk_dataset(
        self,
        dataset_id: str,
        strategy: str = "fixed",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> List[Chunk]:
        """
        Apply a chunking strategy to all documents in a dataset.
        Returns the list of chunks.
        """
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset '{dataset_id}' not found.")

        chunker = get_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        dataset = self._datasets[dataset_id]
        all_chunks: List[Chunk] = []

        for doc in dataset["documents"]:
            doc_id = doc.get("id", str(uuid.uuid4())[:8])
            text = doc.get("text", "")
            if not text:
                continue
            chunks = chunker.chunk(text, doc_id)
            for chunk in chunks:
                chunk.metadata.update(doc.get("metadata", {}))
            all_chunks.extend(chunks)

        dataset["chunks"] = all_chunks
        logger.info(
            f"Dataset '{dataset_id}' chunked into {len(all_chunks)} chunks "
            f"using '{strategy}' strategy."
        )
        return all_chunks

    def get_dataset(self, dataset_id: str) -> Dict:
        if dataset_id not in self._datasets:
            raise KeyError(f"Dataset '{dataset_id}' not found.")
        return self._datasets[dataset_id]

    def get_queries(self, dataset_id: str) -> List[Dict]:
        return self.get_dataset(dataset_id).get("queries", [])

    def get_chunks(self, dataset_id: str) -> List[Chunk]:
        return self.get_dataset(dataset_id).get("chunks", [])

    def list_datasets(self) -> List[Dict]:
        return [
            {
                "id": ds["id"],
                "path": ds["path"],
                "num_documents": len(ds["documents"]),
                "num_queries": len(ds["queries"]),
                "num_chunks": len(ds["chunks"]),
            }
            for ds in self._datasets.values()
        ]
