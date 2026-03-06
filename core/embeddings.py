"""
Embedding pipeline supporting SentenceTransformers and OpenAI.
Includes batching and caching.
"""
from __future__ import annotations
import hashlib
import logging
from collections import OrderedDict
from typing import List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """In-memory LRU embedding cache using an OrderedDict."""

    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self.max_size = max_size

    def _key(self, text: str, model: str) -> str:
        return hashlib.md5(f"{model}::{text}".encode()).hexdigest()

    def get(self, text: str, model: str) -> Optional[List[float]]:
        key = self._key(text, model)
        if key in self._cache:
            # Move to end to mark as recently used
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, text: str, model: str, embedding: List[float]) -> None:
        key = self._key(text, model)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                # Evict least recently used (first item)
                self._cache.popitem(last=False)
        self._cache[key] = embedding

    def clear(self) -> None:
        self._cache.clear()


_cache = EmbeddingCache()


class SentenceTransformerEmbedder:
    """Embedding model backed by sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        return self._model

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        model = self._load_model()
        results: List[Optional[List[float]]] = [None] * len(texts)

        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cached = _cache.get(text, self.model_name)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[batch_start:batch_start + batch_size]
            batch_indices = uncached_indices[batch_start:batch_start + batch_size]
            embeddings = model.encode(batch_texts, batch_size=batch_size, show_progress_bar=False)
            for idx, emb in zip(batch_indices, embeddings):
                emb_list = emb.tolist()
                _cache.set(texts[idx], self.model_name, emb_list)
                results[idx] = emb_list

        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()


class OpenAIEmbedder:
    """Embedding model backed by OpenAI API."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key

    def _client(self):
        try:
            from openai import OpenAI
            return OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

    def embed(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        client = self._client()
        results: List[Optional[List[float]]] = [None] * len(texts)

        uncached_indices = []
        uncached_texts = []
        for i, text in enumerate(texts):
            cached = _cache.get(text, self.model_name)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        for batch_start in range(0, len(uncached_texts), batch_size):
            batch_texts = uncached_texts[batch_start:batch_start + batch_size]
            batch_indices = uncached_indices[batch_start:batch_start + batch_size]
            response = client.embeddings.create(model=self.model_name, input=batch_texts)
            for idx, item in zip(batch_indices, response.data):
                _cache.set(texts[idx], self.model_name, item.embedding)
                results[idx] = item.embedding

        return results  # type: ignore[return-value]

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def dimension(self) -> int:
        dim_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return dim_map.get(self.model_name, 1536)


def get_embedder(model_name: str, api_key: Optional[str] = None):
    """Factory function to get an embedder by model name."""
    openai_models = {"text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"}
    if model_name in openai_models:
        return OpenAIEmbedder(model_name=model_name, api_key=api_key)
    return SentenceTransformerEmbedder(model_name=model_name)
