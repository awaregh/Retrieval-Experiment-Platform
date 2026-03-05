"""
Chunking strategies for document ingestion.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


class FixedTokenChunker:
    """
    Splits text into fixed-size token chunks.
    Uses a simple whitespace tokenizer; swap tokenizer for production use.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, tokenizer: str = "whitespace"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer == "whitespace":
            return text.split()
        # extend with tiktoken etc.
        return text.split()

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        tokens = self._tokenize(text)
        if not tokens:
            return []

        chunks: List[Chunk] = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        char_positions = self._build_char_positions(text, tokens)

        for i, start in enumerate(range(0, len(tokens), step)):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)
            start_char = char_positions[start] if start < len(char_positions) else len(text)
            end_char = char_positions[end - 1] + len(tokens[end - 1]) if end > 0 and end - 1 < len(char_positions) else len(text)
            chunks.append(Chunk(
                chunk_id=f"{document_id}_chunk_{i}",
                document_id=document_id,
                text=chunk_text,
                start_char=start_char,
                end_char=end_char,
                chunk_index=i,
            ))
            if end >= len(tokens):
                break

        return chunks

    def _build_char_positions(self, text: str, tokens: List[str]) -> List[int]:
        """Find character start position for each token, advancing sequentially."""
        positions = []
        pos = 0
        for token in tokens:
            # Skip whitespace to find the exact token boundary
            while pos < len(text) and text[pos].isspace():
                pos += 1
            idx = text.find(token, pos)
            if idx == -1:
                idx = pos
            positions.append(idx)
            pos = idx + len(token)
        return positions


class SlidingWindowChunker:
    """
    Overlapping sliding window chunker with configurable overlap.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100, tokenizer: str = "whitespace"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base = FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap, tokenizer=tokenizer)

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        # SlidingWindowChunker is essentially a FixedTokenChunker with overlap
        return self.base.chunk(text, document_id)


class SemanticChunker:
    """
    Semantic chunker that splits text at sentence boundaries,
    then groups sentences until they exceed the chunk_size threshold.
    """

    def __init__(self, chunk_size: int = 500, similarity_threshold: float = 0.5):
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s for s in sentences if s.strip()]

    def chunk(self, text: str, document_id: str) -> List[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: List[Chunk] = []
        current_sentences: List[str] = []
        current_len = 0
        chunk_index = 0
        char_pos = 0

        for sentence in sentences:
            word_count = len(sentence.split())
            if current_len + word_count > self.chunk_size and current_sentences:
                chunk_text = " ".join(current_sentences)
                start_char = text.find(current_sentences[0], char_pos)
                if start_char == -1:
                    start_char = char_pos
                end_char = start_char + len(chunk_text)
                chunks.append(Chunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    text=chunk_text,
                    start_char=max(0, start_char),
                    end_char=end_char,
                    chunk_index=chunk_index,
                ))
                char_pos = end_char
                chunk_index += 1
                current_sentences = []
                current_len = 0

            current_sentences.append(sentence)
            current_len += word_count

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            start_char = text.find(current_sentences[0], char_pos)
            if start_char == -1:
                start_char = char_pos
            end_char = start_char + len(chunk_text)
            chunks.append(Chunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                text=chunk_text,
                start_char=max(0, start_char),
                end_char=end_char,
                chunk_index=chunk_index,
            ))

        return chunks


def get_chunker(strategy: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Factory function to get a chunker by strategy name."""
    strategies = {
        "fixed": FixedTokenChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        "sliding_window": SlidingWindowChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        "semantic": SemanticChunker(chunk_size=chunk_size),
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: {list(strategies.keys())}")
    return strategies[strategy]
