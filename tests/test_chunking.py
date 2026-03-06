import pytest
from core.chunking import (
    FixedTokenChunker,
    SlidingWindowChunker,
    SemanticChunker,
    get_chunker,
    Chunk,
)

SAMPLE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
    "Sphinx of black quartz, judge my vow."
)

LONG_TEXT = " ".join([f"word{i}" for i in range(200)])


class TestFixedTokenChunker:
    def test_basic_chunking(self):
        chunker = FixedTokenChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT, "doc_1")
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_ids_are_unique(self):
        chunker = FixedTokenChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(LONG_TEXT, "doc_1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_size_respected(self):
        chunker = FixedTokenChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk(LONG_TEXT, "doc_1")
        for chunk in chunks[:-1]:  # last chunk may be smaller
            token_count = len(chunk.text.split())
            assert token_count <= 5

    def test_overlap_creates_more_chunks(self):
        chunker_no_overlap = FixedTokenChunker(chunk_size=10, chunk_overlap=0)
        chunker_overlap = FixedTokenChunker(chunk_size=10, chunk_overlap=5)
        chunks_no_overlap = chunker_no_overlap.chunk(LONG_TEXT, "doc_1")
        chunks_overlap = chunker_overlap.chunk(LONG_TEXT, "doc_1")
        assert len(chunks_overlap) >= len(chunks_no_overlap)

    def test_chunk_document_id(self):
        chunker = FixedTokenChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(LONG_TEXT, "my_document")
        assert all(c.document_id == "my_document" for c in chunks)

    def test_chunk_indices_sequential(self):
        chunker = FixedTokenChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk(LONG_TEXT, "doc_1")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_empty_text_returns_no_chunks(self):
        chunker = FixedTokenChunker(chunk_size=10, chunk_overlap=0)
        chunks = chunker.chunk("", "doc_1")
        assert chunks == []

    def test_text_shorter_than_chunk_size(self):
        chunker = FixedTokenChunker(chunk_size=1000, chunk_overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT, "doc_1")
        assert len(chunks) == 1
        assert chunks[0].text == SAMPLE_TEXT.strip() or SAMPLE_TEXT in chunks[0].text

    def test_chunk_text_covers_all_tokens(self):
        chunker = FixedTokenChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(LONG_TEXT, "doc_1")
        # All words from the original text should appear in at least one chunk
        all_words = set(LONG_TEXT.split())
        chunk_words = set()
        for c in chunks:
            chunk_words.update(c.text.split())
        assert all_words == chunk_words


class TestSlidingWindowChunker:
    def test_creates_chunks(self):
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=3)
        chunks = chunker.chunk(LONG_TEXT, "doc_1")
        assert len(chunks) > 0

    def test_overlap_behavior(self):
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=5)
        chunks = chunker.chunk(LONG_TEXT, "doc_1")
        # With overlap, adjacent chunks should share tokens
        if len(chunks) >= 2:
            tokens_0 = set(chunks[0].text.split())
            tokens_1 = set(chunks[1].text.split())
            assert len(tokens_0 & tokens_1) > 0

    def test_chunk_ids_unique(self):
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=3)
        chunks = chunker.chunk(LONG_TEXT, "doc_1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestSemanticChunker:
    def test_creates_chunks(self):
        chunker = SemanticChunker(chunk_size=10)
        chunks = chunker.chunk(SAMPLE_TEXT, "doc_1")
        assert len(chunks) > 0

    def test_chunks_contain_sentences(self):
        chunker = SemanticChunker(chunk_size=100)
        # With large chunk_size, entire text should fit in one chunk
        chunks = chunker.chunk(SAMPLE_TEXT, "doc_1")
        assert len(chunks) >= 1

    def test_small_chunk_size_splits(self):
        long_text = ". ".join([f"This is sentence number {i} about topic {i}" for i in range(20)]) + "."
        chunker = SemanticChunker(chunk_size=10)
        chunks = chunker.chunk(long_text, "doc_1")
        assert len(chunks) > 1

    def test_empty_text(self):
        chunker = SemanticChunker(chunk_size=10)
        chunks = chunker.chunk("", "doc_1")
        assert chunks == []

    def test_chunk_document_id(self):
        chunker = SemanticChunker(chunk_size=10)
        chunks = chunker.chunk(SAMPLE_TEXT, "my_doc")
        assert all(c.document_id == "my_doc" for c in chunks)

    def test_chunk_ids_unique(self):
        long_text = ". ".join([f"Sentence {i} about something important" for i in range(30)]) + "."
        chunker = SemanticChunker(chunk_size=5)
        chunks = chunker.chunk(long_text, "doc_1")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestGetChunkerFactory:
    def test_fixed_strategy(self):
        chunker = get_chunker("fixed")
        assert isinstance(chunker, FixedTokenChunker)

    def test_sliding_window_strategy(self):
        chunker = get_chunker("sliding_window")
        assert isinstance(chunker, SlidingWindowChunker)

    def test_semantic_strategy(self):
        chunker = get_chunker("semantic")
        assert isinstance(chunker, SemanticChunker)

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("invalid_strategy")

    def test_custom_chunk_size(self):
        chunker = get_chunker("fixed", chunk_size=200, chunk_overlap=20)
        assert chunker.chunk_size == 200
        assert chunker.chunk_overlap == 20
