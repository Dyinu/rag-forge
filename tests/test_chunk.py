"""Tests for chunking functions."""

from rag_forge.chunk import fixed_chunk, recursive_chunk, semantic_chunk


class TestFixedChunk:
    def test_basic_split(self):
        text = "a" * 1000
        chunks = fixed_chunk(text, chunk_size=300, overlap=0)
        assert len(chunks) == 4  # 300+300+300+100
        assert all(len(c) <= 300 for c in chunks)

    def test_overlap(self):
        text = "word " * 200  # 1000 chars
        chunks = fixed_chunk(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 2
        # second chunk should start with tail of first
        # (approximately — depends on word boundaries in overlap)

    def test_empty_input(self):
        assert fixed_chunk("") == []
        assert fixed_chunk("   ") == []

    def test_short_text(self):
        chunks = fixed_chunk("hello world", chunk_size=500)
        assert chunks == ["hello world"]


class TestRecursiveChunk:
    def test_splits_on_paragraphs(self):
        text = ("Paragraph one with some content.\n\n"
                "Paragraph two has more words.\n\n"
                "Paragraph three is here too.")
        chunks = recursive_chunk(text, chunk_size=40, overlap=0)
        assert len(chunks) >= 2

    def test_falls_back_to_sentences(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = recursive_chunk(text, chunk_size=30, overlap=0)
        assert len(chunks) >= 2

    def test_empty_input(self):
        assert recursive_chunk("") == []

    def test_short_text_no_split(self):
        chunks = recursive_chunk("short text", chunk_size=100)
        assert chunks == ["short text"]


class TestSemanticChunk:
    def test_merges_small_paragraphs(self):
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = semantic_chunk(text, max_chunk_size=1000, min_chunk_size=100)
        # all paragraphs are small, should merge into one chunk
        assert len(chunks) == 1

    def test_splits_large_paragraphs(self):
        text = ("Long sentence. " * 50) + "\n\n" + ("Another sentence. " * 50)
        chunks = semantic_chunk(text, max_chunk_size=200)
        assert len(chunks) >= 2

    def test_empty_input(self):
        assert semantic_chunk("") == []
        assert semantic_chunk("  \n\n  ") == []

    def test_single_paragraph(self):
        text = "Just one paragraph with enough content to be a chunk."
        chunks = semantic_chunk(text, max_chunk_size=1000, min_chunk_size=10)
        assert len(chunks) == 1
        assert chunks[0] == text
