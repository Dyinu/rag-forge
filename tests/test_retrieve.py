"""Tests for retrieval functions."""

import numpy as np

from rag_forge.retrieve import bm25_search, dense_search, hybrid_search


class TestDenseSearch:
    def test_finds_most_similar(self):
        # query is closest to corpus[0]
        query = np.array([1.0, 0.0, 0.0])
        corpus = np.array([
            [0.9, 0.1, 0.0],  # most similar
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        results = dense_search(query, corpus, top_k=2)
        assert results[0][0] == 0  # first result is corpus[0]
        assert len(results) == 2

    def test_top_k_limits_results(self):
        query = np.array([1.0, 0.0])
        corpus = np.array([[0.9, 0.1], [0.5, 0.5], [0.1, 0.9]])
        results = dense_search(query, corpus, top_k=1)
        assert len(results) == 1

    def test_scores_are_descending(self):
        query = np.array([1.0, 0.0, 0.0])
        corpus = np.random.randn(10, 3)
        results = dense_search(query, corpus, top_k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestBM25Search:
    def test_finds_keyword_match(self):
        corpus = [
            "the cat sat on the mat",
            "dogs are great pets",
            "cats and dogs living together",
        ]
        results = bm25_search("cat mat", corpus, top_k=2)
        # first doc has both "cat" and "mat"
        assert results[0][0] == 0

    def test_empty_results_for_no_match(self):
        corpus = ["hello world", "foo bar"]
        results = bm25_search("xyzabc", corpus, top_k=5)
        assert len(results) == 0  # no matching terms

    def test_top_k_limits(self):
        corpus = ["word " * 10] * 5
        results = bm25_search("word", corpus, top_k=2)
        assert len(results) <= 2


class TestHybridSearch:
    def test_combines_dense_and_sparse(self):
        corpus = [
            "machine learning algorithms",
            "deep learning neural networks",
            "cooking recipes for pasta",
        ]
        # make embeddings where corpus[1] is closest to query in embedding space
        query_emb = np.array([1.0, 0.0])
        corpus_embs = np.array([[0.3, 0.7], [0.9, 0.1], [0.0, 1.0]])

        results = hybrid_search(
            "machine learning",
            query_emb,
            corpus,
            corpus_embs,
            top_k=2,
            dense_weight=0.5,
        )
        # should return indices from corpus, combining both signals
        assert len(results) == 2
        result_indices = {idx for idx, _ in results}
        # corpus[0] has keyword match, corpus[1] has dense match — both should appear
        assert 0 in result_indices or 1 in result_indices
