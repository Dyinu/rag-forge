"""Integration tests for the benchmark runner.

These test the pipeline end-to-end with minimal config
(skipping heavy models to keep tests fast).
"""

import csv

from rag_forge.bench import BenchConfig, RunResult, load_documents, load_qa_pairs


class TestLoadDocuments:
    def test_loads_txt_files(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("Document one content")
        (tmp_path / "doc2.txt").write_text("Document two content")
        docs = load_documents(tmp_path)
        assert len(docs) == 2

    def test_loads_md_files(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Markdown doc")
        docs = load_documents(tmp_path)
        assert len(docs) == 1

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("")
        (tmp_path / "real.txt").write_text("content")
        docs = load_documents(tmp_path)
        assert len(docs) == 1

    def test_raises_if_no_files(self, tmp_path):
        import pytest

        with pytest.raises(FileNotFoundError):
            load_documents(tmp_path)


class TestLoadQAPairs:
    def test_loads_csv(self, tmp_path):
        csv_path = tmp_path / "qa.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerow(["What is ML?", "Machine learning is..."])
            writer.writerow(["What is DL?", "Deep learning is..."])
        questions, answers = load_qa_pairs(csv_path)
        assert len(questions) == 2
        assert questions[0] == "What is ML?"
        assert answers[1] == "Deep learning is..."

    def test_skips_empty_rows(self, tmp_path):
        csv_path = tmp_path / "qa.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerow(["Valid question", "Valid answer"])
            writer.writerow(["", ""])
            writer.writerow(["Another question", "Another answer"])
        questions, answers = load_qa_pairs(csv_path)
        assert len(questions) == 2


class TestBenchConfig:
    def test_default_config(self):
        config = BenchConfig()
        assert len(config.chunkers) > 0
        assert len(config.embedders) > 0
        assert config.top_k == 5

    def test_custom_config(self):
        config = BenchConfig(
            chunkers=["fixed_512"],
            embedders=["bge-small"],
            retrievers=["dense"],
            rerankers=["none"],
            top_k=3,
        )
        assert config.chunkers == ["fixed_512"]
        assert config.top_k == 3


class TestRunResult:
    def test_config_id(self):
        from rag_forge.evaluate import EvalResult

        result = RunResult(
            chunker="fixed_512",
            embedder="bge-small",
            retriever="dense",
            reranker="none",
            eval=EvalResult(0.8, None, None, 0.9, 0.85),
            latency_ms=50.0,
            num_chunks=100,
        )
        assert result.config_id == "fixed_512|bge-small|dense|none"
