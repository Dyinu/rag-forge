# rag-forge

Opinionated RAG pipeline benchmarking. Stop guessing your chunking/embedding/retrieval config — test them all and pick the winner.

## Why I built this

At my last role I spent weeks manually testing different chunking strategies, embedding models, and retrieval methods for our RAG pipeline. Every time we changed one thing, we had to re-run eval manually. I wanted a tool that would just test every combination and tell me which one works best.

This isn't a framework. It's a benchmark runner with three hardcoded embedding models, three retrieval methods, four chunking strategies, and one reranker. It's opinionated because most of the "configurable" RAG frameworks I've seen are more work to configure than to just write the code yourself.

## What it does

Give it your documents and a CSV of question/answer pairs. It runs every combination of:

- **Chunking:** fixed (256), fixed (512), recursive (512), semantic
- **Embeddings:** BGE-small, E5-small, OpenAI text-embedding-3-small
- **Retrieval:** dense, BM25, hybrid (0.7 dense + 0.3 sparse)
- **Reranking:** cross-encoder (ms-marco-MiniLM) or none

That's 4 × 3 × 3 × 2 = **72 configurations** (48 without OpenAI).

For each config it measures hit rate, MRR, context precision, and latency. Then it ranks them and generates a report.

## Quick start

```bash
pip install rag-forge

# run on the included sample dataset
rag-forge run --docs ./data/sample --qa ./data/sample/qa.csv --skip-openai
```

Output:
```
rag-forge — finding your optimal RAG config

Loading documents from ./data/sample...
  → 3 documents loaded
Loading QA pairs from ./data/sample/qa.csv...
  → 20 QA pairs loaded

Running 48 configurations...
  4 chunkers × 2 embedders × 3 retrievers × 2 rerankers
  ...

Top 5 Configurations:

┌───┬──────────────┬───────────┬──────────┬────────────────┬──────────┬───────┬─────────┐
│ # │ Chunker      │ Embedder  │ Retriever│ Reranker       │ Hit Rate │ MRR   │ Latency │
├───┼──────────────┼───────────┼──────────┼────────────────┼──────────┼───────┼─────────┤
│ 1 │ recursive_512│ bge-small │ hybrid   │ cross-encoder  │ 0.850    │ 0.783 │ 127ms   │
│ 2 │ semantic     │ bge-small │ hybrid   │ cross-encoder  │ 0.800    │ 0.742 │ 118ms   │
│ 3 │ recursive_512│ e5-small  │ hybrid   │ cross-encoder  │ 0.800    │ 0.717 │ 134ms   │
│ 4 │ fixed_512    │ bge-small │ hybrid   │ cross-encoder  │ 0.750    │ 0.683 │ 112ms   │
│ 5 │ recursive_512│ bge-small │ dense    │ cross-encoder  │ 0.750    │ 0.650 │ 95ms    │
└───┴──────────────┴───────────┴──────────┴────────────────┴──────────┴───────┴─────────┘
```

Results saved to `./results/results.md` and `./results/pareto.png`.

## The sample results

On the included 3-document dataset (20 QA pairs), the patterns are consistent with what I've seen on larger datasets:

1. **Hybrid retrieval + cross-encoder reranking wins.** Every time. The combo of dense + BM25 + reranking is the strongest.
2. **Recursive chunking slightly beats fixed.** Respecting sentence boundaries matters.
3. **BGE-small and E5-small are close.** The difference is usually <5% — pick whichever you prefer.
4. **Reranking adds ~30-50ms but boosts hit rate by 10-15%.** Worth it for most use cases.

These are small-scale results. On a production dataset with 5000+ documents, the gaps get bigger — especially the hybrid vs dense-only gap.

## CLI reference

```bash
# full benchmark
rag-forge run --docs ./my_docs --qa ./my_qa.csv

# skip OpenAI (no API key needed)
rag-forge run --docs ./my_docs --qa ./my_qa.csv --skip-openai

# skip reranker (faster, fewer configs)
rag-forge run --docs ./my_docs --qa ./my_qa.csv --skip-reranker

# custom output dir and top-k
rag-forge run --docs ./my_docs --qa ./my_qa.csv --output ./my_results --top-k 10
```

## QA file format

CSV with `question` and `answer` columns:

```csv
question,answer
What is RAG?,Retrieval-Augmented Generation combines retrieval with generation
What embedding model is best?,It depends on your use case
```

The answer should be a string that appears somewhere in your documents. The evaluation checks whether the retrieved chunks contain the ground truth answer.

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## How it works

The benchmark loop is straightforward:

1. **Chunk** all documents with each chunking strategy (cached — only done once per strategy)
2. **Embed** chunks with each embedding model (cached — only done once per chunker+embedder pair)
3. For each query, **retrieve** top-k chunks, optionally **rerank**, and collect the final chunks
4. **Evaluate** hit rate, MRR, and context precision against ground truth
5. **Rank** all configurations and generate the report

Caching embeddings is the key optimization — embedding is the slowest step, and we reuse the same embeddings across retrieval methods and rerankers.

## Limitations

- Only handles `.txt` and `.md` files (no PDF parsing — use a separate tool for that)
- The local embedding models (BGE-small, E5-small) are small variants — the large versions would score higher but take longer
- Evaluation is retrieval-only (no generation evaluation unless you bring an OpenAI key for RAGAS)
- Designed for English text

## License

MIT
