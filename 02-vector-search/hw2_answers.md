# Homework 2: Vector Search Answers

Generated with:

```bash
UV_CACHE_DIR=../.uv-cache HF_HOME=../.hf-cache uv run python hw2.py
```

Before running the script, download the ONNX model:

```bash
UV_CACHE_DIR=../.uv-cache HF_HOME=../.hf-cache uv run python download.py
```

## Answers

1. Embedding a query, first value: `-0.02`
2. Cosine similarity: `0.37`
3. Chunking and search by hand: `02-vector-search/lessons/07-sqlitesearch-vector.md`
4. Vector search with minsearch: `04-evaluation/lessons/05-search-metrics.md`
5. Text search vs vector search: `02-vector-search/lessons/08-pgvector.md`
6. Hybrid search: `01-agentic-rag/lessons/13-function-calling.md`

## Run Output

The measured values from this run:

- Q1 `v[0]`: `-0.020582036807885073`
- Q2 similarity: `0.361070280302606`
- Documents: `72`
- Chunks: `295`
- Q3 top score: `0.648901732433228`
- Q3 top chunk start: `1000`
- Q5 vector-only file: `02-vector-search/lessons/08-pgvector.md`
- Q6 RRF first file: `01-agentic-rag/lessons/13-function-calling.md`

