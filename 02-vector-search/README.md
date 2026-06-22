# Homework 2: Vector Search

This homework uses the course lesson pages as the knowledge base and compares vector search, text search, and hybrid search.

## Files

- `download.py`: downloads the ONNX embedding model from Hugging Face
- `embedder.py`: lightweight ONNX `Embedder` used for the homework
- `hw2.py`: script used to compute the answers
- `hw2_answers.md`: final answers and measured run output

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Download the ONNX embedding model:

```bash
uv run python 02-vector-search/download.py
```

## Run

From the repository root:

```bash
uv run python 02-vector-search/hw2.py
```

## Final Answers

See [hw2_answers.md](hw2_answers.md).

