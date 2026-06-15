# Homework 1: Agentic RAG

This homework builds a RAG system over the LLM Zoomcamp lesson pages, measures prompt token usage, adds chunking, and turns the search flow into a simple tool-calling agent.

## Files

- `hw1.py`: script used to compute the answers
- `hw1_answers.md`: final answers and measured run output

## Setup

Create a local `.env` file in the repository root:

```bash
OPENAI_API_KEY=your-api-key
```

Install dependencies with `uv`:

```bash
uv sync
```

## Run

From the repository root:

```bash
uv run python 01-agentic-rag/hw1.py
```

To run only the non-LLM questions:

```bash
uv run python 01-agentic-rag/hw1.py --skip-llm
```

## Final Answers

See [hw1_answers.md](hw1_answers.md).

