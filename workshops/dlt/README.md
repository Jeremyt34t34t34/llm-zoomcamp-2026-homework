# dlt Workshop Homework

This homework instruments the Module 1 FAQ agent with Pydantic Logfire,
then pulls trace data into DuckDB with dlt for analysis.

## Files

- `agent.py`: Pydantic AI FAQ agent and `search` tool
- `ingest.py`: downloads the FAQ data and builds a minsearch index
- `main.py`: runs the instrumented agent and sends traces to Logfire
- `logfire_pipeline.py`: pulls Logfire `records` into DuckDB with dlt
- `analyze.py`: queries DuckDB to compute the homework answers
- `dlt_answers.md`: final answers and run evidence

## Setup

Create `workshops/dlt/.env` or put the same variables in the repo-level
`.env`:

```bash
OPENAI_API_KEY=sk-...
LOGFIRE_API_KEY=...
LOGFIRE_REGION=us
```

`LOGFIRE_API_KEY` is used by `logfire_pipeline.py` to query/export
Logfire records. `main.py` also passes it to Logfire as the API key, and
uses it as the write token if `LOGFIRE_TOKEN` is not set.

`LOGFIRE_TOKEN` is optional for older Logfire projects that still expose a
separate write token.

## Run

```bash
uv run python workshops/dlt/main.py --question "How do I run Ollama locally?"
uv run python workshops/dlt/logfire_pipeline.py --hours 24
uv run python workshops/dlt/analyze.py
```

The dlt pipeline writes to `logfire_agent_traces.duckdb` and the dataset
name is `agent_traces`.
