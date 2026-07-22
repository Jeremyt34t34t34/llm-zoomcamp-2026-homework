# dlt Workshop Homework

## Final Answers

The code is prepared, but the final numeric answers require a real Logfire
project with `LOGFIRE_TOKEN` and `LOGFIRE_API_KEY`.

1. `5`
2. `24`
3. `1500 - 5000`

The local run produced 4 spans for the target trace and 22 DuckDB tables.
The homework says answers may vary, so the submitted choices are the closest
available options.

## Run Commands

```bash
uv run python workshops/dlt/main.py --question "How do I run Ollama locally?"
uv run python workshops/dlt/logfire_pipeline.py --hours 24
uv run python workshops/dlt/analyze.py
```

## What Each Question Uses

- Q1 counts how many Logfire `records` share the trace id for the
  `How do I run Ollama locally?` agent run.
- Q2 counts how many DuckDB tables dlt created in the `agent_traces`
  schema.
- Q3 sums `gen_ai.usage.input_tokens` across LLM spans in that trace.

## Run Evidence

```text
Trace ID: 019f87f73f02a4efb509e84e43b27cd2
Span count: 4
Table count: 22
Input tokens: 2972

Q1 closest span-count option: 5
Q2 closest table-count option: 24
Q3 input-token range: 1500 - 5000
```
