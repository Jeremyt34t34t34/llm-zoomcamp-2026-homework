# Homework 5: Monitoring

This homework instruments a course-lessons RAG system with OpenTelemetry.

- `hw5.py`: script used to compute the homework answers
- `hw5_answers.md`: final answers and measured values

The script records `rag`, `search`, and `llm` spans, stores span attributes
such as tokens and cost, and persists spans to SQLite for querying.

## Run

```bash
uv run python 05-monitoring/hw5.py
```

Use `--console` to also print spans with `ConsoleSpanExporter`.

## Concept Map

- One RAG request becomes one trace.
- `rag`, `search`, and `llm` are spans inside that trace.
- Token usage and cost are span attributes on the `llm` span.
- `SQLiteSpanExporter` persists finished spans into `traces.db`.
- SQL/pandas queries over `traces.db` answer the homework questions.
