# Homework 5: Monitoring

## Final Answers

1. `3`
2. `7000`
3. `500-2000ms`
4. `rag`, `search`, and `llm`
5. `llm`
6. `They're identical`

## Run Command

```bash
UV_CACHE_DIR=/Users/jeremyzhong/Desktop/llm-zoomcamp/.uv-cache \
uv run python 05-monitoring/hw5.py
```

## Run Evidence

The script ran the homework query four times:

```text
How does the agentic loop keep calling the model until it stops?
```

Observed output:

```text
Documents: 72
Total spans: 12
Span names: ['llm', 'rag', 'search']
LLM input tokens: [7110, 7110, 7110, 7110]

Duration by span, excluding rag:
llm       8946.341 ms
search       5.413 ms

Q1 span count per trace: 3
Q2 first input tokens: 7110
Q3 typical llm duration: 500-2000ms (1762.8ms)
Q4 span names: llm, rag, search
Q5 slowest child span: llm
Q6 token variation: They're identical
```
