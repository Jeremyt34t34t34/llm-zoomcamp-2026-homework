# Homework 4: Evaluation

## Final Answers

1. Average input tokens: **1400**
   - Measured average: `1349.0`

2. First text-search result: **`01-agentic-rag/lessons/03-rag.md`**

3. First vector-search result: **`01-agentic-rag/lessons/01-intro.md`**

4. Text-search Hit Rate: **0.76**
   - Measured: `0.7583333333333333`

5. Vector-search MRR: **0.55**
   - Measured: `0.5486111111111112`

6. Best hybrid RRF `k`: **1**
   - `k=1`: MRR `0.6481944444444449`
   - `k=50`: MRR `0.637916666666667`
   - `k=100`: MRR `0.637916666666667`
   - `k=200`: MRR `0.637916666666667`

## Run Command

```bash
UV_CACHE_DIR=/Users/jeremyzhong/Desktop/llm-zoomcamp/.uv-cache \
uv run python 04-evaluation/hw4.py \
  --model-path /Users/jeremyzhong/Desktop/llm-zoomcamp/llm-zoomcamp-code/02-vector-search/homework/models/Xenova/all-MiniLM-L6-v2
```

Use `--skip-q1` to avoid the OpenAI calls and recompute only Q2-Q6.
