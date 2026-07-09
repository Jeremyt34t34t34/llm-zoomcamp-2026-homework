import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE2_DIR = ROOT_DIR / "02-vector-search"
COURSE_REPO_DIR = ROOT_DIR.parent

sys.path.insert(0, str(MODULE2_DIR))

from embedder import Embedder  # noqa: E402
from hw2 import (  # noqa: E402
    build_text_index,
    build_vector_index,
    embed_texts,
    rrf,
)


DATA_GEN_INSTRUCTIONS = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()


class Questions(BaseModel):
    questions: list[str]


def load_lesson_documents() -> list[dict[str, str]]:
    try:
        from hw2 import load_lesson_documents as load_from_github

        return load_from_github()
    except Exception as exc:
        print(f"GitHub download failed, falling back to local git checkout: {exc}")

    commit = "8c1834d"
    paths_output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=COURSE_REPO_DIR,
        text=True,
    )

    documents = []

    for filename in paths_output.splitlines():
        if "/lessons/" not in filename or not filename.endswith(".md"):
            continue

        content = subprocess.check_output(
            ["git", "show", f"{commit}:{filename}"],
            cwd=COURSE_REPO_DIR,
            text=True,
        )
        documents.append({"filename": filename, "content": content})

    return documents


def llm_structured(client, instructions, user_prompt, output_type, model="gpt-5.4-mini"):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]

    response = client.responses.parse(
        model=model,
        input=messages,
        text_format=output_type,
    )

    return response.output_parsed, response.usage


def generate_questions_for_first_three(documents, model: str):
    load_dotenv(COURSE_REPO_DIR / "04-evaluation" / "code" / ".env")
    client = OpenAI()

    usage_records = []
    generated = []

    for doc in documents[:3]:
        user_prompt = json.dumps(
            {
                "filename": doc["filename"],
                "content": doc["content"],
            },
            ensure_ascii=False,
        )
        result, usage = llm_structured(
            client,
            DATA_GEN_INSTRUCTIONS,
            user_prompt,
            Questions,
            model=model,
        )
        usage_records.append(usage)
        for question in result.questions:
            generated.append({"question": question, "filename": doc["filename"]})

    avg_input_tokens = sum(u.input_tokens for u in usage_records) / len(usage_records)

    return avg_input_tokens, generated


def compute_relevance(rec, search_function):
    results = search_function(rec["question"])
    return [int(doc["filename"] == rec["filename"]) for doc in results]


def compute_relevance_total(ground_truth, search_function):
    relevance = []
    for rec in tqdm(ground_truth):
        relevance.append(compute_relevance(rec, search_function))
    return relevance


def hit_rate(relevance):
    return sum(1 for line in relevance if 1 in line) / len(relevance)


def mrr(relevance):
    total = 0.0

    for line in relevance:
        for rank, value in enumerate(line):
            if value == 1:
                total += 1 / (rank + 1)
                break

    return total / len(relevance)


def evaluate(ground_truth, search_function):
    relevance = compute_relevance_total(ground_truth, search_function)
    return {
        "hit_rate": hit_rate(relevance),
        "mrr": mrr(relevance),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default=str(
            COURSE_REPO_DIR
            / ".hf-cache/hub/models--Xenova--all-MiniLM-L6-v2/snapshots/751bff37182d3f1213fa05d7196b954e230abad9"
        ),
    )
    parser.add_argument("--llm-model", default="gpt-5.4-mini")
    parser.add_argument("--skip-q1", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    documents = load_lesson_documents()
    print("documents:", len(documents))

    if args.skip_q1:
        print("Q1 skipped")
    else:
        avg_input_tokens, generated = generate_questions_for_first_three(
            documents, args.llm_model
        )
        print("Q1 avg input tokens:", avg_input_tokens)
        print("Q1 generated records:", len(generated))

    ground_truth_path = COURSE_REPO_DIR / "cohorts/2026/04-evaluation/ground-truth.csv"
    df_ground_truth = pd.read_csv(ground_truth_path)
    ground_truth = df_ground_truth.to_dict(orient="records")
    print("ground truth:", len(ground_truth))

    from gitsource import chunk_documents

    chunks = chunk_documents(documents, size=2000, step=1000)
    print("chunks:", len(chunks))

    text_index = build_text_index(chunks)

    embedder = Embedder(args.model_path)
    vectors = embed_texts(embedder, [chunk["content"] for chunk in chunks], args.batch_size)
    vector_index = build_vector_index(np.array(vectors), chunks)

    def text_search(query, num_results=5):
        return text_index.search(query, num_results=num_results)

    def vector_search(query, num_results=5):
        query_vector = embedder.encode(query)
        return vector_index.search(query_vector, num_results=num_results)

    def hybrid_search(query, k=60, num_results=5):
        text_results = text_search(query, num_results=10)
        vector_results = vector_search(query, num_results=10)
        return rrf([text_results, vector_results], k=k, num_results=num_results)

    first_question = ground_truth[0]["question"]
    text_first = text_search(first_question)[0]["filename"]
    vector_first = vector_search(first_question)[0]["filename"]

    print("Q2 text first filename:", text_first)
    print("Q3 vector first filename:", vector_first)

    text_eval = evaluate(ground_truth, text_search)
    print("Q4 text eval:", text_eval)

    vector_eval = evaluate(ground_truth, vector_search)
    print("Q5 vector eval:", vector_eval)

    hybrid_results = {}
    for k in [1, 50, 100, 200]:
        result = evaluate(ground_truth, lambda query, k=k: hybrid_search(query, k=k))
        hybrid_results[k] = result
        print(f"Q6 hybrid k={k}:", result)

    best_k = max(hybrid_results, key=lambda k: hybrid_results[k]["mrr"])
    print("Q6 best k:", best_k)


if __name__ == "__main__":
    main()
