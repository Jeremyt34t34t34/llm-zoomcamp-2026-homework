import argparse
from typing import Any

import numpy as np
from embedder import Embedder
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch
from tqdm.auto import tqdm


COMMIT_ID = "8c1834d"
Q1_QUERY = "How does approximate nearest neighbor search work?"
Q4_QUERY = "What metric do we use to evaluate a search engine?"
Q5_QUERY = "How do I store vectors in PostgreSQL?"
Q6_QUERY = "How do I give the model access to tools?"
TARGET_FILENAME = "02-vector-search/lessons/07-sqlitesearch-vector.md"


def load_lesson_documents() -> list[dict[str, str]]:
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id=COMMIT_ID,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    return [file.parse() for file in reader.read()]


def embed_texts(embedder: Embedder, texts: list[str], batch_size: int = 16) -> np.ndarray:
    vectors = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        batch_vectors = embedder.encode_batch(batch)
        vectors.extend(batch_vectors)

    return np.array(vectors)


def build_vector_index(vectors: np.ndarray, documents: list[dict[str, Any]]) -> VectorSearch:
    index = VectorSearch(keyword_fields=["filename", "start"])
    index.fit(vectors, documents)
    return index


def build_text_index(documents: list[dict[str, Any]]) -> Index:
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename", "start"],
    )
    index.fit(documents)
    return index


def rrf(result_lists: list[list[dict[str, Any]]], k: int = 60, num_results: int = 5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def filenames(results: list[dict[str, Any]]) -> list[str]:
    return [doc["filename"] for doc in results]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Xenova/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    embedder = Embedder(args.model_path)

    q1_vector = embedder.encode(Q1_QUERY)
    print("Q1 v[0]:", q1_vector[0])

    documents = load_lesson_documents()
    print("Documents:", len(documents))

    target_doc = next(doc for doc in documents if doc["filename"] == TARGET_FILENAME)
    target_vector = embedder.encode(target_doc["content"])
    q2_similarity = q1_vector.dot(target_vector)
    print("Q2 similarity:", q2_similarity)

    chunks = chunk_documents(documents, size=2000, step=1000)
    print("Chunks:", len(chunks))

    chunk_texts = [chunk["content"] for chunk in chunks]
    X = embed_texts(embedder, chunk_texts, batch_size=args.batch_size)

    q3_scores = X.dot(q1_vector)
    q3_idx = int(np.argmax(q3_scores))
    print("Q3 top score:", q3_scores[q3_idx])
    print("Q3 filename:", chunks[q3_idx]["filename"])
    print("Q3 start:", chunks[q3_idx]["start"])

    vector_index = build_vector_index(X, chunks)
    text_index = build_text_index(chunks)

    q4_vector = embedder.encode(Q4_QUERY)
    q4_results = vector_index.search(q4_vector, num_results=5)
    print("Q4 vector filenames:", filenames(q4_results))
    print("Q4 first filename:", q4_results[0]["filename"])

    q5_vector = embedder.encode(Q5_QUERY)
    q5_vector_results = vector_index.search(q5_vector, num_results=5)
    q5_text_results = text_index.search(Q5_QUERY, num_results=5)
    q5_vector_files = set(filenames(q5_vector_results))
    q5_text_files = set(filenames(q5_text_results))
    print("Q5 vector filenames:", filenames(q5_vector_results))
    print("Q5 text filenames:", filenames(q5_text_results))
    print("Q5 vector-only filenames:", sorted(q5_vector_files - q5_text_files))

    q6_vector = embedder.encode(Q6_QUERY)
    q6_vector_results = vector_index.search(q6_vector, num_results=5)
    q6_text_results = text_index.search(Q6_QUERY, num_results=5)
    q6_rrf_results = rrf([q6_vector_results, q6_text_results])
    print("Q6 vector filenames:", filenames(q6_vector_results))
    print("Q6 text filenames:", filenames(q6_text_results))
    print("Q6 RRF filenames:", filenames(q6_rrf_results))
    print("Q6 first filename:", q6_rrf_results[0]["filename"])


if __name__ == "__main__":
    main()

