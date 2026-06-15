import argparse
import json
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI


COMMIT_ID = "8c1834d"
RAG_QUESTION = "How does the agentic loop keep calling the model until it stops?"
AGENT_QUESTION = "How does the agentic loop work, and how is it different from plain RAG?"
MODEL = "gpt-5.4-mini"

RAG_INSTRUCTIONS = """
You're a course teaching assistant.
Answer the student's question using only the provided context.
If the context doesn't contain the answer, say that you don't know.
""".strip()

AGENT_INSTRUCTIONS = """
You're a course teaching assistant. Answer the student's question using the
search tool. Make multiple searches with different keywords before answering.
""".strip()


@dataclass
class RagResult:
    answer: str
    input_tokens: int | None


def load_lesson_documents() -> list[dict[str, str]]:
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id=COMMIT_ID,
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    return [file.parse() for file in reader.read()]


def build_index(documents: list[dict[str, Any]]) -> Index:
    index = Index(
        text_fields=["content"],
        keyword_fields=["filename"],
    )
    index.fit(documents)
    return index


def build_context(search_results: list[dict[str, Any]]) -> str:
    lines = []

    for doc in search_results:
        lines.append(f"filename: {doc['filename']}")
        if "start" in doc:
            lines.append(f"start: {doc['start']}")
        lines.append(doc["content"])
        lines.append("")

    return "\n\n".join(lines).strip()


def search_index(index: Index, query: str, num_results: int = 5) -> list[dict[str, Any]]:
    return index.search(
        query,
        boost_dict={"content": 1.0},
        num_results=num_results,
    )


def run_rag(client: OpenAI, index: Index, question: str, model: str = MODEL) -> RagResult:
    search_results = search_index(index, question)
    context = build_context(search_results)
    prompt = f"""
Question:
{question}

Context:
{context}
""".strip()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "developer", "content": RAG_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ],
    )

    return RagResult(
        answer=response.output_text,
        input_tokens=get_input_tokens(response),
    )


def get_input_tokens(response: Any) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None

    for attr in ("input_tokens", "prompt_tokens"):
        value = getattr(usage, attr, None)
        if value is not None:
            return value

    if isinstance(usage, dict):
        return usage.get("input_tokens") or usage.get("prompt_tokens")

    return None


def make_search_tool(index: Index):
    def search(query: str) -> list[dict[str, Any]]:
        """Search the course lesson database for passages matching the query."""
        results = search_index(index, query, num_results=5)
        cleaned = []

        for result in results:
            cleaned.append(
                {
                    "filename": result["filename"],
                    "start": result.get("start"),
                    "content": result["content"][:1200],
                }
            )

        return cleaned

    return search


SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "name": "search",
    "description": "Search the course lesson database for passages matching the query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to run against the course lessons.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}


def run_agent(client: OpenAI, index: Index, question: str, model: str = MODEL) -> tuple[str, int]:
    search = make_search_tool(index)
    messages: list[Any] = [
        {"role": "developer", "content": AGENT_INSTRUCTIONS},
        {"role": "user", "content": question},
    ]
    last_answer = ""
    search_calls = 0

    for _ in range(10):
        response = client.responses.create(
            model=model,
            input=messages,
            tools=[SEARCH_TOOL_SCHEMA],
        )

        messages.extend(response.output)
        has_function_calls = False

        for item in response.output:
            if item.type == "function_call":
                has_function_calls = True
                search_calls += 1
                args = json.loads(item.arguments)
                result = search(**args)
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps(result),
                    }
                )

            elif item.type == "message":
                last_answer = response.output_text

        if not has_function_calls:
            break

    return last_answer, search_calls


def print_non_llm_answers(documents: list[dict[str, str]], chunks: list[dict[str, Any]]) -> None:
    document_index = build_index(documents)
    first_result = search_index(document_index, RAG_QUESTION, num_results=1)[0]

    print("Q1 lesson pages:", len(documents))
    print("Q2 first filename:", first_result["filename"])
    print("Q4 chunks:", len(chunks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    load_dotenv()

    documents = load_lesson_documents()
    chunks = chunk_documents(documents, size=2000, step=1000)
    print_non_llm_answers(documents, chunks)

    if args.skip_llm:
        return

    client = OpenAI()
    document_index = build_index(documents)
    chunk_index = build_index(chunks)

    full_rag = run_rag(client, document_index, RAG_QUESTION, model=args.model)
    chunked_rag = run_rag(client, chunk_index, RAG_QUESTION, model=args.model)
    agent_answer, search_calls = run_agent(client, chunk_index, AGENT_QUESTION, model=args.model)

    print("Q3 full-page RAG input tokens:", full_rag.input_tokens)
    print("Q3 answer:", full_rag.answer)
    print("Q5 chunked RAG input tokens:", chunked_rag.input_tokens)
    if full_rag.input_tokens and chunked_rag.input_tokens:
        ratio = full_rag.input_tokens / chunked_rag.input_tokens
        fewer = full_rag.input_tokens - chunked_rag.input_tokens
        print("Q5 fewer tokens:", fewer)
        print("Q5 ratio:", round(ratio, 2))
    print("Q5 answer:", chunked_rag.answer)
    print("Q6 search calls:", search_calls)
    print("Q6 answer:", agent_answer)


if __name__ == "__main__":
    main()
