import argparse
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gitsource import GithubRepositoryDataReader
from minsearch import Index
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)


COMMIT = "8c1834d"
QUERY = "How does the agentic loop keep calling the model until it stops?"
INPUT_PRICE_PER_MILLION = 0.75
OUTPUT_PRICE_PER_MILLION = 4.50

ROOT_DIR = Path(__file__).resolve().parents[1]
COURSE_REPO_DIR = ROOT_DIR.parent

INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
""".strip()

PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()


def load_env() -> None:
    for env_path in [
        ROOT_DIR / ".env",
        COURSE_REPO_DIR / "04-evaluation" / "code" / ".env",
        COURSE_REPO_DIR / "03-orchestration" / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)


def load_lesson_documents() -> list[dict[str, str]]:
    try:
        reader = GithubRepositoryDataReader(
            repo_owner="DataTalksClub",
            repo_name="llm-zoomcamp",
            commit_id=COMMIT,
            allowed_extensions={"md"},
            filename_filter=lambda path: "/lessons/" in path,
        )
        return [file.parse() for file in reader.read()]
    except Exception as exc:
        print(f"GitHub download failed, falling back to local git checkout: {exc}")

    paths_output = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", COMMIT],
        cwd=COURSE_REPO_DIR,
        text=True,
    )

    documents = []

    for filename in paths_output.splitlines():
        if "/lessons/" not in filename or not filename.endswith(".md"):
            continue

        content = subprocess.check_output(
            ["git", "show", f"{COMMIT}:{filename}"],
            cwd=COURSE_REPO_DIR,
            text=True,
        )
        documents.append({"filename": filename, "content": content})

    return documents


def build_index(documents: list[dict[str, str]]) -> Index:
    index = Index(text_fields=["content"], keyword_fields=["filename"])
    index.fit(documents)
    return index


def calc_cost(usage) -> float:
    input_cost = usage.input_tokens / 1_000_000 * INPUT_PRICE_PER_MILLION
    output_cost = usage.output_tokens / 1_000_000 * OUTPUT_PRICE_PER_MILLION
    return input_cost + output_cost


class RAGBase:
    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-5.4-mini",
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc["filename"])
            lines.append(doc["content"])
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]

        return self.llm_client.responses.create(
            model=self.model,
            input=input_messages,
        )

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.output_text


class RAGTraced(RAGBase):
    def __init__(self, *args, tracer, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracer = tracer

    def search(self, query, num_results=5):
        with self.tracer.start_as_current_span("search") as span:
            span.set_attribute("query", query)
            span.set_attribute("num_results", num_results)
            results = super().search(query, num_results=num_results)
            span.set_attribute("result_count", len(results))
            return results

    def llm(self, prompt):
        with self.tracer.start_as_current_span("llm") as span:
            span.set_attribute("model", self.model)
            response = super().llm(prompt)
            usage = response.usage

            if usage is not None:
                span.set_attribute("input_tokens", usage.input_tokens)
                span.set_attribute("output_tokens", usage.output_tokens)
                span.set_attribute("cost", calc_cost(usage))

            return response

    def rag(self, query):
        with self.tracer.start_as_current_span("rag") as span:
            span.set_attribute("query", query)
            return super().rag(query)


class SQLiteSpanExporter(SpanExporter):
    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
            """
        )
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self, timeout_millis=30_000):
        self.conn.commit()
        return True


def setup_tracing(db_path: Path, console: bool):
    provider = TracerProvider()

    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter(str(db_path))))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("llm-zoomcamp")


def load_spans(db_path: Path) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT
                name,
                start_time,
                end_time,
                input_tokens,
                output_tokens,
                cost,
                (end_time - start_time) / 1000000.0 AS duration_ms
            FROM spans
            """,
            conn,
        )
    return df


def duration_bucket(ms: float) -> str:
    if ms < 100:
        return "Under 100ms"
    if ms < 500:
        return "100-500ms"
    if ms < 2000:
        return "500-2000ms"
    return "Over 2000ms"


def token_variation_bucket(tokens: list[int]) -> str:
    if len(set(tokens)) == 1:
        return "They're identical"

    low = min(tokens)
    high = max(tokens)
    variation = (high - low) / low

    if variation <= 0.10:
        return "Within 10% of each other"
    if variation <= 0.50:
        return "Within 50% of each other"
    return "They vary more than 50%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="traces.db")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--console", action="store_true")
    parser.add_argument("--runs", type=int, default=4)
    parser.add_argument("--keep-db", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if db_path.exists() and not args.keep_db:
        db_path.unlink()

    load_env()
    tracer = setup_tracing(db_path, args.console)

    documents = load_lesson_documents()
    index = build_index(documents)
    client = OpenAI()
    rag = RAGTraced(index=index, llm_client=client, tracer=tracer, model=args.model)

    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}")
        answer = rag.rag(QUERY)
        print(answer[:200].replace("\n", " ").strip())

    provider = trace.get_tracer_provider()
    provider.force_flush()

    df = load_spans(db_path)
    span_names = sorted(df["name"].unique().tolist())
    llm_spans = df[df["name"] == "llm"].copy()
    child_durations = (
        df[df["name"] != "rag"].groupby("name")["duration_ms"].sum().sort_values(ascending=False)
    )
    input_tokens = llm_spans["input_tokens"].dropna().astype(int).tolist()

    q1 = len(df) // args.runs
    q2 = input_tokens[0]
    q3_ms = float(llm_spans["duration_ms"].iloc[1:].median()) if len(llm_spans) > 1 else float(llm_spans["duration_ms"].iloc[0])
    q3 = duration_bucket(q3_ms)
    q4 = span_names
    q5 = child_durations.index[0]
    q6 = token_variation_bucket(input_tokens)

    print("\nSummary")
    print("Documents:", len(documents))
    print("Total spans:", len(df))
    print("Span names:", span_names)
    print("LLM input tokens:", input_tokens)
    print("Duration by span, excluding rag:")
    print(child_durations)
    print()
    print("Q1 span count per trace:", q1)
    print("Q2 first input tokens:", q2)
    print("Q3 typical llm duration:", q3, f"({q3_ms:.1f}ms)")
    print("Q4 span names:", ", ".join(q4))
    print("Q5 slowest child span:", q5)
    print("Q6 token variation:", q6)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
