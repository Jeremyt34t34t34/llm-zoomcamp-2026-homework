import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import logfire


ROOT_DIR = Path(__file__).resolve().parents[2]
COURSE_REPO_DIR = ROOT_DIR.parent
SCRIPT_DIR = Path(__file__).resolve().parent
HOMEWORK_QUERY = "How do I run Ollama locally?"


def load_env() -> None:
    for env_path in [
        SCRIPT_DIR / ".env",
        ROOT_DIR / ".env",
        COURSE_REPO_DIR / "04-evaluation" / "code" / ".env",
        COURSE_REPO_DIR / "03-orchestration" / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)


def configure_observability() -> None:
    api_key = os.environ.get("LOGFIRE_API_KEY")
    write_token = os.environ.get("LOGFIRE_TOKEN") or api_key

    logfire.configure(
        service_name="llm-zoomcamp-dlt-homework",
        token=write_token,
        api_key=api_key,
    )
    logfire.instrument_pydantic_ai()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=HOMEWORK_QUERY)
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()

    load_env()
    configure_observability()

    from agent import SearchDeps, faq_agent
    from ingest import build_index, load_faq_data

    documents = load_faq_data()
    index = build_index(documents)
    deps = SearchDeps(index=index)

    for run in range(args.runs):
        print(f"Run {run + 1}/{args.runs}: {args.question}")
        result = faq_agent.run_sync(args.question, deps=deps)
        print(result.output)


if __name__ == "__main__":
    main()
