import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import dlt
from dotenv import load_dotenv
import requests


ROOT_DIR = Path(__file__).resolve().parents[2]
COURSE_REPO_DIR = ROOT_DIR.parent
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SQL = """
SELECT
    trace_id,
    span_id,
    parent_span_id,
    start_timestamp,
    duration,
    service_name,
    span_name,
    message,
    attributes
FROM records
WHERE service_name = 'llm-zoomcamp-dlt-homework'
ORDER BY start_timestamp DESC
LIMIT 10000
""".strip()


def load_env() -> None:
    for env_path in [
        SCRIPT_DIR / ".env",
        ROOT_DIR / ".env",
        COURSE_REPO_DIR / "04-evaluation" / "code" / ".env",
        COURSE_REPO_DIR / "03-orchestration" / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)


def normalize_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    value_stripped = value.strip()
    if not value_stripped or value_stripped[0] not in "[{":
        return value

    try:
        return json.loads(value_stripped)
    except json.JSONDecodeError:
        return value


def query_logfire(sql: str, hours: int) -> list[dict[str, Any]]:
    api_token = os.environ.get("LOGFIRE_API_KEY") or os.environ.get("LOGFIRE_READ_TOKEN")
    if not api_token:
        raise RuntimeError(
            "LOGFIRE_API_KEY is missing. Add a scoped Logfire API key to .env. "
            "Legacy LOGFIRE_READ_TOKEN is also supported as a fallback."
        )

    region = os.environ.get("LOGFIRE_REGION", "us").lower()
    if region not in {"us", "eu"}:
        raise RuntimeError("LOGFIRE_REGION must be either 'us' or 'eu'.")

    base_url = f"https://logfire-{region}.pydantic.dev"
    min_timestamp = datetime.now(tz=UTC) - timedelta(hours=hours)

    response = requests.post(
        f"{base_url}/v2/query",
        headers={"Authorization": f"Bearer {api_token}", "Accept": "application/json"},
        json={"sql": sql, "min_timestamp": min_timestamp.isoformat(), "limit": 10000},
        timeout=60,
    )
    response.raise_for_status()

    payload = json.loads(response.text)
    rows = payload.get("data", payload.get("rows", []))

    normalized = []
    for row in rows:
        normalized.append({key: normalize_value(value) for key, value in row.items()})

    return normalized


@dlt.resource(name="records", write_disposition="replace")
def logfire_records(sql: str, hours: int):
    yield from query_logfire(sql=sql, hours=hours)


def run_pipeline(sql: str, hours: int):
    pipeline = dlt.pipeline(
        pipeline_name="logfire_agent_traces",
        destination="duckdb",
        dataset_name="agent_traces",
    )
    return pipeline.run(logfire_records(sql=sql, hours=hours))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--sql-file")
    args = parser.parse_args()

    load_env()
    sql = DEFAULT_SQL

    if args.sql_file:
        sql = Path(args.sql_file).read_text()

    load_info = run_pipeline(sql=sql, hours=args.hours)
    print(load_info)


if __name__ == "__main__":
    main()
