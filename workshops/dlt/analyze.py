import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


HOMEWORK_QUERY = "How do I run Ollama locally?"


def table_names(conn: duckdb.DuckDBPyConnection) -> list[str]:
    query = """
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'agent_traces'
    ORDER BY table_name
    """
    return [row[0] for row in conn.execute(query).fetchall()]


def table_columns(conn: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'agent_traces'
      AND table_name = ?
    ORDER BY ordinal_position
    """
    return [row[0] for row in conn.execute(query, [table]).fetchall()]


def read_table(conn: duckdb.DuckDBPyConnection, table: str) -> pd.DataFrame:
    return conn.execute(f'SELECT * FROM agent_traces."{table}"').fetch_df()


def find_trace_id(conn: duckdb.DuckDBPyConnection, query_text: str) -> str:
    records = read_table(conn, "records")
    text_columns = [
        col
        for col in records.columns
        if records[col].dtype == "object" or str(records[col].dtype).startswith("string")
    ]

    for _, row in records.iterrows():
        for col in text_columns:
            value = row.get(col)
            if isinstance(value, str) and query_text in value:
                return str(row["trace_id"])

    if "start_timestamp" in records.columns:
        records = records.sort_values("start_timestamp", ascending=False)

    return str(records.iloc[0]["trace_id"])


def count_spans_for_trace(conn: duckdb.DuckDBPyConnection, trace_id: str) -> int:
    records = read_table(conn, "records")
    return int((records["trace_id"].astype(str) == str(trace_id)).sum())


def coerce_number(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value or value[0] not in "{[":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def nested_lookup(data: Any, dotted_key: str) -> float:
    if not isinstance(data, dict):
        return 0.0

    current = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return 0.0
        current = current[part]

    return coerce_number(current)


def sum_input_tokens(conn: duckdb.DuckDBPyConnection, trace_id: str) -> float:
    total = 0.0

    for table in table_names(conn):
        df = read_table(conn, table)
        if "trace_id" not in df.columns:
            continue

        df = df[df["trace_id"].astype(str) == str(trace_id)]
        if df.empty:
            continue

        for col in df.columns:
            col_lower = col.lower()
            if "input" in col_lower and "token" in col_lower:
                total += df[col].map(coerce_number).sum()

        for col in df.columns:
            if "attributes" not in col.lower():
                continue
            for value in df[col]:
                parsed = parse_jsonish(value)
                total += nested_lookup(parsed, "gen_ai.usage.input_tokens")

    return total


def bucket_span_count(count: int) -> str:
    options = [1, 5, 15, 30]
    return str(min(options, key=lambda option: abs(option - count)))


def bucket_table_count(count: int) -> str:
    options = [1, 3, 24, 100]
    return str(min(options, key=lambda option: abs(option - count)))


def bucket_input_tokens(tokens: float) -> str:
    if tokens < 500:
        return "100 - 500"
    if tokens < 5000:
        return "1500 - 5000"
    if tokens < 20000:
        return "10000 - 20000"
    return "50000 - 100000"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="logfire_agent_traces.duckdb")
    parser.add_argument("--query", default=HOMEWORK_QUERY)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise RuntimeError(f"DuckDB file not found: {db_path}")

    with duckdb.connect(str(db_path), read_only=True) as conn:
        tables = table_names(conn)
        trace_id = find_trace_id(conn, args.query)
        span_count = count_spans_for_trace(conn, trace_id)
        input_tokens = sum_input_tokens(conn, trace_id)

    print("Tables:", tables)
    print("Table count:", len(tables))
    print("Trace ID:", trace_id)
    print("Span count:", span_count)
    print("Input tokens:", int(input_tokens))
    print()
    print("Q1 closest span-count option:", bucket_span_count(span_count))
    print("Q2 closest table-count option:", bucket_table_count(len(tables)))
    print("Q3 input-token range:", bucket_input_tokens(input_tokens))


if __name__ == "__main__":
    main()
