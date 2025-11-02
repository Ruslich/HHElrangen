from dotenv import load_dotenv

load_dotenv(".env.local"); load_dotenv()  # also loads .env if present

import io
import os
import time
from typing import Dict, List, Tuple

import boto3
import pandas as pd


def read_csv_from_s3(bucket, key):
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))


# Lightweight helpers for the MVP
_s3 = boto3.client("s3")


def list_keys(bucket: str, prefix: str = "") -> list[str]:
    keys = []
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            if not item["Key"].endswith("/"):
                keys.append(item["Key"])
    return keys


def read_head_from_s3(bucket: str, key: str, n: int = 20) -> pd.DataFrame:
    obj = _s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(obj["Body"], nrows=n)


# --- Athena + Glue helpers ---


_athena = boto3.client("athena", region_name=os.getenv("AWS_REGION", "eu-central-1"))
_glue = boto3.client("glue", region_name=os.getenv("AWS_REGION", "eu-central-1"))

ATHENA_DATABASE = os.getenv("ATHENA_DATABASE")
ATHENA_WORKGROUP = os.getenv("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT = os.getenv("ATHENA_OUTPUT")
ALLOWED_TABLES = {t.strip().lower() for t in os.getenv("ALLOWED_TABLES", "").split(",") if t.strip()}

# Fail fast with a helpful message instead of a cryptic Glue error
if not ATHENA_DATABASE:
    raise RuntimeError("ATHENA_DATABASE not set. Did you load .env.local?")

def get_table_summaries(limit_cols_per_table: int = 12) -> Dict[str, List[Tuple[str, str]]]:
    """Return {table: [(col, type), ...]} for prompt grounding."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    paginator = _glue.get_paginator("get_tables")
    for page in paginator.paginate(DatabaseName=ATHENA_DATABASE):
        for t in page.get("TableList", []):
            name = t["Name"]
            cols = [(c["Name"], c.get("Type", "")) for c in t.get("StorageDescriptor", {}).get("Columns", [])]
            out[name] = cols[:limit_cols_per_table]
    return out

def is_sql_safe(sql: str) -> bool:
    """Very conservative: only SELECTs, only allowlisted tables, no semicolons chains, no DDL/DML."""
    s = " ".join(sql.strip().lower().split())
    if not s.startswith("select"):
        return False
    bad = [" insert ", " update ", " delete ", " drop ", " create ", " alter ", ";", "--", "/*", "*/", " grant ", " revoke "]
    if any(b in s for b in bad):
        return False
    # crude table allowlist check
    if ALLOWED_TABLES:
        hit = any(f" {t} " in f" {s} " or f" {ATHENA_DATABASE}.{t} " in f" {s} " for t in ALLOWED_TABLES)
        if not hit:
            return False
    return True

def athena_sql_to_df(sql: str, max_wait: int = 45):
    """Run SQL and return a pandas DataFrame."""
    if not ATHENA_DATABASE or not ATHENA_OUTPUT:
        raise RuntimeError("ATHENA_DATABASE/ATHENA_OUTPUT not set")

    start = _athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DATABASE},
        WorkGroup=ATHENA_WORKGROUP,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
    )["QueryExecutionId"]

    waited = 0
    while True:
        q = _athena.get_query_execution(QueryExecutionId=start)["QueryExecution"]["Status"]["State"]
        if q in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(1)
        waited += 1
        if waited > max_wait:
            _athena.stop_query_execution(QueryExecutionId=start)
            raise TimeoutError("Athena query timed out")

    if q != "SUCCEEDED":
        raise RuntimeError(f"Athena query {q}")

    rows = _athena.get_query_results(QueryExecutionId=start, MaxResults=1000)
    cols = [c["VarCharValue"] for c in rows["ResultSet"]["Rows"][0]["Data"]]
    data = []
    for r in rows["ResultSet"]["Rows"][1:]:
        vals = [c.get("VarCharValue", None) for c in r.get("Data", [])]
        data.append(vals)
    return pd.DataFrame(data, columns=cols)

def suggest_chart(df: pd.DataFrame) -> Dict[str, str]:
    """Return {'type': 'line'|'bar'|'scatter'|'pie', 'x': 'col', 'y': 'col', 'group': 'optional'}"""
    if df.empty or len(df.columns) < 1:
        return {"type": "table"}

    # simple heuristics
    numeric = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    dates = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    cats = [c for c in df.columns if c not in numeric and c not in dates]

    if dates and numeric:
        return {"type": "line", "x": dates[0], "y": numeric[0]}
    if len(numeric) >= 2:
        return {"type": "scatter", "x": numeric[0], "y": numeric[1]}
    if cats and numeric:
        # if small cardinality, pie; else bar
        if df[cats[0]].nunique() <= 6:
            return {"type": "pie", "x": cats[0], "y": numeric[0]}
        return {"type": "bar", "x": cats[0], "y": numeric[0]}

    return {"type": "table"}
