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

_SCHEMA_CACHE = {"ts": 0, "data": None}
_SCHEMA_TTL = int(os.getenv("SCHEMA_CACHE_TTL_SECONDS", "300"))

def get_table_summaries(limit_cols_per_table: int = 12) -> Dict[str, List[Tuple[str, str]]]:
    """Return {table: [(col, type), ...]} for prompt grounding, cached briefly."""
    import time as _time
    now = _time.time()
    if _SCHEMA_CACHE["data"] is not None and now - _SCHEMA_CACHE["ts"] < _SCHEMA_TTL:
        return _SCHEMA_CACHE["data"]

    out: Dict[str, List[Tuple[str, str]]] = {}
    paginator = _glue.get_paginator("get_tables")
    for page in paginator.paginate(DatabaseName=ATHENA_DATABASE):
        for t in page.get("TableList", []):
            name = t["Name"]
            cols = [(c["Name"], c.get("Type", "")) for c in t.get("StorageDescriptor", {}).get("Columns", [])]
            out[name] = cols[:limit_cols_per_table]

    _SCHEMA_CACHE["data"] = out
    _SCHEMA_CACHE["ts"] = now
    return out

def is_sql_safe(sql: str) -> bool:
    """
    Conservative SQL guard:
    - Only SELECT
    - No DDL/DML/comments/semicolon chaining
    - If ALLOWED_TABLES is set, FROM/JOIN tables must be on the list
      (handles quoted identifiers and db.table forms).
    """
    import re

    s = " ".join(sql.strip().lower().split())
    if not s.startswith("select"):
        return False

    bad = [" insert ", " update ", " delete ", " drop ", " create ", " alter ",
           ";", "--", "/*", "*/", " grant ", " revoke "]
    if any(b in s for b in bad):
        return False

    if ALLOWED_TABLES:
        # find table tokens after FROM / JOIN (grab 1st word-ish token)
        # examples it will match: data, health_db.data, "data", "health_db"."data"
        candidates = re.findall(r'(?:from|join)\s+((?:"[^"]+"|\w+)(?:\.(?:"[^"]+"|\w+))?)', s)
        def norm(tok: str) -> str:
            # strip quotes and keep only the table part
            tok = tok.replace('"', "")
            return tok.split(".")[-1]
        tables = {norm(c) for c in candidates}
        if not tables:
            return False
        if not any(t in ALLOWED_TABLES for t in tables):
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
        qexec = _athena.get_query_execution(QueryExecutionId=start)["QueryExecution"]
        status = qexec["Status"]
        reason = status.get("StateChangeReason", "Unknown reason")
        workgroup = qexec.get("WorkGroup", ATHENA_WORKGROUP)
        # This ID lets you open the exact run in the Athena console (Query history)
        raise RuntimeError(f"Athena query {q}: {reason} (id={start}, workgroup={workgroup})")


    rows = _athena.get_query_results(QueryExecutionId=start, MaxResults=1000)
    cols = [c["VarCharValue"] for c in rows["ResultSet"]["Rows"][0]["Data"]]
    data = []
    for r in rows["ResultSet"]["Rows"][1:]:
        vals = [c.get("VarCharValue", None) for c in r.get("Data", [])]
        data.append(vals)
    return pd.DataFrame(data, columns=cols)

def suggest_chart(df: pd.DataFrame) -> Dict[str, str]:
    """
    Smarter chart suggestions:
    - donut for small-category counts
    - stacked bar if a 'group' column already exists
    - line(+trendline) if date + numeric
    - bar otherwise; box/hist suggestions for raw numeric columns
    """
    if df.empty:
        return {"type": "table"}

    # detect types
    numeric = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
    dates   = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    cats    = [c for c in df.columns if c not in numeric and c not in dates]

    # Common names
    freq_like = [c for c in df.columns if c.lower() in ("frequency","count","cnt","n","total")]
    y = (freq_like[0] if freq_like else (numeric[0] if numeric else None))

    # Titles help comprehension
    def ttl(t):
        return t

    # date + numeric => time series
    if dates and numeric:
        return {"type": "line", "x": dates[0], "y": numeric[0], "title": ttl(f"{numeric[0]} over time"), "meanLine": True, "trendline": False}

    # two columns: 1 cat + 1 numeric (typical NL2SQL output)
    if len(df.columns) >= 2 and cats and y:
        x = cats[0]
        smallk = df[x].nunique() <= 6
        if smallk and y in freq_like:
            return {"type": "donut", "x": x, "y": y, "title": ttl(f"Share by {x}"), "labels": True}
        return {"type": "bar", "x": x, "y": y, "title": ttl(f"{y} by {x}"), "meanLine": True, "labels": True}

    # multiple numeric columns => scatter
    if len(numeric) >= 2:
        return {"type": "scatter", "x": numeric[0], "y": numeric[1], "title": ttl(f"{numeric[1]} vs {numeric[0]}"), "trendline": True}

    # only numeric => histogram
    if len(numeric) == 1 and not cats:
        return {"type": "hist", "x": numeric[0], "y": numeric[0], "title": ttl(f"Distribution of {numeric[0]}")}

    # fallback
    return {"type": "table"}

