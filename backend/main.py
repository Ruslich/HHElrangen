from dotenv import load_dotenv

load_dotenv(".env.local"); load_dotenv()  # also loads .env if present

import os  # noqa: E402
from typing import Optional  # noqa: E402

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from .utils import (  # noqa: E402
    athena_sql_to_df,
    get_table_summaries,
    is_sql_safe,
    list_keys,
    read_csv_from_s3,
    read_head_from_s3,
    suggest_chart,
)

app = FastAPI(title="Health data Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKET = os.getenv("DATA_BUCKET", "health-demo-hherlangen")
PREFIX = os.getenv("DATA_PREFIX", "data/")

@app.get("/hello")
def read_root():
    return {"msg": "Hello AWS!"}

@app.post("/avg")
def average(data: dict = Body(...)):
    values = [1, 2, 3, 4, data.get("value", 0)]
    mean = sum(values) / len(values)
    return {"mean": mean}

@app.get("/s3demo")
def s3_demo():
    df = read_csv_from_s3(BUCKET, f"{PREFIX}labs.csv")
    print(df.head())  # helps confirm what’s loaded
    return {
        "head": df.head(5).to_dict(orient="records")
    }

# --- New MVP endpoints ---

@app.get("/datasets")
def list_datasets():
    keys = [k for k in list_keys(BUCKET, PREFIX) if k.lower().endswith(".csv")]
    return {"bucket": BUCKET, "keys": keys}


@app.get("/head")
def get_head(key: str, n: int = 20):
    df = read_head_from_s3(BUCKET, key, n=n)
    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}


class AnalyzeRequest(BaseModel):
    key: str
    date_col: str
    value_col: str
    filter_col: Optional[str] = None
    filter_value: Optional[str] = None
    # Sprint 1 additions
    date_from: Optional[str] = None   # ISO date, e.g. "2021-01-01"
    date_to: Optional[str] = None     # ISO date
    group_by: Optional[str] = None    # categorical column, e.g. "Gender"
    agg: str = "mean"                 # one of: mean,sum,count,median
    limit: int = 6                    # top-N groups to show if grouped
    use_ai: Optional[bool] = None  # if provided, overrides BEDROCK_ENABLED for this request


# Add a planning request model
class PlanRequest(BaseModel):
    text: str
    defaults: Optional[dict] = None  # current UI selections (key/date_col/value_col/etc.)

def _fallback_rule_based_plan(text: str, defaults: dict) -> dict:
    t = (text or "").lower()
    payload = {
        "key": defaults.get("key"),
        "date_col": defaults.get("date_col"),
        "value_col": defaults.get("value_col"),
        "filter_col": defaults.get("filter_col"),
        "filter_value": defaults.get("filter_value"),
        "date_from": defaults.get("date_from"),
        "date_to": defaults.get("date_to"),
        "group_by": defaults.get("group_by"),
        "agg": (defaults.get("agg") or "mean").lower(),
        "use_ai": defaults.get("use_ai"),
    }
    if " by " in t:
        guess = t.split(" by ", 1)[1].split()[0].strip(",. ")
        if guess:
            payload["group_by"] = guess.title()
    if "trend" in t or "over time" in t:
        payload["group_by"] = None
    for word, agg in [("average", "mean"), ("mean", "mean"), ("sum", "sum"), ("count", "count"), ("median", "median")]:
        if word in t:
            payload["agg"] = agg
            break
    import re
    m = re.search(r"(20\d{2})", t)
    if m:
        y = int(m.group(1))
        payload["date_from"] = f"{y}-01-01"
        payload["date_to"] = f"{y}-12-31"
    return payload

def _plan_with_bedrock(text: str, defaults: dict) -> dict:
    env_enabled = os.getenv("BEDROCK_ENABLED", "false").lower() == "true"
    use_ai = defaults.get("use_ai", False)
    if not env_enabled or not use_ai:
        return _fallback_rule_based_plan(text, defaults)
    try:
        import json

        import boto3
        from botocore.config import Config

        model_id   = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
        max_tokens = int(os.getenv("BEDROCK_MAX_OUTPUT_TOKENS", "180"))
        temperature = float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))
        retries     = int(os.getenv("BEDROCK_RETRIES", "1"))

        cfg = Config(
            region_name=os.getenv("BEDROCK_REGION", "eu-central-1"),
            retries={"max_attempts": max(1, retries), "mode": "standard"},
            read_timeout=10, connect_timeout=3,
        )
        brt = boto3.client("bedrock-runtime", config=cfg)

        columns = defaults.get("columns") or []
        system = [{
            "text": (
                "You are a planner that turns a user's request about a CSV into parameters for an analysis API. "
                "Only output STRICT JSON with keys: key,date_col,value_col,filter_col,filter_value,date_from,date_to,group_by,agg,use_ai. "
                "Preserve provided defaults unless the text clearly overrides them. "
                "Use ISO dates (YYYY-MM-DD). agg is one of: mean,sum,count,median. "
                f"Known columns: {columns}. Do not add extra keys."
            )
        }]
        messages = [{
            "role": "user",
            "content": [{"text": f"User text:\n{text}\n\nDefaults JSON:\n{json.dumps(defaults)}"}]
        }]

        resp = brt.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.9},
        )
        parts = resp.get("output", {}).get("message", {}).get("content", [])
        content = parts[0].get("text", "") if parts else ""
        import json as _json
        payload = _json.loads(content)
        merged = dict(defaults)
        for k in ["key","date_col","value_col","filter_col","filter_value","date_from","date_to","group_by","agg","use_ai"]:
            if k in payload:
                merged[k] = payload[k]
        return merged
    except Exception:
        return _fallback_rule_based_plan(text, defaults)

@app.post("/plan")
def plan(req: PlanRequest):
    defaults = req.defaults or {}
    payload = _plan_with_bedrock(req.text, defaults)
    for k in ("key", "date_col", "value_col"):
        if not payload.get(k):
            raise HTTPException(status_code=400, detail=f"Missing required field in plan: {k}")
    return {"payload": payload}

def _summarize_with_bedrock(summary_inputs: dict, force_off: bool = False) -> str:
    """
    Use Amazon Bedrock (Converse API) if enabled; otherwise return a deterministic fallback.
    Enforces tiny prompts, low tokens, short timeouts, and one retry to control cost.
    """
    # ---- No-cost path
    if force_off or os.getenv("BEDROCK_ENABLED", "false").lower() != "true":
        ts = summary_inputs.get("timeseries") or summary_inputs.get("series") or []
        # count points whether single or multi-series
        if ts and isinstance(ts, list) and isinstance(ts[0], dict) and "points" in ts[0]:
            n = sum(len(s.get("points", [])) for s in ts)
        else:
            n = len(ts)
        agg = summary_inputs.get("agg", "mean")
        mean_val = summary_inputs.get("mean")
        return f"Computed monthly {agg} over {n} points. Overall average: {mean_val}."

    try:
        import json

        import boto3
        from botocore.config import Config

        # Trim inputs to keep prompt tiny
        def _tiny(d: dict) -> dict:
            d = dict(d)
            if "timeseries" in d and isinstance(d["timeseries"], list):
                d["timeseries"] = d["timeseries"][:40]
            if "series" in d and isinstance(d["series"], list):
                d["series"] = [
                    {"group": s.get("group"), "points": s.get("points", [])[:20]}
                    for s in d["series"][:5]
                ]
            return d

        model_id   = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0")
        max_tokens = int(os.getenv("BEDROCK_MAX_OUTPUT_TOKENS", "140"))
        temperature = float(os.getenv("BEDROCK_TEMPERATURE", "0.2"))
        retries     = int(os.getenv("BEDROCK_RETRIES", "1"))

        cfg = Config(
            region_name=os.getenv("BEDROCK_REGION", "eu-central-1"),
            retries={"max_attempts": max(1, retries), "mode": "standard"},
            read_timeout=10, connect_timeout=3,
        )
        brt = boto3.client("bedrock-runtime", config=cfg)

        system = [{
            "text": ("You are a clinical data assistant. Write a SHORT (<= 2 sentences) plain-English summary. "
                     "Name the metric, filters(if any), date window, and aggregation. No medical advice.")
        }]
        messages = [{
            "role": "user",
            "content": [{"text": f"Inputs JSON:\n{json.dumps(_tiny(summary_inputs))}"}]
        }]

        resp = brt.converse(
            modelId=model_id,
            system=system,
            messages=messages,
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature, "topP": 0.9},
        )
        parts = resp.get("output", {}).get("message", {}).get("content", [])
        if parts and "text" in parts[0]:
            return parts[0]["text"].strip()
        return "Summary generated."
    except Exception:
        ts = summary_inputs.get("timeseries") or summary_inputs.get("series") or []
        if ts and isinstance(ts, list) and isinstance(ts[0], dict) and "points" in ts[0]:
            n = sum(len(s.get("points", [])) for s in ts)
        else:
            n = len(ts)
        agg = summary_inputs.get("agg", "mean")
        mean_val = summary_inputs.get("mean")
        return f"Computed monthly {agg} over {n} points. Overall average: {mean_val}."



@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    import pandas as pd
    df0 = read_csv_from_s3(BUCKET, req.key)
    ops: list[str] = []
    rows_in = len(df0)
    df = df0.copy()

    # Validate required columns
    for col in (req.date_col, req.value_col):
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column not found: {col}")

    # Optional filter
    if req.filter_col:
        if req.filter_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Filter column not found: {req.filter_col}")
        if req.filter_value is not None:
            df = df[df[req.filter_col].astype(str) == str(req.filter_value)]
            ops.append(f"filtered {req.filter_col} == '{req.filter_value}'")

    # Parse dates
    df[req.date_col] = pd.to_datetime(df[req.date_col], errors="coerce")
    ops.append(f"parsed dates in {req.date_col} with errors='coerce'")
    df = df.dropna(subset=[req.date_col])

    # Date range
    if req.date_from:
        df = df[df[req.date_col] >= pd.to_datetime(req.date_from)]
        ops.append(f"applied date_from >= {req.date_from}")
    if req.date_to:
        df = df[df[req.date_col] <= pd.to_datetime(req.date_to)]
        ops.append(f"applied date_to <= {req.date_to}")

    if df.empty:
        raise HTTPException(status_code=400, detail="No rows left after date parsing/range filtering.")

    # Ensure numeric values
    df[req.value_col] = pd.to_numeric(df[req.value_col], errors="coerce")
    df = df.dropna(subset=[req.value_col])
    if df.empty:
        raise HTTPException(status_code=400, detail="Value column has no numeric values after conversion.")

    # Aggregation function
    agg = req.agg.lower()
    if agg not in {"mean", "sum", "count", "median"}:
        raise HTTPException(status_code=400, detail=f"Unsupported agg: {req.agg}")
    ops.append(f"aggregated with {agg}")

    # Build result
    coverage = {
        "rows_in": int(rows_in),
        "rows_after_filter": int(len(df)),
        "date_min": df[req.date_col].min().strftime("%Y-%m-%d"),
        "date_max": df[req.date_col].max().strftime("%Y-%m-%d"),
    }

    if req.group_by:
        if req.group_by not in df.columns:
            raise HTTPException(status_code=400, detail=f"Group-by column not found: {req.group_by}")
        ops.append("grouped by month (M) and group-by column")

        grouped = (
            df.set_index(req.date_col)
              .groupby([pd.Grouper(freq="M"), df[req.group_by]])[req.value_col]
        )
        if agg == "mean":
            grouped = grouped.mean()
        elif agg == "sum":
            grouped = grouped.sum()
        elif agg == "count":
            grouped = grouped.count()
        else:
            grouped = grouped.median()

        grouped = grouped.reset_index().rename(
            columns={req.value_col: "value", req.date_col: "date", req.group_by: "group"}
        )
        # Keep top-N groups by overall mean to reduce clutter
        top = (
            grouped.groupby("group")["value"]
            .mean()
            .sort_values(ascending=False)
            .head(max(1, req.limit))
            .index
        )
        grouped = grouped[grouped["group"].isin(top)]
        grouped["date"] = grouped["date"].dt.strftime("%Y-%m-%d")

        series = [
            {"group": g, "points": gdf[["date", "value"]].to_dict(orient="records")}
            for g, gdf in grouped.groupby("group", sort=False)
        ]
        coverage["rows_in_ts"] = int(len(grouped))
        mean_per_group = (
            df.groupby(req.group_by)[req.value_col].mean().to_dict()
            if agg == "mean" else None
        )

        # In the grouped branch, before calling _summarize_with_bedrock:
        env_enabled = os.getenv("BEDROCK_ENABLED", "false").lower() == "true"
        force_off = not env_enabled if req.use_ai is None else (not req.use_ai)

        summary = _summarize_with_bedrock(
            {
                "key": req.key,
                "date_col": req.date_col,
                "value_col": req.value_col,
                "filter_col": req.filter_col,
                "filter_value": req.filter_value,
                "group_by": req.group_by,
                "agg": agg,
                "mean": mean_per_group,
                "timeseries": series,  # multi-series
            },
            force_off=force_off,
        )
        return {
            "series": series,
            "mean": mean_per_group,
            "summary": summary,
            "audit": {
                "inputs": req.model_dump(),
                "operations": ops,
                "coverage": coverage,
            },
        }

    # Single series (monthly)
    ops.append("grouped by month (M)")
    monthly = (
        df.set_index(req.date_col)
          .groupby(pd.Grouper(freq="M"))[req.value_col]
    )
    if agg == "mean":
        monthly = monthly.mean()
    elif agg == "sum":
        monthly = monthly.sum()
    elif agg == "count":
        monthly = monthly.count()
    else:
        monthly = monthly.median()

    monthly = monthly.reset_index().rename(columns={req.value_col: "value", req.date_col: "date"})
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")
    timeseries = monthly.to_dict(orient="records")
    coverage["rows_in_ts"] = int(len(monthly))
    mean_val = float(df[req.value_col].mean())

    # In the single-series branch, before calling _summarize_with_bedrock:
    env_enabled = os.getenv("BEDROCK_ENABLED", "false").lower() == "true"
    force_off = not env_enabled if req.use_ai is None else (not req.use_ai)

    summary = _summarize_with_bedrock(
        {
            "key": req.key,
            "date_col": req.date_col,
            "value_col": req.value_col,
            "filter_col": req.filter_col,
            "filter_value": req.filter_value,
            "timeseries": timeseries,
            "mean": mean_val,
            "agg": agg,
        },
        force_off=force_off,
    )
    return {
        "timeseries": timeseries,
        "mean": mean_val,
        "summary": summary,
        "audit": {
            "inputs": req.model_dump(),
            "operations": ops,
            "coverage": coverage,
        },
        "rows_preview": df.head(20).to_dict(orient="records"),
    }

class NLQRequest(BaseModel):
    question: str
    # optional guardrails
    max_rows: int = 1000

def _build_sql_prompt(question: str) -> str:
    schema = get_table_summaries()
    allow = ", ".join(sorted(os.getenv("ALLOWED_TABLES", "").split(",")))
    schema_lines = []
    for t, cols in schema.items():
        cols_str = ", ".join([f"{c}:{typ}" for c, typ in cols])
        schema_lines.append(f"- {t}: {cols_str}")
    return (
        "You are a clinical analytics SQL generator for Amazon Athena (Presto/Trino dialect). "
        "Return ONLY a single SELECT statement. No comments, no backticks. "
        f"Database: {os.getenv('ATHENA_DATABASE')}. "
        f"Allowed tables: {allow or '(any)'}.\n"
        "Schema (first few columns per table):\n"
        + "\n".join(schema_lines)
        + "\n\nQuestion: " + question
        + "\nRules: Prefer COUNT, GROUP BY for frequencies; use CAST as needed; "
          "if asking for 'most frequent', ORDER BY count desc LIMIT 1; for time trends, group by date truncs."
    )

@app.post("/nlq")
def nlq(req: NLQRequest):
    # 1) Ask Bedrock for SQL
    import re

    import boto3
    from botocore.config import Config

    if os.getenv("BEDROCK_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=400, detail="Bedrock disabled for NLQ")

    prompt = _build_sql_prompt(req.question)
    cfg = Config(region_name=os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "eu-central-1")),
                 read_timeout=10, connect_timeout=3)
    brt = boto3.client("bedrock-runtime", config=cfg)
    resp = brt.converse(
        modelId=os.getenv("BEDROCK_MODEL_ID", "amazon.nova-lite-v1:0"),
        system=[{"text": "Answer with valid Athena SQL only."}],
        messages=[{"role":"user","content":[{"text":prompt}]}],
        inferenceConfig={"maxTokens": 220, "temperature": float(os.getenv("BEDROCK_TEMPERATURE","0.2"))}
    )
    sql = resp["output"]["message"]["content"][0]["text"].strip()

    # --- normalize fences / prefixes / trailing semicolon
    sql = re.sub(r"^```(?:\w+)?\s*", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"\s*```$", "", sql).strip()
    sql = re.sub(r"^\s*sql\s*:\s*", "", sql, flags=re.IGNORECASE).strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()

    # 2) Safety check
    if not is_sql_safe(sql):
        raise HTTPException(status_code=400, detail=f"Unsafe or unsupported SQL generated: {sql}")

    # 3) Execute
    try:
        df = athena_sql_to_df(sql)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Athena error: {e}")

    # 4) Pick a chart + summary
    chart = suggest_chart(df)
    preview = df.head(50).to_dict(orient="records")

    summary = _summarize_with_bedrock(
        {"question": req.question, "sql": sql, "columns": list(df.columns)[:6], "rows": len(df)}
    )

    return {"sql": sql, "chart": chart, "rows": preview, "columns": list(df.columns), "summary": summary}