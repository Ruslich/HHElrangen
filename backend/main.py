import base64
import datetime as dt
import re
from hashlib import md5

from dotenv import load_dotenv

load_dotenv(".env.local"); load_dotenv()  # also loads .env if present

import os  # noqa: E402
from typing import Optional  # noqa: E402

from cachetools import TTLCache
from fastapi import Body, FastAPI, HTTPException, Response  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse
from pydantic import BaseModel  # noqa: E402

from utils import (  # noqa: E402
    athena_sql_to_df,
    fetch_medications,
    fhir_get,
    fhir_or_synth_observations,
    get_table_summaries,
    is_sql_safe,
    list_keys,
    read_csv_from_s3,
    read_head_from_s3,
    suggest_chart,
)
from pdf_prefill import PDFPrefillError, PDFPrefillService

app = FastAPI(title="Health data Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BUCKET = os.getenv("DATA_BUCKET", "health-demo-hherlangen")
PREFIX = os.getenv("DATA_PREFIX", "data/")

# FHIR base URL used by the FHIR helper; configurable via environment.
# Try FHIR_BASE_URL first, then a legacy FHIR_BASE, else default to localhost.
FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", os.getenv("FHIR_BASE", "http://localhost:8080"))

SESSION_TTL = int(os.getenv("SESSION_TTL", "3600"))  # 1 hour default
SESSIONS = TTLCache(maxsize=1000, ttl=SESSION_TTL)

pdf_prefill_service = PDFPrefillService()


def put_session(sid: str, **kv):
    s = SESSIONS.get(sid, {})
    s.update(kv)
    SESSIONS[sid] = s
    return s

def get_session(sid: str) -> dict:
    return SESSIONS.get(sid, {})


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


class PatientChatRequest(BaseModel):
    patient_id: str
    text: str
    days_back: int = 7
    session_id: Optional[str] = None  # SMART mock session id


from fastapi import Body


@app.post("/smart/mock_login")
def smart_mock_login(patient_id: str = Body(..., embed=True)):
    """
    Issue a mock access token bound to a session. In real SMART, this would be the OAuth redirect/callback.
    """
    sid = md5(f"{patient_id}:{os.urandom(4)}".encode()).hexdigest()[:16]
    # In real life, 'access_token' comes from the hospital's authorization server.
    put_session(sid, access_token="mock-access-token", patient_id=patient_id)
    return {"session_id": sid, "patient_id": patient_id}

@app.post("/smart/logout")
def smart_logout(session_id: str = Body(..., embed=True)):
    SESSIONS.pop(session_id, None)
    return {"ok": True}



# --- Bedrock helpers (model selection, review/fix, limits) ---

def _pick_sql_model_id() -> str:
    """
    Choose which model generates SQL.
    - If NLQ_FORCE_PRO=true -> always Pro
    - Else: start with Lite for simple intents, escalate on failure
    """
    use_pro = os.getenv("NLQ_FORCE_PRO", "true").lower() == "true"
    return os.getenv("BEDROCK_PRO_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0") if use_pro \
           else os.getenv("BEDROCK_LITE_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")

def _bedrock_converse(model_id: str, system_text: str, user_text: str,
                      max_tokens: int | None = None, temperature: float | None = None) -> str:
    import boto3
    from botocore.config import Config
    cfg = Config(region_name=os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "eu-central-1")),
                 read_timeout=12, connect_timeout=3)
    brt = boto3.client("bedrock-runtime", config=cfg)
    resp = brt.converse(
        modelId=model_id,
        system=[{"text": system_text}],
        messages=[{"role": "user", "content": [{"text": user_text}]}],
        inferenceConfig={
            "maxTokens": int(os.getenv("NLQ_MAX_TOKENS", max_tokens or 220)),
            "temperature": float(os.getenv("NLQ_TEMPERATURE", temperature or 0.1))
            # Note: topP removed - Claude Sonnet 4.5 doesn't allow both temperature and topP
        },
    )
    parts = resp.get("output", {}).get("message", {}).get("content", [])
    return (parts[0].get("text", "") if parts else "").strip()

def _normalize_sql_fences(sql: str) -> str:
    import re
    sql = re.sub(r"^```(?:\w+)?\s*", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"\s*```$", "", sql).strip()
    sql = re.sub(r"^\s*sql\s*:\s*", "", sql, flags=re.IGNORECASE).strip()
    return sql[:-1].strip() if sql.endswith(";") else sql

def _enforce_row_limit(sql: str, default_limit: int) -> str:
    """Append LIMIT if the query seems unbounded."""
    import re
    if re.search(r'\blimit\s+\d+\b', sql, flags=re.I):
        return sql
    # Only add when there's no aggregation limit pattern already
    return sql.rstrip() + f" LIMIT {max(1, default_limit)}"

def _review_and_fix_sql(question: str, sql: str, error: str | None = None) -> str:
    """
    Ask the model (Pro) to check if SQL answers the NL question.
    If not, or if there was an execution error, return a corrected SQL.
    """
    system = "You are a senior data engineer. Output ONLY a valid Athena SQL SELECT, nothing else."
    critique = (
        "User question:\n"
        f"{question}\n\n"
        "Candidate SQL:\n"
        f"{sql}\n\n"
        "If the SQL does not answer the question or caused an error, fix it. "
        "Rules:\n"
        "- Use exact column/table names from the provided schema snapshot in the SQL prompt.\n"
        "- If the user asks 'Top/Most' of X among Y with condition Z, group by X and filter Z in WHERE, "
        "  ORDER BY COUNT(*) DESC and LIMIT N.\n"
        "- Use LOWER(...) for case-insensitive text filters.\n"
        "- Quote identifiers with spaces using double quotes."
    )
    if error:
        critique += f"\n\nExecution error to avoid next time:\n{error}\n"

    model_id = os.getenv("BEDROCK_PRO_MODEL_ID", "amazon.nova-pro-v1:0")
    fixed = _bedrock_converse(model_id, system, critique)
    return _normalize_sql_fences(fixed)




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

        model_id   = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
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
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
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

        model_id   = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0")
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
            inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
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


def _make_text_filters_case_insensitive(sql: str) -> str:
    """
    Rewrite WHERE ... = 'text' and WHERE ... LIKE 'text'
    to be case-insensitive using LOWER(...).
    """
    # protect string literals
    parts = re.split(r'(\"[^\"]*\"|\'.*?\')', sql)

    def wrap_col(m):
        col = m.group("col")
        val = m.group("val")
        return f"LOWER({col}) LIKE LOWER({val})"

    for i in range(0, len(parts), 2):  # only outside quoted strings
        # = 'text'  ->  LIKE
        parts[i] = re.sub(
            r"(?P<col>\"[^\"]+\"|\w+(?:\.\w+)?)[ \t]*=[ \t]*(?P<val>'[^']*')",
            wrap_col,
            parts[i],
            flags=re.IGNORECASE
        )
        # LIKE 'text' -> LOWER(col) LIKE LOWER('text')
        parts[i] = re.sub(
            r"(?P<col>\"[^\"]+\"|\w+(?:\.\w+)?)[ \t]+LIKE[ \t]+(?P<val>'[^']*')",
            wrap_col,
            parts[i],
            flags=re.IGNORECASE
        )
    return "".join(parts)


def _athena_fix_common(sql: str) -> str:
    import re
    s = sql

    # PERCENTILE_CONT(...) WITHIN GROUP (ORDER BY x)  -> approx_percentile(x, q)
    s = re.sub(
        r"PERCENTILE_CONT\(\s*0\.5\s*\)\s*WITHIN\s+GROUP\s*\(\s*ORDER\s+BY\s+([^)]+)\)",
        r"approx_percentile(\1, 0.5)",
        s, flags=re.I
    )
    s = re.sub(
        r"PERCENTILE_CONT\(\s*0\.9\s*\)\s*WITHIN\s+GROUP\s*\(\s*ORDER\s+BY\s+([^)]+)\)",
        r"approx_percentile(\1, 0.9)",
        s, flags=re.I
    )
    s = re.sub(
        r"PERCENTILE_CONT\(\s*([01](?:\.\d+)?)\s*\)\s*WITHIN\s+GROUP\s*\(\s*ORDER\s+BY\s+([^)]+)\)",
        r"approx_percentile(\2, \1)",
        s, flags=re.I
    )

    # DATEDIFF(day, start, end)  -> date_diff('day', start, end)
    s = re.sub(
        r"DATEDIFF\s*\(\s*day\s*,\s*([^,]+)\s*,\s*([^)]+)\)",
        r"date_diff('day', \1, \2)",
        s, flags=re.I
    )

    # DATE_SUB(CURRENT_DATE, INTERVAL 6 MONTH) -> date_add('month', -6, current_date)
    s = re.sub(
        r"DATE_SUB\s*\(\s*CURRENT_DATE\s*,\s*INTERVAL\s+(\d+)\s+MONTH\s*\)",
        r"date_add('month', -\1, current_date)",
        s, flags=re.I
    )

    # EXTRACT(DOW FROM x) -> day_of_week(x)
    s = re.sub(
        r"EXTRACT\s*\(\s*DOW\s+FROM\s+([^)]+)\)",
        r"day_of_week(\1)",
        s, flags=re.I
    )

    return s


def _build_sql_prompt(question: str, prev_sql: str | None = None) -> str:
    """
    Build a schema-aware prompt using the *current* Glue/Athena schema.
    - Picks column names dynamically (medication/condition/billing/gender/etc)
    - Generates 1–3 short examples only with columns that actually exist
    - Instructs the model to quote names with spaces and use case-insensitive filters
    """
    schema = get_table_summaries()  # {table: [(col, type), ...]}
    allowed = [t.strip() for t in os.getenv("ALLOWED_TABLES", "").split(",") if t.strip()]

    # choose the table we will reference in examples
    table = None
    if allowed:
        for t in allowed:
            if t in schema:
                table = t
                break
    if not table and schema:
        table = next(iter(schema.keys()))  # first available

    cols = schema.get(table, [])
    colnames = [c for c, _ in cols]

    # helpers to pick best matching column from synonyms
    def _norm(s: str) -> str:
        return s.lower().strip()

    lc_map = {_norm(c): c for c in colnames}
    squish_map = {re.sub(r"[\s_]", "", c.lower()): c for c in colnames}

    def pick(*candidates):
        # exact (case-insensitive)
        for k in candidates:
            k2 = _norm(k)
            if k2 in lc_map:
                return lc_map[k2]
        # fuzzy (ignore spaces/underscores)
        for k in candidates:
            k2 = re.sub(r"[\s_]", "", k.lower())
            if k2 in squish_map:
                return squish_map[k2]
        return None

    # try to identify common roles from your dataset
    COL_GENDER     = pick("Gender", "Sex")
    COL_BILLING    = pick("Billing Amount", "Billing", "Cost", "Charge", "Amount")
    COL_CONDITION  = pick("Medical Condition", "Diagnosis", "Condition")
    COL_MEDICATION = pick("Medication", "Drug", "Medicine", "Prescription")
    COL_HOSPITAL   = pick("Hospital", "Facility", "Clinic")

    db = os.getenv("ATHENA_DATABASE", "default")

    # examples: only include when all required columns are found
    examples = []
    if table and COL_GENDER and COL_BILLING:
        examples.append(
            f'SELECT "{COL_GENDER}", AVG(CAST("{COL_BILLING}" AS double)) AS avg_billing '
            f'FROM {db}.{table} '
            f'GROUP BY "{COL_GENDER}" '
            f'ORDER BY avg_billing DESC;'
        )

    if table and COL_MEDICATION and COL_CONDITION:
        examples.append(
            f'SELECT "{COL_MEDICATION}", COUNT(*) AS frequency '
            f'FROM {db}.{table} '
            f'WHERE LOWER("{COL_CONDITION}") LIKE ''%asthma%'' '
            f'GROUP BY "{COL_MEDICATION}" '
            f'ORDER BY frequency DESC '
            f'LIMIT 1;'
        )

    if table and COL_HOSPITAL:
        examples.append(
            f'SELECT "{COL_HOSPITAL}", COUNT(*) AS frequency '
            f'FROM {db}.{table} '
            f'GROUP BY "{COL_HOSPITAL}" '
            f'ORDER BY frequency DESC '
            f'LIMIT 3;'
        )

    if table:
        examples.append(
            f'SELECT "medication", COUNT(*) AS frequency '
            f'FROM {db}.{table} '
            f'WHERE LOWER("medical condition") LIKE ''%diabetes%'' '
            f'GROUP BY "medication" '
            f'ORDER BY frequency DESC LIMIT 5;'
        )

    # include a short, current schema view (only the chosen table for brevity)
    schema_lines = []
    if table:
        preview = ", ".join([f'{c}:{t}' for c, t in cols[:15]])
        schema_lines.append(f"- {table}: {preview}")

    allow_text = ", ".join(allowed) if allowed else "(any)"

    # deterministic hash of the schema we used (handy to debug/calc size)
    schema_hash = md5(("|".join([table or ""] + colnames)).encode()).hexdigest()

    # final prompt (lean + prescriptive)
    prompt = (
        "You are a clinical analytics SQL generator for Amazon Athena (Trino/Presto).\n"
        "Return ONLY one SELECT statement. No comments, no backticks.\n"
        f"Database: {db}.   Allowed tables: {allow_text}.   Schema hash: {schema_hash}\n"
        "Schema:\n" + ("\n".join(schema_lines) if schema_lines else "- (no schema)") + "\n\n"
        "Rules:\n"
        "- Use table and column NAMES EXACTLY as shown. If a name has spaces/mixed case, wrap it in double quotes (e.g., \"Billing Amount\").\n"
        "- Prefer fully qualified FROM {db}.{table}. Use LOWER(col) for case-insensitive text filters.\n"
        "- For 'most popular' or 'most frequent', GROUP BY the target and ORDER BY COUNT(*) DESC LIMIT N.\n"
        "- CAST to double when averaging/summing text numeric columns.\n\n"
        "- Use Athena/Presto syntax: date_diff('day', start, end); date_add('month', -N, current_date); approx_percentile(x, 0.5 or 0.9); day_of_week(date_col).\n"
        "- For LOS: define length_of_stay_days := date_diff('day', \"Date of Admission\", \"Discharge Date\").\n"
    )

    if examples:
        prompt += "Examples:\n" + "\n".join(examples) + "\n\n"

    prompt += "Question: " + question + "\n"

    if prev_sql:
        prompt += (
            "\nPrior SQL (context – update or extend if needed, do not merely repeat):\n"
            f"{prev_sql}\n"
        )
    return prompt


def _pick_col(schema_cols, *candidates):
    # schema_cols is a list like ['Name','Age',...]
    lc = {c.lower(): c for c in schema_cols}
    squish = {re.sub(r'[\s_]', '', c.lower()): c for c in schema_cols}
    for k in candidates:
        if k.lower() in lc:
            return lc[k.lower()]
    for k in candidates:
        kk = re.sub(r'[\s_]', '', k.lower())
        if kk in squish:
            return squish[kk]
    return None


def _semantic_sql_adjust(question: str, sql: str) -> str:
    import re
    summaries = get_table_summaries()
    allowed = [t.strip() for t in os.getenv("ALLOWED_TABLES", "").split(",") if t.strip()]
    table = next((t for t in allowed if t in summaries), (next(iter(summaries)) if summaries else None))
    cols = [c for c, _ in (summaries.get(table) or [])]

    # 1) find columns
    col_med   = _pick_col(cols, "Medication", "Medications", "Drug", "Drugs", "Medicine", "Prescription", "Medication Name", "Drug Name")
    col_cond  = _pick_col(cols, "Medical Condition", "Diagnosis", "Condition")

    # 2) robust intent (plurals + slang)
    q = (question or "").lower()
    wants_med = any(k in q for k in ["medication", "medications", "drug", "drugs", "medicine", "meds"])
    if not wants_med or not col_med:
        return sql

    # 3) determine N
    m = re.search(r'\btop\s+(\d+)', q)
    top_n = int(m.group(1)) if m else (1 if any(k in q for k in ["most", "top", "popular"]) else 5)

    # 4) if grouped by condition, flip to medication
    if re.search(r'group\s+by\s+"?medical condition"?', sql, flags=re.I) or \
       (col_cond and re.search(fr'group\s+by\s+"?{re.escape(col_cond)}"?', sql, flags=re.I)):
        sql = re.sub(r'SELECT\s+.*?\s+FROM', f'SELECT "{col_med}", COUNT(*) AS frequency FROM', sql, flags=re.I|re.S)
        sql = re.sub(r'GROUP\s+BY\s+.*?(ORDER|LIMIT|$)', f'GROUP BY "{col_med}" \\1', sql, flags=re.I|re.S)

    # 5) ensure ORDER BY count desc
    if not re.search(r'order\s+by\s+count\s*\(\s*\*\s*\)\s*desc', sql, flags=re.I):
        sql = re.sub(r'(GROUP\s+BY\s+[^;]+)', r'\1 ORDER BY COUNT(*) DESC', sql, flags=re.I)

    # 6) ensure LIMIT N
    if re.search(r'\blimit\s+\d+', sql, flags=re.I):
        sql = re.sub(r'\blimit\s+\d+', f'LIMIT {top_n}', sql, flags=re.I)
    else:
        sql = sql.rstrip() + f' LIMIT {top_n}'

    # 7) replace guessed identifiers with the exact ones we found
    sql = re.sub(r'\bmedication(s)?\b', f'"{col_med}"', sql, flags=re.I)
    if col_cond:
        sql = re.sub(r'"?medical condition"?', f'"{col_cond}"', sql, flags=re.I)

    return sql

def _counts_summary_from_df(df, sql: str) -> str | None:
    """
    If the query result looks like: <group_col>, <count_col> (e.g., gender + frequency),
    return a short sentence: 'Among N records [with X filter]: Female: 51.2% (2777), Male: 48.8% (2643).'
    Otherwise return None.
    """
    import re

    import pandas as pd

    if df is None or df.empty or len(df.columns) < 2:
        return None

    # Pick a count column
    def is_numeric(col):
        return pd.to_numeric(df[col], errors="coerce").notna().mean() > 0.95

    # Prefer common names first, else first numeric col
    preferred = [c for c in df.columns if c.lower() in ("frequency","count","cnt","n","total")]
    count_col = next((c for c in preferred if c in df.columns and is_numeric(c)), None)
    if not count_col:
        numeric_cols = [c for c in df.columns if is_numeric(c)]
        if not numeric_cols:
            return None
        count_col = numeric_cols[0]

    # Group column = first non-numeric column different from count
    group_col = next((c for c in df.columns if c != count_col and not is_numeric(c)), None)
    if not group_col:
        # fallback: any other column
        candidates = [c for c in df.columns if c != count_col]
        group_col = candidates[0] if candidates else None
    if not group_col:
        return None

    # Clean + compute totals/percentages
    d = df[[group_col, count_col]].copy()
    d[count_col] = pd.to_numeric(d[count_col], errors="coerce").fillna(0)
    total = float(d[count_col].sum())
    if total <= 0:
        return None

    # Extract an English-y filter from SQL, e.g. LOWER("medical condition") LIKE LOWER('%diabetes%')
    filt_text = None
    try:
        m = re.search(r'where\s+(.+?)\s+(group|order|limit|$)', sql, flags=re.I|re.S)
        if m:
            where_block = m.group(1)
            m2 = re.search(r'lower\("([^"]+)"\)\s+like\s+lower\(''%?([^'']+)%?''\)', where_block, flags=re.I)
            if m2:
                filt_text = f'{m2.group(1)} containing "{m2.group(2)}"'
            else:
                # catch simple equals too
                m3 = re.search(r'lower\("([^"]+)"\)\s*=\s*''([^'']+)''', where_block, flags=re.I)
                if m3:
                    filt_text = f'{m3.group(1)} = "{m3.group(2)}"'
    except Exception:
        pass

    # Build parts
    parts = []
    for _, r in d.iterrows():
        n = int(r[count_col])
        p = (r[count_col] / total) * 100.0
        parts.append(f'{r[group_col]}: {n} ({p:.1f}%)')

    head = f'Among {int(total)} records'
    if filt_text:
        head += f' with {filt_text}'
    return head + ': ' + "; ".join(parts) + '.'


def _python_los_answer(question: str) -> dict | None:
    """
    Fallback for LOS-type questions when Athena fails.
    Supports:
      - median/p90 LOS by Medical Condition (optionally 'last N months')
      - median LOS by weekday of Date of Admission
    Returns a dict shaped like /nlq output (chart + rows + summary) or None if not applicable.
    """
    import re

    import numpy as np
    import pandas as pd

    from .utils import read_csv_from_s3

    q = (question or "").lower()
    # Only trigger on clear LOS intents
    if not any(k in q for k in ["length of stay", "los", "median los", "p90 los"]):
        return None

    # Load the demo CSV quickly
    try:
        df = read_csv_from_s3(BUCKET, f"{PREFIX}labs.csv")
    except Exception:
        return None

    # Required columns
    for c in ["Date of Admission", "Discharge Date"]:
        if c not in df.columns:
            return None

    df["Date of Admission"] = pd.to_datetime(df["Date of Admission"], errors="coerce")
    df["Discharge Date"] = pd.to_datetime(df["Discharge Date"], errors="coerce")
    df = df.dropna(subset=["Date of Admission", "Discharge Date"])
    df["LOS_days"] = (df["Discharge Date"] - df["Date of Admission"]).dt.days

    # Last N months filter if present
    m = re.search(r"last\s+(\d+)\s*month", q)
    if m:
        n = int(m.group(1))
        cutoff = pd.Timestamp.today().normalize() - pd.DateOffset(months=n)
        df = df[df["Date of Admission"] >= cutoff]

    # 1) By condition: “median and 90th percentile by Medical Condition”
    if "by medical condition" in q and "weekday" not in q:
        group_col = "Medical Condition" if "Medical Condition" in df.columns else None
        if not group_col:
            return None
        g = (df.groupby(group_col)["LOS_days"]
                .agg(median_los="median",
                     p90_los=lambda s: float(np.nanpercentile(s.dropna(), 90)),
                     patient_count="count")
                .reset_index()
                .sort_values("p90_los", ascending=False)
             )
        rows = g.rename(columns={group_col: "group", "p90_los": "p90", "median_los": "median"}).to_dict(orient="records")
        chart = {"type": "bar", "x": "group", "y": "p90", "title": "90th percentile LOS by condition", "labels": True}
        summary = f"Computed LOS (Discharge−Admission). Top group shows longest p90 LOS; medians included in table. N={int(df.shape[0])}."
        return {"rows": rows, "chart": chart, "columns": list(g.columns), "summary": summary}

    # 2) Weekend effect: “median LOS by weekday of Date of Admission”
    if "weekday" in q or "day of week" in q:
        w = df.groupby(df["Date of Admission"].dt.dayofweek)["LOS_days"].median().reset_index()
        w.columns = ["weekday", "median_los"]
        # Map 0..6 -> Mon..Sun
        names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        w["weekday"] = w["weekday"].map(lambda i: names[int(i)] if 0 <= int(i) <= 6 else str(i))
        rows = w.to_dict(orient="records")
        chart = {"type": "bar", "x": "weekday", "y": "median_los", "title": "Median LOS by weekday of admission", "labels": True}
        summary = f"Median LOS by admission weekday. Overall median ≈ {float(df['LOS_days'].median()):.1f} days."
        return {"rows": rows, "chart": chart, "columns": list(w.columns), "summary": summary}

    return None



@app.post("/nlq")
def nlq(req: NLQRequest, prev_sql: str | None = None):
    if os.getenv("BEDROCK_ENABLED", "false").lower() != "true":
        raise HTTPException(status_code=400, detail="Bedrock disabled for NLQ")

    # 1) Build prescriptive, schema-aware prompt
    prompt = _build_sql_prompt(req.question, prev_sql)

    # 2) Generate SQL (Lite or Pro based on env)
    model_for_sql = _pick_sql_model_id()
    sql_raw = _bedrock_converse(
        model_for_sql,
        system_text="Answer with valid Athena SQL only.",
        user_text=prompt,
    )
    sql = _normalize_sql_fences(sql_raw)

    # 3) Post-processing: case-insensitive text + task-specific fixes
    sql = _make_text_filters_case_insensitive(sql)

    # DEBUG: did we rewrite for medication frequency?
    before = sql
    sql = _semantic_sql_adjust(req.question, sql)
    sql = _athena_fix_common(sql)
    sql = _enforce_row_limit(sql, int(os.getenv("NLQ_DEFAULT_LIMIT", str(req.max_rows or 1000))))
    rewrote_medication = (sql != before)

    # 3b) Safety + LIMIT
    sql = _enforce_row_limit(sql, int(os.getenv("NLQ_DEFAULT_LIMIT", str(req.max_rows or 1000))))
    if not is_sql_safe(sql):
        # Try one repair pass if unsafe
        sql = _review_and_fix_sql(req.question, sql, error="Rejected by safety rules")
        sql = _make_text_filters_case_insensitive(sql)
        sql = _semantic_sql_adjust(req.question, sql)
        sql = _enforce_row_limit(sql, int(os.getenv("NLQ_DEFAULT_LIMIT", str(req.max_rows or 1000))))
        if not is_sql_safe(sql):
            raise HTTPException(status_code=400, detail=f"Unsafe SQL after repair: {sql}")

    # 4) Execute with one self-heal retry on Athena error
    try:
        df = athena_sql_to_df(sql)
    except Exception as e1:
        # One repair pass with the execution error and then retry
        repaired = _review_and_fix_sql(req.question, sql, error=str(e1))
        repaired = _make_text_filters_case_insensitive(repaired)
        repaired = _semantic_sql_adjust(req.question, repaired)
        repaired = _athena_fix_common(repaired)
        repaired = _enforce_row_limit(repaired, int(os.getenv("NLQ_DEFAULT_LIMIT", str(req.max_rows or 1000))))
        if not is_sql_safe(repaired):
            raise HTTPException(status_code=500, detail=f"Athena error: {e1}. Also produced unsafe repair: {repaired}")
        try:
            df = athena_sql_to_df(repaired)
            sql = repaired  # use the fixed version going forward
        except Exception as e2:
            # ---- NEW: Python fallback for LOS-type questions ----
            fallback = _python_los_answer(req.question)
            if fallback:
                return {
                    "chart": fallback.get("chart", {"type": "table"}),
                    "rows": fallback.get("rows", []),
                    "columns": fallback.get("columns", []),
                    "summary": fallback.get("summary", "Computed locally due to Athena error."),
                    "sql": "(python fallback; Athena failed)",
                }
            raise HTTPException(status_code=500, detail=f"Athena error: {e2}. Original: {e1}. SQL: {sql}")


    # 5) Visualization + summary (unchanged)
    chart = suggest_chart(df)
    preview = df.head(50).to_dict(orient="records")

    smart_summary = _counts_summary_from_df(df, sql)
    summary = _summarize_with_bedrock(
        {"question": req.question, "sql": sql, "columns": list(df.columns)[:6], "rows": len(df)}
    ) if not smart_summary else smart_summary


    return {
        "sql": sql,
        "chart": chart,
        "rows": preview,
        "columns": list(df.columns),
        "summary": summary,
        "model_used": model_for_sql,           # <<< added
        "medication_rewrite": rewrote_medication  # <<< added
    }



class ChatRequest(BaseModel):
    text: str
    max_rows: int = 1000
    history: list[dict] = []          # NEW: [{"role":"user"|"assistant", "content": "..."}]
    session_id: str | None = None     # NEW
    context: dict | None = None       # NEW: {"last_sql": "...", "columns": [...]}


class PDFGenerateRequest(BaseModel):
    template_name: str
    data: dict
    user_id: Optional[str] = None
    mapping_override: Optional[dict] = None
    persist: bool = True


class PDFAutoMapRequest(BaseModel):
    csv_columns: list[str]
    user_id: Optional[str] = None


class PDFMappingSaveRequest(BaseModel):
    mapping: dict



# very small intent heuristic: treat messages with analytic cues as data questions
_ANALYTICS_HINTS = re.compile(
    r"\b(compare|trend|over time|average|mean|sum|count|median|which|most|top|by\s+\w+|chart|plot|vs|per|group by)\b",
    re.I,
)

def detect_intent(text: str) -> str:
    return "data" if _ANALYTICS_HINTS.search(text or "") else "chitchat"

def _smalltalk_reply(text: str, history: list[dict]) -> str:
    if os.getenv("BEDROCK_ENABLED", "false").lower() == "true":
        import boto3
        from botocore.config import Config
        brt = boto3.client("bedrock-runtime", config=Config(
            region_name=os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "eu-central-1")),
            read_timeout=6, connect_timeout=3,
        ))

        # Build a short message list that ALWAYS starts with a user message.
        # 1) current user first
        msgs = [{"role": "user", "content": [{"text": (text or "").strip()}]}]

        # 2) then a compact slice of prior turns (user/assistant), if any
        for h in (history or [])[-6:]:
            role = "user" if h.get("role") == "user" else "assistant"
            content = (h.get("content") or "").strip()
            if not content:
                continue
            msgs.append({"role": role, "content": [{"text": content}]})


        resp = brt.converse(
            modelId=os.getenv("BEDROCK_LITE_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            system=[{"text": "You are a friendly, concise assistant. Keep replies under two sentences."}],
            messages=msgs,
            inferenceConfig={"maxTokens": 120, "temperature": 0.5},
        )
        parts = resp.get("output", {}).get("message", {}).get("content", [])
        msg = (parts[0].get("text", "") if parts else "").strip()
        return msg or "Hi! How can I help with your health data?"
    return "Hi! How can I help with your health data?"


@app.post("/chat")
def chat(req: ChatRequest):
    sid = req.session_id or "anon"
    session = SESSIONS.get(sid, {"history": [], "last_sql": None, "columns": []})

    intent = detect_intent(req.text)
    if intent == "data":
        prev_sql = (req.context or {}).get("last_sql") or session.get("last_sql")
        res = nlq(NLQRequest(question=req.text, max_rows=req.max_rows), prev_sql=prev_sql)

        # keep a tiny session memory
        session["last_sql"] = res.get("sql")
        session["columns"] = res.get("columns", [])
        session["history"].extend([
            {"role": "user", "content": req.text},
            {"role": "assistant", "content": f"Ran SQL. Columns: {', '.join(res.get('columns', [])[:4])}..."}
        ])
        session["history"] = session["history"][-12:]
        SESSIONS[sid] = session
        return res

    # smalltalk path — pass a short history
    reply = _smalltalk_reply(req.text, history=(req.history or session.get("history", [])))
    session["history"].extend([
        {"role":"user", "content": req.text},
        {"role":"assistant", "content": reply}
    ])
    session["history"] = session["history"][-12:]
    SESSIONS[sid] = session
    return {"text": reply}


# --- PDF pre-fill endpoints ---------------------------------------------------


@app.get("/pdf/templates")
def pdf_list_templates(refresh: bool = False):
    templates = pdf_prefill_service.list_templates(refresh=refresh)
    return {"templates": templates}


@app.get("/pdf/templates/{template_name}")
def pdf_get_template(template_name: str, user_id: str | None = None):
    template = pdf_prefill_service.get_template_config(template_name, user_id=user_id)
    if not template:
        raise HTTPException(status_code=404, detail=f"Template '{template_name}' not found")
    return {"template": template}


@app.get("/pdf/templates/{template_name}/scan")
def pdf_scan_template(template_name: str):
    try:
        scan = pdf_prefill_service.scan_template(template_name)
    except PDFPrefillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"template": template_name, "scan": scan}


@app.post("/pdf/templates/{template_name}/auto-map")
def pdf_auto_map(template_name: str, req: PDFAutoMapRequest):
    try:
        result = pdf_prefill_service.auto_map(
            template_name,
            req.csv_columns,
            user_id=req.user_id,
        )
    except PDFPrefillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"template": template_name, "auto_mapping": result}


@app.get("/pdf/mappings/{user_id}")
def pdf_list_mappings(user_id: str):
    mappings = pdf_prefill_service.list_user_mappings(user_id)
    return {"user_id": user_id, "mappings": mappings}


@app.post("/pdf/mappings/{user_id}/{template_name}")
def pdf_save_mapping(user_id: str, template_name: str, req: PDFMappingSaveRequest):
    pdf_prefill_service.save_mapping(user_id, template_name, req.mapping)
    return {"ok": True}


@app.post("/pdf/generate")
def pdf_generate(req: PDFGenerateRequest):
    try:
        result = pdf_prefill_service.generate_pdf(
            req.template_name,
            data=req.data,
            user_id=req.user_id,
            mapping_override=req.mapping_override,
            persist=req.persist,
        )
    except PDFPrefillError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    pdf_bytes = result["bytes"]
    encoded = base64.b64encode(pdf_bytes).decode("ascii")
    return {
        "metadata": result["metadata"],
        "pdf_base64": encoded,
    }


class PDFSaveEditsRequest(BaseModel):
    pdf_id: str
    patient_id: str
    form_data: dict
    template_name: str


class PDFExtractRequest(BaseModel):
    pdf_base64: str
    patient_id: str
    template_name: str


@app.post("/pdf/extract-and-save")
def pdf_extract_and_save(req: PDFExtractRequest):
    """
    Extract form data from an edited PDF and save to database.
    Returns the extracted data.
    """
    try:
        import io
        from pypdf import PdfReader
        
        # Decode the PDF
        pdf_bytes = base64.b64decode(req.pdf_base64)
        reader = PdfReader(io.BytesIO(pdf_bytes))
        
        # Extract form field values
        form_data = reader.get_form_text_fields() or {}
        
        # Save extracted data (in a real system, save to database)
        # For now, we'll store in session or return it
        session_key = f"pdf_edits_{req.patient_id}_{req.template_name}"
        SESSIONS[session_key] = {
            "form_data": form_data,
            "template_name": req.template_name,
            "patient_id": req.patient_id,
            "saved_at": dt.datetime.utcnow().isoformat() + "Z"
        }
        
        return {
            "ok": True,
            "message": f"Extracted and saved {len(form_data)} fields",
            "field_count": len(form_data),
            "sample_fields": dict(list(form_data.items())[:5])
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to extract PDF data: {str(exc)}") from exc


@app.post("/pdf/save-edits")
def pdf_save_edits(req: PDFSaveEditsRequest):
    """
    Save edited PDF form data back to database/storage.
    This allows nurses to edit PDFs in the browser and persist changes.
    """
    try:
        # Store the edited form data in session/database
        session_key = f"pdf_edits_{req.patient_id}_{req.template_name}"
        SESSIONS[session_key] = {
            "form_data": req.form_data,
            "template_name": req.template_name,
            "patient_id": req.patient_id,
            "pdf_id": req.pdf_id,
            "saved_at": dt.datetime.utcnow().isoformat() + "Z"
        }
        
        print(f"[SAVE] Saved {len(req.form_data)} fields for patient {req.patient_id}, template {req.template_name}")
        print(f"[SAVE] Session key: {session_key}")
        print(f"[SAVE] Sample data: {dict(list(req.form_data.items())[:3])}")
        
        return {
            "ok": True,
            "message": "PDF edits saved successfully to database",
            "patient_id": req.patient_id,
            "template_name": req.template_name,
            "saved_at": SESSIONS[session_key]["saved_at"]
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/pdf/get-saved-edits")
def pdf_get_saved_edits(patient_id: str, template_name: str):
    """
    Retrieve saved PDF form edits for a patient and template.
    Returns the saved form data so edits persist across sessions.
    """
    session_key = f"pdf_edits_{patient_id}_{template_name}"
    saved = SESSIONS.get(session_key)
    
    if saved:
        print(f"[LOAD] Found saved edits for patient {patient_id}, template {template_name}")
        print(f"[LOAD] {len(saved.get('form_data', {}))} fields saved")
        return {
            "ok": True,
            "form_data": saved.get("form_data", {}),
            "saved_at": saved.get("saved_at")
        }
    else:
        print(f"[LOAD] No saved edits found for {session_key}")
        return {
            "ok": False,
            "form_data": {},
            "message": "No saved edits found"
        }


@app.get("/pdf/{pdf_id}")
def pdf_get_pdf(pdf_id: str):
    """Download a generated PDF by ID"""
    record = pdf_prefill_service.get_pdf(pdf_id)
    if not record:
        raise HTTPException(status_code=404, detail="PDF not found")
    pdf_bytes = record["bytes"]
    headers = {"Content-Disposition": f'attachment; filename="{pdf_id}.pdf"'}
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)


from collections import Counter


@app.post("/patient_chat")
def patient_chat(req: PatientChatRequest):
    """
    FHIR-backed patient assistant (hackathon router) with synthetic fallback.
    Supports:
      - "CRP ... last X days"
      - "creatinine ... last X days"
      - "glucose ... last X days"
      - "recent medications / antibiotics"
    Returns chart/table payloads your Streamlit app already renders.
    """
    try:
        t = (req.text or "").lower()
        days = _parse_days_back(req.text, req.days_back)

                # ---- CORRELATION: Antibiotics vs CRP ----
        if ("crp" in t) and ("antibiotic" in t or "antibiotics" in t or "cef" in t or "piperacillin" in t or "tazo" in t or "vanco" in t):
            token = get_session(req.session_id).get("access_token") if req.session_id else None

            # Try to capture a specific drug name from free text
            abx_name = _extract_abx_name(req.text)

            # 1) CRP time series (FHIR first, synthetic fallback)
            crp_points, is_synth = fhir_or_synth_observations(
                req.patient_id,
                ["1988-5", "30522-7"],
                days,
                synth={"metric": "CRP", "unit": "mg/L", "start": 7.5, "drift": -0.05, "noise": 0.4},
                access_token=token,
            )

            # 2) Antibiotic doses per day, filtered if a drug was named
            from .utils import antibiotic_doses_per_day
            abx_daily = antibiotic_doses_per_day(
                req.patient_id,
                days_back=days,
                access_token=token,
                filter_names=[abx_name] if abx_name else None,
            )

            # 3) Scale antibiotics to CRP axis so both fit on one chart
            crp_peak = max([p["value"] for p in crp_points] + [1.0])
            max_dose = max([p["value"] for p in abx_daily] + [1])
            abx_scaled = [
                {"date": p["date"], "value": (p["value"] / max_dose) * crp_peak * 0.9}
                for p in abx_daily
            ]

            # 4) Combine series
            series_label = f"{abx_name} (doses/day, scaled)" if abx_name else "Antibiotics (doses/day, scaled)"
            rows = []
            rows += [{"date": p["date"], "value": p["value"], "group": "CRP (mg/L)"} for p in crp_points]
            rows += [{"date": p["date"], "value": p["value"], "group": series_label} for p in abx_scaled]

            # 5) Doctor-like explanation with expected behavior
            drug_txt = f" **{abx_name}**" if abx_name else ""
            explanation = (
                f"CRP vs{drug_txt} exposure over {days} days. Antibiotic daily counts are scaled to the CRP axis "
                f"(max CRP≈{crp_peak:.1f} mg/L; max daily doses={max_dose}). In effective therapy, CRP typically begins to "
                f"decline about **48–72 h after starting treatment**; look for a downward CRP trend **lagging** the days with dosing."
            )

            chart_title = f"CRP vs {abx_name}" if abx_name else "CRP vs Antibiotics"
            chart = {"type": "line", "x": "date", "y": "value", "group": "group", "title": chart_title}

            return {
                "chart": chart,
                "timeseries": rows,
                "explanation": explanation,
                "metric": "CRP vs Antibiotics",
                "source": "synthetic" if is_synth else "fhir",
            }



        # ---- LOINC map + units + synthetic defaults (self-contained) ----
        LOINC = {
            "crp":        {"codes": ["1988-5", "30522-7"], "title": "CRP",        "unit": "mg/L",  "synth": {"start": 7.5, "drift": -0.05, "noise": 0.4}},
            "creatinine": {"codes": ["2160-0"],            "title": "Creatinine", "unit": "mg/dL", "synth": {"start": 1.2, "drift":  0.00, "noise": 0.08}},
            "glucose":    {"codes": ["2345-7", "2339-0"],  "title": "Glucose",    "unit": "mg/dL", "synth": {"start": 118, "drift":  0.10, "noise": 4.0}},
        }

        # ----- LAB INTENTS -----
        for key, meta in LOINC.items():
            if key in t or meta["title"].lower() in t:
                # Try FHIR first; if empty or request fails, we synthesize a realistic series
                token = get_session(req.session_id).get("access_token") if req.session_id else None

                points, is_synth = fhir_or_synth_observations(
                    req.patient_id,
                    meta["codes"],
                    days,
                    synth={"metric": meta["title"], "unit": meta["unit"], **meta["synth"]},
                    access_token=token,
                )

                values = [p["value"] for p in points]
                if not values:
                    return {
                        "answer": f"I couldn’t find any {meta['title']} results for this patient in the last {days} days.",
                        "timeseries": [],
                    }

                first, last, peak = values[0], values[-1], max(values)
                mean_val = sum(values) / len(values) if values else None
                change_pct = (100.0 * (last - first) / first) if first else 0.0

                # Prefer Bedrock; fall back to deterministic string if disabled/fails
                summary_inputs = {
                    "metric": meta["title"],
                    "unit": meta["unit"],
                    "timeseries": points,
                    "first": first,
                    "last": last,
                    "peak": peak,
                    "mean": mean_val,
                    "change_pct": change_pct,
                    "days": days,
                    "source": "synthetic" if is_synth else "fhir",
                }
                try:
                    explanation = _summarize_with_bedrock(summary_inputs)
                except Exception:
                    explanation = (
                        f"{meta['title']} over {days} days "
                        f"({'synthetic demo' if is_synth else 'FHIR'}): "
                        f"first={first} {meta['unit']}, last={last} {meta['unit']}, "
                        f"peak={peak} {meta['unit']}, mean≈{mean_val:.2f} {meta['unit']} "
                        f"(Δ≈{change_pct:.1f}%)."
                    )

                chart = {
                    "type": "line",
                    "x": "date",
                    "y": "value",
                    "title": f"{meta['title']} over time",
                    "meanLine": True,
                    "yUnit": meta["unit"],
                }

                return {
                    "chart": chart,
                    "timeseries": points,
                    "explanation": explanation,
                    "metric": meta["title"],
                    "source": "synthetic" if is_synth else "fhir",
                }

        # ----- MEDICATION / ANTIBIOTIC INTENT -----
        if any(w in t for w in ["antibiotic", "antibiotics", "medication", "meds", "drugs"]):
            token = get_session(req.session_id).get("access_token") if req.session_id else None
            meds = fetch_medications(req.patient_id, days_back=days, access_token=token)

            rows = []
            for m in meds:
                med_code = m.get("medicationCodeableConcept", {})
                text = med_code.get("text") or (med_code.get("coding", [{}])[0].get("display"))
                when = m.get("effectiveDateTime") or m.get("effectivePeriod", {}).get("start")
                rows.append({"medication": text or "Unknown", "when": (when or "")[:16]})

            if not rows:
                return {"answer": f"No recent medication administrations found (last {days} days).", "rows": []}

            cnt = Counter(r["medication"] for r in rows)
            explanation = "Recent meds (last " + str(days) + " days): " + "; ".join(f"{k}: {n}x" for k, n in cnt.most_common())

            return {"rows": rows, "explanation": explanation}

        # ----- FALLBACK: Use Claude for general queries -----
        bedrock_enabled = os.getenv("BEDROCK_ENABLED", "false").lower()
        print(f"[PATIENT_CHAT] Fallback triggered. BEDROCK_ENABLED={bedrock_enabled}")
        print(f"[PATIENT_CHAT] Query text: {req.text}")
        
        if bedrock_enabled == "true":
            print("[PATIENT_CHAT] Calling Claude Sonnet 4.5 for general response...")
            try:
                # Get patient context
                token = get_session(req.session_id).get("access_token") if req.session_id else None
                
                # Try to fetch some patient data for context
                patient_context = f"Patient ID: {req.patient_id}"
                try:
                    # Try to get recent meds for context
                    meds = fetch_medications(req.patient_id, days_back=7, access_token=token)
                    if meds:
                        med_list = []
                        for m in meds[:3]:  # Just first 3
                            med_code = m.get("medicationCodeableConcept", {})
                            text = med_code.get("text") or (med_code.get("coding", [{}])[0].get("display"))
                            if text:
                                med_list.append(text)
                        if med_list:
                            patient_context += f"\nRecent medications: {', '.join(med_list)}"
                except Exception:
                    pass  # Continue without meds context
                
                # Call Claude Sonnet 4.5 for intelligent response
                system_prompt = """You are a helpful clinical AI assistant in a hospital EHR system. 
You help healthcare professionals with patient data queries, lab results, and general medical questions.

IMPORTANT FORMATTING RULES:
- Always respond in English
- Use clear markdown formatting with headers (##), bullet points, and bold text
- Keep responses concise but informative
- Be professional and clinically relevant

If asked about specific patient data you don't have access to, guide users on how to query it with examples like:
- "Show CRP last 7 days"
- "Recent medications" 
- "Creatinine last 30 days"
"""
                
                user_prompt = f"{patient_context}\n\nUser question: {req.text}"
                
                model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0")
                response = _bedrock_converse(
                    model_id,
                    system_text=system_prompt,
                    user_text=user_prompt,
                    max_tokens=300,
                    temperature=0.7
                )
                
                return {
                    "answer": response,
                    "explanation": response
                }
            except Exception as e:
                print(f"[PATIENT_CHAT] ❌ Bedrock fallback failed: {e}")
                print(f"[PATIENT_CHAT] Error type: {type(e).__name__}")
                # Continue to deterministic fallback below
        else:
            print("[PATIENT_CHAT] Bedrock is disabled, using deterministic fallback")
        
        # Deterministic fallback if Bedrock is disabled or fails
        print("[PATIENT_CHAT] Returning deterministic fallback response")
        return {
            "answer": 'I can help you with specific patient data queries. Try: "Show CRP last 7 days", "Creatinine last 30 days", "Glucose last 14 days", or "Recent antibiotics."',
            "explanation": 'Use specific medical queries to get patient data and lab results.'
        }

    except HTTPException:
        raise
    except Exception as e:
        # keep the API resilient for the demo
        return {"answer": f"Something went wrong while processing the request: {e}"}


@app.get("/bedrock_test")
def bedrock_test():
    """Test endpoint to verify Bedrock configuration and connectivity"""
    bedrock_enabled = os.getenv("BEDROCK_ENABLED", "false")
    bedrock_model = os.getenv("BEDROCK_MODEL_ID", "not-set")
    bedrock_region = os.getenv("BEDROCK_REGION", "not-set")
    
    result = {
        "bedrock_enabled": bedrock_enabled,
        "bedrock_model": bedrock_model,
        "bedrock_region": bedrock_region,
        "test_call": None,
        "error": None
    }
    
    if bedrock_enabled.lower() == "true":
        try:
            response = _bedrock_converse(
                bedrock_model,
                system_text="You are a test assistant.",
                user_text="Say 'Hello, Bedrock is working!' in exactly 5 words.",
                max_tokens=50,
                temperature=0.1
            )
            result["test_call"] = response
        except Exception as e:
            result["error"] = str(e)
    
    return result


@app.get("/fhir_ping")
def fhir_ping(session_id: Optional[str] = None):
    try:
        from .utils import fhir_get
        token = get_session(session_id).get("access_token") if session_id else None
        bundle = fhir_get("Patient", params={"_count": 3}, access_token=token)
        names = []
        for e in bundle.get("entry", []):
            res = e.get("resource", {})
            if res.get("resourceType") == "Patient":
                names.append(res.get("id"))
        return {"ok": True, "patients": names}
    except Exception as e:
        return {"ok": False, "error": str(e)}



# --- Add in main.py, near other helpers ---

Loinc = {
    "crp": ["1988-5", "30522-7"],          # CRP mass concentration (serum) and alt
    "creatinine": ["2160-0"],              # Creatinine [Mass/volume] in Serum/Plasma
    "glucose": ["2345-7", "2339-0"],       # Glucose (serum/plasma / blood)
}

def _parse_days_back(text: str, default: int = 7) -> int:
    import re
    m = re.search(r"last\s+(\d+)\s*(day|days|d|week|weeks|w)", (text or "").lower())
    if not m:
        return default
    n = int(m.group(1))
    return n * 7 if m.group(2).startswith("w") else n

_ABX_CANON = [
    "ceftriaxone", "piperacillin/tazobactam", "piperacillin tazobactam", "pip/tazo",
    "meropenem", "imipenem", "ertapenem",
    "amoxicillin", "amoxiclav", "augmentin",
    "vancomycin", "linezolid",
    "ciprofloxacin", "levofloxacin", "moxifloxacin",
    "doxycycline", "clindamycin", "metronidazole",
    "gentamicin", "amikacin", "trimethoprim", "sulfamethoxazole", "cotrimoxazole"
]

def _extract_abx_name(text: str) -> str | None:
    t = (text or "").lower()
    for n in _ABX_CANON:
        if n in t:
            return n
    # soft fallback: catch “ceftriax”, “pip/tazo”, “vanco”, etc.
    soft = ["ceftri", "pip/tazo", "piptazo", "vanco", "clinda", "metro", "levo", "cipro", "mero"]
    for n in soft:
        if n in t:
            return n
    return None



def _obs_series_for(patient_id: str, loinc_keys: list[str], days_back: int, session_id: str | None = None) -> list[dict]:
    from .utils import fetch_observations_by_code
    # expand keys like ["crp"] → codes list
    codes = []
    for k in loinc_keys:
        if k.lower() in Loinc:
            codes.extend(Loinc[k.lower()])
        else:
            codes.append(k)
    # Use provided session_id to look up an access token if available; otherwise call without token
    token = get_session(session_id).get("access_token") if session_id else None
    obs = fetch_observations_by_code(patient_id, codes, days_back, access_token=token)
    pts = []
    for o in obs:
        # numeric value only
        v = None
        if "valueQuantity" in o:
            v = o["valueQuantity"].get("value")
        if v is None:
            continue
        eff = o.get("effectiveDateTime") or o.get("issued")
        if not eff:
            continue
        pts.append({"date": eff[:10], "value": float(v)})
    return sorted(pts, key=lambda p: p["date"])


@app.get("/smart_demo")
def smart_demo():
    """
    Tiny SMART-like launcher: lists 3 patients and gives Launch links.
    """
    try:
        bundle = fhir_get("Patient", params={"_count": 3})
        entries = bundle.get("entry", []) or []
        html = ["<h2>Demo EHR Launcher</h2><ul>"]
        for e in entries:
            res = e.get("resource", {})
            pid = res.get("id")
            nm = res.get("name")
            label = (nm[0].get("text") if nm else pid) or pid
            # Streamlit runs at 8501 by default; pass patient id as a param
            href = f"http://127.0.0.1:8501/?demo_launch=1&patient_id={pid}"
            html.append(f'<li>{label} &nbsp; <a href="{href}" target="_blank">Launch app for this patient</a></li>')
        html.append("</ul><p>Tip: change 127.0.0.1 if your app runs elsewhere.</p>")
        return HTMLResponse("\n".join(html))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/find_demo_patient")
def find_demo_patient(code: str = "1988-5"):
    """
    Try to find a patient with this LOINC code present.
    """
    from utils import find_any_patient_with_observation
    pid = find_any_patient_with_observation(code)
    return {"patient_id": pid, "code": code}
