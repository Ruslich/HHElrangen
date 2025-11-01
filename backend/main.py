import json
import os
from typing import Optional

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .utils import list_keys, read_csv_from_s3, read_head_from_s3

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


def _summarize_with_bedrock(summary_inputs: dict) -> str:
    try:
        import boto3
        brt = boto3.client("bedrock-runtime")
        prompt = (
            "You are a clinical data assistant. Create a short, plain-English summary "
            "of the time series trend and its average. Include how it was computed."
            f"\nInputs: {json.dumps(summary_inputs)}"
        )
        body = json.dumps({"prompt": prompt, "max_tokens": 300})
        resp = brt.invoke_model(
            modelId=os.getenv("BEDROCK_MODEL_ID", "cohere.command-r-plus-v1:0"),
            accept="application/json",
            contentType="application/json",
            body=body,
        )
        out = json.loads(resp["body"].read().decode())
        return out.get("text") or out.get("generations", [{}])[0].get("text") or "Summary generated."
    except Exception:
        ts = summary_inputs["timeseries"]
        n = len(ts)
        start = ts[0]["date"] if n else "N/A"
        end = ts[-1]["date"] if n else "N/A"
        try:
            avg = float(summary_inputs.get("mean", 0.0))
        except Exception:
            avg = 0.0
        filt = ""
        if summary_inputs.get("filter_col"):
            filt = f" Filter: {summary_inputs['filter_col']} == {summary_inputs['filter_value']}."
        return (
            f"Computed monthly mean of {summary_inputs['value_col']} over {n} points "
            f"from {start} to {end}.{filt} Average value: {avg:.3f}."
        )


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
            }
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
        }
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