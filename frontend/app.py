# app.py
import os
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Health Chat MVP", layout="wide")
BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# ------------- Sidebar: data picking -------------

st.sidebar.title("Data")

if st.sidebar.button("List datasets"):
    r = requests.get(f"{BACKEND}/datasets", timeout=30)
    st.session_state["datasets"] = r.json().get("keys", [])

datasets = st.session_state.get("datasets", [])
key = st.sidebar.selectbox("Select CSV key", datasets) if datasets else st.sidebar.text_input("Enter CSV key")

if key:
    if st.sidebar.button("Preview head"):
        r = requests.get(f"{BACKEND}/head", params={"key": key, "n": 20}, timeout=60)
        if r.ok:
            st.sidebar.write("Columns:")
            st.sidebar.json(r.json().get("columns", []))
            st.sidebar.dataframe(pd.DataFrame(r.json().get("rows", [])))
        else:
            st.sidebar.error(f"Head failed: {r.text}")

# ------------- Main: Chat + chips -------------

st.title("Analyze a metric")

# default column hints (works with your sample)
DEFAULT_DATE = "Date of Admission"
DEFAULT_VAL = "Billing Amount"
DEFAULT_FILTER_COL = "Medical Condition"

with st.container():
    c1, c2 = st.columns([3, 2])
    with c1:
        date_col = st.text_input("Date column", value=DEFAULT_DATE)
        value_col = st.text_input("Value column", value=DEFAULT_VAL)
        filter_col = st.text_input("Optional filter column (e.g., test)", value=DEFAULT_FILTER_COL)
        filter_value = st.text_input("Optional filter value (e.g., Obesity)", value="")
    with c2:
        date_start = st.text_input("Optional start date (YYYY-MM-DD)", value="")
        date_end = st.text_input("Optional end date (YYYY-MM-DD)", value="")
        group_by = st.text_input("Optional group-by column (e.g., Gender)", value="")
        agg = st.selectbox("Aggregation", ["mean", "sum", "count", "median"], index=0)

st.divider()

# Template chips (safe actions)
st.caption("Quick templates")
tcol1, tcol2, tcol3 = st.columns(3)
trigger_payload: Dict[str, Any] | None = None

if tcol1.button("Trend by month"):
    trigger_payload = {
        "key": key,
        "date_col": date_col,
        "value_col": value_col,
        "filter_col": filter_col or None,
        "filter_value": filter_value or None,
        "date_start": date_start or None,
        "date_end": date_end or None,
        "group_by": None,
        "agg": agg,
    }
if tcol2.button("Compare groups (month + group)"):
    gb = group_by or "Gender"
    trigger_payload = {
        "key": key,
        "date_col": date_col,
        "value_col": value_col,
        "filter_col": filter_col or None,
        "filter_value": filter_value or None,
        "date_start": date_start or None,
        "date_end": date_end or None,
        "group_by": gb,
        "agg": agg,
    }
if tcol3.button("Topline average"):
    trigger_payload = {
        "key": key,
        "date_col": date_col,
        "value_col": value_col,
        "filter_col": filter_col or None,
        "filter_value": filter_value or None,
        "date_start": date_start or None,
        "date_end": date_end or None,
        "group_by": None,
        "agg": "mean",
    }

# Lightweight "chat" that maps to templates
prompt = st.chat_input("Ask e.g. 'trend of Billing Amount for Obesity in 2024'")

def map_prompt_to_template(text: str) -> Dict[str, Any] | None:
    if not text:
        return None
    t = text.lower()
    payload = {
        "key": key,
        "date_col": date_col,
        "value_col": value_col,
        "filter_col": filter_col or None,
        "filter_value": filter_value or None,
        "date_start": date_start or None,
        "date_end": date_end or None,
        "group_by": None,
        "agg": agg,
    }
    # crude cues
    if "compare" in t or "by " in t:
        # try to guess a group column after "by "
        if "by " in t:
            guess = t.split("by ", 1)[1].split()[0].strip(",. ")
            if guess:
                payload["group_by"] = guess.title()
        else:
            payload["group_by"] = group_by or "Gender"
        return payload
    if "trend" in t or "over time" in t:
        payload["group_by"] = None
        return payload
    if "average" in t or "mean" in t:
        payload["group_by"] = None
        payload["agg"] = "mean"
        return payload
    return None

if prompt and not trigger_payload:
    trigger_payload = map_prompt_to_template(prompt)

# Manual analyze button as a fallback
if st.button("Analyze") and not trigger_payload:
    trigger_payload = {
        "key": key,
        "date_col": date_col,
        "value_col": value_col,
        "filter_col": filter_col or None,
        "filter_value": filter_value or None,
        "date_start": date_start or None,
        "date_end": date_end or None,
        "group_by": group_by or None,
        "agg": agg,
    }

# ------------- Call backend & render -------------

if trigger_payload:
    if not key:
        st.error("Please select or enter a dataset key.")
        st.stop()

    with st.spinner("Analyzing..."):
        r = requests.post(f"{BACKEND}/analyze", json=trigger_payload, timeout=120)

    if not r.headers.get("content-type", "").startswith("application/json"):
        st.error(f"Backend error {r.status_code}: {r.text}")
        st.stop()

    j = r.json()
    ts = pd.DataFrame(j.get("timeseries", []))

    # Chart(s)
    if not ts.empty:
        ts["date"] = pd.to_datetime(ts["date"])
        if "group" in ts.columns:
            # Multi-series line chart
            for g, gdf in ts.groupby("group"):
                st.line_chart(gdf.set_index("date")["value"], height=260, width=1024, use_container_width=True)
                st.caption(f"Series: {g}")
        else:
            st.line_chart(ts.set_index("date")["value"], height=300, use_container_width=True)

    st.write("**Aggregate value**:", j.get("mean"))

    # Summary
    st.subheader("Summary")
    st.write(j.get("summary"))

    # How computed
    with st.expander("How we computed this"):
        st.write("**Audit trail**")
        st.code("\n".join(j.get("audit", [])) or "(none)")
        st.write("**Data coverage**")
        st.json(j.get("coverage", {}))
        st.write("**Parameters**")
        st.json({k: v for k, v in trigger_payload.items() if v})

    # Preview + downloads
    preview = pd.DataFrame(j.get("rows_preview", []))
    st.subheader("Preview rows")
    st.dataframe(preview)

    # Download the current timeseries and preview slice
    cdl1, cdl2 = st.columns(2)
    if not ts.empty:
        csv_bytes = ts.to_csv(index=False).encode("utf-8")
        cdl1.download_button("Download time series CSV", csv_bytes, file_name="timeseries.csv", mime="text/csv")
    if not preview.empty:
        csv2 = preview.to_csv(index=False).encode("utf-8")
        cdl2.download_button("Download slice CSV", csv2, file_name="slice.csv", mime="text/csv")

st.caption(f"Backend: {BACKEND}")
