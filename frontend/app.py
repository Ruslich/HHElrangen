from dotenv import load_dotenv

load_dotenv(".env.local"); load_dotenv()  # also loads .env if present

# app.py
import os
from typing import Any, Dict

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Health Chat MVP", layout="wide")

# Chat message store
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            cols = r.json().get("columns", [])
            st.session_state["columns"] = cols
            st.sidebar.json(cols)
            st.sidebar.dataframe(pd.DataFrame(r.json().get("rows", [])))
        else:
            st.sidebar.error(f"Head failed: {r.text}")

# ------------- Main: Chat-first UI -------------
st.title("Healthcare Helper")

# Chat transcript
st.subheader("Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def render_chart(chart, rows):
    import altair as alt
    df = pd.DataFrame(rows)
    if df.empty or "type" not in chart:
        st.dataframe(df)
        return
    t = chart.get("type")
    if t == "line" and chart.get("x") in df and chart.get("y") in df:
        df[chart["x"]] = pd.to_datetime(df[chart["x"]], errors="coerce")
        c = alt.Chart(df).mark_line().encode(x=chart["x"], y=chart["y"])
        st.altair_chart(c, use_container_width=True)
    elif t == "bar" and chart.get("x") in df and chart.get("y") in df:
        c = alt.Chart(df).mark_bar().encode(x=chart["x"], y=chart["y"])
        st.altair_chart(c, use_container_width=True)
    elif t == "scatter" and chart.get("x") in df and chart.get("y") in df:
        c = alt.Chart(df).mark_point().encode(x=chart["x"], y=chart["y"])
        st.altair_chart(c, use_container_width=True)
    elif t == "pie" and chart.get("x") in df and chart.get("y") in df:
        c = alt.Chart(df).mark_arc().encode(
            theta=alt.Theta(field=chart["y"], type="quantitative"),
            color=alt.Color(field=chart["x"], type="nominal")
        )
        st.altair_chart(c, use_container_width=True)
    else:
        st.dataframe(df)

prompt = st.chat_input("Ask me about your data (e.g., 'Which blood type is most frequent for males?')")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                r = requests.post(f"{BACKEND}/nlq", json={"question": prompt}, timeout=120)
                r.raise_for_status()
                j = r.json()
                st.markdown("**SQL**")
                st.code(j.get("sql", ""), language="sql")
                render_chart(j.get("chart", {}), j.get("rows", []))
                st.markdown(f"**Summary:** {j.get('summary', '')}")
            except Exception as e:
                st.error(f"Oops: {e}")
    st.session_state.messages.append({"role": "assistant", "content": "Answer rendered above."})

# ------------- Advanced controls (legacy form posting to /analyze) -------------
with st.expander("Advanced controls (legacy)", expanded=False):
    st.caption("Original metric form is available here if needed.")

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
    use_ai = st.toggle("Use AI summary (Bedrock Nova Lite)", value=False)

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
            "date_from": date_start or None,
            "date_to": date_end or None,
            "group_by": None,
            "agg": agg,
            "use_ai": use_ai,
        }
    if tcol2.button("Compare groups (month + group)"):
        gb = group_by or "Gender"
        trigger_payload = {
            "key": key,
            "date_col": date_col,
            "value_col": value_col,
            "filter_col": filter_col or None,
            "filter_value": filter_value or None,
            "date_from": date_start or None,
            "date_to": date_end or None,
            "group_by": gb,
            "agg": agg,
            "use_ai": use_ai,
        }
    if tcol3.button("Topline average"):
        trigger_payload = {
            "key": key,
            "date_col": date_col,
            "value_col": value_col,
            "filter_col": filter_col or None,
            "filter_value": filter_value or None,
            "date_from": date_start or None,
            "date_to": date_end or None,
            "group_by": None,
            "agg": "mean",
            "use_ai": use_ai,
        }

    # Manual analyze button as a fallback (no legacy chat here)
    if st.button("Analyze") and not trigger_payload:
        trigger_payload = {
            "key": key,
            "date_col": date_col,
            "value_col": value_col,
            "filter_col": filter_col or None,
            "filter_value": filter_value or None,
            "date_from": date_start or None,
            "date_to": date_end or None,
            "group_by": group_by or None,
            "agg": agg,
            "use_ai": use_ai,
        }

    # Call backend & render (legacy /analyze)
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
        # ---- charts: handle single series or grouped series
        if "series" in j and isinstance(j["series"], list):
            # grouped multi-series
            for s in j["series"]:
                gname = s.get("group", "Group")
                gdf = pd.DataFrame(s.get("points", []))
                if not gdf.empty:
                    gdf["date"] = pd.to_datetime(gdf["date"])
                    st.line_chart(gdf.set_index("date")["value"], height=260, use_container_width=True)
                    st.caption(f"Series: {gname}")
            ts = pd.DataFrame()  # keep downloads logic unchanged
        else:
            # single series
            ts = pd.DataFrame(j.get("timeseries", []))
            if not ts.empty:
                ts["date"] = pd.to_datetime(ts["date"])
                st.line_chart(ts.set_index("date")["value"], height=300, use_container_width=True)

        st.write("**Aggregate value**:", j.get("mean"))

        # Summary
        st.subheader("Summary")
        st.write(j.get("summary"))

        # How computed
        with st.expander("How we computed this"):
            st.write("**Audit trail**")
            # j['audit'] is an object; show it verbatim
            st.json(j.get("audit", {}))
            st.write("**Data coverage**")
            st.json(j.get("coverage", {}))
            st.write("**Parameters**")
            st.json({k: v for k, v in trigger_payload.items() if v})

        # Preview + downloads
        preview = pd.DataFrame(j.get("rows_preview", []))
        st.subheader("Preview rows")
        st.dataframe(preview)

        # Download buttons
        cdl1, cdl2 = st.columns(2)
        if not ts.empty:
            csv_bytes = ts.to_csv(index=False).encode("utf-8")
            cdl1.download_button("Download time series CSV", csv_bytes, file_name="timeseries.csv", mime="text/csv")
        if not preview.empty:
            csv2 = preview.to_csv(index=False).encode("utf-8")
            cdl2.download_button("Download slice CSV", csv2, file_name="slice.csv", mime="text/csv")

st.caption(f"Backend: {BACKEND}")
