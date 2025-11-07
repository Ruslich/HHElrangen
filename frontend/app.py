from dotenv import load_dotenv

load_dotenv(".env.local"); load_dotenv()  # also loads .env if present

# app.py
import os

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Health Chat MVP", layout="wide")

import uuid

# --- Demo HIS config (left panel) ---
DEMO_HIS_MODE = True  # flip to False if you want the old layout

# A tiny synthetic roster you control for the pitch
DEMO_PATIENTS = [
    {"id": "DEMO-CRP-001", "name": "Anna Schmidt (ICU)"},
    {"id": "DEMO-CRP-002", "name": "Jonny Kramer (Ward)"},
    {"id": "DEMO-CRP-003", "name": "Karla Núñez (Surg)"},
]


# Parse query params for SMART-like demo launch
params = st.query_params
if "patient_id" in params and params.get("patient_id"):
    # seed the sidebar input on first load
    st.session_state.setdefault("patient_result", None)
    st.session_state.setdefault("messages", [])
    st.session_state["launched_patient_id"] = params.get("patient_id")


# Chat message store
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar reset
st.sidebar.button("Reset chat", on_click=lambda: (
    st.session_state.update(messages=[], session_id=str(uuid.uuid4()), last_sql=None, columns=None)
))


BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# ------------- Sidebar: data picking -------------
# st.sidebar.title("Data")

# if st.sidebar.button("List datasets"):
#     r = requests.get(f"{BACKEND}/datasets", timeout=30)
#     st.session_state["datasets"] = r.json().get("keys", [])

# datasets = st.session_state.get("datasets", [])
# key = st.sidebar.selectbox("Select CSV key", datasets) if datasets else st.sidebar.text_input("Enter CSV key")

# if key:
#     if st.sidebar.button("Preview head"):
#         r = requests.get(f"{BACKEND}/head", params={"key": key, "n": 20}, timeout=60)
#         if r.ok:
#             st.sidebar.write("Columns:")
#             cols = r.json().get("columns", [])
#             st.session_state["columns"] = cols
#             st.sidebar.json(cols)
#             st.sidebar.dataframe(pd.DataFrame(r.json().get("rows", [])))
#         else:
#             st.sidebar.error(f"Head failed: {r.text}")


# ------------- LEFT: Demo HIS (patient selector) -------------
st.sidebar.title("Demo HIS")

if DEMO_HIS_MODE:
    # Doctor "chooses" the patient here, like inside the EHR
    names = [p["name"] for p in DEMO_PATIENTS]
    idx = st.sidebar.selectbox("Select patient", options=list(range(len(names))), format_func=lambda i: names[i])
    selected_patient = DEMO_PATIENTS[idx]
    st.session_state["selected_patient_id"] = selected_patient["id"]
    st.session_state["selected_patient_name"] = selected_patient["name"]

    st.sidebar.caption(f"Selected: **{selected_patient['name']}**  \nID: `{selected_patient['id']}`")

    # Quick action buttons (optional)
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("CRP 7d"):
        st.session_state["patient_quick"] = "Show CRP last 7 days"
    if col2.button("Creat 30d"):
        st.session_state["patient_quick"] = "Show creatinine last 30 days"
    if col3.button("Meds 7d"):
        st.session_state["patient_quick"] = "Recent antibiotics"

    # st.sidebar.divider()
    # patient_mode = st.sidebar.toggle("Ask about selected patient", value=True, help="If on, the chat on the right will query FHIR/synthetic for this patient.")
    # st.session_state["patient_mode"] = patient_mode

    st.sidebar.markdown("### SMART (mock) Connect")
    connect_pid = st.sidebar.text_input("Patient ID to bind (FHIR)", value=st.session_state["selected_patient_id"] or "")
    if st.sidebar.button("Connect (mock SMART)"):
        r = requests.post(f"{BACKEND}/smart/mock_login", json={"patient_id": connect_pid}, timeout=15)
        if r.ok:
            sid = r.json().get("session_id")
            st.session_state["sid"] = sid
            st.session_state["patient_id"] = connect_pid
            st.sidebar.success(f"Connected as session {sid[:6]}…")
        else:
            st.sidebar.error(r.text)

    sid = st.session_state.get("sid")


else:
    st.sidebar.info("Demo HIS mode is off.")


# ------------- Main: Chat-first UI -------------
st.title("Healthcare Helper")

sid = st.session_state.get("sid")
bound_pid = st.session_state.get("patient_id")
if bound_pid:
    st.caption(f"SMART context active · patient `{bound_pid}`")


# Chat transcript
st.subheader("Chat")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def render_chart(chart, rows, question: str | None = None):
    import altair as alt
    import pandas as pd

    alt.data_transformers.disable_max_rows()  # avoids 5k row cap
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("No data to chart.")
        return

    # Pull common options with sensible fallbacks
    t   = (chart or {}).get("type", "table")
    x   = chart.get("x")
    y   = chart.get("y")
    g   = chart.get("group")          # optional series/group
    ttl = chart.get("title") or (question or "").strip() or None
    yfmt = chart.get("yFormat", ",")  # thousands by default
    mean_line = bool(chart.get("meanLine", t in {"bar", "line"}))
    labels = bool(chart.get("labels", t in {"bar", "donut"}))
    trend = bool(chart.get("trendline", False))
    height = int(chart.get("height", 360))

    # Helper: legend/axis tuned for readability
    def _legend(title):
        return alt.Legend(title=title, orient="right", labelLimit=180)

    def _axis(title, fmt=None):
        ax = alt.Axis(title=title)
        if fmt:
            ax.format = fmt
        return ax

    if t == "table" or not x or (t != "pie" and not y):
        st.caption(ttl or "Table")
        st.dataframe(df)
        return

    # ------- CHART TYPES -------
    if t == "bar":
        base = alt.Chart(df, title=ttl, height=height).mark_bar().encode(
            x=alt.X(x, sort='-y', title=x),
            y=alt.Y(y, axis=_axis(y, yfmt)),
            tooltip=[x, alt.Tooltip(y, format=yfmt)] + ([g] if g else [])
        )
        if g:
            base = base.encode(color=alt.Color(g, legend=_legend(g)))

        chart_viz = base
        if labels:
            chart_viz += base.mark_text(dy=-6).encode(text=alt.Text(y, format=yfmt))
        if mean_line:
            chart_viz += alt.Chart(df).mark_rule(strokeDash=[4,4]).encode(
                y=f"mean({y}):Q",
                tooltip=[alt.Tooltip(f"mean({y}):Q", title=f"Mean {y}", format=yfmt)]
            )
        st.altair_chart(chart_viz, use_container_width=True)
        return

    if t == "stacked_bar":
        chart_viz = alt.Chart(df, title=ttl, height=height).mark_bar().encode(
            x=alt.X(x, sort='-y', title=x),
            y=alt.Y(y, stack="normalize", axis=_axis(f"{y} (%)"), title=None),
            color=alt.Color(g or x, legend=_legend(g or x)),
            tooltip=[x, g, alt.Tooltip(y, format=yfmt)]
        )
        st.altair_chart(chart_viz, use_container_width=True)
        return

    if t == "donut" or t == "pie":
        # donut with % labels
        base = alt.Chart(df, title=ttl, height=height).transform_joinaggregate(
            total=f"sum({y})"
        ).transform_calculate(
            pct=f"datum['{y}'] / datum.total"
        ).encode(
            theta=alt.Theta(y, stack=True),
            color=alt.Color(x, legend=_legend(x)),
            tooltip=[x, alt.Tooltip(y, format=yfmt), alt.Tooltip("pct:Q", title="Share", format=".1%")]
        )
        ring = base.mark_arc(innerRadius=60)
        chart_viz = ring
        if labels:
            chart_viz += base.mark_text(radius=90).encode(text=alt.Text("pct:Q", format=".1%"))
        st.altair_chart(chart_viz, use_container_width=True)
        return

    if t == "line":
        # auto-parse dates for x
        try:
            df[x] = pd.to_datetime(df[x], errors="coerce")
        except Exception:
            pass
        base = alt.Chart(df, title=ttl, height=height).mark_line(point=True).encode(
            x=alt.X(x, title=x),
            y=alt.Y(y, axis=_axis(y, yfmt)),
            tooltip=[x, alt.Tooltip(y, format=yfmt)] + ([g] if g else [])
        )
        if g:
            base = base.encode(color=alt.Color(g, legend=_legend(g)))
        chart_viz = base
        if mean_line:
            chart_viz += alt.Chart(df).mark_rule(strokeDash=[4,4]).encode(
                y=f"mean({y}):Q",
                tooltip=[alt.Tooltip(f"mean({y}):Q", title=f"Mean {y}", format=yfmt)]
            )
        if trend:
            chart_viz += alt.Chart(df).transform_regression(x, y).mark_line(strokeDash=[2,2])
        st.altair_chart(chart_viz, use_container_width=True)
        return

    if t == "scatter":
        chart_viz = alt.Chart(df, title=ttl, height=height).mark_point(filled=True).encode(
            x=alt.X(x, axis=_axis(x)),
            y=alt.Y(y, axis=_axis(y, yfmt)),
            color=alt.Color(g, legend=_legend(g)) if g else alt.value("#3277"),
            tooltip=[x, alt.Tooltip(y, format=yfmt)] + ([g] if g else [])
        )
        if trend:
            chart_viz += alt.Chart(df).transform_regression(x, y).mark_line()
        st.altair_chart(chart_viz, use_container_width=True)
        return

    if t == "hist":
        chart_viz = alt.Chart(df, title=ttl, height=height).mark_bar().encode(
            x=alt.X(alt.repeat("layer"), bin=alt.Bin(maxbins=30), title=y or x),
            y=alt.Y('count()', title='Count')
        ).repeat(layer=[y or x])
        st.altair_chart(chart_viz, use_container_width=True)
        return

    if t == "box":
        chart_viz = alt.Chart(df, title=ttl, height=height).mark_boxplot(size=40).encode(
            x=alt.X(g or x, title=g or x),
            y=alt.Y(y, axis=_axis(y, yfmt)),
            color=alt.Color(g or x, legend=_legend(g or x))
        )
        st.altair_chart(chart_viz, use_container_width=True)
        return

    if t == "heatmap":
        chart_viz = alt.Chart(df, title=ttl, height=height).mark_rect().encode(
            x=alt.X(x, sort="-y", title=x),
            y=alt.Y(g or "group:N", title=g or "Group"),
            color=alt.Color(y, scale=alt.Scale(scheme="blues"), legend=_legend(y)),
            tooltip=[x, g, alt.Tooltip(y, format=yfmt)] if g else [x, alt.Tooltip(y, format=yfmt)]
        )
        st.altair_chart(chart_viz, use_container_width=True)
        return

    # Fallback
    st.dataframe(df)


# ------------- RIGHT: Chat box (routes to /patient_chat when patient mode is on) -------------
# If a quick action was pressed on the left, use that as the next prompt
queued = st.session_state.pop("patient_quick", None)
prompt = queued or st.chat_input("Ask me about your data (e.g., 'Which blood type is most frequent for males?')")

if prompt:
    # show user bubble + store
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # -------- Intent-based routing (no sidebar toggle) --------
    # 1) choose a patient id if we have SMART, else (if Demo HIS on) use selection
    sel_pid = st.session_state.get("patient_id") or st.session_state.get("selected_patient_id")

    # 2) detect simple patient intents (labs/meds) from text
    t = (prompt or "").lower()
    is_patient_intent = any(w in t for w in [
        "crp", "creatinine", "glucose",
        "antibiotic", "antibiotics", "medication", "meds", "drugs"
    ])

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                if is_patient_intent and sel_pid:
                    # ---- Patient route → /patient_chat ----
                    payload = {
                        "patient_id": sel_pid,
                        "text": prompt,
                        "days_back": 30,
                        "session_id": sid,  # SMART (mock) session token threading
                    }
                    r = requests.post(f"{BACKEND}/patient_chat", json=payload, timeout=60)
                    if not r.ok:
                        try:
                            st.error(r.json().get("detail", r.text))
                        except Exception:
                            st.error(r.text)
                        j = {"answer": r.text}
                    else:
                        j = r.json()

                    if "timeseries" in j or "chart" in j:
                        st.subheader("Patient chart")
                        render_chart(j.get("chart", {"type": "table"}), j.get("timeseries", []), j.get("metric"))
                        if j.get("explanation"):
                            st.write("**Explanation:** ", j["explanation"])
                        assistant_msg = j.get("explanation") or "See the chart above."
                    elif "rows" in j:
                        st.subheader("Patient data")
                        st.dataframe(pd.DataFrame(j["rows"]))
                        if j.get("explanation"):
                            st.write("**Summary:** ", j["explanation"])
                        assistant_msg = j.get("explanation") or "See the table above."
                    else:
                        st.write(j.get("answer", "No answer."))
                        assistant_msg = j.get("answer", "No answer.")
                else:
                    # ---- Cohort/smalltalk route → /chat ----
                    history = st.session_state.messages[-6:]
                    payload = {
                        "text": prompt,
                        "history": history,
                        "session_id": st.session_state.session_id,
                        "context": {
                            "last_sql": st.session_state.get("last_sql"),
                            "columns": st.session_state.get("columns"),
                        },
                    }
                    r = requests.post(f"{BACKEND}/chat", json=payload, timeout=120)
                    if not r.ok:
                        try:
                            st.error(r.json().get("detail", r.text))
                        except Exception:
                            st.error(r.text)
                        j = {"text": r.text}
                    else:
                        j = r.json()

                    if "text" in j and "sql" not in j:
                        st.markdown(j["text"])
                        assistant_msg = j["text"]
                    else:
                        st.markdown("**SQL**")
                        st.code(j.get("sql", ""), language="sql")
                        render_chart(j.get("chart", {}), j.get("rows", []), prompt)
                        st.markdown(f"**Summary:** {j.get('summary', '')}")
                        st.session_state.last_sql = j.get("sql")
                        st.session_state.columns = j.get("columns")
                        assistant_msg = j.get("summary") or "See the chart and SQL above."
            except Exception as e:
                st.error(f"Oops: {e}")
                assistant_msg = f"Oops: {e}"


    # append assistant message to transcript
    st.session_state.messages.append({"role": "assistant", "content": assistant_msg})



# ------------- Sidebar: patient (FHIR) -------------
# st.sidebar.title("Patient (FHIR)")
# pat_ping = st.sidebar.button("Ping FHIR (list 3 patients)")
# if pat_ping:
#     r = requests.get(f"{BACKEND}/fhir_ping", timeout=30)
#     if r.ok:
#         st.session_state["fhir_ping"] = r.json()
#     else:
#         st.sidebar.error(r.text)

# ping = st.session_state.get("fhir_ping")
# patients = []
# if ping:
#     patients = ping.get("patients", [])
#     st.sidebar.caption(f"Server: {ping.get('base_url')}")
#     for p in patients:
#         st.sidebar.write(f"• {p.get('id')} – {p.get('name')}")

# patient_id = st.sidebar.text_input(
#     "Patient ID",
#     value=st.session_state.get("launched_patient_id") or (patients[0]["id"] if patients else "")
# )
# days_back = st.sidebar.slider("Days back", 1, 365, 7)

# st.sidebar.divider()
# st.sidebar.caption("Ask the patient assistant:")
# pat_q = st.sidebar.text_input("e.g., Show CRP last 7 days")
# if st.sidebar.button("Run patient query") and patient_id and pat_q:
#     with st.spinner("Querying FHIR..."):
#         r = requests.post(f"{BACKEND}/patient_chat", json={"patient_id": patient_id, "text": pat_q, "days_back": days_back}, timeout=60)
#         st.session_state["patient_result"] = r.json() if r.ok else {"error": r.text}

# ------------- Main: patient result (if any) -------------
res = st.session_state.get("patient_result")
if res:
    if "error" in res:
        st.error(res["error"])
    elif "timeseries" in res or "chart" in res:
        st.subheader("Patient chart")
        render_chart(res.get("chart", {"type": "table"}), res.get("timeseries", []), res.get("metric"))
        if res.get("explanation"):
            st.write("**Explanation:** ", res["explanation"])
    elif "rows" in res:
        st.subheader("Patient data")
        st.dataframe(pd.DataFrame(res["rows"]))
        if res.get("explanation"):
            st.write("**Summary:** ", res["explanation"])
    elif "answer" in res:
        st.info(res["answer"])


# ------------- Advanced controls (legacy form posting to /analyze) -------------
# with st.expander("Advanced controls (legacy)", expanded=False):
#     st.caption("Original metric form is available here if needed.")

#     # default column hints (works with your sample)
#     DEFAULT_DATE = "Date of Admission"
#     DEFAULT_VAL = "Billing Amount"
#     DEFAULT_FILTER_COL = "Medical Condition"

#     with st.container():
#         c1, c2 = st.columns([3, 2])
#         with c1:
#             date_col = st.text_input("Date column", value=DEFAULT_DATE)
#             value_col = st.text_input("Value column", value=DEFAULT_VAL)
#             filter_col = st.text_input("Optional filter column (e.g., test)", value=DEFAULT_FILTER_COL)
#             filter_value = st.text_input("Optional filter value (e.g., Obesity)", value="")
#         with c2:
#             date_start = st.text_input("Optional start date (YYYY-MM-DD)", value="")
#             date_end = st.text_input("Optional end date (YYYY-MM-DD)", value="")
#             group_by = st.text_input("Optional group-by column (e.g., Gender)", value="")
#             agg = st.selectbox("Aggregation", ["mean", "sum", "count", "median"], index=0)

#     st.divider()
#     use_ai = st.toggle("Use AI summary (Bedrock Nova Lite)", value=False)

#     # Template chips (safe actions)
#     st.caption("Quick templates")
#     tcol1, tcol2, tcol3 = st.columns(3)
#     trigger_payload: Dict[str, Any] | None = None

#     if tcol1.button("Trend by month"):
#         trigger_payload = {
#             "key": key,
#             "date_col": date_col,
#             "value_col": value_col,
#             "filter_col": filter_col or None,
#             "filter_value": filter_value or None,
#             "date_from": date_start or None,
#             "date_to": date_end or None,
#             "group_by": None,
#             "agg": agg,
#             "use_ai": use_ai,
#         }
#     if tcol2.button("Compare groups (month + group)"):
#         gb = group_by or "Gender"
#         trigger_payload = {
#             "key": key,
#             "date_col": date_col,
#             "value_col": value_col,
#             "filter_col": filter_col or None,
#             "filter_value": filter_value or None,
#             "date_from": date_start or None,
#             "date_to": date_end or None,
#             "group_by": gb,
#             "agg": agg,
#             "use_ai": use_ai,
#         }
#     if tcol3.button("Topline average"):
#         trigger_payload = {
#             "key": key,
#             "date_col": date_col,
#             "value_col": value_col,
#             "filter_col": filter_col or None,
#             "filter_value": filter_value or None,
#             "date_from": date_start or None,
#             "date_to": date_end or None,
#             "group_by": None,
#             "agg": "mean",
#             "use_ai": use_ai,
#         }

#     # Manual analyze button as a fallback (no legacy chat here)
#     if st.button("Analyze") and not trigger_payload:
#         trigger_payload = {
#             "key": key,
#             "date_col": date_col,
#             "value_col": value_col,
#             "filter_col": filter_col or None,
#             "filter_value": filter_value or None,
#             "date_from": date_start or None,
#             "date_to": date_end or None,
#             "group_by": group_by or None,
#             "agg": agg,
#             "use_ai": use_ai,
#         }

#     # Call backend & render (legacy /analyze)
#     if trigger_payload:
#         if not key:
#             st.error("Please select or enter a dataset key.")
#             st.stop()

#         with st.spinner("Analyzing..."):
#             r = requests.post(f"{BACKEND}/analyze", json=trigger_payload, timeout=120)

#         if not r.headers.get("content-type", "").startswith("application/json"):
#             st.error(f"Backend error {r.status_code}: {r.text}")
#             st.stop()

#         j = r.json()
#         # ---- charts: handle single series or grouped series
#         if "series" in j and isinstance(j["series"], list):
#             # grouped multi-series
#             for s in j["series"]:
#                 gname = s.get("group", "Group")
#                 gdf = pd.DataFrame(s.get("points", []))
#                 if not gdf.empty:
#                     gdf["date"] = pd.to_datetime(gdf["date"])
#                     st.line_chart(gdf.set_index("date")["value"], height=260, use_container_width=True)
#                     st.caption(f"Series: {gname}")
#             ts = pd.DataFrame()  # keep downloads logic unchanged
#         else:
#             # single series
#             ts = pd.DataFrame(j.get("timeseries", []))
#             if not ts.empty:
#                 ts["date"] = pd.to_datetime(ts["date"])
#                 st.line_chart(ts.set_index("date")["value"], height=300, use_container_width=True)

#         st.write("**Aggregate value**:", j.get("mean"))

#         # Summary
#         st.subheader("Summary")
#         st.write(j.get("summary"))

#         # How computed
#         with st.expander("How we computed this"):
#             st.write("**Audit trail**")
#             # j['audit'] is an object; show it verbatim
#             st.json(j.get("audit", {}))
#             st.write("**Data coverage**")
#             st.json(j.get("coverage", {}))
#             st.write("**Parameters**")
#             st.json({k: v for k, v in trigger_payload.items() if v})

#         # Preview + downloads
#         preview = pd.DataFrame(j.get("rows_preview", []))
#         st.subheader("Preview rows")
#         st.dataframe(preview)

#         # Download buttons
#         cdl1, cdl2 = st.columns(2)
#         if not ts.empty:
#             csv_bytes = ts.to_csv(index=False).encode("utf-8")
#             cdl1.download_button("Download time series CSV", csv_bytes, file_name="timeseries.csv", mime="text/csv")
#         if not preview.empty:
#             csv2 = preview.to_csv(index=False).encode("utf-8")
#             cdl2.download_button("Download slice CSV", csv2, file_name="slice.csv", mime="text/csv")

# st.caption(f"Backend: {BACKEND}")
