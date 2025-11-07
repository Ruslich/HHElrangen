from dotenv import load_dotenv

load_dotenv(".env.local"); load_dotenv()  # also loads .env if present
import io
import os
import time
from typing import Dict, List, Tuple

import boto3
import pandas as pd
import requests


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


FHIR_BASE_URL = os.getenv("FHIR_BASE_URL", "").rstrip("/")
FHIR_AUTH_TOKEN = os.getenv("FHIR_AUTH_TOKEN", "")  # fallback for local dev

def _auth_headers(access_token: str | None):
    h = {"Accept": "application/fhir+json"}
    tok = access_token or FHIR_AUTH_TOKEN
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h

def fhir_get(path: str, params: dict | None = None, access_token: str | None = None) -> dict:
    if not FHIR_BASE_URL:
        raise RuntimeError("FHIR_BASE_URL not set")
    url = f"{FHIR_BASE_URL}/{path.lstrip('/')}"
    r = requests.get(url, headers=_auth_headers(access_token), params=params or {}, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_observations(patient_id: str, loinc_code: str | None = None, days_back: int = 7,
                       access_token: str | None = None) -> list[dict]:
    import datetime as dt
    since = (dt.date.today() - dt.timedelta(days=days_back)).isoformat()
    params = {"patient": patient_id, "_count": 200, "date": f"ge{since}"}
    if loinc_code:
        params["code"] = loinc_code
    bundle = fhir_get("Observation", params=params, access_token=access_token)
    return [e["resource"] for e in bundle.get("entry", []) if "resource" in e]
    

def fetch_medications(
    patient_id: str,
    days_back: int = 7,
    access_token: str | None = None,
) -> list[dict]:
    """
    Try MedicationAdministration first (effective-time), then fallback to MedicationRequest (authoredon),
    then MedicationStatement (effective). Returns a list of raw FHIR resources.
    """
    import datetime as dt
    since = (dt.date.today() - dt.timedelta(days=days_back)).isoformat()

    # 1) MedicationAdministration (R4 search param: effective-time)
    try:
        params = {"patient": patient_id, "effective-time": f"ge{since}", "_count": 200}
        bundle = fhir_get("MedicationAdministration", params=params, access_token=access_token)
        entries = [e["resource"] for e in bundle.get("entry", []) if "resource" in e]
        if entries:
            return entries
    except Exception:
        pass

    # 2) MedicationRequest (R4 search param: authoredon)
    try:
        params = {"patient": patient_id, "authoredon": f"ge{since}", "_count": 200}
        bundle = fhir_get("MedicationRequest", params=params, access_token=access_token)
        entries = [e["resource"] for e in bundle.get("entry", []) if "resource" in e]
        if entries:
            return entries
    except Exception:
        pass

    # 3) MedicationStatement (R4 search param: effective)
    try:
        params = {"patient": patient_id, "effective": f"ge{since}", "_count": 200}
        bundle = fhir_get("MedicationStatement", params=params, access_token=access_token)
        entries = [e["resource"] for e in bundle.get("entry", []) if "resource" in e]
        if entries:
            return entries
    except Exception:
        pass

    return []


# --- Antibiotics per day (simple keyword match + Medication* timestamps) ---
def antibiotic_doses_per_day(
    patient_id: str,
    days_back: int,
    access_token: str | None = None,
    synth_if_empty: bool = True,
    filter_names: list[str] | None = None,   # << NEW
) -> list[dict]:
    """
    Return daily antibiotic exposure counts for the last N days:
      [{"date":"YYYY-MM-DD","value":<doses/day>}, ...]
    Looks in MedicationAdministration, MedicationRequest, MedicationStatement.
    Detects antibiotics via ATC J01 codes OR name keywords.
    If filter_names is provided (e.g., ["ceftriaxone"]), only count matching antibiotics.
    Expands effective periods into 1/day across the covered date range.
    """
    import datetime as dt
    from collections import Counter

    # ---- window ----
    today = dt.date.today()
    start_date = today - dt.timedelta(days=days_back)

    # Normalize provided filter names to lowercase tokens
    filter_tokens = []
    if filter_names:
        for n in filter_names:
            t = (n or "").lower().strip()
            if t:
                filter_tokens.append(t)

    # ---- antibiotic detection helpers ----
    abx_keywords = (
        # β-lactams & others (broad and brand/common fragments)
        "antibi", "cef", "ceph", "cefaz", "ceftri", "ceftriax", "ceftaz", "cefep",
        "amox", "augmentin", "amoxiclav", "amoxicillin",
        "penicillin", "piperacillin", "tazobactam", "pip/tazo", "tazo",
        "meropenem", "imipenem", "ertapenem", "carbapenem",
        "cipro", "ciproflox", "levofloxacin", "moxifloxacin", "ofloxacin", "quinolone",
        "azithro", "clarithro", "erythro", "macrolide",
        "doxy", "doxycycline", "tetracycline",
        "clinda", "clindamycin",
        "metronidazole", "flagyl",
        "vanco", "vancomycin",
        "linezolid", "zyvox",
        "gentamicin", "amikacin", "tobramycin", "aminoglycoside",
        "trimethoprim", "sulfamethoxazole", "co-trim", "cotrimoxazole", "tmp/smx"
    )

    def _is_abx_code(coding: dict) -> bool:
        sys = (coding.get("system") or "").lower()
        code = (coding.get("code") or "").upper()
        # ATC J01 = antibacterials for systemic use
        return ("whocc" in sys or "atc" in sys) and code.startswith("J01")

    def _name_from_cc(cc: dict) -> str:
        if not isinstance(cc, dict):
            return ""
        txt = (cc.get("text") or "").strip()
        if txt:
            return txt
        for cd in cc.get("coding", []) or []:
            disp = (cd.get("display") or "").strip()
            if disp:
                return disp
        return ""

    def _is_abx_by_name(name: str) -> bool:
        s = (name or "").lower()
        return "antibi" in s or any(k in s for k in abx_keywords)

    def _match_filter(name: str) -> bool:
        if not filter_tokens:
            return True
        s = (name or "").lower()
        return any(tok in s for tok in filter_tokens)

    def _antibiotic_resource(res: dict) -> tuple[bool, str]:
        """Return (is_abx, display_name) and honor filter_names if provided."""
        cc = res.get("medicationCodeableConcept")
        nm = _name_from_cc(cc) if cc else ""
        # code-based detection
        for cd in (cc.get("coding", []) if isinstance(cc, dict) else []) or []:
            if _is_abx_code(cd):
                return (_match_filter(nm), nm if nm else (cd.get("display") or cd.get("code") or "antibiotic"))
        # name-based detection
        if _is_abx_by_name(nm):
            return (_match_filter(nm), nm)
        # fallback: sometimes resource has `code`
        for cd in (res.get("code", {}) or {}).get("coding", []) or []:
            if _is_abx_code(cd):
                disp = cd.get("display") or cd.get("code") or "antibiotic"
                return (_match_filter(disp), disp)
        return (False, "")

    meds = fetch_medications(patient_id, days_back=days_back, access_token=access_token)

    # ---- collect daily counts ----
    per_day = Counter()

    def _date(s: str | None):
        import datetime as dt
        return dt.date.fromisoformat(s[:10]) if s else None

    def _clamp(d):
        if d is None:
            return None
        if d < start_date:
            return start_date
        if d > today:
            return today
        return d

    def _add_range(d0, d1):
        if d0 is None:
            return
        d0 = _clamp(d0)
        d1 = _clamp(d1 or d0)
        if d1 < d0:
            return
        # guard: very long spans are likely errors
        if (d1 - d0).days > 120:
            return
        cur = d0
        while cur <= d1:
            per_day[cur] += 1
            cur = cur + dt.timedelta(days=1)

    matched_any = False

    for m in meds:
        try:
            ok, dispname = _antibiotic_resource(m)
            if not ok:
                continue
            matched_any = True

            # Timestamps from different resource flavors
            effP = m.get("effectivePeriod") or {}
            eff_dt = m.get("effectiveDateTime")
            authored = m.get("authoredOn")
            di = (m.get("dosageInstruction") or [])
            boundsP = (((di[0] or {}).get("timing") or {}).get("repeat") or {}).get("boundsPeriod", {}) if di else {}
            stmtP = (m.get("effectivePeriod") or {})
            stmt_dt = m.get("effectiveDateTime")

            # prefer periods first
            for s, e in [
                (effP.get("start"), effP.get("end")),
                (boundsP.get("start"), boundsP.get("end")),
                (stmtP.get("start"), stmtP.get("end")),
            ]:
                if s:
                    _add_range(_date(s), _date(e) or _date(s))
                    break
            else:
                # single-day event
                for one in (eff_dt, authored, stmt_dt):
                    if one:
                        d = _date(one)
                        if d and start_date <= d <= today:
                            per_day[d] += 1
                            break
        except Exception:
            continue

    out = []
    for i in range(days_back, -1, -1):
        d = today - dt.timedelta(days=i)
        out.append({"date": d.isoformat(), "value": int(per_day.get(d, 0))})

    if synth_if_empty and all(p["value"] == 0 for p in out):
        # small 3-day synthetic course ending 5 days ago to keep the demo illustrative
        for k in range(7, 4, -1):
            idx = days_back - k
            if 0 <= idx < len(out):
                out[idx]["value"] = 1

    return out





# --- Add to utils.py (near other FHIR helpers) ---

# utils.py

def _fhir_collect_all(path: str, params: dict, access_token: str | None = None) -> list[dict]:
    """Follow Bundle.link[next] to collect all entries with auth."""
    out = []
    clean = {str(k).strip(): v for k, v in params.items() if v is not None}
    url = f"{FHIR_BASE_URL}/{path.lstrip('/')}"
    r = requests.get(url, headers=_auth_headers(access_token), params=clean, timeout=15)
    r.raise_for_status()
    bundle = r.json()

    while True:
        out.extend([e["resource"] for e in bundle.get("entry", []) if "resource" in e])

        # follow next link
        next_link = None
        for l in bundle.get("link", []):
            if l.get("relation") == "next":
                next_link = l.get("url")
                break

        if not next_link:
            break

        r = requests.get(next_link, headers=_auth_headers(access_token), timeout=15)
        r.raise_for_status()
        bundle = r.json()

    return out



def fetch_observations_by_code(patient_id: str, loinc_codes: list[str], days_back: int = 7,
                               access_token: str | None = None) -> list[dict]:
    import datetime as dt
    since = (dt.date.today() - dt.timedelta(days=days_back)).isoformat()
    params = {"patient": patient_id, "date": f"ge{since}", "_count": 200, "code": ",".join(loinc_codes)}
    return _fhir_collect_all("Observation", params, access_token=access_token)



def find_any_patient_with_observation(loinc_code: str, days_back: int = 365,
                                      access_token: str | None = None) -> str | None:
    import datetime as dt
    since = (dt.date.today() - dt.timedelta(days=days_back)).isoformat()
    params = {"code": loinc_code, "date": f"ge{since}", "_include": "Observation:patient", "_count": 100}
    bundle = fhir_get("Observation", params=params, access_token=access_token)
    # try to extract patient from included resources
    for e in bundle.get("entry", []):
        res = e.get("resource", {})
        if res.get("resourceType") == "Patient":
            return res.get("id")
    return None


# utils.py

def synth_timeseries(days: int, start_value: float, drift: float = 0.0, noise: float = 0.2):
    """Generate a simple daily timeseries for demos."""
    import datetime as dt
    import random
    today = dt.date.today()
    v = start_value
    out = []
    for i in range(days, -1, -1):
        d = (today - dt.timedelta(days=i)).isoformat()
        v = max(0.0, v + drift + random.uniform(-noise, noise))
        out.append({"date": d, "value": round(v, 2)})
    return out


def fhir_or_synth_observations(patient_id: str, loinc_codes: list[str],
                               days_back: int, synth: dict,
                               access_token: str | None = None):    
    """
    Try FHIR. If nothing (or request fails), return synthetic series.
    'synth' = {"metric": "CRP", "unit": "mg/L", "start": 7.2, "drift": -0.1, "noise": 0.6}
    """
    try:
        obs = fetch_observations_by_code(patient_id, loinc_codes, days_back, access_token=access_token)
        pts = []
        for o in obs:
            vq = o.get("valueQuantity")
            if not vq or vq.get("value") is None:
                continue
            eff = o.get("effectiveDateTime") or o.get("issued")
            if not eff:
                continue
            pts.append({"date": eff[:10], "value": float(vq["value"])})
        pts.sort(key=lambda p: p["date"])
        if pts:
            return pts, False  # from FHIR
    except Exception:
        pass

    # synth fallback
    pts = synth_timeseries(days_back, synth.get("start", 5.0), synth.get("drift", 0.0), synth.get("noise", 0.3))
    return pts, True
