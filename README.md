# 🏥 Health-Chat (SMART-on-FHIR Clinical Copilot)

An embeddable side-panel that lives inside an EHR and gives clinicians:

- **Patient insights** from FHIR (labs, meds, vitals) in one chart + one-line summary.
- **Cohort analytics** over the hospital’s analytics DB (**Athena + S3 by default**).
- **Automated document pre-fill** for common nurse/doctor forms to cut paperwork.

**Interoperable by design:** SMART-on-FHIR for identity + context, FHIR R4 reads for clinical data, and a swappable analytics adapter for SQL engines.  
**LLM optional:** runs fully without a model; Bedrock adds planning/summaries/NL→SQL.

---

## 🎯 Goals

1. **Drop-in to any SMART-capable EHR.**  
   App receives a token + patient context and uses standard FHIR REST reads.

2. **Zero vendor lock-in for analytics.**  
   Ships with an **Athena+S3** adapter; replace 2–3 functions to run on Snowflake, BigQuery, SQL Server, or Postgres.

3. **Give minutes back, safely.**  
   Read-only FHIR + **SELECT-only** SQL with table allow-lists and query linting.  
   Partial pre-fill today targets ~**2 minutes saved per form** (~5%); specific templates can reach much higher later.

---

## 🔌 Interoperability: how it works

### 1) SMART context & tokens (EHR connector)
- **Launch:** EHR performs SMART launch (OIDC/OAuth2).
- **Token:** backend stores a **short-lived Bearer token** and **patient context** in a session cache.
- **Demo mode:** a **SMART mock** endpoint simulates the launch locally (same contract as production).

> **Portability:** moving between EHRs changes only the issuer/scopes and the FHIR base URL.

### 2) FHIR R4 client (clinical connector)
- **Config:** `FHIR_BASE_URL` from env; every request adds `Authorization: Bearer <token>`.
- **Resources (read-only):** `Patient`, `Observation` (e.g., CRP/creatinine),  
  `MedicationAdministration`, `MedicationRequest`, `MedicationStatement`.
- **Pagination:** follows `Bundle.link[next]`.

### 3) Analytics adapter (SQL engine)
- **Default path:** S3 (data lake) → Glue Catalog (schema) → Athena (SQL).
- **NL→SQL:** guardrails enforce **SELECT-only** and allow-listed tables; one self-repair retry for simple syntax issues.
- **Swap the engine:** implement equivalents of:
  - `get_table_summaries()` – list tables/columns for grounding
  - `is_sql_safe(sql)` – enforce your read-only policy
  - `sql_to_df(sql)` – run a query and return rows (DataFrame/JSON)

UI, prompts, and charts stay unchanged.

---

## 🧾 Automated PDF Pre-Fill (nurses & doctors’ forms)

We ship a small module that **prefills AcroForm fields** in common clinical PDFs using data pulled from FHIR and the current encounter context.

- **Where it lives:** [`backend/pdf_prefill/`](https://github.com/Ruslich/HHElrangen/tree/main/backend/pdf_prefill)  
- **Current templates (examples):**
  - `Ärztlicher_Entlassungsbericht_template.pdf` / `Entlassungsbericht_template.pdf`
  - `Verlaufsbericht_template.pdf`
  - `Sturzereignisprotokoll_template.pdf`
  - `Medikationsfehler_template.pdf`
  - `Einwilligungserklärung_template.pdf`
  - `Durchgangsarztbericht_template.pdf` (D-Arzt, F1000)

**What we pre-fill today (read-only, clinician reviews & signs):**
- Patient identifiers (name, DOB, MRN), encounter dates, location/ward
- Attending/author information (where available)
- Recent vitals, key labs (e.g., CRP/creatinine) with timestamps
- Active medications / recent administrations
- Common boilerplate fields (date/time, checkboxes, signatures placeholders)

**Time impact now:** we target **~2 minutes saved per form** (~5%) by removing retyping/copy-paste.  
**Future:** where sections become fully structured, savings can grow substantially on specific templates.

> **How it works (high-level):** the prefill routine maps FHIR fields → PDF form keys (AcroForm). Templates and mappings are kept alongside the PDFs, so adding a new hospital form is a *configuration task* rather than an engineering project.

---

## 🧱 Tech Stack

**Backend**
- Python 3.10+, FastAPI, Pydantic  
- Pandas / PyArrow for tabular data  
- Boto3 (Athena, Glue, S3)  
- Optional: Amazon Bedrock (planning, summaries, NL→SQL)

**Frontend**
- **React 18**, **TypeScript**  
- **Tailwind CSS**  
- **shadcn/ui** components  
- **lucide-react** icons  
- **Recharts** for charts  
- Lightweight **embeddable sidebar widget** (can mount into host EHR UI)  
- **Streamlit** demo app for local testing

**Interoperability**
- SMART-on-FHIR (OIDC/OAuth2)  
- FHIR R4 REST (read-only)

**Build/Dev**
- Vite + ESLint + Prettier (embed package)  
- Uvicorn for backend dev server

---

## ⚙️ Configuration

Create `.env.local` in repo root:

```env
# --- App ---
BACKEND_URL=http://127.0.0.1:8000
ENV=local
LOG_LEVEL=info

# --- SMART / FHIR ---
FHIR_BASE_URL=https://hapi.server/fhir             # hospital or demo FHIR
SMART_ISSUER_URL=https://idp.example.org           # prod IdP (unused in mock)
SMART_MOCK_ENABLED=true                            # local SMART flow
SMART_SESSION_TTL_SECONDS=900

# --- Analytics (Athena adapter) ---
AWS_REGION=eu-central-1
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
ATHENA_WORKGROUP=primary
ATHENA_DATABASE=health_db
ATHENA_OUTPUT_S3=s3://your-athena-results/
GLUE_CATALOG_DATABASE=health_db

# --- Data lake (if you serve files) ---
DATA_BUCKET=your-bucket
DATA_PREFIX=data/

# --- LLM (optional) ---
BEDROCK_ENABLED=false
BEDROCK_REGION=eu-central-1
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0             # or your inference profile


## 🚀 Quick Start

```bash
# 1) Python env
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt

# 2) (Optional) build the React sidebar widget from /embed
cd embed
npm install
npm run build
cd ..

# 3) Run backend
uvicorn main:app --reload --port 8000

# 4) Run the Streamlit demo (separate terminal)
streamlit run app.py
```

- **API docs:** <http://127.0.0.1:8000/docs>  
- **Streamlit demo:** <http://127.0.0.1:8501>

---

## 🔑 SMART & Token Flows

**Local (mocked)**

1. `POST /smart/mock_login` with a test user/patient.  
2. Backend creates a session (`session_id`) and caches `{access_token, patient_id}` (TTL).  
3. Frontend includes `session_id` on subsequent requests.  
4. `GET /fhir_ping` proves tokened FHIR reads.  
5. `POST /smart/logout` clears the session.

**Production (real SMART launch)**

- Replace the mock with the EHR’s **OIDC authorization code + PKCE** flow.  
- On redirect, backend exchanges the code for tokens and stores them in the session cache.  
- All FHIR helpers already expect a Bearer token—**no code changes**.

---

## 🔎 Key Endpoints

### Patient insights (FHIR)

- `POST /patient_chat` – NL intents such as:
  - “CRP last 7 days”
  - “Creatinine trend + nephro meds”
  - Returns a structured payload for charts/tables.
- `GET /fhir_ping` – verifies FHIR base URL + token reachability.

### Cohort analytics (SQL)

- `POST /nlq` – NL → **safe SQL** (SELECT-only, allow-list) → chart/table via analytics adapter.  
- `POST /analyze` – direct analysis with charting over a provided dataset.

### SMART & demo

- `POST /smart/mock_login` – create local SMART session (token + patient).  
- `POST /smart/logout` – end session.  
- `GET /demo` – simple page embedding the sidebar.

---

## 🧾 Forms Pre-Fill Module

- Templates live in `backend/pdf_prefill/`.  
- Prefills AcroForm fields using FHIR/context (IDs, dates, meds, vitals, author, boilerplate).  
- Produces a **filled PDF** for clinician review/signature.

**Extend it:** drop a new PDF template in the folder and add field mappings.  
*(If exposing an endpoint, use e.g. `POST /forms/prefill` with `{ template, patient_id, encounter_id }`.)*

---

## 🔐 Security & Compliance

- **Read-only by default:** no FHIR writes; SQL guardrails block DDL/DML.  
- **Scoped access:** request **minimum SMART scopes** for reads.  
- **Short-lived sessions:** token + context cached with TTL; logout purges cache.  
- **Auditability:** NL prompts and final SQL logged (no PHI by default).

> For production: deploy behind hospital SSO, private networking (VPC), and your standard logging/monitoring.

---

## 📈 Present Limits (Hackathon Build)

- FHIR focus: Observations + Medications.  
- Pre-fill: subset of high-value forms; mapping coverage varies by site.  
- NL→SQL: common aggregates/filters; advanced SQL may need adapter tuning.

---

## 🗺️ Roadmap

- Full SMART production launch (auth code + PKCE) & refresh-token rotation.  
- More FHIR resources: AllergyIntolerance, Condition, Procedure, DiagnosticReport.  
- Additional form templates + deeper auto-fill (structured sections).  
- Analytics adapters: Snowflake / BigQuery / MS-SQL / Postgres.  
- Fine-grained audit and export of generated summaries.


**Built with ❤️ for healthcare professionals**
