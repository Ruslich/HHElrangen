# 🏥 Smart Clinical Copilot

Healthcare hackathon project - AI assistant for doctors and nurses.

## Quick Start

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env.local` file in the project root:

```env
AWS_REGION=eu-central-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
DATA_BUCKET=health-demo-hherlangen
DATA_PREFIX=data/
BEDROCK_ENABLED=true
BEDROCK_REGION=eu-central-1
BEDROCK_MODEL_ID=amazon.nova-lite-v1:0
BACKEND_URL=http://127.0.0.1:8000
```

### 4. Start Backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Start Frontend (in new terminal)

```bash
cd frontend
streamlit run app.py
```

## Project Structure

```
HHElrangen/
├── backend/
│   ├── main.py
│   └── utils.py
├── frontend/
│   └── app.py
├── data/
│   └── labs.csv
├── requirements.txt
└── README.md
```

## API Endpoints

- `POST /chat` - Main chat endpoint
- `POST /nlq` - Natural language queries
- `GET /datasets` - List datasets
- `GET /head` - Preview data

Backend API docs: http://127.0.0.1:8000/docs
