# 🏥 Smart Clinical Copilot

Healthcare hackathon project - AI assistant for doctors and nurses. Combined project integrating **HHElrangen** backend with **CarePilot** embeddable widget.

## ✨ Features

### Backend (FastAPI)
- 💬 Intelligent chat interface with natural language queries
- 📊 Data visualization and analytics
- 🔍 SQL generation from natural language (NLQ)
- 🏥 Patient data analysis with FHIR support
- 📈 Time series analysis and charting

### Frontend Options
1. **Streamlit App** - Full-featured web application
2. **CarePilot Widget** - Embeddable React sidebar for EHR systems

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm (for CarePilot widget)
- **AWS credentials** (for S3/Bedrock access)

### 1. Clone Repository

```bash
git clone https://github.com/Ruslich/HHElrangen.git
cd HHElrangen
git submodule update --init --recursive
```

### 2. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Setup CarePilot Widget

```bash
# Install Node.js dependencies
cd carepilot-embed
npm install

# Build widget
npm run build
cd ..
```

### 4. Configure Environment

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

### 5. Start Everything

**Option A: Use the startup script (recommended)**

```bash
# Linux/macOS
./start.sh

# Windows
start.bat
```

**Option B: Manual startup**

```bash
# Terminal 1: Backend
cd backend
uvicorn main:app --reload --port 8000

# Terminal 2: Streamlit (optional)
cd frontend
streamlit run app.py
```

## 📍 Access Points

Once started, access the application at:

- **Backend API**: http://127.0.0.1:8000
- **API Documentation**: http://127.0.0.1:8000/docs
- **CarePilot Demo**: http://127.0.0.1:8000/demo
- **Streamlit App**: http://127.0.0.1:8501

## 📁 Project Structure

```
HHElrangen/
├── backend/
│   ├── main.py              # FastAPI backend
│   └── utils.py             # Utility functions
├── frontend/
│   └── app.py               # Streamlit frontend
├── carepilot-embed/         # React widget (submodule)
│   ├── src/                 # React source code
│   ├── dist/                # Built widget files
│   └── package.json
├── ideas/                   # Documentation and planning
├── requirements.txt         # Python dependencies
├── .env.local              # Environment variables
├── start.sh                # Startup script (Linux/macOS)
├── start.bat               # Startup script (Windows)
└── README.md
```

## 🔌 CarePilot Widget Integration

### Embedding in Your EHR System

Add CarePilot to any HTML page with a few lines:

```html
<!-- CarePilot Widget Container -->
<div id="carepilot-sidebar"></div>

<!-- CarePilot Scripts -->
<link rel="stylesheet" href="http://localhost:8000/widget.css">
<script type="module" src="http://localhost:8000/widget.js"></script>
<script>
  window.CarePilot.init({
    container: '#carepilot-sidebar',
    apiUrl: 'http://localhost:8000',
    patientId: 'patient-123',
    patientName: 'John Doe',
    department: 'ICU',
    mrn: 'MRN-001234'
  });
</script>
```

### Dynamic Patient Updates

```javascript
// When patient changes in your EHR
window.CarePilot.updatePatient({
  patientId: 'patient-456',
  patientName: 'Jane Smith',
  department: 'Cardiology',
  mrn: 'MRN-005678'
});
```

### Configuration Options

| Option | Type | Required | Description |
|--------|------|----------|-------------|
| `container` | string | Yes | CSS selector for widget container |
| `apiUrl` | string | Yes | Backend API URL |
| `patientId` | string | Yes | Unique patient identifier |
| `patientName` | string | No | Patient full name |
| `department` | string | No | Hospital department |
| `mrn` | string | No | Medical Record Number |
| `fhirUrl` | string | No | FHIR server endpoint |
| `theme` | string | No | UI theme: 'light' or 'dark' |

## 🔌 API Endpoints

### Core Endpoints

- `POST /chat` - Main chat endpoint
- `POST /patient_chat` - Patient-specific chat
- `POST /nlq` - Natural language queries (SQL generation)
- `POST /analyze` - Data analysis with charting
- `GET /datasets` - List available datasets
- `GET /head` - Preview data

### SMART/FHIR Endpoints

- `POST /smart/mock_login` - Mock SMART login
- `POST /smart/logout` - SMART logout
- `GET /fhir_ping` - Test FHIR connection
- `GET /smart_demo` - SMART demo launcher

### Widget Serving

- `GET /widget.js` - CarePilot widget JavaScript
- `GET /widget.css` - CarePilot widget styles
- `GET /embed.js` - Embed loader script
- `GET /demo` - Demo integration page

**Full API Documentation**: http://127.0.0.1:8000/docs

## 🛠️ Development

### Rebuilding CarePilot Widget

```bash
cd carepilot-embed
npm run build
cd ..
```

### Running Widget in Development Mode

```bash
cd carepilot-embed
npm run dev
# Widget available at http://localhost:5173
```

## 📚 Additional Resources

- **CarePilot Widget Docs**: See `carepilot-embed/README.md`
- **Implementation Roadmap**: See `ideas/IMPLEMENTATION_ROADMAP.md`
- **Integration Architecture**: See `ideas/INTEGRATION_ARCHITECTURE.md`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is part of the Healthcare Hackathon Bayern.

---

**Built with ❤️ for healthcare professionals**
