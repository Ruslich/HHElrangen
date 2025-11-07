# 🏥 Clinical Copilot - Easy Integration Architecture

## Overview

Design for **one-day deployment** by hospital IT, with **script-tag embedding** similar to AnythingLLM-Embed.

---

## 🎯 Integration Goals

1. **Frontend Embed**: Single script tag (`<script>` or `<iframe>`)
2. **Backend Deployment**: Docker-based, 1-day setup
3. **Zero External Dependencies**: Everything self-hosted
4. **FHIR Integration**: Connect to existing hospital FHIR server
5. **Security**: All data stays within hospital network

---

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Hospital EHR System                    │
│  (Epic, Cerner, etc.)                                    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │  EHR Patient View                                │  │
│  │                                                   │  │
│  │  <div id="clinical-copilot"></div>               │  │
│  │  <script src="http://internal:8080/embed.js">    │  │
│  │  <script>                                        │  │
│  │    ClinicalCopilot.init({                        │  │
│  │      apiUrl: 'http://internal:8000',             │  │
│  │      patientId: window.EHR.patientId,            │  │
│  │      fhirUrl: 'http://fhir-server:8080/fhir'     │  │
│  │    });                                           │  │
│  │  </script>                                       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
                          │
                          │ HTTP/HTTPS
                          ▼
┌─────────────────────────────────────────────────────────┐
│         Clinical Copilot Backend (Self-Hosted)          │
│                                                          │
│  ┌──────────────┐         ┌──────────────┐            │
│  │  FastAPI     │────────▶│  AWS Bedrock │            │
│  │  (Port 8000) │         │  (optional)  │            │
│  └──────────────┘         └──────────────┘            │
│         │                                                │
│         │ FHIR Queries                                   │
│         ▼                                                │
│  ┌──────────────┐                                       │
│  │ Hospital FHIR│                                       │
│  │   Server     │                                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
                          │
                          │ Serves
                          ▼
┌─────────────────────────────────────────────────────────┐
│      Clinical Copilot Frontend (Embeddable Widget)      │
│                                                          │
│  ┌──────────────┐                                       │
│  │  React App   │  (Static files: embed.js, CSS)       │
│  │  (Port 8080) │  or CDN-hosted                       │
│  └──────────────┘                                       │
└─────────────────────────────────────────────────────────┘
```

---

## 🔌 Integration Methods

### Method 1: Script Tag Embed (Recommended - Like AnythingLLM-Embed)

**Hospital adds to EHR:**

```html
<!-- In EHR's patient view HTML -->
<div id="clinical-copilot-sidebar"></div>

<script src="http://internal-copilot-server:8080/embed.js"></script>
<script>
  ClinicalCopilot.init({
    container: '#clinical-copilot-sidebar',
    apiUrl: 'http://internal-copilot-server:8000',
    patientId: window.currentPatientId,  // From EHR
    fhirUrl: 'http://hospital-fhir:8080/fhir',
    theme: 'light',  // or 'dark'
    position: 'right'  // 'left' or 'right'
  });
</script>
```

**How it works:**
- `embed.js` loads React widget asynchronously
- Widget mounts into `#clinical-copilot-sidebar`
- Automatically connects to backend API
- Gets patient context from EHR

### Method 2: Iframe Embed (Simpler, More Isolated)

```html
<iframe 
  src="http://internal-copilot-server:8080/widget?patientId=123&apiUrl=http://internal:8000"
  width="400"
  height="600"
  frameborder="0"
  style="border: 1px solid #ccc;">
</iframe>
```

### Method 3: Browser Extension (Advanced)

- Chrome/Firefox extension
- Injects sidebar into EHR pages
- Reads patient ID from DOM
- Communicates with backend

---

## 🚀 Backend Deployment (1-Day Setup)

### Option A: Docker Compose (Recommended - Easiest)

**Structure:**
```
clinical-copilot/
├── docker-compose.yml
├── .env.example
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   └── build/  (React build)
└── setup.sh
```

**Deployment Steps (for Hospital IT):**

1. **Download package:**
   ```bash
   git clone https://github.com/yourorg/clinical-copilot
   cd clinical-copilot
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env with hospital-specific values
   ```

3. **Start with Docker:**
   ```bash
   docker-compose up -d
   ```

4. **Verify:**
   ```bash
   curl http://localhost:8000/hello
   # Should return: {"msg":"Hello AWS!"}
   ```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - FHIR_BASE_URL=${FHIR_BASE_URL}
      - BEDROCK_ENABLED=${BEDROCK_ENABLED}
      - AWS_REGION=${AWS_REGION}
    env_file:
      - .env
    volumes:
      - ./backend:/app
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "8080:80"
    environment:
      - REACT_APP_API_URL=${API_URL:-http://localhost:8000}
    restart: unless-stopped
```

### Option B: Single Docker Image (Even Simpler)

**One Docker image containing both frontend + backend:**

```dockerfile
# Dockerfile (all-in-one)
FROM python:3.11-slim

# Install backend
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy backend
COPY backend/ .

# Build and copy frontend static files
COPY frontend/build /app/static

# Serve both API and static files
CMD uvicorn main:app --host 0.0.0.0 --port 8000
```

**Hospital deployment:**
```bash
docker run -d \
  -p 8000:8000 \
  -e FHIR_BASE_URL=http://fhir-server:8080/fhir \
  -e BEDROCK_ENABLED=true \
  clinical-copilot:latest
```

---

## 📦 Frontend Embed Architecture

### React Widget Structure

```
frontend-embed/
├── src/
│   ├── index.js              # Entry point (exports ClinicalCopilot)
│   ├── App.jsx               # Main widget component
│   ├── components/
│   │   ├── Sidebar.jsx       # Sidebar container
│   │   ├── Chat.jsx          # Chat interface
│   │   ├── PatientSnapshot.jsx
│   │   └── Chart.jsx
│   ├── services/
│   │   └── api.js            # Backend API client
│   └── utils/
│       └── fhir.js           # FHIR helpers
├── public/
│   └── embed.js              # Loader script (synchronous)
├── build/                    # Production build
└── package.json
```

### embed.js (Loader Script)

```javascript
// public/embed.js
(function() {
  'use strict';
  
  // Configuration from script tag
  const config = window.ClinicalCopilotConfig || {};
  
  // Load React widget asynchronously
  const script = document.createElement('script');
  script.src = config.widgetUrl || 'http://localhost:8080/widget.js';
  script.async = true;
  
  script.onload = function() {
    // Initialize widget
    if (window.ClinicalCopilot) {
      window.ClinicalCopilot.init(config);
    }
  };
  
  document.head.appendChild(script);
  
  // CSS
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = config.cssUrl || 'http://localhost:8080/widget.css';
  document.head.appendChild(link);
})();
```

### Widget Initialization

```javascript
// src/index.js
import React from 'react';
import ReactDOM from 'react-dom';
import App from './App';

window.ClinicalCopilot = {
  init: function(config) {
    const container = document.querySelector(config.container || '#clinical-copilot-sidebar');
    if (!container) {
      console.error('Clinical Copilot: Container not found');
      return;
    }
    
    ReactDOM.render(
      <App 
        apiUrl={config.apiUrl}
        patientId={config.patientId}
        fhirUrl={config.fhirUrl}
        theme={config.theme || 'light'}
      />,
      container
    );
  },
  
  destroy: function() {
    const container = document.querySelector('#clinical-copilot-sidebar');
    if (container) {
      ReactDOM.unmountComponentAtNode(container);
    }
  }
};
```

---

## 🔐 Security & Configuration

### Environment Variables (.env)

```env
# FHIR Server (Hospital's existing)
FHIR_BASE_URL=http://hospital-fhir:8080/fhir

# AWS Bedrock (Optional - hospital's own AWS account)
BEDROCK_ENABLED=true
AWS_REGION=eu-central-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# API Configuration
API_URL=http://localhost:8000
CORS_ORIGINS=http://ehr-hospital.local,https://ehr-hospital.com

# Security
JWT_SECRET=...  # For future auth
ALLOWED_ORIGINS=*
```

### CORS Configuration (Backend)

```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

---

## 📋 Hospital Integration Checklist

### Day 1 Setup (4-6 hours)

**Morning (2-3 hours):**
- [ ] Download/Clone repository
- [ ] Configure `.env` file
- [ ] Test Docker Compose locally
- [ ] Verify FHIR server connectivity

**Afternoon (2-3 hours):**
- [ ] Deploy to hospital server (VM/Docker host)
- [ ] Configure firewall rules (ports 8000, 8080)
- [ ] Test API endpoints
- [ ] Embed script in EHR test environment
- [ ] Verify patient context passing

**Verification:**
- [ ] Backend responds: `curl http://server:8000/hello`
- [ ] Frontend loads: Open `http://server:8080/embed.js`
- [ ] Widget appears in EHR test page
- [ ] Patient queries work: Test "Show CRP last 7 days"

---

## 🎨 Widget Customization

### Theming

```javascript
ClinicalCopilot.init({
  theme: {
    primaryColor: '#0066cc',      // Hospital brand color
    sidebarWidth: '400px',
    position: 'right',
    backgroundColor: '#ffffff'
  }
});
```

### Custom Styling (CSS Overrides)

```css
/* In EHR's CSS */
#clinical-copilot-sidebar {
  border-left: 1px solid #ccc;
  box-shadow: -2px 0 5px rgba(0,0,0,0.1);
}

.clinical-copilot-chat {
  font-family: 'Hospital Font', sans-serif;
}
```

---

## 🔄 EHR Integration Patterns

### Pattern 1: Patient Context Injection

**Epic/Cerner injects patient ID:**

```javascript
// EHR provides global variable
window.currentPatient = {
  id: 'patient-123',
  name: 'John Doe',
  mrn: 'MRN-456'
};

// Copilot reads it
ClinicalCopilot.init({
  patientId: window.currentPatient.id,
  patientContext: window.currentPatient
});
```

### Pattern 2: SMART on FHIR Launch

```javascript
// SMART Launch Context
FHIR.oauth2.ready(function(smart) {
  const patient = smart.patient;
  
  ClinicalCopilot.init({
    patientId: patient.id,
    fhirUrl: smart.server.serviceUrl,
    accessToken: smart.state.tokenResponse.access_token
  });
});
```

### Pattern 3: URL Parameters

```
http://ehr-hospital.com/patient/123
# Copilot reads: window.location.pathname
# Extracts patient ID: /patient/123 → "123"
```

---

## 📊 Deployment Options

### Option 1: On-Premises VM
- Hospital's own server
- Full control
- Requires VM setup

### Option 2: Docker Container
- Easiest deployment
- Portable
- Requires Docker host

### Option 3: Kubernetes
- Scalable
- Production-ready
- More complex setup

### Option 4: Cloud (Hospital's AWS/Azure)
- Managed infrastructure
- Auto-scaling
- Requires cloud access

---

## 🧪 Testing Integration

### Test Script (for Hospital IT)

```bash
#!/bin/bash
# test-integration.sh

echo "Testing Clinical Copilot Integration..."

# Test backend
echo "1. Testing backend..."
curl -f http://localhost:8000/hello || exit 1

# Test FHIR connection
echo "2. Testing FHIR connection..."
curl -f http://localhost:8000/fhir_ping || exit 1

# Test frontend
echo "3. Testing frontend..."
curl -f http://localhost:8080/embed.js || exit 1

# Test widget initialization
echo "4. Testing widget..."
curl -f http://localhost:8080/widget.js || exit 1

echo "✅ All tests passed!"
```

---

## 📚 Integration Documentation (For Hospitals)

### Quick Start Guide

1. **Prerequisites:**
   - Docker installed
   - Access to FHIR server
   - Ports 8000, 8080 available

2. **Installation:**
   ```bash
   git clone https://github.com/yourorg/clinical-copilot
   cd clinical-copilot
   cp .env.example .env
   # Edit .env
   docker-compose up -d
   ```

3. **Embed in EHR:**
   ```html
   <script src="http://your-server:8080/embed.js"></script>
   <script>
     ClinicalCopilot.init({
       apiUrl: 'http://your-server:8000',
       patientId: window.currentPatientId
     });
   </script>
   ```

4. **Done!** Widget appears in EHR sidebar.

---

## 🎯 Next Steps for Implementation

1. **Build React Widget** (frontend-embed/)
2. **Create embed.js loader**
3. **Dockerize backend**
4. **Create docker-compose.yml**
5. **Write setup scripts**
6. **Create integration docs**
7. **Test with mock EHR**

---

## 💡 Advantages of This Architecture

✅ **One-Day Setup**: Docker Compose = 4-6 hours  
✅ **Script Tag Embed**: Like AnythingLLM-Embed  
✅ **Self-Hosted**: All data stays in hospital  
✅ **FHIR Native**: Works with existing FHIR servers  
✅ **Customizable**: Themes, positioning, styling  
✅ **Isolated**: Widget doesn't interfere with EHR  
✅ **Secure**: CORS, internal network only  
✅ **Maintainable**: Easy updates via Docker pull  

---

This architecture gives you **maximum integrability** with **minimal setup effort** for hospitals! 🚀

