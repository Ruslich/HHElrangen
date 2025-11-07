# 🔌 Integration Examples

## Example 1: Simple Script Tag Embed (Like AnythingLLM-Embed)

### Hospital EHR HTML

```html
<!DOCTYPE html>
<html>
<head>
    <title>Patient View - EHR System</title>
</head>
<body>
    <h1>Patient: John Doe (ID: 12345)</h1>
    
    <!-- Existing EHR content -->
    <div id="patient-chart">...</div>
    
    <!-- Clinical Copilot Sidebar -->
    <div id="clinical-copilot-sidebar"></div>
    
    <!-- Embed Script (One line!) -->
    <script src="http://internal-copilot:8080/embed.js" 
            data-api-url="http://internal-copilot:8000"
            data-patient-id="12345"
            data-fhir-url="http://fhir-server:8080/fhir">
    </script>
</body>
</html>
```

### embed.js (Auto-initializes from data attributes)

```javascript
// embed.js - Auto-loads widget based on script tag attributes
(function() {
  const script = document.currentScript;
  const apiUrl = script.getAttribute('data-api-url') || 'http://localhost:8000';
  const patientId = script.getAttribute('data-patient-id') || null;
  const fhirUrl = script.getAttribute('data-fhir-url') || null;
  
  // Load widget CSS
  const css = document.createElement('link');
  css.rel = 'stylesheet';
  css.href = apiUrl.replace(':8000', ':8080') + '/widget.css';
  document.head.appendChild(css);
  
  // Load widget JS
  const js = document.createElement('script');
  js.src = apiUrl.replace(':8000', ':8080') + '/widget.js';
  js.onload = function() {
    if (window.ClinicalCopilot && patientId) {
      window.ClinicalCopilot.init({
        container: '#clinical-copilot-sidebar',
        apiUrl: apiUrl,
        patientId: patientId,
        fhirUrl: fhirUrl
      });
    }
  };
  document.head.appendChild(js);
})();
```

---

## Example 2: Iframe Embed (Simplest)

### Hospital EHR HTML

```html
<!-- Sidebar iframe -->
<iframe 
  id="copilot-iframe"
  src="http://internal-copilot:8080/widget?patientId=12345&apiUrl=http://internal-copilot:8000"
  width="400"
  height="100%"
  frameborder="0"
  style="position: fixed; right: 0; top: 0; bottom: 0; z-index: 1000;">
</iframe>

<!-- Communicate with iframe -->
<script>
  // Send patient context to iframe
  window.addEventListener('message', function(event) {
    if (event.origin === 'http://internal-copilot:8080') {
      // Handle messages from copilot
      console.log('Message from copilot:', event.data);
    }
  });
  
  // Notify iframe of patient change
  function updatePatient(patientId) {
    const iframe = document.getElementById('copilot-iframe');
    iframe.contentWindow.postMessage({
      type: 'PATIENT_CHANGED',
      patientId: patientId
    }, 'http://internal-copilot:8080');
  }
</script>
```

---

## Example 3: React Widget Component (Embeddable)

### Widget Entry Point (widget.js)

```javascript
// widget.js - React widget that can be embedded
import React from 'react';
import ReactDOM from 'react-dom';
import ClinicalCopilotWidget from './App';

window.ClinicalCopilot = {
  instances: [],
  
  init: function(config) {
    const container = document.querySelector(config.container || '#clinical-copilot-sidebar');
    if (!container) {
      console.error('Clinical Copilot: Container not found:', config.container);
      return null;
    }
    
    const instance = ReactDOM.createRoot(container);
    instance.render(
      React.createElement(ClinicalCopilotWidget, {
        apiUrl: config.apiUrl,
        patientId: config.patientId,
        fhirUrl: config.fhirUrl,
        theme: config.theme || 'light',
        onClose: () => this.destroy(config.container)
      })
    );
    
    this.instances.push({ container: config.container, instance });
    return instance;
  },
  
  destroy: function(containerSelector) {
    const index = this.instances.findIndex(i => i.container === containerSelector);
    if (index !== -1) {
      const { instance } = this.instances[index];
      instance.unmount();
      this.instances.splice(index, 1);
    }
  }
};
```

### Widget App Component

```jsx
// App.jsx - Main widget component
import React, { useState, useEffect } from 'react';
import Chat from './components/Chat';
import PatientSnapshot from './components/PatientSnapshot';
import './styles/widget.css';

function ClinicalCopilotWidget({ apiUrl, patientId, fhirUrl, theme, onClose }) {
  const [activeTab, setActiveTab] = useState('chat');
  
  return (
    <div className={`clinical-copilot-widget theme-${theme}`}>
      <div className="widget-header">
        <h3>Clinical Copilot</h3>
        <button onClick={onClose}>×</button>
      </div>
      
      <div className="widget-tabs">
        <button 
          className={activeTab === 'snapshot' ? 'active' : ''}
          onClick={() => setActiveTab('snapshot')}>
          Patient Snapshot
        </button>
        <button 
          className={activeTab === 'chat' ? 'active' : ''}
          onClick={() => setActiveTab('chat')}>
          Chat
        </button>
      </div>
      
      <div className="widget-content">
        {activeTab === 'snapshot' && (
          <PatientSnapshot apiUrl={apiUrl} patientId={patientId} />
        )}
        {activeTab === 'chat' && (
          <Chat apiUrl={apiUrl} patientId={patientId} fhirUrl={fhirUrl} />
        )}
      </div>
    </div>
  );
}

export default ClinicalCopilotWidget;
```

---

## Example 4: Docker Compose Setup

### docker-compose.yml

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - FHIR_BASE_URL=${FHIR_BASE_URL}
      - BEDROCK_ENABLED=${BEDROCK_ENABLED:-false}
      - AWS_REGION=${AWS_REGION}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    env_file:
      - .env
    volumes:
      - ./backend:/app
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/hello"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend-embed
      dockerfile: Dockerfile
    ports:
      - "8080:80"
    environment:
      - REACT_APP_API_URL=${API_URL:-http://localhost:8000}
    depends_on:
      - backend
    restart: unless-stopped
```

### Backend Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile

```dockerfile
# Build stage
FROM node:18-alpine AS build

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# Build
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built files
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

---

## Example 5: One-Click Setup Script

### setup.sh

```bash
#!/bin/bash

set -e

echo "🏥 Clinical Copilot - One-Day Setup"
echo "===================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env with your hospital's configuration"
    echo "   Key settings:"
    echo "   - FHIR_BASE_URL (your FHIR server)"
    echo "   - AWS credentials (if using Bedrock)"
    read -p "Press Enter after editing .env..."
fi

# Build and start
echo "🐳 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 10

# Health check
echo "🔍 Checking services..."
if curl -f http://localhost:8000/hello > /dev/null 2>&1; then
    echo "✅ Backend is running!"
else
    echo "❌ Backend is not responding"
    exit 1
fi

if curl -f http://localhost:8080/embed.js > /dev/null 2>&1; then
    echo "✅ Frontend is running!"
else
    echo "❌ Frontend is not responding"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Add this to your EHR HTML:"
echo ""
echo '   <script src="http://your-server:8080/embed.js"'
echo '           data-api-url="http://your-server:8000"'
echo '           data-patient-id="YOUR_PATIENT_ID">'
echo '   </script>'
echo ""
echo "2. Test the integration"
echo "3. Configure firewall rules if needed"
echo ""
echo "📊 Services:"
echo "   Backend API: http://localhost:8000"
echo "   Frontend: http://localhost:8080"
echo "   API Docs: http://localhost:8000/docs"
echo ""
```

---

## Example 6: EHR Integration Patterns

### Epic Integration

```javascript
// Epic provides patient context via global
if (window.Epic && window.Epic.currentPatient) {
  ClinicalCopilot.init({
    patientId: window.Epic.currentPatient.id,
    patientName: window.Epic.currentPatient.name,
    mrn: window.Epic.currentPatient.mrn
  });
}
```

### Cerner Integration

```javascript
// Cerner PowerChart integration
if (window.Cerner && window.Cerner.PowerChart) {
  window.Cerner.PowerChart.onPatientChange(function(patient) {
    ClinicalCopilot.init({
      patientId: patient.personId,
      encounterId: patient.encounterId
    });
  });
}
```

### SMART on FHIR Launch

```javascript
// SMART Launch
FHIR.oauth2.ready(function(smart) {
  const patient = smart.patient;
  const encounter = smart.encounter;
  
  ClinicalCopilot.init({
    patientId: patient.id,
    encounterId: encounter.id,
    fhirUrl: smart.server.serviceUrl,
    accessToken: smart.state.tokenResponse.access_token
  });
});
```

---

## Example 7: API Client (Frontend)

```javascript
// services/api.js
class ClinicalCopilotAPI {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }
  
  async chat(text, sessionId, patientId, context = {}) {
    const response = await fetch(`${this.baseUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        session_id: sessionId,
        patient_id: patientId,
        context
      })
    });
    return response.json();
  }
  
  async patientChat(patientId, text, daysBack = 7) {
    const response = await fetch(`${this.baseUrl}/patient_chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        patient_id: patientId,
        text,
        days_back: daysBack
      })
    });
    return response.json();
  }
  
  async fhirPing() {
    const response = await fetch(`${this.baseUrl}/fhir_ping`);
    return response.json();
  }
}

export default ClinicalCopilotAPI;
```

---

## Example 8: Widget Styling

```css
/* widget.css */
.clinical-copilot-widget {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  width: 400px;
  height: 100vh;
  background: #ffffff;
  border-left: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  position: fixed;
  right: 0;
  top: 0;
  z-index: 1000;
  box-shadow: -2px 0 10px rgba(0,0,0,0.1);
}

.clinical-copilot-widget.theme-dark {
  background: #1e1e1e;
  color: #ffffff;
  border-left-color: #333;
}

.widget-header {
  padding: 16px;
  border-bottom: 1px solid #e0e0e0;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.widget-tabs {
  display: flex;
  border-bottom: 1px solid #e0e0e0;
}

.widget-tabs button {
  flex: 1;
  padding: 12px;
  border: none;
  background: transparent;
  cursor: pointer;
  border-bottom: 2px solid transparent;
}

.widget-tabs button.active {
  border-bottom-color: #0066cc;
  color: #0066cc;
}

.widget-content {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
}
```

---

## Example 9: Testing Integration

### test.html (Local testing)

```html
<!DOCTYPE html>
<html>
<head>
    <title>Clinical Copilot Test</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 0;
            padding: 20px;
        }
        #patient-info {
            background: #f0f0f0;
            padding: 20px;
            margin-bottom: 20px;
        }
        #clinical-copilot-sidebar {
            position: fixed;
            right: 0;
            top: 0;
            width: 400px;
            height: 100vh;
        }
    </style>
</head>
<body>
    <div id="patient-info">
        <h1>Test Patient: John Doe</h1>
        <p>Patient ID: <strong>test-patient-123</strong></p>
        <p>MRN: MRN-456</p>
    </div>
    
    <div id="clinical-copilot-sidebar"></div>
    
    <script src="http://localhost:8080/embed.js" 
            data-api-url="http://localhost:8000"
            data-patient-id="test-patient-123"
            data-fhir-url="http://localhost:8080/fhir">
    </script>
    
    <script>
        // Simulate EHR patient context
        window.currentPatientId = 'test-patient-123';
        window.currentPatientName = 'John Doe';
    </script>
</body>
</html>
```

---

## Summary

These examples show:

1. **Script Tag Embed** - Simplest, like AnythingLLM-Embed
2. **Iframe Embed** - Most isolated
3. **React Widget** - Most flexible
4. **Docker Setup** - One-day deployment
5. **EHR Integration** - Real-world patterns
6. **API Client** - Clean communication
7. **Styling** - Customizable theme
8. **Testing** - Local test page

**Key Takeaway**: Hospital IT adds **one script tag** and it just works! 🚀

