# 🗺️ Implementation Roadmap

## Goal: Build Embeddable Clinical Copilot (Like AnythingLLM-Embed)

**Target**: One-day deployment, script-tag embedding, self-hosted

---

## Phase 1: Frontend Widget (Week 1) 🎨

### 1.1 Create React Widget Structure
```
frontend-embed/
├── src/
│   ├── index.js           # Entry point, exports ClinicalCopilot
│   ├── App.jsx            # Main widget component
│   ├── components/
│   │   ├── Sidebar.jsx
│   │   ├── Chat.jsx
│   │   ├── PatientSnapshot.jsx
│   │   ├── FormGenerator.jsx
│   │   └── Chart.jsx
│   ├── services/
│   │   └── api.js         # Backend API client
│   └── styles/
│       └── widget.css
├── public/
│   └── embed.js           # Loader script
├── package.json
└── vite.config.js         # Or Create React App
```

**Tasks:**
- [ ] Set up React project (Vite or CRA)
- [ ] Create widget entry point (`index.js`)
- [ ] Build main App component
- [ ] Create embed.js loader script
- [ ] Build API service layer
- [ ] Style widget (CSS)
- [ ] Test widget initialization

**Deliverable**: Working React widget that can be embedded via script tag

---

### 1.2 Build Embed Loader Script
**File**: `frontend-embed/public/embed.js`

**Features:**
- Auto-loads from script tag attributes
- Loads CSS and JS asynchronously
- Initializes widget automatically
- Handles errors gracefully

**Example Usage:**
```html
<script src="http://server:8080/embed.js" 
        data-api-url="http://server:8000"
        data-patient-id="123">
</script>
```

---

### 1.3 Build Widget Components

**Chat Component:**
- Message history
- Input field
- Send button
- Loading states

**Patient Snapshot:**
- Patient info display
- Quick actions
- Recent labs/meds

**Form Generator:**
- Form type selector
- Auto-filled fields
- Edit capability
- Export PDF

**Chart Component:**
- Time series charts
- Lab values visualization
- Medication timeline

---

## Phase 2: Backend Dockerization (Week 1) 🐳

### 2.1 Create Dockerfile
**File**: `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Tasks:**
- [ ] Create Dockerfile
- [ ] Test Docker build
- [ ] Verify all dependencies work
- [ ] Test container startup

---

### 2.2 Create Docker Compose
**File**: `docker-compose.yml`

**Services:**
- Backend (FastAPI)
- Frontend (Nginx serving React build)

**Tasks:**
- [ ] Create docker-compose.yml
- [ ] Configure environment variables
- [ ] Set up volume mounts
- [ ] Add health checks
- [ ] Test full stack

---

### 2.3 Environment Configuration
**File**: `.env.example`

**Variables:**
- FHIR_BASE_URL
- AWS credentials (optional)
- CORS_ORIGINS
- API_URL

**Tasks:**
- [ ] Create .env.example
- [ ] Document all variables
- [ ] Add validation
- [ ] Create setup script

---

## Phase 3: Deployment Scripts (Week 1) 🚀

### 3.1 Setup Script
**File**: `setup.sh`

**Features:**
- Check prerequisites (Docker)
- Create .env from template
- Build Docker images
- Start services
- Health checks
- Print instructions

**Tasks:**
- [ ] Write setup.sh
- [ ] Add error handling
- [ ] Add progress indicators
- [ ] Test on clean system

---

### 3.2 Health Check Script
**File**: `test-integration.sh`

**Checks:**
- Backend API responds
- Frontend serves files
- FHIR connection works
- Widget loads correctly

**Tasks:**
- [ ] Write test script
- [ ] Test all endpoints
- [ ] Add failure reporting

---

## Phase 4: Documentation (Week 2) 📚

### 4.1 Hospital Integration Guide
**File**: `HOSPITAL_SETUP.md`

**Contents:**
- Prerequisites
- Installation steps
- Configuration
- EHR integration
- Troubleshooting

**Tasks:**
- [ ] Write setup guide
- [ ] Add screenshots
- [ ] Create video tutorial
- [ ] Add FAQ section

---

### 4.2 API Documentation
**File**: `API_DOCS.md`

**Contents:**
- Endpoint list
- Request/response formats
- Authentication
- Error codes

**Tasks:**
- [ ] Document all endpoints
- [ ] Add examples
- [ ] Create Postman collection

---

### 4.3 Widget API Documentation
**File**: `WIDGET_API.md`

**Contents:**
- Initialization options
- Methods
- Events
- Customization

**Tasks:**
- [ ] Document widget API
- [ ] Add code examples
- [ ] Show customization options

---

## Phase 5: Testing & Polish (Week 2) ✅

### 5.1 Integration Testing
**Tests:**
- Widget loads in test HTML
- API calls work
- Patient context passes
- Charts render
- Forms generate

**Tasks:**
- [ ] Create test HTML page
- [ ] Test all features
- [ ] Fix bugs
- [ ] Performance optimization

---

### 5.2 EHR Integration Testing
**Tests:**
- Epic integration (mock)
- Cerner integration (mock)
- SMART on FHIR launch
- Patient context injection

**Tasks:**
- [ ] Create mock EHR pages
- [ ] Test integration patterns
- [ ] Verify patient context
- [ ] Test error handling

---

### 5.3 Security Review
**Checks:**
- CORS configuration
- Input validation
- XSS prevention
- CSRF protection
- Data encryption

**Tasks:**
- [ ] Security audit
- [ ] Fix vulnerabilities
- [ ] Add security headers
- [ ] Test penetration

---

## Phase 6: Hackathon Demo (Week 2) 🎯

### 6.1 Demo Setup
**Components:**
- Working widget in test EHR
- Docker Compose running
- Sample patient data
- Demo script

**Tasks:**
- [ ] Prepare demo environment
- [ ] Create demo script
- [ ] Practice presentation
- [ ] Prepare backup plan

---

### 6.2 Pitch Materials
**Contents:**
- Integration demo video
- Architecture diagram
- One-day setup timeline
- Benefits for hospitals

**Tasks:**
- [ ] Record demo video
- [ ] Create slides
- [ ] Write pitch script
- [ ] Prepare Q&A answers

---

## Implementation Priority

### Must Have (for Hackathon):
1. ✅ React widget (basic)
2. ✅ embed.js loader
3. ✅ Docker setup
4. ✅ Basic integration guide
5. ✅ Working demo

### Nice to Have:
- Advanced theming
- Multiple EHR integrations
- Performance optimization
- Comprehensive docs
- Video tutorials

---

## Timeline Estimate

**Week 1:**
- Days 1-2: React widget + embed.js
- Days 3-4: Docker setup
- Day 5: Testing + fixes

**Week 2:**
- Days 1-2: Documentation
- Days 3-4: Testing + polish
- Day 5: Demo prep

**Total: ~10 days for MVP**

---

## Quick Start (For Development)

```bash
# 1. Create React widget
cd frontend-embed
npm create vite@latest . -- --template react
npm install
npm run build

# 2. Dockerize backend
cd backend
docker build -t clinical-copilot-backend .

# 3. Test locally
docker-compose up -d
open http://localhost:8080/test.html
```

---

## Key Decisions

1. **Widget Framework**: React (most embeddable)
2. **Build Tool**: Vite (fast builds)
3. **Deployment**: Docker Compose (easiest)
4. **Integration**: Script tag (simplest)
5. **Styling**: CSS (no framework dependency)

---

## Success Criteria

✅ Hospital IT can set up in **one day**  
✅ Integration requires **one script tag**  
✅ Widget works in **any EHR**  
✅ All data stays **self-hosted**  
✅ **Zero external dependencies**  

---

## Next Steps

1. **Start with React widget** (Phase 1.1)
2. **Build embed.js** (Phase 1.2)
3. **Dockerize backend** (Phase 2)
4. **Test integration** (Phase 5)
5. **Prepare demo** (Phase 6)

---

**Ready to build? Start with Phase 1.1! 🚀**

