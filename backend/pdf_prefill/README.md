# PDF Pre-fill Integration

## Overview
This module provides PDF form pre-filling capabilities for the Elrangen Hackathon project, allowing automatic generation of medical reports using real PDF templates.

## Features

### ✅ Implemented
1. **Template Management** - Scan and manage PDF form templates
2. **Automatic Field Mapping** - AI-powered mapping of patient data to PDF fields
3. **PDF Generation** - Fill templates with patient data
4. **Edit Before Generate** - Review and edit data before PDF creation
5. **First/Last Name Separation** - Properly parse patient names into components
6. **English Output** - All UI and data in English
7. **Download Functionality** - Download generated PDFs directly

## Available Templates

1. **Entlassungsbericht_template** - Discharge Report
2. **Ärztlicher_Entlassungsberich_template** - Medical Discharge Report
3. **Verlaufsbericht_template** - Progress Report
4. **Durchgangsarztbericht_template** - Accident Report
5. **Einwilligungserklärung_template** - Consent Form
6. **Sturzereignisprotokoll_template** - Fall Incident Report
7. **Medikationsfehler_template** - Medication Error Report

## Data Structure

The system expects patient data in this format:

```javascript
{
  // Names (properly separated)
  first_name: "Anna",
  last_name: "Schmidt",
  full_name: "Anna Schmidt",
  
  // Demographics
  age: 45,
  gender: "F",
  
  // Location
  department: "Intensive Care Unit",
  room: "1A",
  
  // IDs
  id: "DEMO-CRP-001",
  mrn: "DEMO-CRP-001",
  
  // Dates
  admissionDate: "2025-10-31",
  
  // Clinical (English)
  diagnosis: "Appendicitis",
  medications_text: "Medication A 500mg - 2x daily, Medication B 250mg - 1x daily",
  vitals_text: "Blood Pressure: 120/80 mmHg, Heart Rate: 70 bpm...",
  labs_text: "CRP: 25 mg/L, Leukocytes: 7 /nL...",
  notes: "Patient in stable condition."
}
```

## API Endpoints

### GET /pdf/templates
List all available PDF templates
```bash
curl http://localhost:8000/pdf/templates
```

### POST /pdf/generate
Generate a filled PDF
```bash
curl -X POST http://localhost:8000/pdf/generate \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "Durchgangsarztbericht_template",
    "data": { ... patient data ... },
    "user_id": "optional_user_id",
    "persist": true
  }'
```

### GET /pdf/{pdf_id}
Download a generated PDF
```bash
curl http://localhost:8000/pdf/abc-123 --output report.pdf
```

## Workflow

1. **User clicks "Prepare Form"**
   - System fetches patient data
   - Separates first/last name
   - Formats medications, vitals, labs as text (English)
   - Shows editable form

2. **User reviews/edits data**
   - Can modify any field
   - All fields shown in English
   - Changes reflected in final PDF

3. **User clicks "Generate PDF"**
   - Calls `/patient_chat` for AI context (Claude Sonnet 4.5)
   - Calls `/pdf/generate` to fill template
   - Returns downloadable PDF

4. **User downloads PDF**
   - Click "Download PDF" button
   - File saved with descriptive name

## Key Fixes Applied

### 1. Name Separation
- **Before**: Both "Vorname" and "Name" got full name
- **After**: "Vorname" → first_name, "Name" → last_name
- Parser splits on whitespace: "Anna Schmidt" → first:"Anna", last:"Schmidt"

### 2. English Output
- **UI**: All labels in English (Form Type, Download PDF, etc.)
- **Data**: Patient info formatted in English (Blood Pressure, Heart Rate, etc.)
- **Messages**: Success/error messages in English

### 3. Edit Button
- **New workflow**: Prepare → Edit → Generate
- **Button**: "Prepare Form" (opens edit view)
- **Edit view**: Editable fields for all key data
- **Actions**: Cancel or Generate PDF

## Testing

Backend test:
```bash
cd '/Users/aliguliyev/Desktop/Elrangen Hackathon'
curl -X POST http://localhost:8000/pdf/generate \
  -H "Content-Type: application/json" \
  -d '{"template_name": "Durchgangsarztbericht_template", "data": {"first_name": "Anna", "last_name": "Schmidt", "diagnosis": "Appendicitis"}, "persist": false}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); print('✅ Works!' if 'pdf_base64' in r else '❌ Failed')"
```

## Dependencies

- `openai` - For semantic field mapping (embeddings)
- `pypdf` - PDF form filling
- `PyMuPDF` - Font size fixes and advanced PDF operations
- `reportlab` - PDF generation fallback
- `python-dotenv` - Environment variables

## Configuration

Set these in `.env.local`:
```bash
OPENAI_API_KEY=sk-...  # For auto-mapping
PDF_PREFILL_BASE_DIR=/path/to/backend/pdf_prefill  # Optional, auto-detected
```

## Notes

- Templates are in `backend/pdf_prefill/pdf_templates/`
- Mappings are in `backend/pdf_prefill/field_mappings/`
- Generated PDFs stored in `backend/pdf_prefill/generated/`
- Uses Claude Sonnet 4.5 for patient data gathering via `/patient_chat`
- Auto-mapping uses OpenAI embeddings for intelligent field matching

