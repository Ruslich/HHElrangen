# PDF Templates Directory

Place your empty PDF form templates here.

## Supported Templates:

### 1. verlaufsbericht.pdf
Progress report template with fillable form fields.

**Required form fields:**
- `patient_first_name` - Patient's first name
- `patient_last_name` - Patient's last name  
- `patient_full_name` - Full name
- `date_of_birth` - Birth date
- `ward` - Hospital ward/station
- `diagnoses` - Medical diagnoses
- `medications` - Current medications
- `current_date` - Report generation date
- `verlauf_text` - Progress description (AI-generated)
- `plan_text` - Treatment plan (AI-generated)
- `glucose_summary` - Glucose data summary

## How to Create Form Fields in PDF:

### Using Adobe Acrobat:
1. Open your PDF template
2. Go to Tools → Prepare Form
3. Add text fields with the exact names above
4. Save the PDF

### Using LibreOffice:
1. Create document in Writer
2. Insert → Form → Text Box
3. Right-click → Control Properties → set Name
4. Export as PDF with "Create PDF form" checked

## For Testing:
If you don't have a template yet, the system will generate a simple PDF for you.