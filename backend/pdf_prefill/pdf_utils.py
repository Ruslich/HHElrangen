"""
Low-level PDF utilities for filling AcroForm-based templates.

The implementation is adapted from the standalone PDF pre-fill prototype and
focuses on robustly filling form fields, handling date normalization, and
fixing font sizes to avoid oversized text in rendered PDFs.
"""

from __future__ import annotations

import datetime as dt
import io
import logging
from pathlib import Path
from typing import Dict, Optional

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def fill_pdf_template(
    template_path: Path,
    field_mapping: Dict[str, str],
    data: Dict[str, object],
    field_constraints: Optional[Dict[str, Dict[str, int]]] = None,
) -> Optional[bytes]:
    """
    Fill an existing PDF template with form fields.

    Args:
        template_path: Path to the PDF template file.
        field_mapping: Mapping of PDF field name -> key in `data`.
        data: Data available for filling (e.g. patient record).
        field_constraints: Optional dict of field-specific constraints such as
            {"FieldName": {"max_length": 120}} to prevent overflow.

    Returns:
        Bytes of the generated PDF, or None if no fields were filled.
    """

    try:
        reader = PdfReader(str(template_path), strict=False)
        form_fields = reader.get_form_text_fields()
        if not form_fields:
            logger.warning("PDF %s has no fillable form fields", template_path.name)
            return None

        fill_data: Dict[str, str] = {}
        field_constraints = field_constraints or {}

        print(f"[PDF_UTILS] Processing {len(field_mapping)} field mappings for template {template_path.name}")
        print(f"[PDF_UTILS] PDF has {len(form_fields)} total form fields")
        print(f"[PDF_UTILS] Data keys provided: {list(data.keys())}")
        
        for pdf_field, data_key in field_mapping.items():
            if pdf_field not in form_fields:
                print(f"[PDF_UTILS] ❌ Field '{pdf_field}' not in template (available: {list(form_fields.keys())[:3]}...)")
                continue

            raw_value = data.get(data_key, "")
            print(f"[PDF_UTILS] Mapping '{pdf_field}' ← '{data_key}' = '{raw_value}'")
            
            value = _normalize_field_value(pdf_field, data_key, raw_value, data)
            if value is None:
                print(f"[PDF_UTILS] Value normalized to None, skipping")
                continue

            value = _apply_constraints(pdf_field, value, field_constraints)
            if value:
                fill_data[pdf_field] = value
                print(f"[PDF_UTILS] ✅ Will fill '{pdf_field}' = '{value[:30]}'")

        print(f"[PDF_UTILS] ======> Matched {len(fill_data)} fields out of {len(field_mapping)} mappings")
        
        if not fill_data:
            print(f"[PDF_UTILS] ❌ ERROR: No fields matched!")
            print(f"[PDF_UTILS] Available PDF fields: {list(form_fields.keys())[:15]}")
            return None

        try:
            writer = PdfWriter(clone_from=reader)
        except (TypeError, AttributeError):
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)

        try:
            writer.update_page_form_field_values(writer.pages[0], fill_data)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to fill form fields: %s", exc, exc_info=True)
            raise

        buffer = io.BytesIO()
        writer.write(buffer)
        buffer.seek(0)
        result = buffer.getvalue()

        fixed = _try_fix_font_sizes(result)
        if fixed is not None:
            result = fixed

        logger.info("Filled template %s (%d fields)", template_path.name, len(fill_data))
        return result

    except Exception as exc:  # pragma: no cover
        logger.error("Error filling PDF template %s: %s", template_path, exc, exc_info=True)
        return None


def _normalize_field_value(pdf_field: str, data_key: str, raw_value, data: Dict[str, object]) -> Optional[str]:
    """Apply domain-specific conversions for dates and numeric fields."""
    value = raw_value

    is_date_field = any(keyword in pdf_field.lower() for keyword in ["datum", "date", "birth", "geburts", "unfall", "tag"])

    if data_key == "Age" and is_date_field and value:
        try:
            age = int(value)
            if 0 < age < 120:
                birth_year = dt.date.today().year - age
                value = f"{birth_year}-01-01"
        except (ValueError, TypeError):
            pass

    if data_key == "Admission Type" and is_date_field:
        admission_date = data.get("Date of Admission") or data.get("date_of_admission")
        if admission_date:
            value = admission_date

    if data_key in ("dob", "date_of_birth") and value:
        value = _format_iso_date(value)

    if is_date_field and isinstance(value, str):
        value = _format_iso_date(value)

    if value is None:
        return None

    if isinstance(value, list):
        value = " ".join(str(v) for v in value if v is not None)

    value_str = str(value).strip()
    return value_str or None


def _format_iso_date(value: str) -> str:
    """Convert YYYY-MM-DD style dates to DD.MM.YYYY for German forms."""
    try:
        if "-" in value:
            parts = value.split("-")
            if len(parts) == 3 and len(parts[0]) == 4:
                return f"{parts[2]}.{parts[1]}.{parts[0]}"
    except Exception:  # pragma: no cover
        pass
    return value


def _apply_constraints(pdf_field: str, value: str, constraints: Dict[str, Dict[str, int]]) -> str:
    config = constraints.get(pdf_field, {})
    max_length = config.get("max_length")
    if max_length and len(value) > max_length:
        truncated = value[: max_length - 3]
        last_space = truncated.rfind(" ")
        if last_space > max_length * 0.8:
            truncated = truncated[:last_space]
        value = truncated + "..."
    return value


def _try_fix_font_sizes(pdf_bytes: bytes) -> Optional[bytes]:
    """
    Use PyMuPDF (if available) to reset text field font sizes to a reasonable value.
    Returns new bytes or None if the operation failed.
    """
    try:
        import fitz  # type: ignore
    except Exception:
        logger.debug("PyMuPDF not available; skipping font size fix.")
        return None

    try:
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        updated = False
        for page in pdf_doc:
            for widget in page.widgets():
                if widget.field_type_string == "text":
                    widget.text_fontsize = 11
                    widget.update()
                    updated = True
        if updated:
            result = pdf_doc.tobytes()
        else:
            result = None
        pdf_doc.close()
        return result
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to adjust font sizes: %s", exc, exc_info=True)
        return None

