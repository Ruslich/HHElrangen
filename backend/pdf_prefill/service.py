"""
High-level service that exposes PDF pre-fill capabilities to the FastAPI layer.

Responsibilities
----------------
* manage template metadata (global + per-user overrides)
* auto-map CSV columns onto PDF form fields
* fill templates with patient data and optionally persist the result
* keep a small in-memory cache for generated PDFs
"""

from __future__ import annotations

import datetime as dt
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3

from .auto_mapper import AutomaticFieldMapper
from .pdf_utils import fill_pdf_template
from .template_scanner import TemplateScanner
from .user_manager import UserManager


class PDFPrefillError(RuntimeError):
    """Domain-specific exception for service errors."""


class PDFPrefillService:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        openai_api_key: Optional[str] = None,
        s3_client=None,
    ):
        self.base_dir = Path(
            base_dir
            or os.getenv("PDF_PREFILL_BASE_DIR")
            or Path(__file__).resolve().parent
        )
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.templates_dir = self.base_dir / "pdf_templates"
        self.field_mappings_dir = self.base_dir / "field_mappings"
        self.generated_dir = self.base_dir / "generated"
        self.generated_dir.mkdir(parents=True, exist_ok=True)

        self.user_manager = UserManager(self.base_dir)
        self.template_scanner = TemplateScanner(self.templates_dir)
        self.field_mapper = AutomaticFieldMapper(openai_api_key or os.getenv("OPENAI_API_KEY"))

        self.report_templates: Dict[str, Dict] = {}
        self._pdf_cache: Dict[str, Dict] = {}

        self.s3_bucket = os.getenv("PDF_PREFILL_S3_BUCKET")
        self.s3_prefix = os.getenv("PDF_PREFILL_S3_PREFIX", "pdf-prefill/")
        self.s3 = s3_client
        if self.s3_bucket and self.s3 is None:
            self.s3 = boto3.client("s3")

        self.load_report_templates()

    # ------------------------------------------------------------------
    # Template + mapping management
    # ------------------------------------------------------------------
    def load_report_templates(self) -> None:
        self.report_templates.clear()
        if not self.field_mappings_dir.exists():
            return
        for mapping_file in self.field_mappings_dir.glob("*.json"):
            try:
                with mapping_file.open("r", encoding="utf-8") as f:
                    config = json.load(f)
                template_name = mapping_file.stem
                self.report_templates[template_name] = config
            except Exception:  # pragma: no cover - ignore malformed file
                continue

    def list_templates(self, refresh: bool = False) -> List[Dict]:
        if refresh:
            self.load_report_templates()
        results = []
        for name, config in self.report_templates.items():
            results.append(
                {
                    "name": name,
                    "description": config.get("description"),
                    "keywords": config.get("keywords", []),
                    "field_count": len(config.get("field_mapping", {})),
                    "template_file": config.get("template_file"),
                }
            )
        return results

    def get_template_config(self, template_name: str, user_id: Optional[str] = None) -> Optional[Dict]:
        # User override takes priority
        if user_id:
            user_mapping = self.user_manager.load_mapping(user_id, template_name)
            if user_mapping:
                merged = dict(user_mapping)
                merged.setdefault("name", template_name)
                return merged
            if not template_name.endswith("_template"):
                alt = f"{template_name}_template"
                user_mapping = self.user_manager.load_mapping(user_id, alt)
                if user_mapping:
                    merged = dict(user_mapping)
                    merged.setdefault("name", template_name)
                    return merged

        # Try exact match first
        config = self.report_templates.get(template_name)
        if config:
            merged = dict(config)
            merged.setdefault("name", template_name)
            return merged
        
        # Try without _template suffix (e.g., Durchgangsarztbericht_template -> durchgangsarztbericht)
        if template_name.endswith("_template"):
            base_name = template_name.replace("_template", "").lower()
            config = self.report_templates.get(base_name)
            if config:
                merged = dict(config)
                merged.setdefault("name", template_name)
                return merged
        
        # Return empty config to allow PDF generation even without mapping
        return {"name": template_name, "field_mapping": {}, "field_labels": {}}

    def list_user_mappings(self, user_id: str) -> Dict[str, Dict]:
        return self.user_manager.list_user_mappings(user_id)

    def save_mapping(self, user_id: str, template_name: str, mapping: Dict) -> None:
        self.user_manager.save_mapping(user_id, template_name, mapping)

    # ------------------------------------------------------------------
    # Auto-mapping + scanning
    # ------------------------------------------------------------------
    def scan_template(self, template_name: str) -> Dict:
        pdf_path = self._resolve_template_path(template_name)
        return self.template_scanner.scan_template(pdf_path)

    def auto_map(
        self,
        template_name: str,
        csv_columns: List[str],
        user_id: Optional[str] = None,
    ) -> Dict:
        template_config = self.get_template_config(template_name, user_id=user_id)
        if not template_config:
            raise PDFPrefillError(f"Unknown template '{template_name}'")

        pdf_path = self._resolve_template_path(template_name, template_config)
        scan = self.template_scanner.scan_template(pdf_path)
        if not scan.get("has_fields"):
            raise PDFPrefillError(f"Template '{template_name}' exposes no fillable fields.")

        existing_config = dict(template_config)
        field_labels = scan.get("field_labels", {})
        auto_config = self.field_mapper.generate_mapping_json(
            template_name,
            csv_columns,
            scan.get("form_fields", []),
            existing_config=existing_config,
            field_labels=field_labels,
        )
        auto_config["field_labels"] = field_labels
        auto_config["template_file"] = str(pdf_path.name)
        return auto_config

    # ------------------------------------------------------------------
    # PDF generation / storage
    # ------------------------------------------------------------------
    def generate_pdf(
        self,
        template_name: str,
        data: Dict[str, object],
        user_id: Optional[str] = None,
        mapping_override: Optional[Dict[str, str]] = None,
        persist: bool = True,
    ) -> Dict:
        config = self.get_template_config(template_name, user_id=user_id)
        if not config:
            raise PDFPrefillError(f"Template '{template_name}' not found.")

        mapping = mapping_override or config.get("field_mapping", {})
        
        # If no mapping exists, try to auto-generate one
        if not mapping:
            try:
                pdf_path = self._resolve_template_path(template_name, config)
                scan = self.template_scanner.scan_template(pdf_path)
                if scan.get("has_fields"):
                    csv_columns = list(data.keys())
                    auto_config = self.field_mapper.generate_mapping_json(
                        template_name,
                        csv_columns,
                        scan.get("form_fields", []),
                        existing_config=config,
                        field_labels=scan.get("field_labels", {}),
                    )
                    mapping = auto_config.get("field_mapping", {})
            except Exception:
                pass  # Continue with empty mapping if auto-map fails
        
        if not mapping:
            raise PDFPrefillError(f"Template '{template_name}' has no field mapping. Please configure mappings first or provide mapping_override.")

        pdf_path = self._resolve_template_path(template_name, config)
        field_constraints = config.get("field_constraints", {})
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Generating PDF for template '{template_name}' with {len(mapping)} field mappings")
        logger.info(f"PDF path: {pdf_path}")
        logger.info(f"Data keys: {list(data.keys())[:10]}")
        logger.info(f"Mapping sample: {dict(list(mapping.items())[:5])}")
        
        pdf_bytes = fill_pdf_template(pdf_path, mapping, data, field_constraints=field_constraints)
        if not pdf_bytes:
            logger.error(f"fill_pdf_template returned None for template {template_name}")
            raise PDFPrefillError("Failed to fill PDF template – no output generated.")

        pdf_id = str(uuid.uuid4())
        metadata = {
            "pdf_id": pdf_id,
            "template_name": template_name,
            "generated_at": dt.datetime.utcnow().isoformat() + "Z",
            "filename": f"{template_name}_{data.get('full_name', 'export')}.pdf".replace(" ", "_"),
            "user_id": user_id,
        }

        if persist:
            storage = self._store_pdf(pdf_id, metadata["filename"], pdf_bytes)
            metadata.update(storage)

        self._pdf_cache[pdf_id] = {
            "metadata": metadata,
            "bytes": pdf_bytes,
        }

        return {"metadata": metadata, "bytes": pdf_bytes}

    def get_pdf(self, pdf_id: str) -> Optional[Dict]:
        entry = self._pdf_cache.get(pdf_id)
        if entry:
            return entry

        file_path = self.generated_dir / f"{pdf_id}.pdf"
        if file_path.exists():
            with file_path.open("rb") as f:
                pdf_bytes = f.read()
            metadata_path = file_path.with_suffix(".json")
            metadata = {}
            if metadata_path.exists():
                with metadata_path.open("r", encoding="utf-8") as f:
                    metadata = json.load(f)
            entry = {"metadata": metadata, "bytes": pdf_bytes}
            self._pdf_cache[pdf_id] = entry
            return entry

        if self.s3_bucket:
            key = f"{self.s3_prefix}{pdf_id}.pdf"
            try:
                obj = self.s3.get_object(Bucket=self.s3_bucket, Key=key)
                pdf_bytes = obj["Body"].read()
                entry = {"metadata": {"s3_key": key}, "bytes": pdf_bytes}
                self._pdf_cache[pdf_id] = entry
                return entry
            except Exception:  # pragma: no cover
                return None
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_template_path(self, template_name: str, config: Optional[Dict] = None) -> Path:
        config = config or self.report_templates.get(template_name, {})
        filename = config.get("template_file") or f"{template_name}.pdf"
        
        # Try exact filename first
        candidate = self.templates_dir / filename
        if candidate.exists():
            return candidate

        # Try with _template suffix if not already present
        if not template_name.endswith("_template"):
            alt_filename = f"{template_name}_template.pdf"
            candidate = self.templates_dir / alt_filename
            if candidate.exists():
                return candidate

        # Try a more permissive search (case-insensitive, partial match)
        template_base = template_name.replace("_template", "")
        for pdf_file in self.templates_dir.glob("*.pdf"):
            # Case-insensitive comparison
            if template_base.lower() in pdf_file.stem.lower():
                return pdf_file
        
        raise PDFPrefillError(f"PDF template file for '{template_name}' not found in {self.templates_dir}")

    def _store_pdf(self, pdf_id: str, filename: str, pdf_bytes: bytes) -> Dict:
        storage_meta: Dict[str, str] = {}
        safe_filename = filename or f"{pdf_id}.pdf"

        if self.s3_bucket:
            key = f"{self.s3_prefix}{pdf_id}/{safe_filename}"
            self.s3.put_object(Bucket=self.s3_bucket, Key=key, Body=pdf_bytes, ContentType="application/pdf")
            storage_meta.update({"s3_bucket": self.s3_bucket, "s3_key": key})
        else:
            target = self.generated_dir / f"{pdf_id}.pdf"
            with target.open("wb") as f:
                f.write(pdf_bytes)
            metadata_path = target.with_suffix(".json")
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump({"filename": safe_filename}, f, indent=2)
            storage_meta.update({"file_path": str(target)})
        return storage_meta

