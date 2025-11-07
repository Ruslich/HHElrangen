"""
User and mapping storage helpers for the PDF pre-fill service.

Adapted from the standalone PDF pre-fill prototype. Persists per-user
configuration (mappings, settings) under `user_data/` inside the package
directory by default, or a configurable base path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


logger = logging.getLogger(__name__)


@dataclass
class UserSettings:
    """User settings and preferences for PDF pre-fill."""

    user_id: str
    pdf_prefill_enabled: bool = False
    database_connected: bool = False
    database_info: Optional[Dict] = None  # {filename, row_count, columns, last_updated}
    last_mapping_setup: Optional[str] = None  # ISO datetime
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class UserManager:
    """Handles per-user directories, settings, and mappings."""

    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.user_data_dir = self.base_dir / "user_data"
        self.user_data_dir.mkdir(parents=True, exist_ok=True)

    def get_user_dir(self, user_id: str) -> Path:
        user_dir = self.user_data_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def get_mappings_dir(self, user_id: str) -> Path:
        mappings_dir = self.get_user_dir(user_id) / "mappings"
        mappings_dir.mkdir(parents=True, exist_ok=True)
        return mappings_dir

    def load_user_settings(self, user_id: str) -> UserSettings:
        settings_file = self.get_user_dir(user_id) / "settings.json"
        if settings_file.exists():
            try:
                with settings_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    return UserSettings(**data)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.error("Error loading user settings for %s: %s", user_id, exc)
        return UserSettings(user_id=user_id)

    def save_user_settings(self, settings: UserSettings) -> None:
        settings_file = self.get_user_dir(settings.user_id) / "settings.json"
        try:
            with settings_file.open("w", encoding="utf-8") as f:
                json.dump(asdict(settings), f, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.error("Error saving user settings for %s: %s", settings.user_id, exc)

    def update_database_info(self, user_id: str, filename: str, row_count: int, columns: list) -> None:
        settings = self.load_user_settings(user_id)
        settings.database_connected = True
        settings.database_info = {
            "filename": filename,
            "row_count": row_count,
            "columns": columns,
            "last_updated": datetime.now().isoformat(),
        }
        self.save_user_settings(settings)

    def enable_pdf_prefill(self, user_id: str) -> None:
        settings = self.load_user_settings(user_id)
        settings.pdf_prefill_enabled = True
        self.save_user_settings(settings)

    def disable_pdf_prefill(self, user_id: str) -> None:
        settings = self.load_user_settings(user_id)
        settings.pdf_prefill_enabled = False
        self.save_user_settings(settings)

    def mark_mapping_setup_complete(self, user_id: str) -> None:
        settings = self.load_user_settings(user_id)
        settings.last_mapping_setup = datetime.now().isoformat()
        self.save_user_settings(settings)

    def save_mapping(self, user_id: str, template_name: str, mapping_config: Dict) -> None:
        mapping_file = self.get_mappings_dir(user_id) / f"{template_name}.json"
        try:
            with mapping_file.open("w", encoding="utf-8") as f:
                json.dump(mapping_config, f, indent=2)
        except Exception as exc:  # pragma: no cover
            logger.error("Error saving mapping for %s/%s: %s", user_id, template_name, exc)

    def load_mapping(self, user_id: str, template_name: str) -> Optional[Dict]:
        mapping_file = self.get_mappings_dir(user_id) / f"{template_name}.json"
        if mapping_file.exists():
            try:
                with mapping_file.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:  # pragma: no cover
                logger.error("Error loading mapping for %s/%s: %s", user_id, template_name, exc)
        return None

    def list_user_mappings(self, user_id: str) -> Dict[str, Dict]:
        mappings: Dict[str, Dict] = {}
        mappings_dir = self.get_mappings_dir(user_id)
        for mapping_file in mappings_dir.glob("*.json"):
            try:
                with mapping_file.open("r", encoding="utf-8") as f:
                    mappings[mapping_file.stem] = json.load(f)
            except Exception as exc:  # pragma: no cover
                logger.error("Error listing mapping %s for %s: %s", mapping_file.name, user_id, exc)
        return mappings

    def get_database_columns(self, user_id: str) -> Optional[list]:
        settings = self.load_user_settings(user_id)
        if settings.database_info:
            return settings.database_info.get("columns", [])
        return None

    def delete_mapping(self, user_id: str, template_name: str) -> bool:
        mapping_file = self.get_mappings_dir(user_id) / f"{template_name}.json"
        if mapping_file.exists():
            try:
                mapping_file.unlink()
                return True
            except Exception as exc:  # pragma: no cover
                logger.error("Error deleting mapping for %s/%s: %s", user_id, template_name, exc)
        return False

    def delete_all_mappings(self, user_id: str) -> int:
        mappings_dir = self.get_mappings_dir(user_id)
        if not mappings_dir.exists():
            return 0
        deleted = 0
        for mapping_file in mappings_dir.glob("*.json"):
            try:
                mapping_file.unlink()
                deleted += 1
            except Exception as exc:  # pragma: no cover
                logger.error("Error deleting mapping %s for %s: %s", mapping_file, user_id, exc)
        return deleted

