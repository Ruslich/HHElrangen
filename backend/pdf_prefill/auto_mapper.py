"""
Automatic Field Mapping Engine

Intelligently matches CSV columns to PDF form fields using:
1. Exact/synonym matching
2. Semantic similarity (AI)
3. Pattern matching
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from openai import OpenAI
import os
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class FieldMapping:
    """Represents a mapping between CSV column and PDF field"""
    pdf_field: str
    csv_column: str
    confidence: float  # 0-100
    method: str  # "exact", "synonym", "semantic", "pattern", "manual"
    description: Optional[str] = None

class AutomaticFieldMapper:
    """Automatically maps CSV columns to PDF form fields"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        
        # Medical field synonyms (German ↔ English)
        self.synonyms = {
            # Names
            "first_name": ["vorname", "firstname", "given name", "vorname der person"],
            "last_name": ["name", "nachname", "lastname", "family name", "surname", 
                         "name der versicherten person", "name der person"],
            "full_name": ["name", "vollständiger name", "complete name"],
            
            # Dates
            "dob": ["geburtsdatum", "date of birth", "birth date", "geburtstag", 
                   "date of birth", "geb.", "geburtsdatum der person"],
            "date_of_admission": ["eingetroffen am", "admission date", "aufnahmedatum",
                                 "date of admission", "aufnahme", "unfalltag", "1 unfalltag"],
            
            # Medical
            "diagnoses": ["diagnose", "medical condition", "diagnosis", "erkrankung",
                         "krankheit", "diagnosen", "condition"],
            "medications": ["medikation", "medication", "medikamente", "drugs",
                          "arzneimittel", "meds"],
            "gender": ["geschlecht", "sex", "gender", "geschlecht der person"],
            
            # Insurance
            "insurance_provider": ["krankenkasse", "insurance", "insurance provider",
                                 "versicherung", "versicherungsträger", "kasse"],
            
            # Address
            "full_address": ["vollständige anschrift", "address", "adresse", 
                           "complete address", "anschrift", "wohnadresse"],
            "street": ["straße", "street", "strasse", "str."],
            "city": ["stadt", "city", "ort", "wohnort"],
            "zip_code": ["postleitzahl", "zip code", "plz", "postal code", "zip"],
            
            # Hospital
            "ward": ["station", "ward", "abteilung", "department", "hospital",
                    "krankenhaus", "hospital name"],
            "room_number": ["zimmer", "room", "room number", "zimmernummer"],
            "doctor": ["arzt", "doctor", "doktor", "attending physician", "behandler"],
            
            # Other
            "age": ["alter", "age", "years old"],
            "blood_type": ["blutgruppe", "blood type", "blood group"],
            "admission_type": ["aufnahmetyp", "admission type", "type of admission"],
        }
        
        # Pattern matchers
        self.patterns = {
            "date": [
                r".*date.*",
                r".*datum.*",
                r".*tag.*",
                r"dob",
                r"birth",
                r"admission",
                r"discharge",
                r"unfalltag",
                r"eingetroffen"
            ],
            "name": [
                r".*name.*",
                r".*vorname.*",
                r".*nachname.*",
                r".*person.*"
            ],
            "address": [
                r".*address.*",
                r".*anschrift.*",
                r".*adresse.*",
                r".*street.*",
                r".*straße.*",
                r".*city.*",
                r".*stadt.*",
                r".*zip.*",
                r".*plz.*"
            ],
            "medical": [
                r".*diagnos.*",
                r".*condition.*",
                r".*medication.*",
                r".*medikation.*",
                r".*krankheit.*",
                r".*erkrankung.*"
            ],
            "insurance": [
                r".*insurance.*",
                r".*krankenkasse.*",
                r".*versicherung.*",
                r".*kasse.*"
            ]
        }
    
    def normalize_string(self, s: str) -> str:
        """Normalize string for comparison"""
        if not s:
            return ""
        # Lowercase, remove extra spaces, remove special chars
        s = s.lower().strip()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', '_', s)
        return s
    
    def exact_match(self, csv_col: str, pdf_field: str) -> Optional[float]:
        """Check for exact match (after normalization)"""
        csv_norm = self.normalize_string(csv_col)
        pdf_norm = self.normalize_string(pdf_field)
        
        if csv_norm == pdf_norm:
            return 100.0
        return None
    
    def synonym_match(self, csv_col: str, pdf_field: str) -> Optional[float]:
        """Check for synonym match"""
        csv_norm = self.normalize_string(csv_col)
        pdf_norm = self.normalize_string(pdf_field)
        
        # Check if csv_col matches any synonym that also matches pdf_field
        for standard_field, synonyms in self.synonyms.items():
            csv_matches = any(csv_norm == self.normalize_string(syn) for syn in synonyms)
            pdf_matches = any(pdf_norm == self.normalize_string(syn) for syn in synonyms)
            
            if csv_matches and pdf_matches:
                return 95.0  # High confidence for synonym match
        
        # Direct synonym check
        all_synonyms = []
        for synonyms in self.synonyms.values():
            all_synonyms.extend(synonyms)
        
        csv_in_synonyms = any(csv_norm == self.normalize_string(s) for s in all_synonyms)
        pdf_in_synonyms = any(pdf_norm == self.normalize_string(s) for s in all_synonyms)
        
        if csv_in_synonyms and pdf_in_synonyms:
            # Check if they're related (same category)
            for standard_field, synonyms in self.synonyms.items():
                if csv_norm in [self.normalize_string(s) for s in synonyms]:
                    if pdf_norm in [self.normalize_string(s) for s in synonyms]:
                        return 90.0
        
        return None
    
    def pattern_match(self, csv_col: str, pdf_field: str) -> Optional[Tuple[float, str]]:
        """Check for pattern-based match"""
        csv_norm = self.normalize_string(csv_col)
        pdf_norm = self.normalize_string(pdf_field)
        
        for pattern_type, patterns in self.patterns.items():
            csv_match = any(re.match(pattern, csv_norm, re.IGNORECASE) for pattern in patterns)
            pdf_match = any(re.match(pattern, pdf_norm, re.IGNORECASE) for pattern in patterns)
            
            if csv_match and pdf_match:
                confidence = 70.0 if pattern_type in ["date", "name"] else 60.0
                return (confidence, pattern_type)
        
        return None
    
    def batch_semantic_match(self, csv_columns: List[str], pdf_fields: List[str]) -> Dict[tuple, float]:
        """
        Batch semantic matching - get embeddings for all fields at once, then compute similarities.
        Handles large batches by chunking if needed (OpenAI limit is ~2048 items per batch).
        Returns: Dict mapping (csv_col, pdf_field) tuples to confidence scores
        """
        if not self.openai_client:
            return {}
        
        try:
            all_texts = csv_columns + pdf_fields
            logger.debug(f"Getting embeddings for {len(all_texts)} fields in batch...")
            
            # OpenAI embeddings API can handle up to 2048 items per request
            # If we have more, we need to chunk (but for our use case, this is unlikely)
            max_batch_size = 2048
            csv_embeddings = []
            pdf_embeddings = []
            
            if len(all_texts) <= max_batch_size:
                # Single batch - fast path
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=all_texts
                )
                
                # Split embeddings: first len(csv_columns) are CSV, rest are PDF fields
                csv_embeddings = [np.array(item.embedding) for item in response.data[:len(csv_columns)]]
                pdf_embeddings = [np.array(item.embedding) for item in response.data[len(csv_columns):]]
            else:
                # Chunk if needed (shouldn't happen for normal templates, but handle it)
                logger.warning(f"Large batch ({len(all_texts)} items), chunking...")
                # Get CSV embeddings
                for i in range(0, len(csv_columns), max_batch_size):
                    chunk = csv_columns[i:i+max_batch_size]
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    csv_embeddings.extend([np.array(item.embedding) for item in response.data])
                
                # Get PDF field embeddings
                for i in range(0, len(pdf_fields), max_batch_size):
                    chunk = pdf_fields[i:i+max_batch_size]
                    response = self.openai_client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    pdf_embeddings.extend([np.array(item.embedding) for item in response.data])
            
            # Compute all pairwise similarities
            similarities = {}
            for i, csv_emb in enumerate(csv_embeddings):
                for j, pdf_emb in enumerate(pdf_embeddings):
                    # Cosine similarity
                    dot_product = np.dot(csv_emb, pdf_emb)
                    norm1 = np.linalg.norm(csv_emb)
                    norm2 = np.linalg.norm(pdf_emb)
                    
                    if norm1 > 0 and norm2 > 0:
                        similarity = dot_product / (norm1 * norm2)
                        confidence = (similarity + 1) * 50  # Convert to 0-100 scale
                        
                        if confidence > 60:  # Only store if above threshold
                            similarities[(csv_columns[i], pdf_fields[j])] = confidence
            
            logger.debug(f"Computed {len(similarities)} semantic matches from batch (1 API call instead of {len(csv_columns) * len(pdf_fields)})")
            return similarities
            
        except Exception as e:
            logger.error(f"Batch semantic matching failed: {e}")
            return {}
    
    def semantic_match(self, csv_col: str, pdf_field: str) -> Optional[float]:
        """Use AI to check semantic similarity (single match - for backward compatibility)"""
        if not self.openai_client:
            return None
        
        try:
            # Create embeddings for both
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[csv_col, pdf_field]
            )
            
            # Calculate cosine similarity using numpy
            embedding1 = np.array(response.data[0].embedding)
            embedding2 = np.array(response.data[1].embedding)
            
            # Cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return None
            
            similarity = dot_product / (norm1 * norm2)
            
            # Convert to 0-100 scale
            confidence = (similarity + 1) * 50  # similarity is -1 to 1, convert to 0-100
            
            # Only return if confidence is reasonable
            if confidence > 60:
                return confidence
            
        except Exception as e:
            logger.debug(f"Semantic matching failed: {e}")
        
        return None
    
    def find_best_mapping(
        self, 
        csv_col: str, 
        pdf_fields: List[str], 
        semantic_cache: Dict[tuple, float] = None,
        field_labels: Dict[str, str] = None,
        field_display_names: Dict[str, str] = None
    ) -> Optional[FieldMapping]:
        """
        Find best mapping for a CSV column across all PDF fields.
        Uses visible field labels for matching when available.
        """
        best_mapping = None
        best_confidence = 0
        semantic_cache = semantic_cache or {}
        field_labels = field_labels or {}
        field_display_names = field_display_names or {}
        
        # First, try quick matches (exact, synonym) - these are instant
        for pdf_field in pdf_fields:
            # Get display name (visible label) for matching
            display_name = field_display_names.get(pdf_field, field_labels.get(pdf_field, pdf_field))
            
            # Try exact match first (on both field name and display name)
            confidence = self.exact_match(csv_col, pdf_field)
            method = "exact"
            
            if not confidence:
                confidence = self.exact_match(csv_col, display_name)
                if confidence:
                    method = "exact (label)"
            
            # Try synonym match (on both)
            if not confidence:
                confidence = self.synonym_match(csv_col, pdf_field)
                method = "synonym"
            
            if not confidence:
                confidence = self.synonym_match(csv_col, display_name)
                if confidence:
                    method = "synonym (label)"
            
            # Try pattern match (also instant)
            if not confidence:
                result = self.pattern_match(csv_col, pdf_field)
                if result:
                    confidence, pattern_type = result
                    method = f"pattern:{pattern_type}"
            
            # Early exit if we found a very good match (>= 90% confidence)
            if confidence and confidence >= 90:
                return FieldMapping(
                    pdf_field=pdf_field,
                    csv_column=csv_col,
                    confidence=confidence,
                    method=method,
                    description=f"Auto-mapped using {method}"
                )
            
            # Try semantic match from cache (instant lookup)
            # Cache uses field names as keys
            if not confidence or confidence < 70:
                semantic_key = (csv_col, pdf_field)
                if semantic_key in semantic_cache:
                    semantic_conf = semantic_cache[semantic_key]
                    if semantic_conf and (not confidence or semantic_conf > confidence):
                        confidence = semantic_conf
                        method = "semantic"
            
            # Update best match
            if confidence and confidence > best_confidence:
                best_confidence = confidence
                best_mapping = FieldMapping(
                    pdf_field=pdf_field,
                    csv_column=csv_col,
                    confidence=confidence,
                    method=method,
                    description=f"Auto-mapped using {method}"
                )
        
        return best_mapping if best_confidence >= 50 else None
    
    def auto_map_template(
        self, 
        csv_columns: List[str], 
        pdf_fields: List[str],
        existing_mapping: Optional[Dict] = None,
        field_labels: Optional[Dict[str, str]] = None
    ) -> Dict[str, FieldMapping]:
        """
        Automatically map CSV columns to PDF fields for a template.
        Uses batch semantic matching for speed.
        Uses visible field labels for better matching quality.
        
        Args:
            csv_columns: List of CSV column names
            pdf_fields: List of PDF field names (AcroForm names)
            existing_mapping: Existing mappings to preserve
            field_labels: Dict mapping PDF field names to visible labels
        
        Returns: Dict mapping PDF field names to FieldMapping objects
        """
        field_labels = field_labels or {}
        logger.info(f"Auto-mapping {len(csv_columns)} CSV columns to {len(pdf_fields)} PDF fields")
        if field_labels:
            logger.debug(f"Using {len(field_labels)} field labels for improved matching")
        
        existing_mapping = existing_mapping or {}
        results = {}
        
        # Create mapping: field_name -> display_name (for matching)
        # Use visible label if available, otherwise use field name
        field_display_names = {
            field: field_labels.get(field, field) 
            for field in pdf_fields
        }
        display_names_list = list(field_display_names.values())
        
        # Get batch semantic similarities using DISPLAY NAMES (visible labels)
        # This dramatically improves matching quality!
        logger.debug("Computing batch semantic similarities using field labels...")
        semantic_cache = self.batch_semantic_match(csv_columns, display_names_list)
        logger.debug(f"Got {len(semantic_cache)} semantic matches from batch API call")
        
        # Map back: (csv_col, display_name) -> (csv_col, field_name)
        # Create reverse lookup: display_name -> field_name
        display_to_field = {label: field for field, label in field_display_names.items()}
        
        # Convert semantic_cache to use field names
        field_semantic_cache = {}
        for (csv_col, display_name), confidence in semantic_cache.items():
            field_name = display_to_field.get(display_name)
            if field_name:
                field_semantic_cache[(csv_col, field_name)] = confidence
        
        # Try to map each CSV column
        used_pdf_fields = set(existing_mapping.values())
        
        for csv_col in csv_columns:
            # Skip if already mapped
            if csv_col in existing_mapping.values():
                continue
            
            # Find best match (using semantic cache with field names)
            available_fields = [f for f in pdf_fields if f not in used_pdf_fields]
            mapping = self.find_best_mapping(
                csv_col, 
                available_fields, 
                field_semantic_cache,
                field_labels=field_labels,
                field_display_names=field_display_names
            )
            
            if mapping and mapping.confidence >= 50:
                results[mapping.pdf_field] = mapping
                used_pdf_fields.add(mapping.pdf_field)
                display_name = field_labels.get(mapping.pdf_field, mapping.pdf_field)
                logger.debug(f"Mapped '{csv_col}' → '{display_name}' ({mapping.pdf_field}) "
                           f"(confidence: {mapping.confidence:.1f}%, method: {mapping.method})")
        
        logger.info(f"Auto-mapped {len(results)} fields (confidence >= 50%)")
        return results
    
    def generate_mapping_json(
        self,
        template_name: str,
        csv_columns: List[str],
        pdf_fields: List[str],
        existing_config: Optional[Dict] = None,
        field_labels: Optional[Dict[str, str]] = None
    ) -> Dict:
        """Generate complete mapping JSON configuration"""
        existing_config = existing_config or {}
        existing_field_mapping = existing_config.get("field_mapping", {})
        field_labels = field_labels or existing_config.get("field_labels", {})
        
        # Get auto-mappings (using field labels for better matching)
        auto_mappings = self.auto_map_template(
            csv_columns, 
            pdf_fields, 
            existing_field_mapping,
            field_labels=field_labels
        )
        
        # Build field_mapping dict
        field_mapping = dict(existing_field_mapping)  # Keep existing manual mappings
        
        for pdf_field, mapping in auto_mappings.items():
            field_mapping[pdf_field] = mapping.csv_column
        
        # Build result
        result = {
            "template_file": existing_config.get("template_file", ""),
            "description": existing_config.get("description", f"Auto-generated mapping for {template_name}"),
            "field_mapping": field_mapping,
            "auto_mapped_fields": {
                pdf_field: {
                    "csv_column": mapping.csv_column,
                    "confidence": mapping.confidence,
                    "method": mapping.method
                }
                for pdf_field, mapping in auto_mappings.items()
            },
            "keywords": existing_config.get("keywords", []),
            "field_constraints": existing_config.get("field_constraints", {})
        }
        
        return result

