"""
PDF Template Scanner

Scans all PDF templates and extracts form fields with visible labels.
"""

import logging
import re
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# Try to import OpenAI for AI-powered label cleaning
try:
    from openai import OpenAI
    HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
except ImportError:
    HAS_OPENAI = False

# Try to import PyMuPDF for better field label extraction
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available. Field label extraction will be limited. Install with: pip install pymupdf")

class TemplateScanner:
    """Scans PDF templates for form fields"""
    
    def __init__(self, templates_dir: Path):
        self.templates_dir = Path(templates_dir)
    
    def _extract_field_labels_pymupdf(self, pdf_file: Path) -> Dict[str, str]:
        """
        Extract visible field labels using PyMuPDF with intelligent text analysis.
        Uses PDF structure, formatting, and text patterns to identify real labels.
        Returns dict mapping field_name -> visible_label
        """
        if not HAS_PYMUPDF:
            return {}
        
        try:
            doc = fitz.open(str(pdf_file))
            field_labels = {}
            
            # First pass: Extract all text with formatting info for better analysis
            for page_num in range(len(doc)):
                page = doc[page_num]
                widgets = list(page.widgets())
                
                if not widgets:
                    continue
                
                # Get text blocks with formatting (for detecting bold/labels)
                text_dict = page.get_text("dict")
                blocks = text_dict.get("blocks", [])
                
                # Build a map of text positions for smarter label detection
                # Also extract text in reading order for better context
                text_positions = []
                full_page_text_lines = []  # For context analysis
                
                for block in blocks:
                    if "lines" not in block:
                        continue
                    for line in block["lines"]:
                        line_text_parts = []
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            
                            bbox = span["bbox"]  # [x0, y0, x1, y1]
                            font_flags = span.get("flags", 0)
                            is_bold = (font_flags & 16) > 0  # Bit 4 = bold
                            font_size = span.get("size", 0)
                            
                            text_positions.append({
                                "rect": fitz.Rect(bbox),
                                "text": text,
                                "is_bold": is_bold,
                                "font_size": font_size,
                                "y": bbox[1],  # Top of text
                                "y_bottom": bbox[3],  # Bottom of text
                                "x_left": bbox[0],
                                "x_right": bbox[2]
                            })
                            
                            line_text_parts.append(text)
                        
                        if line_text_parts:
                            # Store full line for context (helps identify labels in paragraphs)
                            full_page_text_lines.append({
                                "text": ' '.join(line_text_parts),
                                "y": line.get("spans", [{}])[0].get("bbox", [0, 0, 0, 0])[1] if line.get("spans") else 0
                            })
                
                for widget in widgets:
                    field_name = widget.field_name
                    if not field_name:
                        continue
                    
                    # Get field bounding box
                    field_rect = widget.rect
                    
                    label = None
                    
                    # 1. Try alternate name (TU) - most reliable
                    if widget.field_label and widget.field_label.strip():
                        label = widget.field_label.strip()
                    
                    # 2. Intelligent text extraction near the field
                    if not label:
                        # Strategy: Look for text that's likely a label
                        # - Text above the field (most common)
                        # - Bold text (likely labels)
                        # - Text with specific patterns (numbered sections, etc.)
                        # - Text to the left (less common)
                        
                        candidate_labels = []
                        
                        # Search for text above the field (priority area)
                        search_rect_above = fitz.Rect(
                            field_rect.x0 - 300,  # Wide search area
                            max(0, field_rect.y0 - 80),  # Up to 80 points above
                            field_rect.x1 + 300,
                            field_rect.y0  # Top of field
                        )
                        
                        # Search for text to the left
                        search_rect_left = fitz.Rect(
                            max(0, field_rect.x0 - 250),
                            field_rect.y0 - 15,
                            field_rect.x0,
                            field_rect.y1 + 15
                        )
                        
                        # Strategy: Extract text intelligently
                        # 1. Get ALL text in the search area
                        # 2. Look for the best label candidate
                        # 3. Use formatting cues (bold, font size) to identify labels
                        
                        # Collect all nearby text with context
                        nearby_texts = []
                        
                        for text_info in text_positions:
                            text_rect = text_info["rect"]
                            text = text_info["text"]
                            
                            # Check if text is in search area
                            if (search_rect_above.intersects(text_rect) or 
                                search_rect_left.intersects(text_rect)):
                                distance = abs(text_rect.y1 - field_rect.y0)
                                nearby_texts.append({
                                    "text": text,
                                    "rect": text_rect,
                                    "distance": distance,
                                    "is_bold": text_info["is_bold"],
                                    "font_size": text_info["font_size"],
                                    "y": text_rect.y1,  # Bottom of text (for sorting)
                                    "is_above": search_rect_above.intersects(text_rect)
                                })
                        
                        # Sort by position: prefer text above, closer to field
                        nearby_texts.sort(key=lambda x: (
                            not x["is_above"],  # Text above first
                            x["distance"]  # Then by distance
                        ))
                        
                        # Try to construct label from nearby text
                        # Often labels are split across multiple text blocks
                        # We want to combine them intelligently
                        potential_labels = []
                        
                        # Method 1: Single best text block (for simple cases)
                        for text_info in nearby_texts:
                            clean_text = self._clean_label_text(text_info["text"])
                            if not clean_text or len(clean_text) < 3:
                                continue
                            
                            score = 0
                            
                            # Bold text is very likely a label
                            if text_info["is_bold"]:
                                score += 20
                            
                            # Text that matches label patterns
                            if self._looks_like_label(clean_text):
                                score += 25
                            
                            # Prefer text above
                            if text_info["is_above"]:
                                score += 10
                            
                            # Prefer shorter, focused text
                            if 5 <= len(clean_text) <= 100:
                                score += 5
                            elif len(clean_text) > 150:
                                score -= 15
                            
                            # Prefer closer text
                            if text_info["distance"] < 25:
                                score += 8
                            elif text_info["distance"] < 50:
                                score += 4
                            
                            # Prefer text on same horizontal line (aligned with field)
                            x_overlap = min(text_info["rect"].x1, field_rect.x1) - max(text_info["rect"].x0, field_rect.x0)
                            if x_overlap > 0:
                                score += 5
                            
                            potential_labels.append({
                                "text": clean_text,
                                "score": score,
                                "distance": text_info["distance"],
                                "is_bold": text_info["is_bold"],
                                "original": text_info["text"]
                            })
                        
                        # Method 2: Combine multiple text blocks (for multi-line labels)
                        # Look for text blocks that are vertically aligned and close together
                        if len(nearby_texts) > 1:
                            # Group text blocks by horizontal position
                            text_groups = []
                            for text_info in nearby_texts:
                                if text_info["is_above"]:
                                    # Find or create a group
                                    grouped = False
                                    for group in text_groups:
                                        # Check if text aligns horizontally (within 50px)
                                        if abs(group["x_center"] - (text_info["rect"].x0 + text_info["rect"].x1) / 2) < 50:
                                            group["texts"].append(text_info)
                                            group["y_min"] = min(group["y_min"], text_info["rect"].y0)
                                            group["y_max"] = max(group["y_max"], text_info["rect"].y1)
                                            grouped = True
                                            break
                                    
                                    if not grouped:
                                        x_center = (text_info["rect"].x0 + text_info["rect"].x1) / 2
                                        text_groups.append({
                                            "x_center": x_center,
                                            "texts": [text_info],
                                            "y_min": text_info["rect"].y0,
                                            "y_max": text_info["rect"].y1
                                        })
                            
                            # For each group, combine texts
                            for group in text_groups:
                                # Sort texts by vertical position (top to bottom)
                                group["texts"].sort(key=lambda x: x["rect"].y0)
                                
                                # Combine texts
                                combined_text = ' '.join(t["text"].strip() for t in group["texts"])
                                clean_combined = self._clean_label_text(combined_text)
                                
                                if clean_combined and len(clean_combined) >= 3:
                                    # Score combined text
                                    score = 0
                                    has_bold = any(t["is_bold"] for t in group["texts"])
                                    avg_distance = sum(t["distance"] for t in group["texts"]) / len(group["texts"])
                                    
                                    if has_bold:
                                        score += 15
                                    if self._looks_like_label(clean_combined):
                                        score += 20
                                    if 10 <= len(clean_combined) <= 120:
                                        score += 5
                                    if avg_distance < 40:
                                        score += 5
                                    
                                    potential_labels.append({
                                        "text": clean_combined,
                                        "score": score,
                                        "distance": avg_distance,
                                        "is_bold": has_bold,
                                        "original": combined_text
                                    })
                        
                        # Method 3: If still no good label, try to find it in full page context
                        # Look for text that appears just before the field in reading order
                        if not label and full_page_text_lines:
                            # Find lines near the field's Y position
                            field_y = field_rect.y0
                            nearby_lines = []
                            for line_info in full_page_text_lines:
                                line_y = line_info["y"]
                                # Lines that are above the field (within 100 points)
                                if field_y - 100 <= line_y < field_y:
                                    distance = field_y - line_y
                                    nearby_lines.append({
                                        "text": line_info["text"],
                                        "distance": distance,
                                        "y": line_y
                                    })
                            
                            # Sort by distance (closest first)
                            nearby_lines.sort(key=lambda x: x["distance"])
                            
                            # Try the closest lines as potential labels
                            for line_info in nearby_lines[:3]:  # Check top 3 closest lines
                                clean_line = self._clean_label_text(line_info["text"])
                                if (clean_line and 
                                    len(clean_line) >= 5 and 
                                    len(clean_line) <= 120 and
                                    self._looks_like_label(clean_line) and
                                    not self._is_noise(clean_line)):
                                    label = clean_line
                                    break
                        
                        # Pick the best candidate from all methods
                        if potential_labels and not label:
                            # Sort by score (highest first), then by distance (closest first)
                            potential_labels.sort(key=lambda x: (-x["score"], x["distance"]))
                            best_candidate = potential_labels[0]
                            
                            # Use if score is reasonable
                            if (best_candidate["score"] >= 10 and 
                                not self._is_noise(best_candidate["text"])):
                                label = best_candidate["text"]
                        elif potential_labels and label:
                            # We have a label from Method 3, but check if Method 1/2 has better
                            potential_labels.sort(key=lambda x: (-x["score"], x["distance"]))
                            best_candidate = potential_labels[0]
                            
                            # Override Method 3 if Method 1/2 score is significantly better
                            if (best_candidate["score"] > 30 and 
                                not self._is_noise(best_candidate["text"])):
                                label = best_candidate["text"]
                    
                    # 3. Final cleanup and validation
                    if label:
                        label = self._clean_label_text(label)
                        # Validate: should be meaningful
                        if label and len(label) >= 3 and not self._is_noise(label):
                            field_labels[field_name] = label
            
            doc.close()
            
            # 4. Filter out generic field names that don't have meaningful labels
            filtered_labels = {}
            for field_name, label in field_labels.items():
                # Skip generic field names if label is also generic
                if self._is_generic_field_name(field_name):
                    # Only keep if we found a meaningful label (not generic)
                    if label and not self._is_generic_label(label):
                        filtered_labels[field_name] = label
                    # Skip if label is also generic
                else:
                    # Keep non-generic field names if they have any label
                    if label:
                        filtered_labels[field_name] = label
            
            field_labels = filtered_labels
            
            # 5. AI-powered label refinement (if OpenAI available)
            # This helps identify actual labels from messy extracted text
            if HAS_OPENAI and field_labels:
                field_labels = self._refine_labels_with_ai(pdf_file, field_labels)
            
            # 6. Remove duplicate labels (keep first occurrence, mark others)
            field_labels = self._deduplicate_labels(field_labels)
            
            logger.debug(f"Extracted {len(field_labels)} field labels using intelligent extraction")
            return field_labels
        
        except Exception as e:
            logger.error(f"Error extracting labels with PyMuPDF for {pdf_file.name}: {e}")
            return {}
    
    def _clean_label_text(self, text: str) -> str:
        """Clean extracted text to get a proper label"""
        if not text:
            return ""
        
        # Remove newlines and normalize whitespace
        text = ' '.join(text.split())
        
        # Remove common noise patterns
        # Remove incomplete sentences at the start (common in PDF extraction)
        # Look for patterns like "g g g g" or repeated single characters
        words = text.split()
        if len(words) > 0:
            # Remove leading single-character "words" that are likely noise
            while words and len(words[0]) == 1 and words[0].islower():
                words = words[1:]
        
        text = ' '.join(words)
        
        # Remove trailing incomplete words or noise
        # If text ends with a single lowercase letter, it might be cut off
        if len(text) > 1 and text[-1].islower() and text[-2] == ' ':
            text = text[:-2].rstrip()
        
        # Limit length (labels shouldn't be too long)
        if len(text) > 120:
            # Try to find a good cutoff point (end of sentence, comma, etc.)
            for delimiter in ['.', '?', '!', ';', ',']:
                idx = text.rfind(delimiter, 0, 100)
                if idx > 20:
                    text = text[:idx + 1].strip()
                    break
            else:
                # No good delimiter, just truncate at word boundary
                words = text[:100].split()
                text = ' '.join(words[:-1]) if len(words) > 1 else text[:100]
        
        return text.strip()
    
    def _clean_ai_label(self, label: str) -> str:
        """
        Post-process AI-refined labels to remove section numbers, 
        parenthetical extras, and make them cleaner.
        """
        if not label or label == "NOT_FOUND":
            return label
        
        # Remove leading section numbers (e.g., "1.6 ", "2.3 ", "5.1 ")
        label = re.sub(r'^\d+\.\d+\s+', '', label)
        label = re.sub(r'^\d+\s+', '', label)  # Also handle "1 ", "2 "
        
        # Remove parenthetical explanations that make labels too long
        # Match patterns like "(inklusive Medikation, ggf. weitere Diagnostik)"
        # But keep short, essential ones like "(optional)" or "(Ja/Nein)"
        label = re.sub(r'\([^)]{30,}\)', '', label)  # Remove long parentheticals (>30 chars)
        
        # Remove common verbose suffixes
        label = re.sub(r'\s*\(.*etc[.)]?.*\)', '', label, flags=re.IGNORECASE)
        label = re.sub(r'\s*\(.*inklusive.*\)', '', label, flags=re.IGNORECASE)
        label = re.sub(r'\s*\(.*ggf\.\s+.*\)', '', label, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        label = ' '.join(label.split())
        
        # Truncate if still too long (max 60 chars for display)
        if len(label) > 60:
            # Try to cut at a natural boundary
            for delimiter in [',', ';', ':', ' ']:
                idx = label.rfind(delimiter, 0, 55)
                if idx > 20:
                    label = label[:idx].strip()
                    break
            else:
                label = label[:57].strip() + "..."
        
        return label.strip()
    
    def _looks_like_label(self, text: str) -> bool:
        """
        Check if text looks like a field label (not noise).
        German medical forms often have numbered sections, specific patterns, etc.
        """
        if not text or len(text) < 3:
            return False
        
        # Remove leading/trailing whitespace for analysis
        text_clean = text.strip()
        
        # Strong indicators of a label
        strong_label_patterns = [
            r'^\d+[\.\)]\s+[A-ZÄÖÜ]',  # Numbered section: "5.1 Beschwerden"
            r'^\d+[\.\)]\s+[a-zäöüß]',  # Numbered section (lowercase start): "5.1 beschwerden"
            r'^[A-ZÄÖÜ][a-zäöüß]+:\s*$',  # Ends with colon: "Name:", "Adresse:"
            r'^[A-ZÄÖÜ][A-ZÄÖÜ\s]+$',  # All caps (common in forms): "NAME", "ADRESSE"
        ]
        
        for pattern in strong_label_patterns:
            if re.match(pattern, text_clean):
                return True
        
        # Check if it's definitely noise (exclude these)
        noise_patterns = [
            r'^[a-z]\s+[a-z]\s+[a-z]',  # Multiple single lowercase letters: "g g g"
            r'^g\s+g\s+',  # Repeated "g" (common extraction artifact)
            r'^[^A-Za-z0-9ÄÖÜäöüß]+$',  # Only symbols (no letters/numbers)
            r'^[a-z]{1,2}\s+[a-z]{1,2}\s+',  # Very short lowercase words repeated
        ]
        
        for pattern in noise_patterns:
            if re.match(pattern, text_clean):
                return False
        
        # Medium confidence: check word structure
        words = text_clean.split()
        if 1 <= len(words) <= 20:  # Labels are usually 1-20 words
            # Count words that look like proper German words
            proper_words = 0
            for word in words:
                # Proper German word: starts with capital (noun), or is a common word
                if len(word) > 2:  # At least 3 chars
                    if word[0].isupper() or word[0].isdigit():
                        proper_words += 1
                    elif word.lower() in ['der', 'die', 'das', 'und', 'von', 'für', 'bei', 'nach', 'am', 'zum']:
                        proper_words += 1
            
            # If most words are proper, it's likely a label
            if proper_words >= max(1, len(words) * 0.4):  # At least 40% proper words
                return True
        
        # Check for German medical form common patterns
        medical_patterns = [
            r'.*[Vv]ersichert.*',  # Contains "versichert"
            r'.*[Uu]nfall.*',  # Contains "Unfall"
            r'.*[Gg]eburt.*',  # Contains "Geburt"
            r'.*[Nn]ame.*',  # Contains "Name"
            r'.*[Aa]dresse.*',  # Contains "Adresse"
            r'.*[Dd]atum.*',  # Contains "Datum"
            r'.*[Kk]rank.*',  # Contains "Krank"
        ]
        
        for pattern in medical_patterns:
            if re.search(pattern, text_clean):
                # If it matches a medical pattern and has reasonable structure
                if 3 <= len(text_clean) <= 100:
                    return True
        
        return False
    
    def _is_noise(self, text: str) -> bool:
        """Check if text is likely noise/irrelevant"""
        if not text:
            return True
        
        # Very short text is probably noise
        if len(text) < 3:
            return True
        
        # Check for common noise patterns
        noise_indicators = [
            text.startswith('g ') and 'g ' in text[:20],  # Repeated "g"
            len(set(text.split()[:5])) == 1 and len(text.split()) > 3,  # All same word
            text.count(' ') > 0 and all(len(w) == 1 for w in text.split()[:5]),  # Single char words
        ]
        
        if any(noise_indicators):
            return True
        
        return False
    
    def _is_generic_field_name(self, field_name: str) -> bool:
        """Check if field name is generic (TextXX, Textfield, etc.)"""
        if not field_name:
            return True
        
        field_lower = field_name.lower()
        generic_patterns = [
            r'^text\d+$',  # Text1, Text11, Text123
            r'^textfield\d*$',  # Textfield, Textfield1, Textfield-0
            r'^field\d+$',  # Field1, Field2
            r'^f\d+$',  # F1, F2
        ]
        
        for pattern in generic_patterns:
            if re.match(pattern, field_lower):
                return True
        
        return False
    
    def _is_generic_label(self, label: str) -> bool:
        """Check if label is generic/unhelpful (same as field name or generic text)"""
        if not label:
            return True
        
        label_lower = label.lower().strip()
        
        # If label is same as a generic field name pattern, it's generic
        if self._is_generic_field_name(label):
            return True
        
        # Very short or single-word labels that don't provide context
        words = label_lower.split()
        if len(words) <= 1 and len(label_lower) < 4:
            return True
        
        # Labels that are just the field name (no improvement)
        # This will be checked by caller
        
        return False
    
    def _deduplicate_labels(self, field_labels: Dict[str, str]) -> Dict[str, str]:
        """Handle duplicate labels by adding context to differentiate them"""
        seen_labels = {}  # Maps normalized label -> list of (field_name, label) tuples
        deduplicated = {}
        
        # First pass: collect all occurrences of each label
        for field_name, label in field_labels.items():
            if not label:
                continue
                
            label_lower = label.lower().strip()
            normalized = ' '.join(label_lower.split())
            
            if normalized not in seen_labels:
                seen_labels[normalized] = []
            seen_labels[normalized].append((field_name, label))
        
        # Second pass: add context to duplicates
        for normalized, occurrences in seen_labels.items():
            if len(occurrences) == 1:
                # No duplicates - use as-is
                field_name, label = occurrences[0]
                deduplicated[field_name] = label
            else:
                # Multiple fields with same label - add context
                for i, (field_name, label) in enumerate(occurrences):
                    if not self._is_generic_field_name(field_name):
                        # Use field name for context
                        deduplicated[field_name] = f"{label} [{field_name}]"
                    elif i == 0:
                        # First occurrence - keep original (might be best one)
                        deduplicated[field_name] = label
                    else:
                        # Generic field name in duplicate - skip it (no value)
                        # This prevents showing multiple "Art der Heilbehandlung" with TextXX fields
                        continue
        
        return deduplicated
    
    def _refine_labels_with_ai(self, pdf_file: Path, extracted_labels: Dict[str, str]) -> Dict[str, str]:
        """
        Use AI to refine extracted labels - identify actual field labels from messy text.
        Uses batch processing: analyzes the whole page at once for efficiency.
        """
        if not HAS_OPENAI:
            return extracted_labels
        
        try:
            # Identify problematic fields that need refinement
            problematic_fields = {}
            field_positions = {}  # Store field positions for context
            
            try:
                doc = fitz.open(str(pdf_file))
                all_page_text = ""
                
                # Collect field positions and page text
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    widgets = list(page.widgets())
                    
                    if widgets:
                        page_text = page.get_text("text")
                        all_page_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                        
                        # Check each field
                        for widget in widgets:
                            field_name = widget.field_name
                            if not field_name:
                                continue
                            
                            current_label = extracted_labels.get(field_name, "")
                            
                            # Fields that need AI refinement
                            is_generic = self._is_generic_field_name(field_name)
                            is_noisy = self._is_noise(current_label) if current_label else True
                            is_too_long = len(current_label) > 60 if current_label else False  # Shorter threshold
                            is_empty_or_bad = not current_label or current_label == field_name
                            # Check if label starts with section number (like "1.6 ", "2.3 ")
                            has_section_number = bool(re.match(r'^\d+\.?\d*\s+', current_label)) if current_label else False
                            # Check if label has verbose parentheticals
                            has_long_parenthetical = bool(re.search(r'\([^)]{30,}\)', current_label)) if current_label else False
                            
                            if is_generic or is_noisy or is_too_long or is_empty_or_bad or has_section_number or has_long_parenthetical:
                                field_rect = widget.rect
                                problematic_fields[field_name] = {
                                    "current_label": current_label,
                                    "page": page_num,
                                    "position": f"x={field_rect.x0:.0f}, y={field_rect.y0:.0f}"
                                }
                
                doc.close()
            except Exception as e:
                logger.debug(f"Error reading PDF for AI refinement: {e}")
                return extracted_labels
            
            if not problematic_fields:
                return extracted_labels
            
            # Batch AI call: analyze all problematic fields at once
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            refined_labels = extracted_labels.copy()
            
            # Create prompt for batch processing
            fields_list = "\n".join([
                f"- {field_name} (Position: {info['position']}, Aktuell: '{info['current_label']}')"
                for field_name, info in list(problematic_fields.items())[:30]  # Limit to 30 fields per call
            ])
            
            # Limit page text to avoid token limits (first 4000 chars should be enough)
            page_text_sample = all_page_text[:4000]
            
            prompt = f"""Du analysierst ein deutsches medizinisches PDF-Formular und identifizierst die TATSÄCHLICHEN, KLAREN Feldlabels.

PDF-TEXT (Auszug):
{page_text_sample}

PROBLEMATISCHE FELDER (benötigen bessere Labels):
{fields_list}

AUFGABE:
Für JEDES Feld finde das KURZE, KLARE Feldlabel ohne überflüssige Zusätze.

WICHTIGE REGELN:
1. Labels sind die BESCHRIFTUNGEN direkt vor/neben dem Eingabefeld
2. ENTFERNE Sektionsnummern wie "1.6", "2.3", "5.1" vom Anfang - diese gehören nicht zum Label!
3. ENTFERNE lange Erklärungen in Klammern wie "(inklusive Medikation, ggf. weitere Diagnostik)"
4. Labels sollen KURZ sein (5-40 Zeichen ideal, max 60 Zeichen)
5. Beispiele für GUTE Labels:
   - "Beschwerden/Klagen" (nicht "5.1 Beschwerden/Klagen (inklusive...)")
   - "Aufnahmebefunde" (nicht "1.6 Aufnahmebefunde (funktionell/Bildgebung/Labor, etc.)")
   - "Therapieempfehlungen" (nicht "1.9 Therapieempfehlungen (inklusive Medikation...)")
6. Ignoriere Rauschen, einzelne Buchstaben, "g g g g", etc.
7. Wenn kein gutes Label gefunden: "NOT_FOUND"

Antworte NUR mit einem JSON-Object:
{{
  "field_name_1": "Kurzes, klares Label ohne Nummern/Zusätze",
  "field_name_2": "NOT_FOUND",
  ...
}}

KEINE weiteren Worte, nur das JSON!
"""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=2000,  # Increased to handle larger responses
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content.strip()
                # Try to extract JSON if response_format didn't work perfectly
                if content.startswith("```json"):
                    content = content.replace("```json", "").replace("```", "").strip()
                elif content.startswith("```"):
                    content = content.replace("```", "").strip()
                
                ai_results = json.loads(content)
                
                # Apply AI-refined labels with post-processing
                for field_name, ai_label in ai_results.items():
                    if field_name in problematic_fields:
                        ai_label = str(ai_label).strip()
                        
                        # Post-process: clean up the label
                        ai_label = self._clean_ai_label(ai_label)
                            
                        # Validate AI response
                        if (ai_label and 
                            ai_label != "NOT_FOUND" and 
                            len(ai_label) >= 3 and 
                            len(ai_label) <= 60 and  # Shorter max length
                            not self._is_noise(ai_label) and
                            not self._is_generic_label(ai_label)):
                            refined_labels[field_name] = ai_label
                            old_label = problematic_fields[field_name]["current_label"]
                            logger.debug(f"AI refined '{field_name}': '{old_label}' → '{ai_label}'")
                
                logger.info(f"AI refined {len([v for v in ai_results.values() if v != 'NOT_FOUND'])} field labels")
                
            except Exception as e:
                logger.debug(f"AI batch refinement failed: {e}")
                # Fallback: try individual calls for critical fields
                pass
            
            return refined_labels
            
        except Exception as e:
            logger.debug(f"AI label refinement error: {e}")
            return extracted_labels
    
    def _extract_field_labels_pypdf(self, pdf_file: Path, field_names: List[str]) -> Dict[str, str]:
        """
        Extract field labels using pypdf (fallback method).
        Tries to get /TU (alternate name) or field descriptions.
        """
        labels = {}
        try:
            reader = PdfReader(str(pdf_file), strict=False)
            
            # Access AcroForm fields
            if '/AcroForm' not in reader.trailer.get('/Root', {}):
                return labels
            
            acro_form = reader.trailer['/Root']['/AcroForm']
            if '/Fields' not in acro_form:
                return labels
            
            def extract_field_info(field_ref, visited=None):
                """Recursively extract field information"""
                if visited is None:
                    visited = set()
                
                field_obj = field_ref.get_object()
                if id(field_obj) in visited:
                    return
                visited.add(id(field_obj))
                
                # Get field name (T)
                field_name = field_obj.get('/T')
                if field_name and isinstance(field_name, str):
                    # Try alternate name (TU) - user-friendly label
                    alt_name = field_obj.get('/TU')
                    if alt_name:
                        labels[field_name] = str(alt_name)
                
                # Handle kids (nested fields)
                if '/Kids' in field_obj:
                    for kid_ref in field_obj['/Kids']:
                        extract_field_info(kid_ref, visited)
            
            # Process all fields
            for field_ref in acro_form['/Fields']:
                extract_field_info(field_ref)
            
        except Exception as e:
            logger.debug(f"Error extracting labels with pypdf for {pdf_file.name}: {e}")
        
        return labels
    
    def scan_template(self, pdf_file: Path) -> Optional[Dict]:
        """
        Scan a single PDF template and extract form fields with visible labels.
        
        Returns: {
            "template_file": "filename.pdf",
            "form_fields": ["field1", "field2", ...],
            "field_labels": {"field1": "Visible Label 1", ...},  # Field name -> visible label
            "has_fields": bool
        }
        """
        try:
            reader = PdfReader(str(pdf_file), strict=False)
            form_fields = reader.get_form_text_fields()
            
            if form_fields is None or len(form_fields) == 0:
                return {
                    "template_file": pdf_file.name,
                    "form_fields": [],
                    "field_labels": {},
                    "has_fields": False,
                    "error": "No form fields found"
                }
            
            field_names = list(form_fields.keys())
            
            # Extract visible labels - prefer PyMuPDF, fallback to pypdf
            field_labels = {}
            if HAS_PYMUPDF:
                field_labels = self._extract_field_labels_pymupdf(pdf_file)
                logger.debug(f"Extracted {len(field_labels)} labels using PyMuPDF")
            
            # Fallback: try pypdf method
            if len(field_labels) < len(field_names) * 0.5:  # If we got less than 50% labels
                pypdf_labels = self._extract_field_labels_pypdf(pdf_file, field_names)
                # Merge, prefer PyMuPDF labels
                for name, label in pypdf_labels.items():
                    if name not in field_labels:
                        field_labels[name] = label
                logger.debug(f"Added {len(pypdf_labels)} labels using pypdf fallback")
            
            # Filter out generic field names that don't have meaningful labels
            # Only include fields that either:
            # 1. Have a meaningful extracted label, OR
            # 2. Have a descriptive (non-generic) field name
            final_form_fields = []
            final_field_labels = {}
            
            for field_name in field_names:
                has_label = field_name in field_labels and field_labels[field_name]
                is_generic = self._is_generic_field_name(field_name)
                
                if has_label:
                    # Has a label - check if it's meaningful
                    label = field_labels[field_name]
                    # Clean the label (remove section numbers, parentheticals, etc.)
                    label = self._clean_ai_label(label)
                    
                    if not self._is_generic_label(label) and label and label != "NOT_FOUND":
                        # Good label - include this field (use cleaned version)
                        final_form_fields.append(field_name)
                        final_field_labels[field_name] = label
                    elif not is_generic:
                        # Field name is descriptive, label is generic - use field name
                        final_form_fields.append(field_name)
                        final_field_labels[field_name] = field_name
                    # Else: generic field name with generic/bad label - SKIP IT
                elif not is_generic:
                    # No label but field name is descriptive - use field name as label
                    final_form_fields.append(field_name)
                    final_field_labels[field_name] = field_name
                # Else: generic field name without label - SKIP IT (don't include in mapping)
            
            logger.info(f"Filtered {len(field_names)} fields → {len(final_form_fields)} fields with meaningful labels")
            
            return {
                "template_file": pdf_file.name,
                "form_fields": final_form_fields,  # Only fields with meaningful labels
                "field_labels": final_field_labels,
                "has_fields": len(final_form_fields) > 0,
                "field_count": len(final_form_fields),
                "original_field_count": len(field_names)
            }
        
        except Exception as e:
            logger.error(f"Error scanning template {pdf_file.name}: {e}")
            return {
                "template_file": pdf_file.name,
                "form_fields": [],
                "field_labels": {},
                "has_fields": False,
                "error": str(e)
            }
    
    def _improve_generic_field_names(self, field_names: List[str], extracted_labels: Dict[str, str]) -> Dict[str, str]:
        """
        Improve generic field names (like "Text2", "Textfield1") by:
        1. Using extracted labels where available
        2. Using field position/context to infer meaning
        3. Keeping original if no improvement possible
        """
        improved = {}
        
        for field_name in field_names:
            # Use extracted label if available
            if field_name in extracted_labels and extracted_labels[field_name]:
                improved[field_name] = extracted_labels[field_name]
                continue
            
            # For generic names, try to infer from context
            # This is a fallback - ideally we'd have labels from PyMuPDF
            if field_name.startswith('Text') or field_name.startswith('textfield'):
                # Keep as-is but mark as generic
                improved[field_name] = field_name  # Will be improved by AI mapping
            else:
                # Use field name as label if it's descriptive
                improved[field_name] = field_name
        
        return improved
    
    def scan_all_templates(self) -> Dict[str, Dict]:
        """
        Scan all PDF templates in the templates directory.
        
        Returns: Dict mapping template names to scan results
        """
        results = {}
        
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            return results
        
        for pdf_file in self.templates_dir.glob("*.pdf"):
            # Skip README and other non-template files
            if pdf_file.name.lower() in ["readme.pdf"]:
                continue
            
            template_name = pdf_file.stem
            scan_result = self.scan_template(pdf_file)
            results[template_name] = scan_result
            
            if scan_result["has_fields"]:
                logger.info(f"Scanned {template_name}: {scan_result['field_count']} fields")
            else:
                logger.debug(f"Skipped {template_name}: {scan_result.get('error', 'No fields')}")
        
        return results
    
    def get_templates_with_fields(self) -> Dict[str, Dict]:
        """Get only templates that have form fields"""
        all_templates = self.scan_all_templates()
        return {
            name: info for name, info in all_templates.items()
            if info.get("has_fields", False)
        }
    
    def get_template_fields(self, template_name: str) -> Optional[List[str]]:
        """Get form fields for a specific template"""
        pdf_file = self.templates_dir / f"{template_name}.pdf"
        
        if not pdf_file.exists():
            # Try with different extensions or exact name match
            matching_files = list(self.templates_dir.glob(f"{template_name}*"))
            if matching_files:
                pdf_file = matching_files[0]
            else:
                return None
        
        result = self.scan_template(pdf_file)
        if result and result.get("has_fields"):
            return result["form_fields"]
        
        return None

