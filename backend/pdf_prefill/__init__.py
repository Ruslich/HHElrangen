"""
PDF pre-fill integration package for the Elrangen hackathon backend.

This module bundles reusable utilities for:
  - template scanning and metadata
  - mapping CSV/clinical fields onto PDF form fields
  - generating and storing filled PDFs for downstream consumption

The implementations are adapted from the standalone "PDF pre-fill" prototype.
"""

from .service import PDFPrefillError, PDFPrefillService

__all__ = ["PDFPrefillService", "PDFPrefillError"]

