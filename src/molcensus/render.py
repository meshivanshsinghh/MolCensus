"""Render PDF pages to images.

We detect molecules on whole-page images instead of pulling embedded pictures out of
the PDF, because journals often store structures as vector drawings that image
extraction misses, and extracted figures mix molecules with plots and tables.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def open_pdf(pdf_path: Path):
    """Open a PDF with PyMuPDF (imported as `pymupdf` in new versions, `fitz` in older ones)."""
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    return fitz.open(str(pdf_path))


def render_pages(pdfs: dict[int, Path], pages_dir: Path, dpi: int, force: bool = False) -> pd.DataFrame:
    """Save every page as <pages_dir>/<paper_id>/page_###.png and return an index of the images."""
    records = []
    for paper_id, pdf_path in pdfs.items():
        paper_dir = pages_dir / str(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        with open_pdf(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                image_path = paper_dir / f"page_{page_number:03d}.png"
                if force or not image_path.exists():  # skip pages rendered in an earlier run
                    page.get_pixmap(dpi=dpi).save(str(image_path))
                records.append({"paper_id": paper_id, "page": page_number, "image_path": str(image_path)})

        log.info("Paper %s: %d pages rendered at %d dpi", paper_id, page_number, dpi)

    return pd.DataFrame(records, columns=["paper_id", "page", "image_path"])
