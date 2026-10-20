"""Find molecule drawings on each page and save one cropped image per molecule.

Two methods share the same output format (one row per crop):
  * moldet   - MolDetV2, the YOLO11 document detector that BioMiner uses (default).
  * embedded - fallback that saves raster images embedded in the PDF. It needs no model,
               but misses vector drawings and can return whole multi-molecule figures.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

from .render import open_pdf

log = logging.getLogger(__name__)

CROP_COLUMNS = [
    "crop_id", "paper_id", "page", "x1", "y1", "x2", "y2", "det_confidence", "method", "image_path",
]


def _crop_id(paper_id: int, page: int, index: int) -> str:
    return f"p{paper_id}_pg{page:03d}_c{index:02d}"


def detect_with_moldet(
    pages: pd.DataFrame,
    weights_path: Path,
    crops_dir: Path,
    imgsz: int,
    conf_threshold: float,
    pad_px: int,
    device: str,
) -> pd.DataFrame:
    """Run MolDetV2 on every page image and crop each detected molecule."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Run `pip install -r requirements.txt`, "
            "or set detection.method: embedded in the config."
        ) from exc

    model = YOLO(str(weights_path))
    records = []

    for page in pages.itertuples(index=False):
        try:
            result = model.predict(source=page.image_path, imgsz=imgsz, conf=conf_threshold, device=device, verbose=False)[0]
        except Exception as exc:  # one unreadable page must not stop detection for every paper
            log.warning("Paper %s page %s: detection failed (%s); skipped", page.paper_id, page.page, exc)
            continue
        if result.boxes is None or len(result.boxes) == 0:
            continue  # page with text, plots or tables only

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        # Number crops in reading order: top to bottom, then left to right.
        order = sorted(range(len(boxes)), key=lambda k: (round(boxes[k][1] / 50), boxes[k][0]))

        paper_dir = crops_dir / str(page.paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(page.image_path) as page_image:
            page_rgb = page_image.convert("RGB")
        width, height = page_rgb.size

        for index, k in enumerate(order, start=1):
            x1, y1, x2, y2 = boxes[k]
            # Pad the box so bonds touching its edge are not cut off, but stay inside the page.
            x1, y1 = max(0, int(x1) - pad_px), max(0, int(y1) - pad_px)
            x2, y2 = min(width, int(x2) + pad_px), min(height, int(y2) + pad_px)

            crop_id = _crop_id(page.paper_id, page.page, index)
            crop_path = paper_dir / f"{crop_id}.png"
            page_rgb.crop((x1, y1, x2, y2)).save(crop_path)
            records.append(
                {
                    "crop_id": crop_id, "paper_id": page.paper_id, "page": page.page,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "det_confidence": round(float(scores[k]), 4), "method": "moldet",
                    "image_path": str(crop_path),
                }
            )

    crops = pd.DataFrame(records, columns=CROP_COLUMNS)
    log.info("MolDetV2 found %d molecule crops on %d pages", len(crops), len(pages))
    return crops


def detect_embedded_images(pdfs: dict[int, Path], crops_dir: Path, min_width: int, min_height: int) -> pd.DataFrame:
    """Fallback: save every raster image embedded in the PDFs that is large enough to hold a structure."""
    records = []
    for paper_id, pdf_path in pdfs.items():
        paper_dir = crops_dir / str(paper_id)
        paper_dir.mkdir(parents=True, exist_ok=True)

        with open_pdf(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                index = 0
                for image_info in page.get_images(full=True):
                    try:
                        extracted = document.extract_image(image_info[0])  # image_info[0] is the xref
                    except Exception as exc:
                        log.warning("Paper %s page %s: image could not be extracted (%s)", paper_id, page_number, exc)
                        continue
                    if extracted["width"] < min_width or extracted["height"] < min_height:
                        continue
                    index += 1
                    crop_id = _crop_id(paper_id, page_number, index)
                    crop_path = paper_dir / f"{crop_id}.png"
                    try:
                        # Re-encode as RGB PNG so CMYK or JPX images can be read by the recognizers.
                        Image.open(io.BytesIO(extracted["image"])).convert("RGB").save(crop_path)
                    except Exception as exc:  # an unreadable image should not stop the run
                        log.warning("Skipping image on paper %s page %s: %s", paper_id, page_number, exc)
                        continue
                    records.append(
                        {
                            "crop_id": crop_id, "paper_id": paper_id, "page": page_number,
                            "x1": None, "y1": None, "x2": None, "y2": None,
                            "det_confidence": None, "method": "embedded", "image_path": str(crop_path),
                        }
                    )

    crops = pd.DataFrame(records, columns=CROP_COLUMNS)
    log.info("Embedded-image fallback saved %d images", len(crops))
    return crops
