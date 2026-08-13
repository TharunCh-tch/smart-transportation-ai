"""
License-plate candidate detection + OCR.

Two-stage, classic-CV approach (no trained plate detector — see the
limitation note below):

1. Candidate region proposal: given a vehicle bounding box from
   detector.py, search the lower half of that box (where a plate is on
   almost any front/rear vehicle photo) using the textbook "poor man's
   ANPR" pipeline: grayscale -> bilateral filter (denoise, keep edges)
   -> Canny edge detection -> contour extraction -> filter contours by
   plate-like aspect ratio (~1.5:1 to 5.5:1, covers US/EU plate shapes)
   and minimum area. The largest qualifying contour's bounding rect is
   the plate candidate.
2. OCR: run EasyOCR on the candidate crop (upscaled + thresholded) and
   keep the highest-confidence alphanumeric read.

Why EasyOCR over pytesseract: EasyOCR is pure-Python + PyTorch, so it
installs with `pip install easyocr` and needs no separate system binary.
pytesseract instead wraps the Tesseract OCR *engine*, which has to be
installed separately (apt/brew/an .exe on Windows) and put on PATH —
extra friction for anyone cloning this repo. EasyOCR's tradeoff is a
larger model download (~64MB detection + recognition nets) and slower
cold start.

Known limitation (documented, not hidden): this is a geometric heuristic,
not a trained plate detector. It works reasonably on head-on/near-head-on
shots of a single vehicle with a visible, well-lit, unobstructed plate.
It will miss plates at steep angles, in low light, partially occluded,
or when the vehicle crop is small/blurry — see results.md for measured
numbers on a synthetic test set.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIN_ASPECT = 1.5
MAX_ASPECT = 5.5
MIN_AREA_FRAC = 0.01   # candidate must cover >=1% of the search region area
MAX_AREA_FRAC = 0.60


@dataclass
class PlateRead:
    text: str
    confidence: float
    box: Tuple[float, float, float, float]  # absolute image coords

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
        }


def propose_plate_regions(image_bgr: np.ndarray,
                           search_box: Tuple[float, float, float, float],
                           max_candidates: int = 3) -> List[Tuple[int, int, int, int]]:
    """
    Return up to `max_candidates` candidate plate rectangles (x1,y1,x2,y2
    in absolute image pixel coords), searched within the lower half of
    `search_box` (a vehicle detection box), sorted by contour area desc.
    """
    import cv2

    h_img, w_img = image_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in search_box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return []

    box_h = y2 - y1
    sy1 = y1 + int(box_h * 0.45)   # lower ~55% of the vehicle box
    sy2 = y2
    roi = image_bgr[sy1:sy2, x1:x2]
    if roi.size == 0:
        return []

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    roi_area = roi.shape[0] * roi.shape[1]

    candidates = []
    for c in contours:
        rx, ry, rw, rh = cv2.boundingRect(c)
        if rh == 0:
            continue
        aspect = rw / float(rh)
        area = rw * rh
        area_frac = area / float(roi_area) if roi_area else 0
        if MIN_ASPECT <= aspect <= MAX_ASPECT and MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC:
            # Map back to absolute image coordinates.
            abs_x1 = x1 + rx
            abs_y1 = sy1 + ry
            abs_x2 = abs_x1 + rw
            abs_y2 = abs_y1 + rh
            candidates.append((area, (abs_x1, abs_y1, abs_x2, abs_y2)))

    candidates.sort(key=lambda t: t[0], reverse=True)
    return [box for _, box in candidates[:max_candidates]]


class PlateReader:
    """Lazily-loaded EasyOCR wrapper, process-wide singleton via get_plate_reader()."""

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._reader is not None:
            return
        with self._lock:
            if self._reader is not None:
                return
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError(
                    "easyocr is not installed. Install with "
                    "`pip install -r requirements-cv.txt`."
                ) from exc
            logger.info("Loading EasyOCR reader (languages=%s, gpu=%s)", self.languages, self.gpu)
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)

    @property
    def is_loaded(self) -> bool:
        return self._reader is not None

    def read_plate(self, image_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[PlateRead]:
        """OCR a single candidate box; returns the best alphanumeric read, or None."""
        import cv2

        self._ensure_loaded()
        x1, y1, x2, y2 = box
        crop = image_bgr[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            return None

        # Upscale small crops — OCR nets do much better above ~200px wide.
        h, w = crop.shape[:2]
        if w < 200:
            scale = 200.0 / max(w, 1)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        results = self._reader.readtext(crop, detail=1, paragraph=False)
        if not results:
            return None

        # Keep the highest-confidence, alphanumeric-only read.
        best = None
        for _, text, conf in results:
            cleaned = "".join(ch for ch in text.upper() if ch.isalnum())
            if not cleaned:
                continue
            if best is None or conf > best[1]:
                best = (cleaned, conf)
        if best is None:
            return None
        return PlateRead(text=best[0], confidence=best[1], box=(x1, y1, x2, y2))

    def read_from_vehicle_box(self, image_bgr: np.ndarray,
                               vehicle_box: Tuple[float, float, float, float]) -> Optional[PlateRead]:
        """Full pipeline: propose candidate regions in `vehicle_box`, OCR each, return the best."""
        candidates = propose_plate_regions(image_bgr, vehicle_box)
        best_read = None
        for cand in candidates:
            read = self.read_plate(image_bgr, cand)
            if read and (best_read is None or read.confidence > best_read.confidence):
                best_read = read
        return best_read


_reader_lock = threading.Lock()
_reader_singleton: Optional[PlateReader] = None


def get_plate_reader() -> PlateReader:
    global _reader_singleton
    if _reader_singleton is None:
        with _reader_lock:
            if _reader_singleton is None:
                _reader_singleton = PlateReader()
    return _reader_singleton
