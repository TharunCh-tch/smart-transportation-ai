"""
Orchestrates the vision stack for a single image or a video: detection ->
tracking -> per-vehicle plate OCR + emergency heuristic. This is the
module the FastAPI routes in app/api/routes/vision.py call into, and the
same functions back the offline evaluation script used for results.md.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from app.ai.vision.detector import get_detector
from app.ai.vision.emergency import assess as assess_emergency
from app.ai.vision.plate_ocr import get_plate_reader
from app.ai.vision.tracker import CentroidTracker

logger = logging.getLogger(__name__)


def _read_image(source) -> np.ndarray:
    import cv2

    if isinstance(source, np.ndarray):
        return source
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source))
        if img is None:
            raise ValueError(f"Could not read image: {source}")
        return img
    raise TypeError(f"Unsupported image source type: {type(source)}")


def process_image(source, run_ocr: bool = True, run_emergency: bool = True) -> Dict:
    """
    Full single-frame pipeline. `source` is a path or a BGR numpy array.
    Returns a JSON-serializable dict: detections, plate reads, and
    emergency-heuristic assessments, plus timing.
    """
    t0 = time.perf_counter()
    image = _read_image(source)
    h, w = image.shape[:2]

    detector = get_detector()
    detections = detector.detect(image)
    t_detect = time.perf_counter()

    vehicles = []
    plate_reader = get_plate_reader() if (run_ocr and detections) else None
    for det in detections:
        entry = {"detection": det.to_dict()}
        if run_emergency:
            entry["emergency"] = assess_emergency(image, det.box).to_dict()
        if run_ocr and plate_reader is not None:
            try:
                plate = plate_reader.read_from_vehicle_box(image, det.box)
                entry["plate"] = plate.to_dict() if plate else None
            except Exception as exc:  # OCR failures shouldn't kill the whole request
                logger.warning("Plate OCR failed for a detection: %s", exc)
                entry["plate"] = None
        vehicles.append(entry)
    t_end = time.perf_counter()

    return {
        "image_size": {"width": w, "height": h},
        "vehicle_count": len(detections),
        "vehicles": vehicles,
        "timing_ms": {
            "detect": round((t_detect - t0) * 1000, 1),
            "total": round((t_end - t0) * 1000, 1),
        },
    }


def process_video(source, run_ocr: bool = False, run_emergency: bool = True,
                   sample_every: int = 1, max_frames: Optional[int] = None) -> Dict:
    """
    Frame-by-frame pipeline with tracking. OCR defaults OFF for video
    because per-frame EasyOCR is slow on CPU; it can be enabled but will
    be noticeably slower per frame (see results.md latency numbers).

    `sample_every`: process every Nth frame (tracker still gets called
    per processed frame — this trades temporal resolution for speed).
    """
    import cv2

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source}")

    detector = get_detector()
    plate_reader = get_plate_reader() if run_ocr else None
    tracker = CentroidTracker()

    frame_idx = 0
    processed = 0
    track_summary: Dict[int, Dict] = {}
    frames_out = []
    t0 = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % sample_every != 0:
                frame_idx += 1
                continue

            detections = detector.detect(frame)
            tracks = tracker.update(detections)

            frame_result = {"frame": frame_idx, "tracks": []}
            for t in tracks:
                track_entry = t.to_dict()
                if run_emergency:
                    track_entry["emergency"] = assess_emergency(frame, t.box).to_dict()
                if run_ocr and plate_reader is not None:
                    plate = plate_reader.read_from_vehicle_box(frame, t.box)
                    track_entry["plate"] = plate.to_dict() if plate else None
                frame_result["tracks"].append(track_entry)

                summary = track_summary.setdefault(t.track_id, {
                    "track_id": t.track_id, "label": t.label,
                    "first_frame": frame_idx, "last_frame": frame_idx, "frame_count": 0,
                })
                summary["last_frame"] = frame_idx
                summary["frame_count"] += 1

            frames_out.append(frame_result)
            processed += 1
            frame_idx += 1
            if max_frames and processed >= max_frames:
                break
    finally:
        cap.release()

    t_end = time.perf_counter()
    return {
        "frames_processed": processed,
        "unique_tracks": len(track_summary),
        "track_summary": list(track_summary.values()),
        "frames": frames_out,
        "timing_ms": {"total": round((t_end - t0) * 1000, 1),
                      "avg_per_frame": round((t_end - t0) * 1000 / processed, 1) if processed else None},
    }
