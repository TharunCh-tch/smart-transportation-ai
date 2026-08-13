"""
Computer-vision API: vehicle / emergency-vehicle detection, license-plate
OCR, and multi-object tracking on uploaded images/video or bundled sample
images. Built on top of app/ai/vision/*.

This is deliberately independent of the traffic-simulation engine in
app/ai/{city_graph,route_optimizer,traffic_simulator,fleet_simulator}.py
— the one integration point is that a detected likely-emergency vehicle
can be written into the same `TrafficIncident` table the simulator uses
(see /vision/incidents and POST .../detect/image's lat/lng/edge_id
params), so a real camera-frame detection shows up alongside simulated
incidents in the existing incident data model.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import TrafficIncident
from app.models.schemas import CVIncidentOut, VisionHealthOut, VisionSampleOut

logger = logging.getLogger(__name__)
router = APIRouter()

SAMPLES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "vision_samples"

SAMPLES = {
    "bus.jpg": {
        "description": "Ultralytics stock sample — 1 bus, several pedestrians.",
        "source": "Bundled with the `ultralytics` pip package (Ultralytics assets).",
    },
    "manhattan_50th_st.jpg": {
        "description": "50th Street, Midtown Manhattan — truck, bus, taxi, sedan.",
        "source": "Wikimedia Commons, CC-BY-4.0, photographer DrewWilliam.",
    },
    "auckland_traffic.jpg": {
        "description": "Southern Motorway, Auckland NZ — dense multi-lane traffic (~37 vehicles).",
        "source": "Wikimedia Commons, CC0 1.0 public domain, photographer Kiwiev.",
    },
}

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo", "video/webm"}


def _check_cv_available():
    try:
        import cv2  # noqa: F401
        import ultralytics  # noqa: F401
    except ImportError:
        raise HTTPException(
            503,
            "CV dependencies not installed. Run: "
            "pip install -r requirements-cv.txt",
        )


async def _save_upload(upload: UploadFile, suffix: str) -> Path:
    contents = await upload.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)}MB)")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(contents)
    tmp.close()
    return Path(tmp.name)


def _maybe_log_emergency_incidents(db: Session, vision_result: dict, lat: Optional[float],
                                    lng: Optional[float], edge_id: Optional[str]) -> int:
    """Write a TrafficIncident row for each likely-emergency vehicle found.
    Returns the number of incidents created. No-op if lat/lng not provided
    (we don't want to guess a location)."""
    if lat is None or lng is None:
        return 0
    created = 0
    for v in vision_result.get("vehicles", []):
        emergency = v.get("emergency")
        if emergency and emergency.get("likely_emergency"):
            db.add(TrafficIncident(
                edge_id=edge_id or "unknown",
                severity="high",
                description=(
                    f"CV-detected likely emergency vehicle ({v['detection']['label']}, "
                    f"heuristic score={emergency['emergency_score']:.2f}) - see "
                    f"app/ai/vision/emergency.py for method and limitations."
                ),
                lat=lat, lng=lng, active=1, source="cv-detection",
            ))
            created += 1
    if created:
        db.commit()
    return created


# ── Health / samples ────────────────────────────────────────────────────

@router.get("/vision/health", response_model=VisionHealthOut)
def vision_health():
    ultralytics_available = easyocr_available = opencv_available = False
    try:
        import ultralytics  # noqa: F401
        ultralytics_available = True
    except ImportError:
        pass
    try:
        import easyocr  # noqa: F401
        easyocr_available = True
    except ImportError:
        pass
    try:
        import cv2  # noqa: F401
        opencv_available = True
    except ImportError:
        pass

    from app.ai.vision.detector import get_detector
    from app.ai.vision.plate_ocr import get_plate_reader

    detector = get_detector()
    reader = get_plate_reader()
    return VisionHealthOut(
        detector_loaded=detector.is_loaded,
        detector_weights=detector.weights,
        detector_device=detector.device,
        ocr_loaded=reader.is_loaded,
        ultralytics_available=ultralytics_available,
        easyocr_available=easyocr_available,
        opencv_available=opencv_available,
    )


@router.get("/vision/samples", response_model=List[VisionSampleOut])
def vision_samples():
    return [VisionSampleOut(name=name, **meta) for name, meta in SAMPLES.items()]


# ── Detection endpoints ─────────────────────────────────────────────────

@router.post("/vision/detect/image")
async def detect_image(
    file: Optional[UploadFile] = File(None),
    sample: Optional[str] = Query(None, description="Name of a bundled sample image (see /vision/samples)"),
    run_ocr: bool = Query(True, description="Run license-plate OCR on each detected vehicle"),
    run_emergency: bool = Query(True, description="Run the emergency-vehicle heuristic on each detection"),
    lat: Optional[float] = Query(None, description="If set with lng, logs likely-emergency detections as incidents"),
    lng: Optional[float] = Query(None),
    edge_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """
    Run the vehicle/plate/emergency pipeline on a single image.
    Either upload `file` or pass `sample=<name>` to use a bundled image.
    """
    _check_cv_available()
    from app.ai.vision.pipeline import process_image

    tmp_path = None
    try:
        if file is not None:
            if file.content_type not in ALLOWED_IMAGE_TYPES:
                raise HTTPException(415, f"Unsupported image type: {file.content_type}")
            suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
            tmp_path = await _save_upload(file, suffix)
            image_source = str(tmp_path)
        elif sample is not None:
            if sample not in SAMPLES:
                raise HTTPException(404, f"Unknown sample '{sample}'. See GET /api/vision/samples")
            image_source = str(SAMPLES_DIR / sample)
        else:
            raise HTTPException(400, "Provide either an uploaded `file` or a `sample` name")

        result = process_image(image_source, run_ocr=run_ocr, run_emergency=run_emergency)
        incidents_created = _maybe_log_emergency_incidents(db, result, lat, lng, edge_id)
        result["incidents_logged"] = incidents_created
        return result
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


@router.post("/vision/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    run_ocr: bool = Query(False, description="OCR per tracked vehicle per frame — slow on CPU"),
    run_emergency: bool = Query(True),
    sample_every: int = Query(5, ge=1, le=60, description="Process every Nth frame"),
    max_frames: int = Query(60, ge=1, le=600, description="Cap on frames processed, to bound request time"),
):
    """
    Run detection + tracking across a video, sampling every `sample_every`
    frames (bounded by `max_frames`) to keep request latency reasonable on
    CPU. Returns per-frame tracks plus a per-track summary (first/last
    frame seen, hit count) — this is the "temporal tracking" surface.
    """
    _check_cv_available()
    from app.ai.vision.pipeline import process_video

    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(415, f"Unsupported video type: {file.content_type}")

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    tmp_path = await _save_upload(file, suffix)
    try:
        result = process_video(
            str(tmp_path), run_ocr=run_ocr, run_emergency=run_emergency,
            sample_every=sample_every, max_frames=max_frames,
        )
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


# ── CV-sourced incidents (integration with the existing incident model) ──

@router.get("/vision/incidents", response_model=List[CVIncidentOut])
def cv_incidents(limit: int = 20, db: Session = Depends(get_db)):
    """Incidents created from real CV detections (as opposed to the
    simulated ones from /api/traffic/incidents)."""
    rows = (
        db.query(TrafficIncident)
        .filter(TrafficIncident.source == "cv-detection")
        .order_by(TrafficIncident.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        CVIncidentOut(
            id=r.id, edge_id=r.edge_id, severity=r.severity,
            description=r.description, lat=r.lat, lng=r.lng,
            active=bool(r.active), source=r.source,
        )
        for r in rows
    ]
