"""
Vehicle detection built on Ultralytics YOLOv8 with pretrained COCO weights.

Why YOLOv8 (via the `ultralytics` pip package) instead of YOLOv5:
  - `ultralytics` is the actively maintained successor; the original
    `ultralytics/yolov5` repo is in maintenance mode and its recommended
    install path (torch.hub / cloning the repo) is heavier and less
    reproducible than `pip install ultralytics`.
  - Same COCO-pretrained weight family, same detection classes we need
    (car/motorcycle/bus/truck), a cleaner Python API (`YOLO(...)(image)`),
    and ONNX/TensorRT export baked in if we ever need faster inference.
  - Nothing here is custom-trained. We use the stock COCO checkpoint
    (`yolov8n.pt`), so detections are limited to the 80 COCO classes.

No custom or fine-tuned weights are used anywhere in this module — see
results.md for what that means for accuracy on e.g. emergency vehicles.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# COCO class indices we treat as "vehicles" for this project.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

DEFAULT_WEIGHTS = "yolov8n.pt"  # smallest/fastest COCO-pretrained checkpoint
DEFAULT_CONF = 0.35
DEFAULT_IOU = 0.45


@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    # pixel-space box, (x1, y1, x2, y2), top-left origin
    box: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "class_id": self.class_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": {"x1": round(float(x1), 1), "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1), "y2": round(float(y2), 1)},
        }


class ModelUnavailableError(RuntimeError):
    """Raised when the `ultralytics`/`torch` stack isn't installed."""


class VehicleDetector:
    """
    Thin, lazily-initialized wrapper around an Ultralytics YOLO model.

    The model is loaded once per process (module-level singleton via
    `get_detector()`) since constructing it — and the first-run weight
    download — is the expensive part. Inference itself is CPU-friendly
    for the nano checkpoint (see results.md for measured latency).
    """

    def __init__(self, weights: str = DEFAULT_WEIGHTS, conf: float = DEFAULT_CONF,
                 iou: float = DEFAULT_IOU, device: str = "cpu"):
        self.weights = weights
        self.conf = conf
        self.iou = iou
        self.device = device
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise ModelUnavailableError(
                    "ultralytics is not installed. Install with "
                    "`pip install -r requirements-cv.txt`."
                ) from exc
            logger.info("Loading YOLO weights: %s", self.weights)
            self._model = YOLO(self.weights)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def detect(self, image, classes: Optional[List[int]] = None) -> List[Detection]:
        """
        Run detection on a single image.

        `image` may be a file path, a numpy BGR array (OpenCV convention),
        or a PIL Image — anything Ultralytics' predict() accepts.
        `classes` restricts detection to given COCO class ids; defaults to
        the vehicle classes above.
        """
        self._ensure_loaded()
        target_classes = classes if classes is not None else list(VEHICLE_CLASSES.keys())

        results = self._model.predict(
            source=image,
            conf=self.conf,
            iou=self.iou,
            classes=target_classes,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        r = results[0]
        if r.boxes is None:
            return detections

        names = r.names  # class_id -> label, from the model itself
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            detections.append(Detection(
                class_id=cls_id,
                label=names.get(cls_id, str(cls_id)),
                confidence=conf,
                box=(x1, y1, x2, y2),
            ))
        return detections


_detector_lock = threading.Lock()
_detector_singleton: Optional[VehicleDetector] = None


def get_detector() -> VehicleDetector:
    """Process-wide singleton so we only pay model-load cost once."""
    global _detector_singleton
    if _detector_singleton is None:
        with _detector_lock:
            if _detector_singleton is None:
                _detector_singleton = VehicleDetector()
    return _detector_singleton
