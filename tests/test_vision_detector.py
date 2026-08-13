"""
Tests for vehicle detection (app/ai/vision/detector.py).

- Unit tests mock the Ultralytics YOLO model so they run in milliseconds
  and don't need network access or downloaded weights.
- One @pytest.mark.integration test loads the real yolov8n.pt checkpoint
  (downloaded on first use) and runs actual inference on the bundled
  bus.jpg sample, checking for a sane, known result.
"""
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.ai.vision.detector import VEHICLE_CLASSES, VehicleDetector


def _make_mock_yolo_result(boxes_data):
    """boxes_data: list of (class_id, confidence, (x1,y1,x2,y2))"""
    mock_result = MagicMock()
    mock_result.names = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    mock_boxes = []
    for cls_id, conf, box in boxes_data:
        b = MagicMock()
        b.cls = [cls_id]
        b.conf = [conf]
        b.xyxy = [list(box)]
        mock_boxes.append(b)
    mock_result.boxes = mock_boxes
    return mock_result


def test_detect_returns_empty_list_when_no_boxes():
    detector = VehicleDetector()
    fake_model = MagicMock()
    fake_model.predict.return_value = [_make_mock_yolo_result([])]
    detector._model = fake_model

    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert detections == []


def test_detect_parses_boxes_into_detection_objects():
    detector = VehicleDetector()
    fake_model = MagicMock()
    fake_model.predict.return_value = [_make_mock_yolo_result([
        (2, 0.87, (10.0, 20.0, 100.0, 120.0)),
        (7, 0.55, (200.0, 30.0, 320.0, 150.0)),
    ])]
    detector._model = fake_model

    detections = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert len(detections) == 2
    assert detections[0].label == "car"
    assert detections[0].class_id == 2
    assert detections[0].confidence == pytest.approx(0.87)
    assert detections[0].box == (10.0, 20.0, 100.0, 120.0)
    assert detections[1].label == "truck"


def test_detect_restricts_to_vehicle_classes_by_default():
    detector = VehicleDetector()
    fake_model = MagicMock()
    detector._model = fake_model
    fake_model.predict.return_value = [_make_mock_yolo_result([])]

    detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    _, kwargs = fake_model.predict.call_args
    assert set(kwargs["classes"]) == set(VEHICLE_CLASSES.keys())


def test_detection_to_dict_shape():
    detector = VehicleDetector()
    fake_model = MagicMock()
    fake_model.predict.return_value = [_make_mock_yolo_result([(2, 0.9, (1.0, 2.0, 3.0, 4.0))])]
    detector._model = fake_model

    det = detector.detect(np.zeros((10, 10, 3), dtype=np.uint8))[0]
    d = det.to_dict()
    assert d["label"] == "car"
    assert d["class_id"] == 2
    assert set(d["box"].keys()) == {"x1", "y1", "x2", "y2"}


def test_model_unavailable_error_message_mentions_requirements_file():
    from app.ai.vision.detector import ModelUnavailableError
    with patch.dict("sys.modules", {"ultralytics": None}):
        detector = VehicleDetector()
        with pytest.raises(ModelUnavailableError) as exc_info:
            detector._ensure_loaded()
        assert "requirements-cv.txt" in str(exc_info.value)


@pytest.mark.integration
def test_real_yolo_detects_the_bus_in_bus_jpg(bus_jpg):
    """Ultralytics' own bus.jpg sample: known to contain exactly one bus.
    This is the real inference path — first run downloads yolov8n.pt."""
    detector = VehicleDetector()
    detections = detector.detect(str(bus_jpg))

    assert len(detections) >= 1
    labels = [d.label for d in detections]
    assert "bus" in labels
    bus_det = next(d for d in detections if d.label == "bus")
    assert bus_det.confidence > 0.5
    x1, y1, x2, y2 = bus_det.box
    assert x2 > x1 and y2 > y1


@pytest.mark.integration
def test_real_yolo_finds_no_vehicles_in_zidane_jpg(zidane_jpg):
    """Negative control: zidane.jpg is two soccer players, no vehicles."""
    detector = VehicleDetector()
    detections = detector.detect(str(zidane_jpg))
    assert detections == []
