"""
Tests for license-plate region proposal + OCR (app/ai/vision/plate_ocr.py).

- Region-proposal tests use plain OpenCV logic on synthetic images (fast,
  no model weights).
- The OCR-reading unit test mocks EasyOCR's Reader entirely, so it stays
  fast and deterministic.
- One @pytest.mark.integration test actually loads EasyOCR and reads a
  real rendered-text image, to prove the real path works end-to-end.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.ai.vision.plate_ocr import PlateReader, propose_plate_regions


def _vehicle_with_plate_image(size=(200, 300)):
    """A gray 'vehicle' box with a lighter rectangular 'plate' region in
    the plate-typical aspect-ratio range, placed in the lower half."""
    h, w = size
    img = np.full((h, w, 3), 90, dtype=np.uint8)  # dark gray vehicle body
    plate_w, plate_h = 90, 30  # aspect ratio 3.0, within [1.5, 5.5]
    px = (w - plate_w) // 2
    py = int(h * 0.65)
    img[py:py + plate_h, px:px + plate_w] = (230, 230, 230)  # bright plate
    return img, (px, py, px + plate_w, py + plate_h)


def test_propose_plate_regions_finds_a_plate_like_rectangle():
    img, expected_box = _vehicle_with_plate_image()
    candidates = propose_plate_regions(img, search_box=(0, 0, img.shape[1], img.shape[0]))
    assert len(candidates) >= 1

    # Best candidate should reasonably overlap the synthetic plate we drew.
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union else 0

    best_iou = max(iou(c, expected_box) for c in candidates)
    assert best_iou > 0.3


def test_propose_plate_regions_on_blank_image_finds_nothing():
    img = np.full((200, 300, 3), 128, dtype=np.uint8)  # flat, no edges at all
    candidates = propose_plate_regions(img, search_box=(0, 0, 300, 200))
    assert candidates == []


def test_propose_plate_regions_handles_degenerate_search_box():
    img = np.zeros((200, 300, 3), dtype=np.uint8)
    assert propose_plate_regions(img, search_box=(50, 50, 50, 50)) == []
    assert propose_plate_regions(img, search_box=(-10, -10, 10, 10)) == []


def test_read_plate_returns_none_when_ocr_finds_no_text():
    reader = PlateReader()
    reader._reader = MagicMock()
    reader._reader.readtext.return_value = []
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = reader.read_plate(img, (0, 0, 200, 100))
    assert result is None


def test_read_plate_keeps_highest_confidence_alphanumeric_read():
    reader = PlateReader()
    mock_engine = MagicMock()
    mock_engine.readtext.return_value = [
        ([], "hello", 0.40),        # not alphanumeric-plate-like but still text
        ([], "ABC-1234", 0.91),     # highest confidence, has a hyphen (stripped)
        ([], "abc1234", 0.55),
    ]
    reader._reader = mock_engine
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = reader.read_plate(img, (0, 0, 200, 100))
    assert result is not None
    assert result.text == "ABC1234"
    assert result.confidence == pytest.approx(0.91)


@pytest.mark.integration
def test_easyocr_real_inference_reads_rendered_plate_text():
    """Real end-to-end check: render actual text with PIL, run the real
    EasyOCR model (first call downloads ~64MB of weights), and confirm we
    get back a plausible alphanumeric read. Not asserting exact-match
    here (that's what scripts/eval_vision.py measures formally, see
    results.md) — just that the real pipeline produces sane output."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (400, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 64)
    except OSError:
        pytest.skip("Arial Bold font not available on this system")
    draw.text((30, 20), "ABC1234", font=font, fill=(0, 0, 0))
    img_bgr = np.array(img)[:, :, ::-1].copy()

    reader = PlateReader()
    result = reader.read_plate(img_bgr, (0, 0, 400, 120))

    assert result is not None
    assert len(result.text) >= 4  # got *something* substantial back
    assert result.text.isalnum()
    assert 0.0 <= result.confidence <= 1.0
