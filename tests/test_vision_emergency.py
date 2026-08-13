"""
Tests for the emergency-vehicle color-cue heuristic (app/ai/vision/emergency.py).
Pure OpenCV/NumPy on synthetic images — no model weights involved.
"""
import numpy as np

from app.ai.vision.emergency import EMERGENCY_SCORE_THRESHOLD, assess


def _solid_image(bgr_color, size=(200, 300)):
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr_color
    return img


def test_plain_white_car_is_not_flagged_as_emergency():
    img = _solid_image((235, 235, 235))  # near-white body, no red/blue
    result = assess(img, (0, 0, 300, 200))
    assert result.likely_emergency is False
    assert result.emergency_score < EMERGENCY_SCORE_THRESHOLD


def test_plain_gray_car_is_not_flagged_as_emergency():
    img = _solid_image((120, 120, 120))
    result = assess(img, (0, 0, 300, 200))
    assert result.likely_emergency is False


def test_vehicle_with_red_and_blue_top_strip_is_flagged():
    """Simulate a roof light bar: red/blue stripes across the top 35% of
    the box, plain body below — this is exactly the cue the heuristic
    looks for."""
    h, w = 200, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)
    top_h = int(h * 0.35)
    # OpenCV is BGR: pure red = (0,0,255), pure blue = (255,0,0)
    img[0:top_h, 0:w // 2] = (0, 0, 255)   # red half of the light bar
    img[0:top_h, w // 2:w] = (255, 0, 0)   # blue half of the light bar
    img[top_h:, :] = (250, 250, 250)       # white body below

    result = assess(img, (0, 0, w, h))
    assert result.red_blue_fraction > 0.5
    assert result.likely_emergency is True
    assert result.emergency_score >= EMERGENCY_SCORE_THRESHOLD


def test_score_is_zero_to_one_bounded():
    for color in [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 255), (0, 0, 0)]:
        img = _solid_image(color)
        result = assess(img, (0, 0, 300, 200))
        assert 0.0 <= result.emergency_score <= 1.0
        assert 0.0 <= result.red_blue_fraction <= 1.0
        assert 0.0 <= result.contrast_fraction <= 1.0


def test_degenerate_zero_area_box_returns_safe_default():
    img = _solid_image((0, 0, 255))
    result = assess(img, (50, 50, 50, 50))  # zero-width/height box
    assert result.likely_emergency is False
    assert result.emergency_score == 0.0


def test_to_dict_reports_heuristic_method_not_trained_classifier():
    img = _solid_image((128, 128, 128))
    result = assess(img, (0, 0, 300, 200))
    d = result.to_dict()
    assert "heuristic" in d["method"].lower()
