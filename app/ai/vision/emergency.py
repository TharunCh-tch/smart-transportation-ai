"""
Heuristic "likely emergency vehicle" classifier.

IMPORTANT / documented limitation: this is NOT a trained classifier.
There is no labeled emergency-vehicle dataset in this project, so we do
not claim learned emergency-vehicle detection anywhere. What's here
instead is a deliberately simple, inspectable color-cue heuristic:

  1. Take the top ~35% of a vehicle's detection box (where a roof-mounted
     light bar sits relative to the full vehicle silhouette).
  2. Convert that strip to HSV and measure the fraction of pixels that
     fall in red or blue hue bands (the two colors used by the vast
     majority of US/EU police, fire, and EMS light bars).
  3. Also measure the fraction of near-white/near-black pixels in the
     full box, since many emergency vehicles (US police cruisers,
     ambulances) have high-contrast white bodies with black/dark
     graphics — a weak secondary signal, not used alone.
  4. Combine into a 0-1 "emergency_score"; flag `likely_emergency=True`
     above a threshold.

This will false-positive on e.g. red sports cars or blue vans, and
false-negative on emergency vehicles with lights off, obstructed light
bars, or unusual liveries. It is a reasonable-effort heuristic offered
as a starting point, not a production signal. See results.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

# HSV hue ranges (OpenCV hue is 0-179)
_RED_RANGES = [((0, 70, 60), (10, 255, 255)), ((170, 70, 60), (179, 255, 255))]
_BLUE_RANGE = ((100, 70, 60), (130, 255, 255))

# Tuned empirically against a set of known true-negative images (real
# traffic photos with zero emergency vehicles present — see results.md,
# "Emergency-vehicle heuristic" section) to keep the false-positive rate
# low. At 0.12 the false-positive rate on that negative set was ~21%; at
# 0.35 it drops to ~2%. Still a heuristic, still not a trained classifier.
EMERGENCY_SCORE_THRESHOLD = 0.35


@dataclass
class EmergencyAssessment:
    likely_emergency: bool
    emergency_score: float
    red_blue_fraction: float
    contrast_fraction: float

    def to_dict(self) -> dict:
        return {
            "likely_emergency": self.likely_emergency,
            "emergency_score": round(self.emergency_score, 4),
            "red_blue_fraction": round(self.red_blue_fraction, 4),
            "contrast_fraction": round(self.contrast_fraction, 4),
            "method": "heuristic-color-cue (not a trained classifier)",
        }


def _hsv_mask_fraction(hsv_roi: np.ndarray) -> float:
    import cv2

    if hsv_roi.size == 0:
        return 0.0
    masks = []
    for lo, hi in _RED_RANGES:
        masks.append(cv2.inRange(hsv_roi, np.array(lo), np.array(hi)))
    lo, hi = _BLUE_RANGE
    masks.append(cv2.inRange(hsv_roi, np.array(lo), np.array(hi)))
    combined = masks[0]
    for m in masks[1:]:
        combined = cv2.bitwise_or(combined, m)
    return float(np.count_nonzero(combined)) / float(hsv_roi.shape[0] * hsv_roi.shape[1])


def _contrast_fraction(bgr_roi: np.ndarray) -> float:
    if bgr_roi.size == 0:
        return 0.0
    gray = bgr_roi.mean(axis=2)
    white = gray > 200
    black = gray < 45
    return float(np.count_nonzero(white | black)) / float(gray.size)


def assess(image_bgr: np.ndarray, box: Tuple[float, float, float, float]) -> EmergencyAssessment:
    import cv2

    x1, y1, x2, y2 = [int(v) for v in box]
    h_img, w_img = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    if x2 <= x1 or y2 <= y1:
        return EmergencyAssessment(False, 0.0, 0.0, 0.0)

    full = image_bgr[y1:y2, x1:x2]
    top_h = max(1, int((y2 - y1) * 0.35))
    top_strip = image_bgr[y1:y1 + top_h, x1:x2]

    hsv_strip = cv2.cvtColor(top_strip, cv2.COLOR_BGR2HSV) if top_strip.size else np.zeros((0, 0, 3), np.uint8)
    red_blue_frac = _hsv_mask_fraction(hsv_strip)
    contrast_frac = _contrast_fraction(full)

    # Weighted combination — red/blue light-bar cue dominates, contrast is a
    # weak secondary signal so a plain white car doesn't trip the threshold
    # on its own.
    score = 0.85 * red_blue_frac + 0.15 * min(contrast_frac, 0.5)
    return EmergencyAssessment(
        likely_emergency=score >= EMERGENCY_SCORE_THRESHOLD,
        emergency_score=score,
        red_blue_fraction=red_blue_frac,
        contrast_fraction=contrast_frac,
    )
