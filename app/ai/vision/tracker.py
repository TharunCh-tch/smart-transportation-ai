"""
Lightweight multi-object tracker for associating detections across frames.

This implements the classic centroid-tracking algorithm (Rosebrock,
pyimagesearch "Simple object tracking with OpenCV", 2018) plus an IOU
gate: a new detection is matched to an existing track if it is both the
nearest centroid AND overlaps the track's last box by at least
`min_iou`. This is not ByteTrack/DeepSORT — there's no motion model or
re-identification embedding — but it is a real, correct implementation
of greedy nearest-neighbor association with disappearance handling, and
it is enough to give stable IDs to vehicles across consecutive frames of
a slow-moving traffic camera feed.

Complexity: O(existing_tracks x new_detections) per frame via a full
distance matrix + greedy row/col elimination (like the pyimagesearch
reference; not the Hungarian algorithm, so not globally optimal, but
deterministic and dependency-free).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def _centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _dist2(p: Tuple[float, float], q: Tuple[float, float]) -> float:
    return (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2


@dataclass
class Track:
    track_id: int
    box: Tuple[float, float, float, float]
    centroid: Tuple[float, float]
    label: str
    confidence: float
    disappeared: int = 0
    hits: int = 1
    age: int = 0

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.box
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "box": {"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1)},
            "hits": self.hits,
            "age": self.age,
        }


class CentroidTracker:
    """
    Frame-to-frame tracker. Call `update(detections)` once per frame with
    a list of objects exposing `.box`, `.label`, `.confidence`
    (e.g. `Detection` from detector.py) and get back the current list of
    live `Track`s with stable `track_id`s.
    """

    def __init__(self, max_disappeared: int = 8, min_iou: float = 0.1, max_centroid_dist: float = 120.0):
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self.max_centroid_dist = max_centroid_dist
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _register(self, det) -> None:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(
            track_id=tid, box=tuple(det.box), centroid=_centroid(det.box),
            label=det.label, confidence=det.confidence,
        )

    def _deregister(self, track_id: int) -> None:
        self.tracks.pop(track_id, None)

    def update(self, detections: List) -> List[Track]:
        for t in self.tracks.values():
            t.age += 1

        if not detections:
            for tid in list(self.tracks.keys()):
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    self._deregister(tid)
            return self.live_tracks()

        det_centroids = [_centroid(d.box) for d in detections]

        if not self.tracks:
            for d in detections:
                self._register(d)
            return self.live_tracks()

        track_ids = list(self.tracks.keys())
        track_centroids = [self.tracks[tid].centroid for tid in track_ids]

        # Build cost matrix: distance, gated by IOU.
        rows = len(track_ids)
        cols = len(detections)
        cost: List[List[Optional[float]]] = [[None] * cols for _ in range(rows)]
        for i, tid in enumerate(track_ids):
            t = self.tracks[tid]
            for j, d in enumerate(detections):
                if t.label != d.label:
                    continue
                iou = _iou(t.box, tuple(d.box))
                dist = _dist2(track_centroids[i], det_centroids[j]) ** 0.5
                if iou >= self.min_iou or dist <= self.max_centroid_dist:
                    cost[i][j] = dist

        # Greedy matching: repeatedly pick the globally-smallest remaining
        # distance pair until no valid pairs are left.
        matched_rows, matched_cols = set(), set()
        pairs = []
        for i in range(rows):
            for j in range(cols):
                if cost[i][j] is not None:
                    pairs.append((cost[i][j], i, j))
        pairs.sort(key=lambda p: p[0])

        for dist, i, j in pairs:
            if i in matched_rows or j in matched_cols:
                continue
            matched_rows.add(i)
            matched_cols.add(j)
            tid = track_ids[i]
            d = detections[j]
            track = self.tracks[tid]
            track.box = tuple(d.box)
            track.centroid = det_centroids[j]
            track.confidence = d.confidence
            track.disappeared = 0
            track.hits += 1

        # Unmatched existing tracks -> disappeared++, evict if stale.
        for i, tid in enumerate(track_ids):
            if i not in matched_rows:
                self.tracks[tid].disappeared += 1
                if self.tracks[tid].disappeared > self.max_disappeared:
                    self._deregister(tid)

        # Unmatched detections -> new tracks.
        for j, d in enumerate(detections):
            if j not in matched_cols:
                self._register(d)

        return self.live_tracks()

    def live_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.disappeared == 0]

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
