"""
Tests for the centroid tracker (app/ai/vision/tracker.py). Pure logic,
no model weights involved — should run in milliseconds.
"""
from dataclasses import dataclass

from app.ai.vision.tracker import CentroidTracker, _iou


@dataclass
class FakeDetection:
    label: str
    confidence: float
    box: tuple


def test_iou_identical_boxes_is_one():
    box = (10, 10, 50, 50)
    assert _iou(box, box) == 1.0


def test_iou_disjoint_boxes_is_zero():
    assert _iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0


def test_new_detections_are_registered_with_incrementing_ids():
    tracker = CentroidTracker()
    dets = [FakeDetection("car", 0.9, (0, 0, 10, 10)), FakeDetection("car", 0.9, (100, 0, 110, 10))]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    assert {t.track_id for t in tracks} == {1, 2}


def test_track_id_persists_across_frames_for_slow_motion():
    """A vehicle moving a few pixels per frame should keep the same ID —
    this is the whole point of a tracker vs. re-detecting from scratch."""
    tracker = CentroidTracker()
    track_ids_over_time = []
    for step in range(10):
        x = step * 4  # small per-frame motion
        dets = [FakeDetection("car", 0.9, (x, 0, x + 20, 20))]
        tracks = tracker.update(dets)
        assert len(tracks) == 1
        track_ids_over_time.append(tracks[0].track_id)
    assert len(set(track_ids_over_time)) == 1, "track ID should be stable across frames"


def test_two_vehicles_moving_in_parallel_do_not_swap_ids():
    tracker = CentroidTracker()
    ids_a, ids_b = [], []
    for step in range(8):
        det_a = FakeDetection("car", 0.9, (step * 5, 0, step * 5 + 20, 20))
        det_b = FakeDetection("car", 0.9, (step * 5, 200, step * 5 + 20, 220))
        tracks = tracker.update([det_a, det_b])
        assert len(tracks) == 2
        by_y = sorted(tracks, key=lambda t: t.box[1])
        ids_a.append(by_y[0].track_id)
        ids_b.append(by_y[1].track_id)
    assert len(set(ids_a)) == 1
    assert len(set(ids_b)) == 1
    assert ids_a[0] != ids_b[0]


def test_track_is_dropped_after_max_disappeared_frames():
    tracker = CentroidTracker(max_disappeared=2)
    tracker.update([FakeDetection("car", 0.9, (0, 0, 10, 10))])
    assert len(tracker.tracks) == 1

    tracker.update([])  # frame 2: no detections
    tracker.update([])  # frame 3: still none
    assert len(tracker.tracks) == 1  # not yet evicted (disappeared == 2, threshold is > 2)

    tracker.update([])  # frame 4: disappeared == 3 > max_disappeared(2)
    assert len(tracker.tracks) == 0


def test_different_labels_do_not_get_matched_to_the_same_track():
    tracker = CentroidTracker()
    tracker.update([FakeDetection("car", 0.9, (0, 0, 20, 20))])
    tracks = tracker.update([FakeDetection("truck", 0.9, (2, 2, 22, 22))])
    # A truck detection appearing where the car was should NOT reuse the
    # car's track id — it should register as a new track.
    assert len(tracker.tracks) == 2
    labels = {t.label for t in tracks} | {tracker.tracks[tid].label for tid in tracker.tracks}
    assert "car" in labels and "truck" in labels


def test_reset_clears_all_tracks_and_id_counter():
    tracker = CentroidTracker()
    tracker.update([FakeDetection("car", 0.9, (0, 0, 10, 10))])
    tracker.reset()
    assert tracker.tracks == {}
    tracks = tracker.update([FakeDetection("car", 0.9, (0, 0, 10, 10))])
    assert tracks[0].track_id == 1
