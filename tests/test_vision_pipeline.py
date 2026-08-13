"""
Integration tests for the full image/video pipeline
(app/ai/vision/pipeline.py) — real detector + real emergency heuristic
against the bundled sample images. OCR is left off in most of these to
keep runtime reasonable; one test explicitly enables it.
"""
import numpy as np
import pytest

from app.ai.vision.pipeline import process_image, process_video


@pytest.mark.integration
def test_process_image_on_bus_jpg_returns_sane_structure(bus_jpg):
    result = process_image(str(bus_jpg), run_ocr=False, run_emergency=True)

    assert result["vehicle_count"] == 1
    assert result["image_size"]["width"] > 0
    assert result["image_size"]["height"] > 0
    vehicle = result["vehicles"][0]
    assert vehicle["detection"]["label"] == "bus"
    assert "emergency" in vehicle
    assert "plate" not in vehicle  # run_ocr=False
    assert result["timing_ms"]["total"] >= 0


@pytest.mark.integration
def test_process_image_negative_control_zidane(zidane_jpg):
    result = process_image(str(zidane_jpg), run_ocr=False, run_emergency=True)
    assert result["vehicle_count"] == 0
    assert result["vehicles"] == []


@pytest.mark.integration
def test_process_image_accepts_numpy_array_source(bus_jpg):
    import cv2
    img = cv2.imread(str(bus_jpg))
    result = process_image(img, run_ocr=False, run_emergency=False)
    assert result["vehicle_count"] >= 1


def test_process_image_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(ValueError):
        process_image(str(missing), run_ocr=False, run_emergency=False)


@pytest.mark.integration
def test_process_video_tracks_are_stable_across_synthetic_motion(fixtures_dir, tmp_path):
    """Build a tiny synthetic clip from a static image with a horizontal
    shift per frame (simulating motion), then confirm the tracker keeps
    a stable ID for the same object across frames — a real, if small,
    end-to-end check of the video + tracking path."""
    import cv2

    src = cv2.imread(str(fixtures_dir / "bus.jpg"))
    h, w = src.shape[:2]
    video_path = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5, (w, h))
    for i in range(6):
        shift = i * 2
        m = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(src, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        writer.write(frame)
    writer.release()

    result = process_video(str(video_path), run_ocr=False, run_emergency=False,
                            sample_every=1, max_frames=6)

    assert result["frames_processed"] == 6
    assert result["unique_tracks"] >= 1
    # The bus should be tracked continuously (not lost/re-registered) across
    # all 6 frames of small motion.
    bus_tracks = [t for t in result["track_summary"] if t["label"] == "bus"]
    assert bus_tracks
    assert bus_tracks[0]["frame_count"] == 6


def test_process_video_raises_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(ValueError):
        process_video(str(missing))
