"""
FastAPI endpoint tests. Uses TestClient (httpx under the hood) against
the real app, with a throwaway SQLite DB (see conftest.py). Heavy CV
calls are mocked in most tests for speed; a couple of @pytest.mark.integration
tests hit the real detection pipeline through the actual HTTP endpoint.
"""
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── Core app / existing transport endpoints ─────────────────────────────

def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_list_nodes_returns_twenty_intersections():
    resp = client.get("/api/nodes")
    assert resp.status_code == 200
    assert len(resp.json()) == 20


def test_traffic_edges_have_valid_levels():
    resp = client.get("/api/traffic/edges")
    assert resp.status_code == 200
    edges = resp.json()
    assert len(edges) > 0
    for e in edges:
        assert 0.0 <= e["traffic_level"] <= 1.0


def test_optimize_route_happy_path():
    resp = client.post("/api/routes/optimize", json={
        "origin": "times_square", "destination": "wall_street",
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["path"][0] == "times_square"
    assert body["path"][-1] == "wall_street"


def test_optimize_route_unknown_node_is_400():
    resp = client.post("/api/routes/optimize", json={
        "origin": "not_a_node", "destination": "wall_street",
    })
    assert resp.status_code == 400


# ── Vision endpoints ─────────────────────────────────────────────────────

def test_vision_samples_lists_bundled_images():
    resp = client.get("/api/vision/samples")
    assert resp.status_code == 200
    names = {s["name"] for s in resp.json()}
    assert "bus.jpg" in names


def test_vision_health_reports_dependency_availability():
    resp = client.get("/api/vision/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "detector_loaded" in body
    assert "ultralytics_available" in body


def test_detect_image_without_file_or_sample_is_400():
    resp = client.post("/api/vision/detect/image")
    assert resp.status_code == 400


def test_detect_image_unknown_sample_is_404():
    resp = client.post("/api/vision/detect/image?sample=not_a_real_sample.jpg")
    assert resp.status_code == 404


def test_detect_image_with_mocked_pipeline_returns_result():
    fake_result = {
        "image_size": {"width": 100, "height": 100},
        "vehicle_count": 1,
        "vehicles": [{
            "detection": {"class_id": 2, "label": "car", "confidence": 0.9,
                          "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            "emergency": {"likely_emergency": False, "emergency_score": 0.01,
                          "red_blue_fraction": 0.0, "contrast_fraction": 0.0,
                          "method": "heuristic-color-cue (not a trained classifier)"},
        }],
        "timing_ms": {"detect": 1.0, "total": 1.0},
    }
    with patch("app.ai.vision.pipeline.process_image", return_value=fake_result) as mock_process:
        resp = client.post("/api/vision/detect/image?sample=bus.jpg&run_ocr=false")
    assert resp.status_code == 200
    body = resp.json()
    assert body["vehicle_count"] == 1
    assert body["incidents_logged"] == 0
    mock_process.assert_called_once()


def test_detect_image_logs_cv_incident_when_emergency_flagged_and_location_given():
    fake_result = {
        "image_size": {"width": 100, "height": 100},
        "vehicle_count": 1,
        "vehicles": [{
            "detection": {"class_id": 2, "label": "car", "confidence": 0.9,
                          "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}},
            "emergency": {"likely_emergency": True, "emergency_score": 0.5,
                          "red_blue_fraction": 0.6, "contrast_fraction": 0.4,
                          "method": "heuristic-color-cue (not a trained classifier)"},
        }],
        "timing_ms": {"detect": 1.0, "total": 1.0},
    }
    with patch("app.ai.vision.pipeline.process_image", return_value=fake_result):
        resp = client.post(
            "/api/vision/detect/image"
            "?sample=bus.jpg&run_ocr=false&lat=40.7580&lng=-73.9855&edge_id=times_square__grand_central"
        )
    assert resp.status_code == 200
    assert resp.json()["incidents_logged"] == 1

    incidents_resp = client.get("/api/vision/incidents")
    assert incidents_resp.status_code == 200
    incidents = incidents_resp.json()
    assert any(i["source"] == "cv-detection" for i in incidents)


def test_detect_image_rejects_oversized_or_wrong_type_gracefully():
    # Wrong content type should 415, not 500.
    resp = client.post(
        "/api/vision/detect/image",
        files={"file": ("not_an_image.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 415


@pytest.mark.integration
def test_detect_image_real_inference_on_bundled_sample():
    """Full HTTP round-trip with the real detector against a bundled
    sample image — no mocking."""
    resp = client.post("/api/vision/detect/image?sample=bus.jpg&run_ocr=false&run_emergency=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["vehicle_count"] >= 1
    assert any(v["detection"]["label"] == "bus" for v in body["vehicles"])
