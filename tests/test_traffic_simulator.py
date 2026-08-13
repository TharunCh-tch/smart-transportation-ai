"""Tests for the NumPy-based traffic simulator (app/ai/traffic_simulator.py)."""
import numpy as np

from app.ai.city_graph import EDGE_IDS
from app.ai.traffic_simulator import (
    generate_incidents,
    get_traffic_predictions,
    get_traffic_state,
    traffic_stats,
)


def test_traffic_state_covers_every_edge():
    state = get_traffic_state(hour=8.0)
    assert set(state.keys()) == set(EDGE_IDS)


def test_traffic_state_values_are_bounded_0_to_1():
    for hour in [0.0, 6.5, 8.0, 12.0, 18.0, 23.9]:
        state = get_traffic_state(hour=hour)
        for eid, val in state.items():
            assert 0.0 <= val <= 1.0, f"traffic value out of bounds for {eid} at hour={hour}: {val}"


def test_traffic_state_is_deterministic_for_same_hour():
    """Same hour (rounded to the seed's granularity) should reproduce the
    same traffic snapshot, so the frontend map doesn't flicker."""
    state_a = get_traffic_state(hour=8.0)
    state_b = get_traffic_state(hour=8.0)
    assert state_a == state_b


def test_rush_hour_is_more_congested_than_late_night_on_average():
    morning_rush = get_traffic_state(hour=8.0)
    late_night = get_traffic_state(hour=2.0)
    assert np.mean(list(morning_rush.values())) > np.mean(list(late_night.values()))


def test_traffic_predictions_return_requested_hour_count():
    predictions = get_traffic_predictions(hours=12)
    assert len(predictions) == 12
    for p in predictions:
        assert 0 <= p["hour"] <= 23
        assert 0.0 <= p["avg_traffic"] <= 1.0
        assert p["congested_roads"] >= 0


def test_generate_incidents_returns_two_to_four_incidents():
    traffic = get_traffic_state(hour=18.0)
    incidents = generate_incidents(traffic, seed=42)
    assert 2 <= len(incidents) <= 4
    for inc in incidents:
        assert inc["severity"] in {"low", "moderate", "high"}
        assert inc["edge_id"] in EDGE_IDS


def test_generate_incidents_is_seedable_and_deterministic():
    traffic = get_traffic_state(hour=18.0)
    incidents_a = generate_incidents(traffic, seed=7)
    incidents_b = generate_incidents(traffic, seed=7)
    assert incidents_a == incidents_b


def test_traffic_stats_shape_and_ranges():
    traffic = get_traffic_state(hour=8.0)
    stats = traffic_stats(traffic)
    assert set(stats.keys()) == {"avg_traffic", "congested_count", "free_flow_count", "avg_speed_kmh"}
    assert 0.0 <= stats["avg_traffic"] <= 1.0
    assert stats["congested_count"] + stats["free_flow_count"] <= len(EDGE_IDS)
    assert stats["avg_speed_kmh"] > 0
