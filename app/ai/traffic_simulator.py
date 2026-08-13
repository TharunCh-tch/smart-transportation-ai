"""
Deterministic traffic simulation driven by time-of-day + seeded noise.
Uses numpy for vectorised traffic prediction over a 12-hour window.
"""
import math
import random
import time
from datetime import datetime
from typing import Dict, List

import numpy as np

from app.ai.city_graph import ALL_EDGES, EDGE_IDS

# ── Time-of-day traffic pattern ───────────────────────────────────────────────

def _tod_base(hour: float) -> float:
    """Return baseline congestion level 0–1 for a given hour (0–24)."""
    t = hour / 24.0
    morning = math.exp(-((t - 8 / 24) ** 2) / (2 * (1.5 / 24) ** 2))
    evening = math.exp(-((t - 18 / 24) ** 2) / (2 * (1.5 / 24) ** 2))
    night   = math.exp(-((t - 2 / 24) ** 2) / (2 * (2 / 24) ** 2))
    return max(0.05, min(0.97, 0.15 + 0.7 * (morning + evening) / 2 - 0.1 * night))


# ── Per-edge noise seeds (stable per session) ─────────────────────────────────

_EDGE_SEEDS = {eid: hash(eid) % 1000 for eid in EDGE_IDS}


def get_traffic_state(hour: float = None) -> Dict[str, float]:
    """
    Return {edge_id: traffic_level} for the given hour.
    Deterministic for same hour+minute so the map looks stable.
    """
    if hour is None:
        now = datetime.now()
        hour = now.hour + now.minute / 60.0

    base = _tod_base(hour)
    rng = np.random.default_rng(seed=int(hour * 12) % 10000)
    noise = rng.normal(0, 0.12, len(EDGE_IDS))

    result = {}
    for i, eid in enumerate(EDGE_IDS):
        edge_noise = ((_EDGE_SEEDS[eid] % 100) / 100.0 - 0.5) * 0.18
        val = base + noise[i] + edge_noise
        result[eid] = float(np.clip(val, 0.05, 0.97))
    return result


def get_traffic_predictions(hours: int = 12) -> List[Dict]:
    """Predict average traffic for the next `hours` hours."""
    now = datetime.now()
    predictions = []
    for delta in range(hours):
        h = (now.hour + delta) % 24
        base = _tod_base(h)
        congested = sum(1 for eid in EDGE_IDS
                        if base + ((_EDGE_SEEDS[eid] % 100) / 100.0 - 0.5) * 0.18 > 0.65)
        label = f"{h:02d}:00"
        predictions.append({
            "hour": h, "label": label,
            "avg_traffic": round(base, 3),
            "congested_roads": congested,
        })
    return predictions


# ── Incident simulation ───────────────────────────────────────────────────────

_INCIDENT_DESCRIPTIONS = [
    "Multi-vehicle collision", "Stalled vehicle blocking lane",
    "Road works – lane closure", "Water main break – road flooded",
    "Pedestrian incident", "Signal malfunction", "Debris on road",
    "Emergency vehicle staging", "Protest – partial road closure",
]

_SEVERITIES = ["low", "moderate", "high"]


def generate_incidents(traffic: Dict[str, float], seed: int = None) -> List[Dict]:
    """Return 2–4 simulated incidents on the most congested edges."""
    rng = random.Random(seed if seed is not None else int(time.time() / 300))
    sorted_edges = sorted(traffic.items(), key=lambda x: x[1], reverse=True)
    # Pick from top-10 congested edges
    candidates = [eid for eid, _ in sorted_edges[:10]]
    n = rng.randint(2, 4)
    chosen = rng.sample(candidates, min(n, len(candidates)))

    incidents = []
    for i, eid in enumerate(chosen):
        edge = next((e for e in ALL_EDGES if e["id"] == eid), None)
        if not edge:
            continue
        lat = (edge["mid_lat"] + rng.uniform(-0.003, 0.003))
        lng = (edge["mid_lng"] + rng.uniform(-0.003, 0.003))
        sev_idx = 2 if traffic[eid] > 0.80 else (1 if traffic[eid] > 0.60 else 0)
        incidents.append({
            "id": i + 1,
            "edge_id": eid,
            "severity": _SEVERITIES[sev_idx],
            "description": rng.choice(_INCIDENT_DESCRIPTIONS),
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "active": True,
        })
    return incidents


# ── Analytics helpers ─────────────────────────────────────────────────────────

def traffic_stats(traffic: Dict[str, float]) -> Dict:
    values = list(traffic.values())
    arr = np.array(values)
    return {
        "avg_traffic": float(np.mean(arr)),
        "congested_count": int(np.sum(arr > 0.65)),
        "free_flow_count": int(np.sum(arr < 0.35)),
        "avg_speed_kmh": round(float(60 / (1 + np.mean(arr) * 2)), 1),
    }
