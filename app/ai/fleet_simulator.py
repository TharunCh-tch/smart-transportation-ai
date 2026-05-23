"""
Simulates 6 vehicles moving along fixed routes in real time.
Position is computed deterministically from wall-clock time so
every API call returns a consistent snapshot.
"""
import math
import time
from typing import Dict, List

from app.ai.city_graph import NODES, ADJACENCY, haversine

FLEET: List[Dict] = [
    {"id": "V001", "name": "Delivery Van #1",   "type": "delivery",   "color": "#3498db",
     "route": ["times_square","penn_station","chelsea","soho","world_trade","wall_street","chinatown","brooklyn_bridge"]},
    {"id": "V002", "name": "City Bus M15",       "type": "bus",        "color": "#e74c3c",
     "route": ["upper_east","central_park_s","columbus_circle","lincoln_center","hells_kitchen","times_square","rockefeller","grand_central"]},
    {"id": "V003", "name": "Taxi #7829",          "type": "taxi",       "color": "#f1c40f",
     "route": ["grand_central","rockefeller","times_square","empire_state","madison_sq","flatiron","union_square","east_village"]},
    {"id": "V004", "name": "Police Unit 14",      "type": "emergency",  "color": "#2ecc71",
     "route": ["brooklyn_bridge","chinatown","east_village","union_square","flatiron","chelsea","penn_station","madison_sq"]},
    {"id": "V005", "name": "Freight Truck #3",    "type": "delivery",   "color": "#9b59b6",
     "route": ["soho","chinatown","world_trade","wall_street","brooklyn_bridge","east_village","union_square","soho"]},
    {"id": "V006", "name": "Shuttle Bus A",       "type": "bus",        "color": "#1abc9c",
     "route": ["lincoln_center","columbus_circle","central_park_s","upper_east","grand_central","rockefeller","times_square","lincoln_center"]},
]

# Speed per vehicle type (km/h)
_SPEED = {"delivery": 28, "bus": 22, "taxi": 35, "emergency": 50}


def _route_segments(route: List[str]):
    """Return list of (src, dst, distance_km) for a circular route."""
    segs = []
    for i in range(len(route) - 1):
        src, dst = route[i], route[i + 1]
        dist = ADJACENCY.get(src, {}).get(dst, {}).get("distance_km")
        if dist is None:
            # Straight-line fallback
            n1, n2 = NODES[src], NODES[dst]
            dist = haversine(n1["lat"], n1["lng"], n2["lat"], n2["lng"])
        segs.append({"src": src, "dst": dst, "dist": dist})
    return segs


def _interp(lat1, lng1, lat2, lng2, t):
    """Linear interpolation between two coords at fraction t."""
    return lat1 + (lat2 - lat1) * t, lng1 + (lng2 - lng1) * t


def get_vehicle_positions(traffic: Dict[str, float] = None) -> List[Dict]:
    """
    Return current simulated positions for all fleet vehicles.
    Each vehicle progresses continuously along its route.
    """
    from app.ai.city_graph import edge_id as eid_fn

    now = time.time()
    result = []

    for v in FLEET:
        speed_kmh = _SPEED.get(v["type"], 30)
        # Slow down if traffic is heavy on current segment
        segs = _route_segments(v["route"])
        total_dist = sum(s["dist"] for s in segs)
        total_time_s = (total_dist / speed_kmh) * 3600

        # Phase offset so vehicles start spread out
        phase = hash(v["id"]) % 1000 / 1000.0
        elapsed = (now + phase * total_time_s) % total_time_s

        # Find which segment the vehicle is on
        acc = 0.0
        seg_time = 0.0
        current_seg = segs[0]
        seg_frac = 0.0
        for seg in segs:
            seg_time_s = (seg["dist"] / speed_kmh) * 3600
            if acc + seg_time_s >= elapsed:
                seg_frac = (elapsed - acc) / seg_time_s
                current_seg = seg
                break
            acc += seg_time_s

        # Interpolate position
        n1 = NODES[current_seg["src"]]
        n2 = NODES[current_seg["dst"]]
        lat, lng = _interp(n1["lat"], n1["lng"], n2["lat"], n2["lng"], seg_frac)

        # Actual speed with traffic
        t_level = (traffic or {}).get(eid_fn(current_seg["src"], current_seg["dst"]), 0.3)
        actual_speed = speed_kmh / (1.0 + t_level * 1.5)

        result.append({
            "id": v["id"],
            "name": v["name"],
            "type": v["type"],
            "color": v["color"],
            "status": "moving",
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "speed_kmh": round(actual_speed, 1),
            "current_edge": eid_fn(current_seg["src"], current_seg["dst"]),
            "destination": NODES[current_seg["dst"]]["name"],
        })
    return result
