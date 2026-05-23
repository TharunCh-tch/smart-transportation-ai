"""NYC road network: 20 intersections, ~40 edges."""
import math
from typing import Dict, List, Tuple

NODES: Dict[str, Dict] = {
    "times_square":    {"lat": 40.7580, "lng": -73.9855, "name": "Times Square"},
    "grand_central":   {"lat": 40.7527, "lng": -73.9772, "name": "Grand Central"},
    "penn_station":    {"lat": 40.7506, "lng": -73.9971, "name": "Penn Station"},
    "empire_state":    {"lat": 40.7484, "lng": -73.9857, "name": "Empire State Bldg"},
    "union_square":    {"lat": 40.7359, "lng": -73.9911, "name": "Union Square"},
    "wall_street":     {"lat": 40.7074, "lng": -74.0113, "name": "Wall Street"},
    "world_trade":     {"lat": 40.7127, "lng": -74.0134, "name": "World Trade Center"},
    "brooklyn_bridge": {"lat": 40.7061, "lng": -73.9969, "name": "Brooklyn Bridge"},
    "central_park_s":  {"lat": 40.7644, "lng": -73.9730, "name": "Central Park South"},
    "upper_east":      {"lat": 40.7736, "lng": -73.9566, "name": "Upper East Side"},
    "lincoln_center":  {"lat": 40.7725, "lng": -73.9836, "name": "Lincoln Center"},
    "columbus_circle": {"lat": 40.7682, "lng": -73.9819, "name": "Columbus Circle"},
    "rockefeller":     {"lat": 40.7587, "lng": -73.9787, "name": "Rockefeller Center"},
    "madison_sq":      {"lat": 40.7422, "lng": -73.9878, "name": "Madison Square"},
    "flatiron":        {"lat": 40.7411, "lng": -73.9897, "name": "Flatiron Building"},
    "chelsea":         {"lat": 40.7465, "lng": -74.0014, "name": "Chelsea"},
    "hells_kitchen":   {"lat": 40.7651, "lng": -73.9945, "name": "Hell's Kitchen"},
    "east_village":    {"lat": 40.7265, "lng": -73.9815, "name": "East Village"},
    "soho":            {"lat": 40.7233, "lng": -74.0030, "name": "SoHo"},
    "chinatown":       {"lat": 40.7157, "lng": -73.9971, "name": "Chinatown"},
}

# (source, target, distance_km, base_time_min)
_EDGE_DATA: List[Tuple[str, str, float, float]] = [
    ("times_square",    "grand_central",   0.7,  4.0),
    ("times_square",    "rockefeller",     0.4,  3.0),
    ("times_square",    "hells_kitchen",   0.5,  3.5),
    ("times_square",    "empire_state",    0.5,  4.0),
    ("times_square",    "columbus_circle", 0.8,  5.0),
    ("grand_central",   "upper_east",      1.0,  6.0),
    ("grand_central",   "rockefeller",     0.6,  4.0),
    ("grand_central",   "empire_state",    0.5,  4.0),
    ("grand_central",   "madison_sq",      1.0,  7.0),
    ("penn_station",    "empire_state",    0.3,  2.5),
    ("penn_station",    "chelsea",         0.5,  4.0),
    ("penn_station",    "madison_sq",      0.6,  5.0),
    ("penn_station",    "times_square",    0.7,  5.0),
    ("empire_state",    "madison_sq",      0.7,  5.0),
    ("empire_state",    "flatiron",        0.8,  6.0),
    ("union_square",    "flatiron",        0.3,  2.0),
    ("union_square",    "east_village",    0.5,  4.0),
    ("union_square",    "madison_sq",      0.6,  5.0),
    ("wall_street",     "world_trade",     0.3,  2.0),
    ("wall_street",     "brooklyn_bridge", 0.4,  3.0),
    ("wall_street",     "chinatown",       0.5,  4.0),
    ("world_trade",     "soho",            0.7,  5.0),
    ("world_trade",     "chinatown",       0.5,  4.0),
    ("brooklyn_bridge", "chinatown",       0.4,  3.0),
    ("brooklyn_bridge", "east_village",    0.8,  6.0),
    ("rockefeller",     "central_park_s",  0.8,  5.0),
    ("rockefeller",     "times_square",    0.4,  3.0),
    ("columbus_circle", "lincoln_center",  0.5,  3.0),
    ("columbus_circle", "central_park_s",  0.4,  3.0),
    ("central_park_s",  "upper_east",      1.0,  7.0),
    ("lincoln_center",  "upper_east",      1.2,  8.0),
    ("lincoln_center",  "hells_kitchen",   0.4,  3.0),
    ("hells_kitchen",   "chelsea",         0.6,  4.0),
    ("chelsea",         "soho",            1.0,  7.0),
    ("chelsea",         "penn_station",    0.5,  4.0),
    ("soho",            "east_village",    0.5,  4.0),
    ("soho",            "chinatown",       0.5,  4.0),
    ("east_village",    "chinatown",       0.5,  4.0),
    ("east_village",    "union_square",    0.5,  4.0),
    ("flatiron",        "chelsea",         0.5,  4.0),
    ("madison_sq",      "flatiron",        0.2,  2.0),
    ("upper_east",      "central_park_s",  1.0,  7.0),
]


def haversine(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def build_adjacency() -> Dict[str, Dict[str, Dict]]:
    """Return adjacency dict: adj[src][dst] = {distance_km, base_time_min}"""
    adj: Dict[str, Dict] = {n: {} for n in NODES}
    for src, dst, dist, t in _EDGE_DATA:
        adj[src][dst] = {"distance_km": dist, "base_time_min": t}
        adj[dst][src] = {"distance_km": dist, "base_time_min": t}
    return adj


def edge_id(a: str, b: str) -> str:
    return "__".join(sorted([a, b]))


def all_edges() -> List[Dict]:
    """Return deduplicated edge list."""
    seen = set()
    result = []
    for src, dst, dist, t in _EDGE_DATA:
        eid = edge_id(src, dst)
        if eid not in seen:
            seen.add(eid)
            mid_lat = (NODES[src]["lat"] + NODES[dst]["lat"]) / 2
            mid_lng = (NODES[src]["lng"] + NODES[dst]["lng"]) / 2
            result.append({
                "id": eid, "source": src, "target": dst,
                "distance_km": dist, "base_time_min": t,
                "mid_lat": mid_lat, "mid_lng": mid_lng,
            })
    return result


ADJACENCY = build_adjacency()
ALL_EDGES = all_edges()
EDGE_IDS = [e["id"] for e in ALL_EDGES]
