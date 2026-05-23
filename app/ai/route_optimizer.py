"""A* route optimizer with traffic-weighted travel times."""
import heapq
import math
from typing import Dict, List, Optional, Tuple

from app.ai.city_graph import ADJACENCY, NODES, ALL_EDGES, haversine


def _heuristic(node_id: str, goal_id: str) -> float:
    """Admissible heuristic: straight-line distance converted to min at 30 km/h."""
    n = NODES[node_id]
    g = NODES[goal_id]
    km = haversine(n["lat"], n["lng"], g["lat"], g["lng"])
    return km * 2.0   # 2 min/km ≈ 30 km/h baseline


def astar(
    origin: str,
    destination: str,
    traffic: Dict[str, float],   # edge_id → traffic 0–1
) -> Tuple[Optional[List[str]], float, float]:
    """
    Returns (path_node_ids, total_time_min, total_distance_km).
    Traffic factor multiplies base travel time: congested road = slower.
    """
    def weighted_time(src: str, dst: str) -> float:
        from app.ai.city_graph import edge_id
        data = ADJACENCY[src][dst]
        t_level = traffic.get(edge_id(src, dst), 0.3)
        # Traffic factor: free-flow=1.0x, fully congested=3.0x
        factor = 1.0 + t_level * 2.0
        return data["base_time_min"] * factor

    open_heap: List[Tuple[float, float, str, List[str]]] = []
    heapq.heappush(open_heap, (_heuristic(origin, destination), 0.0, origin, [origin]))
    best: Dict[str, float] = {}

    while open_heap:
        f, g, node, path = heapq.heappop(open_heap)
        if node == destination:
            dist = sum(
                ADJACENCY[path[i]][path[i + 1]]["distance_km"]
                for i in range(len(path) - 1)
            )
            return path, g, dist

        if best.get(node, math.inf) <= g:
            continue
        best[node] = g

        for neighbor, data in ADJACENCY.get(node, {}).items():
            ng = g + weighted_time(node, neighbor)
            nf = ng + _heuristic(neighbor, destination)
            heapq.heappush(open_heap, (nf, ng, neighbor, path + [neighbor]))

    return None, math.inf, 0.0


def baseline_time(origin: str, destination: str) -> float:
    """Shortest time ignoring traffic (Dijkstra with base times)."""
    path, t, _ = astar(origin, destination, traffic={})
    return t
