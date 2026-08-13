"""Tests for the A* route optimizer (app/ai/route_optimizer.py)."""
import math

import pytest

from app.ai.city_graph import ADJACENCY, NODES, edge_id
from app.ai.route_optimizer import astar, baseline_time


def test_astar_finds_a_path_between_connected_nodes():
    path, travel_time, distance = astar("times_square", "wall_street", traffic={})
    assert path is not None
    assert path[0] == "times_square"
    assert path[-1] == "wall_street"
    assert travel_time > 0
    assert distance > 0


def test_astar_path_is_contiguous_in_the_graph():
    """Every consecutive pair in the returned path must be a real edge."""
    path, _, _ = astar("times_square", "chinatown", traffic={})
    assert path is not None
    for i in range(len(path) - 1):
        assert path[i + 1] in ADJACENCY[path[i]], f"No edge {path[i]} -> {path[i+1]}"


def test_astar_same_origin_and_destination_trivial_path():
    path, travel_time, distance = astar("times_square", "times_square", traffic={})
    assert path == ["times_square"]
    assert travel_time == 0.0
    assert distance == 0.0


def test_astar_unknown_node_raises_keyerror():
    # astar() itself trusts its inputs are valid node ids (the API layer
    # validates this before calling it — see app/api/routes/transport.py).
    with pytest.raises(KeyError):
        astar("not_a_real_node", "wall_street", traffic={})


def test_higher_traffic_never_produces_a_faster_or_equal_route_time():
    """Congestion should only ever slow the optimal route down (or leave
    it unchanged if the congested edges aren't on the shortest path)."""
    origin, destination = "times_square", "union_square"
    _, free_flow_time, _ = astar(origin, destination, traffic={})

    heavy_traffic = {eid: 0.95 for eid in _all_edge_ids()}
    _, congested_time, _ = astar(origin, destination, traffic=heavy_traffic)

    assert congested_time >= free_flow_time


def test_astar_matches_dijkstra_baseline_on_empty_traffic():
    """With no traffic dict, baseline_time() and astar() should agree —
    both fall back to base_time_min-only shortest path."""
    origin, destination = "grand_central", "soho"
    _, astar_time, _ = astar(origin, destination, traffic={})
    baseline = baseline_time(origin, destination)
    assert math.isclose(astar_time, baseline, rel_tol=1e-9)


def test_astar_no_path_between_disconnected_or_missing_edges_returns_none():
    # All nodes in this graph are connected, so instead we validate the
    # "no path found" contract directly against a node with no neighbors.
    ADJACENCY["_isolated_test_node_"] = {}
    NODES["_isolated_test_node_"] = {"lat": 0.0, "lng": 0.0, "name": "Isolated"}
    try:
        path, travel_time, distance = astar("_isolated_test_node_", "times_square", traffic={})
        assert path is None
        assert travel_time == math.inf
        assert distance == 0.0
    finally:
        del ADJACENCY["_isolated_test_node_"]
        del NODES["_isolated_test_node_"]


def _all_edge_ids():
    ids = set()
    for src, neighbors in ADJACENCY.items():
        for dst in neighbors:
            ids.add(edge_id(src, dst))
    return ids
