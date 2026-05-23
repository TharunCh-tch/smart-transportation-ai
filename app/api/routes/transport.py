import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.ai.city_graph import ALL_EDGES, NODES
from app.ai.fleet_simulator import get_vehicle_positions
from app.ai.route_optimizer import astar, baseline_time
from app.ai.traffic_simulator import (
    generate_incidents,
    get_traffic_predictions,
    get_traffic_state,
    traffic_stats,
)
from app.core.database import get_db
from app.models.db_models import SavedRoute
from app.models.schemas import (
    AnalyticsSummary,
    EdgeOut,
    IncidentOut,
    NodeOut,
    RouteRequest,
    RouteResult,
    VehicleOut,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# ── Network ───────────────────────────────────────────────────────────────────

@router.get("/nodes", response_model=List[NodeOut])
def list_nodes():
    return [NodeOut(id=k, name=v["name"], lat=v["lat"], lng=v["lng"]) for k, v in NODES.items()]


@router.get("/traffic/edges", response_model=List[EdgeOut])
def traffic_edges():
    """All edges with current traffic levels and computed travel times."""
    traffic = get_traffic_state()
    result = []
    for e in ALL_EDGES:
        t = traffic.get(e["id"], 0.3)
        factor = 1.0 + t * 2.0
        result.append(EdgeOut(
            id=e["id"], source=e["source"], target=e["target"],
            distance_km=e["distance_km"], base_time_min=e["base_time_min"],
            traffic_level=round(t, 3),
            travel_time_min=round(e["base_time_min"] * factor, 2),
        ))
    return result


# ── Route optimization ─────────────────────────────────────────────────────────

@router.post("/routes/optimize", response_model=RouteResult)
def optimize_route(req: RouteRequest, db: Session = Depends(get_db)):
    if req.origin not in NODES:
        raise HTTPException(400, f"Unknown node: {req.origin}")
    if req.destination not in NODES:
        raise HTTPException(400, f"Unknown node: {req.destination}")
    if req.origin == req.destination:
        raise HTTPException(400, "Origin and destination must differ")

    traffic = get_traffic_state()
    path, travel_time, distance = astar(req.origin, req.destination, traffic)

    if path is None:
        raise HTTPException(404, "No route found between these nodes")

    base_t = baseline_time(req.origin, req.destination)
    saved = max(0.0, round(travel_time - base_t, 2))   # AI saves vs naive

    # Persist
    db.add(SavedRoute(
        origin_id=req.origin, destination_id=req.destination,
        origin_name=NODES[req.origin]["name"], destination_name=NODES[req.destination]["name"],
        path_json=str(path), distance_km=round(distance, 2),
        travel_time_min=round(travel_time, 2), time_saved_min=saved,
    ))
    db.commit()

    from app.ai.city_graph import edge_id
    edge_traffics = [traffic.get(edge_id(path[i], path[i + 1]), 0.3) for i in range(len(path) - 1)]
    avg_t = sum(edge_traffics) / len(edge_traffics) if edge_traffics else 0.0

    return RouteResult(
        origin=req.origin, destination=req.destination,
        origin_name=NODES[req.origin]["name"], destination_name=NODES[req.destination]["name"],
        path=path,
        path_names=[NODES[n]["name"] for n in path],
        path_coords=[[NODES[n]["lat"], NODES[n]["lng"]] for n in path],
        distance_km=round(distance, 2),
        travel_time_min=round(travel_time, 2),
        baseline_time_min=round(base_t, 2),
        time_saved_min=saved,
        avg_traffic=round(avg_t, 3),
    )


@router.get("/routes/history")
def route_history(limit: int = 10, db: Session = Depends(get_db)):
    return db.query(SavedRoute).order_by(SavedRoute.created_at.desc()).limit(limit).all()


# ── Traffic & incidents ────────────────────────────────────────────────────────

@router.get("/traffic/incidents", response_model=List[IncidentOut])
def active_incidents():
    traffic = get_traffic_state()
    return [IncidentOut(**i) for i in generate_incidents(traffic)]


@router.get("/traffic/predict")
def predict_traffic():
    return get_traffic_predictions(12)


# ── Fleet ────────────────────────────────────────────────────────────────────

@router.get("/fleet", response_model=List[VehicleOut])
def fleet_positions():
    traffic = get_traffic_state()
    return [VehicleOut(**v) for v in get_vehicle_positions(traffic)]


# ── Analytics ─────────────────────────────────────────────────────────────────

@router.get("/analytics", response_model=AnalyticsSummary)
def analytics():
    traffic = get_traffic_state()
    stats = traffic_stats(traffic)
    incidents = generate_incidents(traffic)
    predictions = get_traffic_predictions(12)
    return AnalyticsSummary(
        total_nodes=len(NODES),
        total_edges=len(ALL_EDGES),
        active_incidents=len(incidents),
        active_vehicles=6,
        predictions=predictions,
        **stats,
    )
