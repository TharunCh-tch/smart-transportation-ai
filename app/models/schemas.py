from typing import List, Optional

from pydantic import BaseModel


class NodeOut(BaseModel):
    id: str
    name: str
    lat: float
    lng: float


class EdgeOut(BaseModel):
    id: str
    source: str
    target: str
    distance_km: float
    base_time_min: float
    traffic_level: float      # 0–1
    travel_time_min: float    # current with traffic


class RouteRequest(BaseModel):
    origin: str
    destination: str


class RouteResult(BaseModel):
    origin: str
    destination: str
    origin_name: str
    destination_name: str
    path: List[str]
    path_names: List[str]
    path_coords: List[List[float]]
    distance_km: float
    travel_time_min: float
    baseline_time_min: float
    time_saved_min: float
    avg_traffic: float


class IncidentOut(BaseModel):
    id: int
    edge_id: str
    severity: str
    description: str
    lat: float
    lng: float
    active: bool

    model_config = {"from_attributes": True}


class VehicleOut(BaseModel):
    id: str
    name: str
    type: str
    color: str
    status: str
    lat: float
    lng: float
    speed_kmh: float
    current_edge: str
    destination: str


class TrafficPrediction(BaseModel):
    hour: int
    label: str
    avg_traffic: float
    congested_roads: int


class VisionSampleOut(BaseModel):
    name: str
    description: str
    source: str


class VisionHealthOut(BaseModel):
    detector_loaded: bool
    detector_weights: str
    detector_device: str
    ocr_loaded: bool
    ultralytics_available: bool
    easyocr_available: bool
    opencv_available: bool


class CVIncidentOut(BaseModel):
    id: int
    edge_id: str
    severity: str
    description: str
    lat: Optional[float] = None
    lng: Optional[float] = None
    active: bool
    source: str

    model_config = {"from_attributes": True}


class AnalyticsSummary(BaseModel):
    total_nodes: int
    total_edges: int
    avg_traffic: float
    congested_count: int
    free_flow_count: int
    active_incidents: int
    active_vehicles: int
    avg_speed_kmh: float
    predictions: List[TrafficPrediction]
