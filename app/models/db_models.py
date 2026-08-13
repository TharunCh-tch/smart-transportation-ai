from sqlalchemy import Column, DateTime, Float, Integer, String, Text, func

from app.core.database import Base


class SavedRoute(Base):
    __tablename__ = "saved_routes"
    id = Column(Integer, primary_key=True)
    origin_id = Column(String, nullable=False)
    destination_id = Column(String, nullable=False)
    origin_name = Column(String)
    destination_name = Column(String)
    path_json = Column(Text)          # JSON list of node IDs
    distance_km = Column(Float)
    travel_time_min = Column(Float)
    time_saved_min = Column(Float)
    created_at = Column(DateTime, server_default=func.now())


class TrafficIncident(Base):
    __tablename__ = "traffic_incidents"
    id = Column(Integer, primary_key=True)
    edge_id = Column(String, nullable=False)   # "node1__node2"
    severity = Column(String, default="moderate")   # low / moderate / high
    description = Column(String)
    lat = Column(Float)
    lng = Column(Float)
    active = Column(Integer, default=1)
    source = Column(String, default="simulated")   # "simulated" | "cv-detection"
    created_at = Column(DateTime, server_default=func.now())
    resolved_at = Column(DateTime, nullable=True)
