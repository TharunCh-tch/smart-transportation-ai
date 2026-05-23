# Smart Transportation AI

Real-time traffic simulation, A* route optimization, fleet tracking, and predictive analytics on an interactive NYC map.

## Features

| Feature | Details |
|---|---|
| **Live Map** | Leaflet.js + OpenStreetMap — 20 NYC intersections, 42 roads colored by congestion |
| **A* Route Optimizer** | Finds optimal path considering live traffic weights; compares vs baseline |
| **Traffic Simulation** | Time-of-day sine model + per-edge noise — peaks at 8 AM and 6 PM |
| **12-Hour Forecast** | NumPy-powered traffic prediction charted with Chart.js |
| **Fleet Tracking** | 6 vehicles (vans, buses, taxis, police) moving in real time along city routes |
| **Incident Detection** | AI-identified congestion hotspots flagged with severity levels |
| **Auto-refresh** | Map, fleet, and analytics update every 10 seconds automatically |

## Architecture

```
Browser (Leaflet.js + Chart.js)
     │  REST/JSON  (10s polling)
     ▼
FastAPI  (app/main.py)
  /api/nodes                  — 20 NYC intersection nodes
  /api/traffic/edges          — 42 roads with live traffic levels
  /api/routes/optimize        — POST: A* with traffic weights
  /api/fleet                  — 6 vehicle positions (time-based simulation)
  /api/traffic/incidents      — AI-detected congestion incidents
  /api/traffic/predict        — 12-hour traffic forecast
  /api/analytics              — network-wide statistics
     │
  ┌──┴──────────────────────────────────┐
  │  city_graph.py      NYC road network │
  │  route_optimizer.py A* + Dijkstra   │
  │  traffic_simulator.py  NumPy model  │
  │  fleet_simulator.py    GPS sim      │
  └─────────────────────────────────────┘
     │
  SQLite (saved routes, incidents)
```

## Setup

```bash
cd smart-transportation-ai
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
uvicorn app.main:app --port 8002 --reload
```

Open **http://localhost:8002** — no API keys required.

## How to Use

### Route Optimizer
1. Select **Origin** and **Destination** from the dropdowns
2. Click **Optimize Route**
3. The A* algorithm finds the fastest path given current traffic
4. The highlighted route appears on the map with time comparison vs baseline

### Live Map
- **Green roads** = free flow, **Yellow** = moderate, **Orange** = heavy, **Red** = congested
- Road thickness increases with congestion
- Vehicle emojis move in real time (🚐🚌🚕🚓)
- Click any road for traffic details

### Traffic Tab
- Network stats, active incidents, 12-hour forecast chart

## Tech Stack

| Component | Technology |
|---|---|
| API | FastAPI |
| Route finding | A* algorithm (pure Python + heapq) |
| Traffic model | NumPy time-series simulation |
| Database | SQLite via SQLAlchemy |
| Map | Leaflet.js + OpenStreetMap (free, no API key) |
| Charts | Chart.js |
| Frontend | Vanilla HTML/CSS/JS |
