"""
Computer-vision module: vehicle / emergency-vehicle detection, license-plate
OCR, and multi-object tracking, built on top of pretrained models.

This package is intentionally decoupled from the traffic-simulation engine
in app/ai/*.py — it can be imported and used standalone, and it is wired
into the API surface via app/api/routes/vision.py.
"""
