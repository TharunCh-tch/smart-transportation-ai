# Smart Transportation AI — full app (traffic sim + CV module).
#
# This image is CPU-only by default. For GPU inference on an EC2 GPU
# instance, see the "Deploying CV inference to an EC2 GPU instance"
# section of README.md — it uses a different base image
# (nvidia/cuda + a CUDA-enabled torch wheel) and a couple of extra flags.
#
# Build:  docker build -t smart-transportation-ai .
# Run:    docker run -p 8002:8002 smart-transportation-ai

FROM python:3.9-slim

# libgl1/libglib2.0-0: required at runtime by opencv-python (a transitive
# dependency of ultralytics) even though we also install the headless
# variant — see requirements-cv.txt for why both end up installed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-cv.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt -r requirements-cv.txt

# Pre-download model weights at build time so the first real request
# isn't slowed down by a cold download. Cached in the image layer.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" \
    && python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

COPY . .

# The dev SQLite file is gitignored/not copied in via .dockerignore;
# create the data dir so SQLAlchemy can create the DB file on startup.
RUN mkdir -p /app/data

EXPOSE 8002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
