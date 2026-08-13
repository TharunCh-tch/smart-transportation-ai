import logging
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.transport import router as transport_router
from app.api.routes.vision import router as vision_router
from app.core.config import settings
from app.core.database import Base, engine

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — creating DB tables…")
    Base.metadata.create_all(bind=engine)
    logger.info("Smart Transportation AI is ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered traffic simulation, A* route optimization, and real-time fleet tracking",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(transport_router, prefix="/api", tags=["Transport"])
app.include_router(vision_router, prefix="/api", tags=["Vision"])

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/", include_in_schema=False)
async def ui():
    idx = FRONTEND / "index.html"
    return FileResponse(str(idx)) if idx.exists() else {"app": settings.APP_NAME}


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy", "app": settings.APP_NAME, "version": settings.APP_VERSION}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8002, reload=True)
