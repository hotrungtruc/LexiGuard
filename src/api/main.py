"""
FastAPI Application Entrypoint: CORS, Lifespan (DB Pool Management), and Route Registration
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from src.api.routes import audit, chat, health, ocr
from src.core.config import async_connection_manager
from src.core.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages application lifecycle: database connections and resource initialization."""
    setup_logging()
    logger.info("Initializing LexiGuard API services & database connection pools...")
    try:
        db_health = await async_connection_manager.check_health()
        logger.info(f"Database initial health status: {db_health}")
    except Exception as exc:
        logger.warning(f"Initial database health probe failed: {exc}")

    yield

    logger.info("Gracefully closing LexiGuard database connections & pools...")
    try:
        await async_connection_manager.close()
        logger.info("LexiGuard database connections closed successfully.")
    except Exception as exc:
        logger.warning(f"Error closing database connections: {exc}")


app = FastAPI(
    title="LexiGuard API",
    description="Multi-Agent Contract Vulnerability Auditing & Legal Cross-Referencing System",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(audit.router, prefix="/api/v1", tags=["Audit"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(ocr.router, prefix="/api/v1", tags=["OCR"])

# Mount Web AI Interface
web_dir = Path(__file__).resolve().parent.parent / "ui" / "web"
web_dir.mkdir(parents=True, exist_ok=True)
app.mount("/app", StaticFiles(directory=str(web_dir), html=True), name="web_app")


@app.get("/")
async def root(request: Request):
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return RedirectResponse(url="/app/")
    return {"service": "LexiGuard API", "version": "0.1.0", "status": "online"}
