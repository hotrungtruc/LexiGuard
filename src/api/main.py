"""
FastAPI Application Entrypoint: CORS, Lifespan (DB Pool Management), and Route Registration
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from src.core.config import settings, async_connection_manager
from src.core.logging import setup_logging
from src.api.routes import health, audit

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
    lifespan=lifespan
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

@app.get("/")
async def root():
    return {
        "service": "LexiGuard API",
        "status": "online",
        "docs": "/docs"
    }
