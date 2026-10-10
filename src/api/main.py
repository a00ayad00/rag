"""
FastAPI Main Application.
Configures lifespan events, CORS, structured request logging middleware,
and mounts API routers.
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import settings
from src.api.routes import router as api_router
from src.ingestion.pipeline import IngestionPipeline, create_sample_reports_if_needed

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("bilingual_rag_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for startup and shutdown procedures."""
    logger.info("Initializing Bilingual Document Q&A Service...")
    logger.info(f"Configuration: Embedding={settings.EMBEDDING_MODEL_NAME}, LLM={settings.LLM_PROVIDER}")

    # Ensure sample reports exist if data directories are unpopulated
    try:
        create_sample_reports_if_needed()
    except Exception as e:
        logger.warning(f"Note on sample report bootstrapping: {e}")

    logger.info("Service initialized and ready to receive requests.")
    yield
    logger.info("Shutting down Bilingual Document Q&A Service.")


app = FastAPI(
    title="Bilingual Document Q&A (RAG) API",
    description=(
        "Production-grade bilingual (English & Arabic) retrieval-augmented generation service. "
        "Built with FastAPI, LangChain, LangGraph, Qdrant, and Cross-Encoder Reranking."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Structured Request Logging Middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    start_time = time.perf_counter()

    logger.info(f"[{request_id}] START {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{latency_ms:.2f}"
        logger.info(
            f"[{request_id}] COMPLETE {request.method} {request.url.path} "
            f"- Status: {response.status_code} ({latency_ms:.2f}ms)"
        )
        return response
    except Exception as exc:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        logger.error(
            f"[{request_id}] FAILED {request.method} {request.url.path} "
            f"after {latency_ms:.2f}ms: {exc}", exc_info=True
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error occurred.", "request_id": request_id}
        )


# Mount Router
app.include_router(api_router)


@app.get("/", tags=["Root"])
async def root():
    """Root landing endpoint with system overview and links."""
    return {
        "service": "Bilingual Document Q&A (RAG) Service",
        "version": "1.0.0",
        "docs": "/docs",
        "supported_languages": ["english", "arabic"],
        "retrieval_modes": ["baseline", "reranked"],
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False
    )
