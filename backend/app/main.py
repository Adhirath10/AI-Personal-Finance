"""
Main FastAPI Application Entry Point

AI-Powered Personal Finance Monitoring and Predictive Financial Management System
"""

import logging

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from .api.routes import router
from .config.settings import settings
from .database.connection import verify_database_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="AI-Powered Personal Finance API",
    description="AI/ML-powered personal finance monitoring and recommendations system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    logger.warning("Database constraint rejected a request on %s.", request.url.path)
    return JSONResponse(status_code=409, content={"detail": "The requested record conflicts with existing data."})


@app.exception_handler(SQLAlchemyError)
async def database_error_handler(request: Request, exc: SQLAlchemyError):
    logger.exception("Database request failed on %s.", request.url.path)
    return JSONResponse(status_code=503, content={"detail": "Database operation failed."})


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - health check."""
    return {
        "status": "healthy",
        "service": "AI Finance API",
        "version": "1.0.0"
    }


@app.get("/api/health", tags=["Health"])
@app.get("/health", tags=["Health"], include_in_schema=False)
async def health_check():
    """Report database and AI-service connectivity independently."""
    try:
        verify_database_connection()
    except SQLAlchemyError as exc:
        logger.exception("Database health check failed")
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable",
        ) from exc

    ai_status = "unavailable"
    try:
        async with httpx.AsyncClient(
            base_url=settings.resolved_ai_service_url,
            timeout=httpx.Timeout(2.0),
        ) as client:
            response = await client.get("/health")
        payload = response.json()
        if (
            response.status_code == 200
            and isinstance(payload, dict)
            and payload.get("status") == "ok"
            and payload.get("database") == "available"
        ):
            ai_status = "connected"
    except (httpx.RequestError, ValueError):
        logger.info("AI service health check failed; database remains available.")
    return {
        "status": "ok",
        "database": "connected",
        "ai_service": ai_status,
    }


app.include_router(router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
