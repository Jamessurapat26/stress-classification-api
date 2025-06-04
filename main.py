# main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import time
import asyncio
import sys


try:
    if sys.platform != 'win32':
        import uvloop
    else:
        uvloop = None
except ImportError:
    uvloop = None

from endpoint.predictEndpoint import router as predict_router
from config.cors import add_cors_middleware
from services.db_service import DBService
from model.model import ONNXModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
model_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events."""
    # Startup
    logger.info("Starting application...")
    logger.info(f"Running on platform: {sys.platform}")
    
    # Set event loop policy for better performance (Unix only)
    if uvloop and hasattr(asyncio, 'set_event_loop_policy'):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        logger.info("Using uvloop event loop policy for enhanced performance")
    else:
        logger.info("Using default asyncio event loop policy")
    
    # Initialize model once at startup
    global model_instance
    try:
        model_instance = ONNXModel()
        logger.info("Model loaded successfully during startup")
    except Exception as e:
        logger.error(f"Failed to load model during startup: {e}")
        raise
    
    # Initialize database connection
    await DBService.initialize()
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    await DBService.close()

app = FastAPI(
    title="Stress Prediction API",
    description="API for stress level prediction using physiological data",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
add_cors_middleware(app)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Include routers
app.include_router(predict_router, prefix="/predict", tags=["predict"])

@app.get("/", tags=["health"])
async def health_check():
    """Health check endpoint with model status."""
    return {
        "status": "ok",
        "model_loaded": model_instance is not None and model_instance.is_loaded,
        "platform": sys.platform,
        "event_loop": "uvloop" if uvloop else "asyncio",
        "timestamp": int(time.time())
    }

@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """Detailed health check including database connectivity."""
    try:
        # Check database connectivity
        db_status = await DBService.health_check()
        
        return {
            "status": "ok",
            "model_loaded": model_instance is not None and model_instance.is_loaded,
            "database_connected": db_status,
            "platform": sys.platform,
            "event_loop": "uvloop" if uvloop else "asyncio",
            "timestamp": int(time.time())
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "error": str(e),
                "platform": sys.platform,
                "timestamp": int(time.time())
            }
        )
