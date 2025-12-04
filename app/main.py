"""
FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import routes
from app.config import API_CONFIG
from app.services.model_service import ModelService

# Create FastAPI app
app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
    docs_url=API_CONFIG["docs_url"],
    redoc_url=API_CONFIG["redoc_url"],
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(routes.router)


@app.on_event("startup")
async def startup_event():
    """Attempt to load PatchTST artifacts on startup (no interactive training)."""
    try:
        model_service = ModelService()
        success = model_service.model_loader.load()
        if success:
            print("✅ PatchTST artifacts loaded successfully")
        else:
            print(
                "⚠️  Warning: Could not load PatchTST artifacts now.\n"
                "   The loader will attempt auto-download from GitHub Releases on first request."
            )
    except Exception as e:
        print(f"⚠️  Warning: Could not load PatchTST artifacts on startup: {e}")
        print("   They will be loaded on first prediction request (with auto-download if missing)")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FPT Stock Price Prediction API",
        "version": API_CONFIG["version"],
        "docs": "/docs",
        "health": "/health",
    }
