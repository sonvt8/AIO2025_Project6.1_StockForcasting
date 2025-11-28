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
    """Load models on startup"""
    from app.utils.model_checker import check_models_exist
    from app.utils.model_trainer import train_and_export_models

    # Check if models exist
    all_exist, missing = check_models_exist()

    if not all_exist:
        print("\n" + "=" * 60)
        print("⚠️  Model files not found!")
        print("=" * 60)
        print("\nMissing files:")
        for file in missing:
            print(f"  - {file}")

        print("\n" + "=" * 60)
        print("Would you like to train models now?")
        print("=" * 60)
        print("\nThis will:")
        print("  - Load data from data/raw/FPT_train.csv")
        print("  - Train model with 2-stage grid search (may take several minutes)")
        print("  - Export models to app/models/artifacts/")
        print("\nNote: You can also download pre-trained models from GitHub releases.")

        # Interactive prompt
        while True:
            try:
                response = input("\nTrain models now? (y/n): ").strip().lower()
                if response in ["y", "yes"]:
                    print("\nStarting model training...")
                    success = train_and_export_models()
                    if success:
                        print("\n✅ Models trained successfully! Loading models...")
                        # Try to load models after training
                        try:
                            model_service = ModelService()
                            load_success = model_service.model_loader.load_models()
                            if load_success:
                                print("✅ Models loaded successfully")
                            else:
                                print("⚠️  Warning: Models trained but failed to load.")
                                print("   Please restart API to load models.")
                        except Exception as e:
                            print(f"⚠️  Warning: Models trained but failed to load: {e}")
                            print("   Please restart API to load models.")
                    else:
                        print(
                            "\n❌ Model training failed. API will start but predictions will fail."
                        )
                        print("   You can try again later or download models from GitHub.")
                    break
                elif response in ["n", "no"]:
                    print("\n⚠️  Skipping model training.")
                    print("API will start, but predictions will fail until models are available.")
                    print("To train models later, run: python export_models.py")
                    print("Or download models from GitHub releases to app/models/artifacts/")
                    break
                else:
                    print("Please enter 'y' or 'n'")
            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️  Skipping model training (interrupted).")
                print("API will start, but predictions will fail until models are available.")
                break
        return

    # Models exist, try to load them
    try:
        model_service = ModelService()
        success = model_service.model_loader.load_models()
        if success:
            print("✅ Models loaded successfully")
        else:
            print("⚠️  Warning: Models not loaded. Please check model files.")
    except Exception as e:
        print(f"⚠️  Warning: Could not load models on startup: {e}")
        print("   Models will be loaded on first prediction request")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FPT Stock Price Prediction API",
        "version": API_CONFIG["version"],
        "docs": "/docs",
        "health": "/health",
    }
