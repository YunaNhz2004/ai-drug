from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse

from typing import Optional
import time
from src.inference import predict_smiles, health_check

app = FastAPI(
    title="Toxicity Prediction API",
    description="API for predicting molecular toxicity from SMILES strings",
    version="1.0.0"
)

# =========================
# CORS - BẮT BUỘC CHO REACTJS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://nhungnguyen0804.github.io",
        "http://localhost:3000",  # React dev
        "http://localhost:5173",  # Vite dev
        "https://yourdomain.com"   # Production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# MODELS
# =========================
class SmilesRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of molecule")
    skip_image: bool = Field(True, description="Skip 2D image generation for faster response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "skip_image": False
            }
        }


class PredictionResponse(BaseModel):
    smiles: str
    binary_toxicity: dict
    organ_toxicity: dict
    adr: dict
    inference_time: Optional[float] = None
    images: dict



class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    error: Optional[str] = None


# =========================
# ENDPOINTS
# =========================

@app.get("/", tags=["Root"])
def root():
    """Root endpoint"""
    return {
        "message": "Toxicity Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Health check endpoint"""
    return health_check()


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"]
)
def predict(req: SmilesRequest):
    """
    Predict toxicity for a given SMILES string
    
    - **smiles**: SMILES string representing the molecule
    - **skip_image**: Set to true for faster response (no 2D image generation)
    """
    start_time = time.time()
    
    try:
        # Validate input
        if not req.smiles or not req.smiles.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SMILES string cannot be empty"
            )
        
        # Predict
        result = predict_smiles(req.smiles.strip(), skip_image=req.skip_image)
        
        # Check for errors
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        # Add inference time
        result["inference_time"] = round(time.time() - start_time, 3)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/predict", tags=["Prediction"])
def predict_get(
    smiles: str = Query(..., description="SMILES string"),
    skip_image: bool = Query(False, description="Skip image generation")
):
    """
    GET version of predict endpoint (for quick testing)
    """
    return predict(SmilesRequest(smiles=smiles, skip_image=skip_image))


# =========================
# STARTUP/SHUTDOWN EVENTS
# =========================

@app.on_event("startup")
async def startup_event():
    """Run on startup"""
    print("Starting Toxicity Prediction API...")
    health_status = health_check()
    if health_status["status"] == "healthy":
        print("✅ Model loaded successfully")
    else:
        print(f"❌ Model loading failed: {health_status.get('error')}")


@app.on_event("shutdown")
async def shutdown_event():
    """Run on shutdown"""
    print("Shutting down API...")


# =========================
# ERROR HANDLERS
# =========================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "detail": "Please check /docs for available endpoints"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "Please contact support if this persists"
        }
    )