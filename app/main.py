from fastapi import FastAPI, UploadFile, File, HTTPException
from app import schemas
import uuid

# Initialize the FastAPI app
app = FastAPI(
    title="TruWage MVP Backend",
    description="API for Video Integrity, AQA, and Fair Wage Prediction",
    version="1.0.0"
)

# --- 1. Check Integrity Endpoint (Deepfake Detection) ---
@app.post("/check_integrity", response_model=schemas.IntegrityCheckResponse)
async def check_integrity(video_id: str):
    """
    Dummy endpoint for Day 1. 
    Day 2 Goal: Replace this dummy logic with the 3D CNN I3D/X3D inference.
    """
    # MOCK LOGIC: Assume video is authentic for now
    return schemas.IntegrityCheckResponse(
        video_id=video_id,
        is_authentic=True,
        confidence_score=0.92,
        message="Video passed integrity check. Proceeding to AQA."
    )

# --- 2. Get Wage Endpoint (GNN Fair-Wage Engine) ---
@app.post("/get_wage", response_model=schemas.WagePredictionResponse)
async def get_wage(request: schemas.WageRequest):
    """
    Dummy endpoint for Day 1.
    Day 2/3 Goal: Replace with Bayesian GNN prediction logic.
    """
    # MOCK LOGIC: Simple hardcoded logic based on complexity
    base_wage = 500
    multiplier = request.complexity * 1.5
    
    min_wage = base_wage * multiplier
    max_wage = min_wage * 1.2 # 20% upper bound for Bayesian range
    
    return schemas.WagePredictionResponse(
        task_type=request.task_type,
        estimated_wage_min=min_wage,
        estimated_wage_max=max_wage,
        confidence_interval="85% confidence (Monte Carlo Dropout)",
        reasoning=f"Wage adjusted for {request.complexity}/5 complexity and {request.location_demand} local demand."
    )

# --- 3. Full Pipeline Endpoint (Upload Video + Voice) ---
@app.post("/upload_video", response_model=schemas.FullPipelineResponse)
async def upload_video(
    file: UploadFile = File(...),
    task_type: str = "plumbing",
    complexity: int = 3
):
    """
    Main endpoint. Checks integrity first. If it fails, it short-circuits 
    and does NOT run the wage prediction (saving compute).
    """
    video_id = str(uuid.uuid4())
    
    # Step 1: Check Integrity
    integrity_result = await check_integrity(video_id=video_id)
    
    # Step 2: Short-circuit logic (Day 3 enhancement, pre-built here)
    if not integrity_result.is_authentic:
        return schemas.FullPipelineResponse(
            integrity_check=integrity_result,
            overall_status="FAILED_INTEGRITY_CHECK"
        )
    
    # Step 3: If authentic, get wage prediction
    wage_request = schemas.WageRequest(
        task_type=task_type,
        complexity=complexity,
        location_demand="high"
    )
    wage_result = await get_wage(wage_request)
    
    return schemas.FullPipelineResponse(
        integrity_check=integrity_result,
        wage_prediction=wage_result,
        overall_status="SUCCESS"
    )

# --- Health Check (Good practice for debugging) ---
@app.get("/")
def read_root():
    return {"message": "TruWage Backend is running!"}