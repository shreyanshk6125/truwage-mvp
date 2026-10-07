from fastapi import FastAPI, UploadFile, File, HTTPException
from app import schemas
from app.deepfake_detector import integrity_checker
import shutil
import uuid
import os
import uuid

# Initialize the FastAPI app
app = FastAPI(
    title="TruWage MVP Backend",
    description="API for Video Integrity, AQA, and Fair Wage Prediction",
    version="1.0.0"
)

# --- 1. Check Integrity Endpoint (Deepfake Detection) ---
@app.post("/check_integrity", response_model=schemas.IntegrityCheckResponse)
async def check_integrity(video_id: str, file_path: str):
    """
    Day 2 Update: Uses the pre-trained 3D CNN + heuristic to check video integrity.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    # Run the AI check
    result = integrity_checker.predict(file_path)
    
    return schemas.IntegrityCheckResponse(
        video_id=video_id,
        is_authentic=result["is_authentic"],
        confidence_score=result["confidence_score"],
        message=result["message"]
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
    Main pipeline: Saves video, checks integrity, short-circuits if fake, else predicts wage.
    """
    video_id = str(uuid.uuid4())
    
    # 1. Save the uploaded file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, f"{video_id}_{file.filename}")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 2. Check Integrity using our new Day 2 model
    integrity_result = await check_integrity(video_id=video_id, file_path=file_path)
    
    # 3. Short-circuit logic: If fake, DO NOT run the heavy AQA/Wage models
    if not integrity_result.is_authentic:
        # Clean up the file
        os.remove(file_path)
        return schemas.FullPipelineResponse(
            integrity_check=integrity_result,
            overall_status="FAILED_INTEGRITY_CHECK"
        )
    
    # 4. If authentic, get wage prediction (Mock for now, Balraj will plug in GNN later)
    wage_request = schemas.WageRequest(
        task_type=task_type,
        complexity=complexity,
        location_demand="high"
    )
    wage_result = await get_wage(wage_request)
    
    # Clean up the file after processing
    os.remove(file_path)
    
    return schemas.FullPipelineResponse(
        integrity_check=integrity_result,
        wage_prediction=wage_result,
        overall_status="SUCCESS"
    )

# --- Health Check (Good practice for debugging) ---
@app.get("/")
def read_root():
    return {"message": "TruWage Backend is running!"}