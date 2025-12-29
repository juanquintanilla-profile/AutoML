"""
FastAPI main application.
"""

# Load environment variables FIRST (before any other imports)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from datetime import datetime

# Try to import logfire, but make it optional
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

from .models import (
    JobResponse,
    JobStatusResponse,
    JobResultsResponse,
    JobListResponse,
    JobStatus,
)
from .database import init_db, get_job, list_jobs, get_job_count
from .jobs import (
    generate_job_id,
    save_uploaded_file,
    start_automl_job,
    get_job_results,
    get_job_logs,
)


# Initialize Logfire if available and configured
if LOGFIRE_AVAILABLE:
    try:
        logfire.configure()
        print("[OK] Logfire configured successfully")
    except Exception as e:
        print(f"[WARN] Logfire not configured: {e}")
        print("       API will run without Logfire instrumentation")
        LOGFIRE_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="AutoML Agent API",
    description="REST API for running AutoML jobs",
    version="0.1.0",
)

# Instrument FastAPI with Logfire if available
if LOGFIRE_AVAILABLE:
    try:
        logfire.instrument_fastapi(app)
        print("[OK] Logfire FastAPI instrumentation enabled")
    except Exception as e:
        print(f"[WARN] Logfire FastAPI instrumentation failed: {e}")
        print("       API will run without FastAPI instrumentation")

# CORS middleware (for Streamlit UI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()
    if LOGFIRE_AVAILABLE:
        logfire.info("API started", message="AutoML Agent API is ready")
    print("[OK] AutoML Agent API is ready")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "AutoML Agent API",
        "version": "0.1.0",
        "status": "running",
    }


@app.post("/api/v1/jobs", response_model=JobResponse)
async def create_job(
    dataset: UploadFile = File(..., description="CSV dataset file"),
    target_column: str = Form(..., description="Name of target column"),
    config_path: str = Form(default="automl_agent/config.yaml", description="Path to config file"),
):
    """
    Create a new AutoML job.

    Upload a dataset and specify the target column to start training.
    """
    # Validate file type
    if not dataset.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Generate job ID
    job_id = generate_job_id()
    if LOGFIRE_AVAILABLE:
        with logfire.span('create_job', target=target_column, filename=dataset.filename):
            logfire.info('job_created', job_id=job_id, filename=dataset.filename)

    # Save uploaded file
    file_content = await dataset.read()
    dataset_path = await save_uploaded_file(file_content, job_id, dataset.filename)

    # Start job in background
    await start_automl_job(
        job_id=job_id,
        dataset_path=dataset_path,
        target_column=target_column,
        config_path=config_path,
    )

    # Get job info
    job = get_job(job_id)

    return JobResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        created_at=datetime.fromisoformat(job["created_at"]),
        target_column=job["target_column"],
    )


@app.get("/api/v1/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get job status.

    Returns current status, progress, and timing information.
    """
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    # Calculate duration if completed
    duration = None
    if job["started_at"] and job["completed_at"]:
        started = datetime.fromisoformat(job["started_at"])
        completed = datetime.fromisoformat(job["completed_at"])
        duration = (completed - started).total_seconds()

    return JobStatusResponse(
        job_id=job["job_id"],
        status=JobStatus(job["status"]),
        created_at=datetime.fromisoformat(job["created_at"]),
        started_at=datetime.fromisoformat(job["started_at"]) if job["started_at"] else None,
        completed_at=datetime.fromisoformat(job["completed_at"]) if job["completed_at"] else None,
        duration_seconds=duration,
        current_iteration=job.get("current_iteration"),
        error=job.get("error"),
    )


@app.get("/api/v1/jobs/{job_id}/results", response_model=JobResultsResponse)
async def get_results(job_id: str):
    """
    Get job results.

    Returns evaluation metrics and best model information.
    Only available for completed jobs.
    """
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed yet (status: {job['status']})"
        )

    results = get_job_results(job_id)

    return JobResultsResponse(**results)


@app.get("/api/v1/jobs/{job_id}/logs")
async def get_logs(job_id: str):
    """
    Get job logs.

    Returns execution logs for the job.
    """
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    logs = get_job_logs(job_id)

    return {
        "job_id": job_id,
        "logs": logs,
        "message": "For detailed traces, check the Logfire dashboard"
    }


@app.get("/api/v1/jobs/{job_id}/download-model")
async def download_model(job_id: str):
    """
    Download trained model.

    Returns the model.joblib file for the job.
    """
    job = get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed yet"
        )

    model_path = Path(job["output_dir"]) / "model.joblib"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found")

    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename=f"model_{job_id}.joblib"
    )


@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_all_jobs(limit: int = 100, offset: int = 0):
    """
    List all jobs.

    Returns a paginated list of all jobs.
    """
    jobs = list_jobs(limit=limit, offset=offset)
    total = get_job_count()

    job_responses = [
        JobResponse(
            job_id=job["job_id"],
            status=JobStatus(job["status"]),
            created_at=datetime.fromisoformat(job["created_at"]),
            started_at=datetime.fromisoformat(job["started_at"]) if job["started_at"] else None,
            completed_at=datetime.fromisoformat(job["completed_at"]) if job["completed_at"] else None,
            target_column=job["target_column"],
            error=job.get("error"),
        )
        for job in jobs
    ]

    return JobListResponse(jobs=job_responses, total=total)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
