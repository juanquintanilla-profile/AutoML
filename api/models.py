"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CreateJobRequest(BaseModel):
    """Request to create a new AutoML job."""
    target_column: str = Field(..., description="Name of the target column")
    config_overrides: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional config overrides (e.g., max_iterations, model_families)"
    )


class JobResponse(BaseModel):
    """Response with job information."""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    target_column: str
    error: Optional[str] = None


class JobStatusResponse(BaseModel):
    """Detailed job status response."""
    job_id: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    current_iteration: Optional[int] = None
    total_iterations: Optional[int] = None
    error: Optional[str] = None


class JobResultsResponse(BaseModel):
    """Job results with metrics and model info."""
    job_id: str
    status: JobStatus
    best_model: Optional[str] = None
    best_score: Optional[float] = None
    primary_metric: Optional[str] = None
    all_metrics: Optional[Dict[str, Any]] = None
    total_iterations: Optional[int] = None
    total_time: Optional[float] = None


class JobListResponse(BaseModel):
    """List of jobs."""
    jobs: List[JobResponse]
    total: int
