"""
Job execution and management.
"""

import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

import logfire

from automl_agent.main import run_automl
from .database import (
    create_job,
    update_job_status,
    update_job_progress,
    get_job,
)
from .models import JobStatus


# Thread pool for running AutoML jobs
executor = ThreadPoolExecutor(max_workers=4)


def generate_job_id() -> str:
    """Generate unique job ID."""
    return str(uuid.uuid4())[:8]


async def save_uploaded_file(file_content: bytes, job_id: str, filename: str) -> str:
    """Save uploaded file to disk."""
    output_dir = Path("outputs") / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    file_path = output_dir / filename

    # Write file asynchronously
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(
        None,
        file_path.write_bytes,
        file_content
    )

    return str(file_path)


def execute_automl_job(
    job_id: str,
    dataset_path: str,
    target_column: str,
    config_path: str,
    output_dir: str,
):
    """
    Execute AutoML job (runs in thread pool).

    This function wraps the existing automl_agent/main.py::run_automl()
    without modifying it.
    """
    with logfire.span(
        'automl_job',
        job_id=job_id,
        target=target_column,
        dataset=dataset_path
    ):
        try:
            # Update status to running
            update_job_status(
                job_id,
                JobStatus.RUNNING,
                started_at=datetime.now().isoformat()
            )

            logfire.info('job_started', job_id=job_id)

            # Run AutoML (this is the existing function, unchanged)
            state = run_automl(
                data_path=dataset_path,
                target_column=target_column,
                config_path=config_path,
                output_dir=output_dir,
            )

            # Update final status
            update_job_status(
                job_id,
                JobStatus.COMPLETED,
                completed_at=datetime.now().isoformat()
            )

            # Update final results
            update_job_progress(
                job_id,
                current_iteration=state.iteration,
                best_model=state.best_model.get('model_name') if state.best_model else None,
                best_score=state.best_score,
            )

            logfire.info(
                'job_completed',
                job_id=job_id,
                best_model=state.best_model.get('model_name') if state.best_model else None,
                best_score=state.best_score,
                iterations=state.iteration,
                duration=state.total_time
            )

        except Exception as e:
            # Update status to failed
            update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e),
                completed_at=datetime.now().isoformat()
            )

            logfire.error('job_failed', job_id=job_id, error=str(e), exc_info=True)
            raise


async def start_automl_job(
    job_id: str,
    dataset_path: str,
    target_column: str,
    config_path: str = "automl_agent/config.yaml",
) -> None:
    """
    Start AutoML job in background.

    Args:
        job_id: Unique job identifier
        dataset_path: Path to dataset file
        target_column: Target column name
        config_path: Path to config file
    """
    output_dir = f"outputs/{job_id}"

    # Create job record
    create_job(
        job_id=job_id,
        target_column=target_column,
        dataset_path=dataset_path,
        config_path=config_path,
        output_dir=output_dir,
    )

    # Execute in thread pool (non-blocking)
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        execute_automl_job,
        job_id,
        dataset_path,
        target_column,
        config_path,
        output_dir,
    )


def get_job_results(job_id: str) -> Optional[dict]:
    """
    Get job results from output directory.

    Reads the metrics.json and run_summary.json files.
    """
    job = get_job(job_id)
    if not job:
        return None

    output_dir = Path(job["output_dir"])

    results = {
        "job_id": job_id,
        "status": job["status"],
        "best_model": job.get("best_model"),
        "best_score": job.get("best_score"),
    }

    # Try to read metrics file
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        import json
        with open(metrics_file) as f:
            metrics_data = json.load(f)
            results["all_metrics"] = metrics_data
            results["primary_metric"] = metrics_data.get("primary_metric")

    # Try to read summary file
    summary_file = output_dir / "run_summary.json"
    if summary_file.exists():
        import json
        with open(summary_file) as f:
            summary_data = json.load(f)
            if "state" in summary_data:
                results["total_iterations"] = summary_data["state"].get("iteration")
                results["total_time"] = summary_data["state"].get("total_time")

    return results


def get_job_logs(job_id: str) -> Optional[str]:
    """Get job logs (if available)."""
    job = get_job(job_id)
    if not job:
        return None

    # For now, we don't capture logs to file
    # This could be enhanced later
    return f"Job {job_id} - Status: {job['status']}\nCheck Logfire dashboard for detailed logs."
