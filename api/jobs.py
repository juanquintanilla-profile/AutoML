"""
Job execution and management.
"""

import uuid
import shutil
import os
from pathlib import Path
from datetime import datetime
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Try to import logfire, but make it optional
try:
    import logfire
    LOGFIRE_AVAILABLE = True
except ImportError:
    LOGFIRE_AVAILABLE = False
    logfire = None

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
    try:
        # Update status to running
        update_job_status(
            job_id,
            JobStatus.RUNNING,
            started_at=datetime.now().isoformat()
        )

        if LOGFIRE_AVAILABLE:
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

        if LOGFIRE_AVAILABLE:
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

        if LOGFIRE_AVAILABLE:
            logfire.error('job_failed', job_id=job_id, error=str(e), exc_info=True)

        print(f"[ERROR] Job {job_id} failed: {str(e)}")
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
        "best_params": None,
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
                state = summary_data["state"]
                results["total_iterations"] = state.get("iteration")
                results["total_time"] = state.get("total_time")

            # Extract best_params from optimized_models or best_model
            best_model_name = results.get("best_model")
            if best_model_name:
                # Check in optimized_models list
                optimized_models = summary_data.get("optimized_models", [])
                for model in optimized_models:
                    if model.get("model_name") == best_model_name:
                        results["best_params"] = model.get("best_params", {})
                        break

                # Also check in best_model dict if present
                if not results["best_params"]:
                    best_model_data = summary_data.get("best_model", {})
                    if isinstance(best_model_data, dict):
                        results["best_params"] = best_model_data.get("best_params", {})

            # Get config info for context
            config = summary_data.get("config", {})
            if config and not results["best_params"]:
                # If FLAML was used, include HPO config as context
                hpo_config = config.get("hpo", {})
                if hpo_config.get("engine") == "flaml":
                    results["best_params"] = {
                        "optimization_method": "FLAML AutoML",
                        "time_budget": hpo_config.get("time_budget"),
                        "metric": config.get("metrics", {}).get("primary"),
                    }

    return results


def get_job_logs(job_id: str) -> Optional[str]:
    """Get job logs (if available)."""
    job = get_job(job_id)
    if not job:
        return None

    # For now, we don't capture logs to file
    # This could be enhanced later
    return f"Job {job_id} - Status: {job['status']}\nCheck Logfire dashboard for detailed logs."
