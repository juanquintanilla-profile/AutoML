"""
Simple SQLite database for job metadata.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from .models import JobStatus


DB_PATH = Path("outputs/jobs.db")


def init_db():
    """Initialize database schema."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                target_column TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                config_path TEXT,
                output_dir TEXT NOT NULL,
                error TEXT,
                current_iteration INTEGER DEFAULT 0,
                best_model TEXT,
                best_score REAL,
                metadata TEXT
            )
        """)
        conn.commit()


@contextmanager
def get_db():
    """Get database connection context manager."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def create_job(
    job_id: str,
    target_column: str,
    dataset_path: str,
    config_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    """Create a new job record."""
    created_at = datetime.now().isoformat()

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (
                job_id, status, created_at, target_column,
                dataset_path, config_path, output_dir
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, JobStatus.PENDING, created_at, target_column,
             dataset_path, config_path, output_dir)
        )
        conn.commit()

    return get_job(job_id)


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE job_id = ?",
            (job_id,)
        ).fetchone()

        if row:
            return dict(row)
        return None


def update_job_status(
    job_id: str,
    status: JobStatus,
    error: Optional[str] = None,
    started_at: Optional[str] = None,
    completed_at: Optional[str] = None,
):
    """Update job status."""
    with get_db() as conn:
        updates = {"status": status}
        if error:
            updates["error"] = error
        if started_at:
            updates["started_at"] = started_at
        if completed_at:
            updates["completed_at"] = completed_at

        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        values = list(updates.values()) + [job_id]

        conn.execute(
            f"UPDATE jobs SET {set_clause} WHERE job_id = ?",
            values
        )
        conn.commit()


def update_job_progress(
    job_id: str,
    current_iteration: int,
    best_model: Optional[str] = None,
    best_score: Optional[float] = None,
):
    """Update job progress."""
    with get_db() as conn:
        conn.execute(
            """
            UPDATE jobs
            SET current_iteration = ?, best_model = ?, best_score = ?
            WHERE job_id = ?
            """,
            (current_iteration, best_model, best_score, job_id)
        )
        conn.commit()


def list_jobs(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """List all jobs."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT * FROM jobs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        ).fetchall()

        return [dict(row) for row in rows]


def get_job_count() -> int:
    """Get total job count."""
    with get_db() as conn:
        result = conn.execute("SELECT COUNT(*) as count FROM jobs").fetchone()
        return result["count"]
