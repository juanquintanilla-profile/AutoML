"""
Streamlit UI for AutoML Agent.
"""

import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path

# API Configuration
API_URL = "http://localhost:8000"


st.set_page_config(
    page_title="AutoML Agent",
    page_icon="🤖",
    layout="wide",
)


def main():
    """Main Streamlit app."""
    st.title("🤖 AutoML Agent")
    st.markdown("Automated Machine Learning with Multi-Agent Orchestration")

    # Sidebar navigation
    page = st.sidebar.radio(
        "Navigation",
        ["🚀 New Job", "📊 Job Monitor", "📈 Results"]
    )

    if page == "🚀 New Job":
        show_new_job_page()
    elif page == "📊 Job Monitor":
        show_monitor_page()
    elif page == "📈 Results":
        show_results_page()


def show_new_job_page():
    """Page for creating new AutoML jobs."""
    st.header("Create New AutoML Job")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Dataset (CSV)",
        type=["csv"],
        help="Upload a CSV file with your training data"
    )

    if uploaded_file:
        # Preview dataset
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10))

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Rows", f"{len(df):,}")
            st.metric("Columns", len(df.columns))

        with col2:
            # Select target column
            target_column = st.selectbox(
                "Target Column",
                options=df.columns.tolist(),
                help="Select the column you want to predict"
            )

        # Advanced options (optional)
        with st.expander("⚙️ Advanced Options (Optional)"):
            config_path = st.text_input(
                "Config Path",
                value="automl_agent/config.yaml",
                help="Path to custom configuration file"
            )

        # Run button
        if st.button("🚀 Start AutoML Job", type="primary", use_container_width=True):
            with st.spinner("Creating job..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)

                    # Create job via API
                    response = requests.post(
                        f"{API_URL}/api/v1/jobs",
                        files={"dataset": uploaded_file},
                        data={
                            "target_column": target_column,
                            "config_path": config_path,
                        }
                    )

                    if response.status_code == 200:
                        job_data = response.json()
                        st.success(f"✅ Job created successfully!")

                        st.info(f"**Job ID:** `{job_data['job_id']}`")
                        st.info("Go to **Job Monitor** to track progress")

                        # Store job ID in session state
                        if 'job_ids' not in st.session_state:
                            st.session_state.job_ids = []
                        st.session_state.job_ids.append(job_data['job_id'])

                    else:
                        st.error(f"Error: {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure the API server is running:\n\n"
                             "`python -m uvicorn api.main:app --reload`")
                except Exception as e:
                    st.error(f"Error: {str(e)}")


def show_monitor_page():
    """Page for monitoring job progress."""
    st.header("Job Monitor")

    # Fetch all jobs
    try:
        response = requests.get(f"{API_URL}/api/v1/jobs")

        if response.status_code == 200:
            data = response.json()
            jobs = data["jobs"]

            if not jobs:
                st.info("No jobs found. Create a new job to get started!")
                return

            # Display jobs in a table
            st.subheader(f"Total Jobs: {data['total']}")

            # Convert to DataFrame
            jobs_df = pd.DataFrame([
                {
                    "Job ID": job["job_id"],
                    "Status": job["status"],
                    "Target": job["target_column"],
                    "Created": datetime.fromisoformat(job["created_at"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "Error": job.get("error", "")[:50] if job.get("error") else ""
                }
                for job in jobs
            ])

            # Color-code status
            def color_status(val):
                colors = {
                    "pending": "background-color: #FFA500",
                    "running": "background-color: #1E90FF; color: white",
                    "completed": "background-color: #32CD32; color: white",
                    "failed": "background-color: #DC143C; color: white",
                }
                return colors.get(val, "")

            styled_df = jobs_df.style.applymap(color_status, subset=["Status"])
            st.dataframe(styled_df, use_container_width=True)

            # Job detail section
            st.subheader("Job Details")
            selected_job = st.selectbox(
                "Select Job",
                options=[job["job_id"] for job in jobs],
                format_func=lambda x: f"{x} ({next(j['status'] for j in jobs if j['job_id'] == x)})"
            )

            if selected_job:
                show_job_details(selected_job)

        else:
            st.error(f"Error fetching jobs: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_job_details(job_id: str):
    """Show detailed information for a specific job."""
    try:
        response = requests.get(f"{API_URL}/api/v1/jobs/{job_id}/status")

        if response.status_code == 200:
            job = response.json()

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Status", job["status"].upper())

            with col2:
                if job.get("duration_seconds"):
                    st.metric("Duration", f"{job['duration_seconds']:.1f}s")
                else:
                    st.metric("Duration", "Running...")

            with col3:
                if job.get("current_iteration"):
                    st.metric("Iteration", job["current_iteration"])

            # Error message
            if job.get("error"):
                st.error(f"**Error:** {job['error']}")

            # Logs
            with st.expander("📋 View Logs"):
                logs_response = requests.get(f"{API_URL}/api/v1/jobs/{job_id}/logs")
                if logs_response.status_code == 200:
                    logs = logs_response.json()
                    st.code(logs.get("logs", "No logs available"))
                    st.info(logs.get("message", ""))

            # Auto-refresh for running jobs
            if job["status"] == "running":
                st.info("🔄 Auto-refreshing every 5 seconds...")
                time.sleep(5)
                st.rerun()

        else:
            st.error(f"Error fetching job details: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_results_page():
    """Page for viewing job results."""
    st.header("Results")

    try:
        # Fetch all jobs
        response = requests.get(f"{API_URL}/api/v1/jobs")

        if response.status_code == 200:
            data = response.json()
            completed_jobs = [
                job for job in data["jobs"]
                if job["status"] == "completed"
            ]

            if not completed_jobs:
                st.info("No completed jobs yet. Check Job Monitor for running jobs.")
                return

            # Select job
            selected_job = st.selectbox(
                "Select Completed Job",
                options=[job["job_id"] for job in completed_jobs],
                format_func=lambda x: f"{x} ({next(j['target_column'] for j in completed_jobs if j['job_id'] == x)})"
            )

            if selected_job:
                show_job_results(selected_job)

        else:
            st.error(f"Error fetching jobs: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_job_results(job_id: str):
    """Display results for a completed job."""
    try:
        response = requests.get(f"{API_URL}/api/v1/jobs/{job_id}/results")

        if response.status_code == 200:
            results = response.json()

            # Key metrics
            st.subheader("🏆 Best Model")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Model", results.get("best_model", "N/A"))

            with col2:
                if results.get("best_score"):
                    st.metric(
                        results.get("primary_metric", "Score"),
                        f"{results['best_score']:.4f}"
                    )

            with col3:
                if results.get("total_time"):
                    st.metric("Total Time", f"{results['total_time']:.1f}s")

            # All metrics
            if results.get("all_metrics"):
                with st.expander("📊 All Metrics"):
                    st.json(results["all_metrics"])

            # Download model
            st.subheader("💾 Download Model")

            download_url = f"{API_URL}/api/v1/jobs/{job_id}/download-model"

            st.markdown(
                f"[⬇️ Download Trained Model]({download_url})",
                unsafe_allow_html=True
            )

            if st.button("Download Model", type="primary"):
                try:
                    model_response = requests.get(download_url)
                    if model_response.status_code == 200:
                        st.download_button(
                            label="Save Model",
                            data=model_response.content,
                            file_name=f"model_{job_id}.joblib",
                            mime="application/octet-stream"
                        )
                    else:
                        st.error("Error downloading model")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        else:
            st.error(f"Error fetching results: {response.text}")

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
