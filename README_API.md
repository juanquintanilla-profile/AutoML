# AutoML Agent API + UI

This extends the AutoML Agent with a REST API and web UI, **without modifying the original `automl_agent/` code**.

## Architecture

```
┌──────────────┐
│ Streamlit UI │  (localhost:8501)
└──────┬───────┘
       │ HTTP
┌──────▼───────┐
│  FastAPI     │  (localhost:8000)
│  + Logfire   │
└──────┬───────┘
       │ Python calls
┌──────▼───────┐
│ automl_agent │  (unchanged)
└──────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
# Install base AutoML dependencies
pip install -r requirements.txt

# Install API + UI dependencies
pip install -r requirements-api.txt
```

### 2. Configure Logfire

```bash
# Create .env file
cp .env.example .env

# Get free Logfire token from https://logfire.pydantic.dev
# Add to .env:
# LOGFIRE_TOKEN=your_token_here
```

### 3. Start the API

```bash
# Terminal 1: Start FastAPI server
python -m uvicorn api.main:app --reload

# API will be available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 4. Start the UI

```bash
# Terminal 2: Start Streamlit
streamlit run ui/app.py

# UI will open at http://localhost:8501
```

## Usage

### Via Web UI (Streamlit)

1. Go to http://localhost:8501
2. Upload a CSV dataset
3. Select target column
4. Click "Start AutoML Job"
5. Monitor progress in "Job Monitor" tab
6. View results in "Results" tab

### Via API (curl)

```bash
# Create job
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -F "dataset=@data/customer_churn.csv" \
  -F "target_column=churn"

# Response: {"job_id": "abc12345", "status": "pending", ...}

# Check status
curl "http://localhost:8000/api/v1/jobs/abc12345/status"

# Get results (when completed)
curl "http://localhost:8000/api/v1/jobs/abc12345/results"

# Download model
curl "http://localhost:8000/api/v1/jobs/abc12345/download-model" -O

# List all jobs
curl "http://localhost:8000/api/v1/jobs"
```

### Via Python

```python
import requests

# Create job
with open('data/customer_churn.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/jobs',
        files={'dataset': f},
        data={'target_column': 'churn'}
    )

job_id = response.json()['job_id']

# Poll status
import time
while True:
    status = requests.get(f'http://localhost:8000/api/v1/jobs/{job_id}/status').json()
    print(f"Status: {status['status']}")

    if status['status'] in ['completed', 'failed']:
        break

    time.sleep(5)

# Get results
results = requests.get(f'http://localhost:8000/api/v1/jobs/{job_id}/results').json()
print(f"Best model: {results['best_model']}")
print(f"Best score: {results['best_score']}")
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/jobs` | Create new AutoML job |
| `GET` | `/api/v1/jobs` | List all jobs |
| `GET` | `/api/v1/jobs/{id}/status` | Get job status |
| `GET` | `/api/v1/jobs/{id}/results` | Get job results |
| `GET` | `/api/v1/jobs/{id}/logs` | Get job logs |
| `GET` | `/api/v1/jobs/{id}/download-model` | Download trained model |

Full API documentation: http://localhost:8000/docs

## Monitoring with Logfire

All API requests and AutoML jobs are automatically traced with Logfire:

1. Sign up at https://logfire.pydantic.dev (free tier available)
2. Get your token and add to `.env`
3. Start the API
4. View real-time traces in the Logfire dashboard

You'll see:
- All HTTP requests/responses
- Job execution timelines
- Errors with full stack traces
- Performance metrics

## Outputs

Job artifacts are saved in `outputs/{job_id}/`:
- `model.joblib` - Trained model
- `metrics.json` - Evaluation metrics
- `run_summary.json` - Full execution summary
- `config.json` - Configuration used

The SQLite database is at `outputs/jobs.db`.

## What Changed?

**Nothing in `automl_agent/`!** The original code is completely untouched.

New files:
- `api/` - FastAPI application
- `ui/` - Streamlit application
- `requirements-api.txt` - Additional dependencies
- `outputs/` - Job outputs and database

## Production Deployment

For production, consider:

1. **Environment variables**: Use proper secret management
2. **Database**: Switch from SQLite to PostgreSQL
3. **File storage**: Use S3 or similar for datasets/models
4. **Queue**: Add Celery + Redis for robust job queue
5. **Auth**: Add authentication/authorization
6. **CORS**: Restrict to specific origins
7. **HTTPS**: Use reverse proxy (nginx) with SSL

Example Docker Compose:
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - LOGFIRE_TOKEN=${LOGFIRE_TOKEN}

  ui:
    build: .
    command: streamlit run ui/app.py
    ports:
      - "8501:8501"
```

## Troubleshooting

**API won't start:**
- Check if port 8000 is available
- Verify `.env` has valid `LOGFIRE_TOKEN`
- Ensure `requirements-api.txt` is installed

**Jobs fail:**
- Check `automl_agent/config.yaml` is valid
- Verify dataset has correct format
- Check Logfire dashboard for detailed errors

**UI can't connect:**
- Ensure API is running on port 8000
- Check CORS settings in `api/main.py`
