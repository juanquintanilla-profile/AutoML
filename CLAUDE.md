# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AutoML Agent is a multi-agent system that uses LLM-based orchestration to automate machine learning workflows. The system uses GPT-4 (via OpenAI or Azure OpenAI) to orchestrate specialized agents that handle data analysis, model selection, hyperparameter optimization, and evaluation.

## Running the Application

### Three Ways to Use AutoML Agent

#### 1. Command Line (Direct)

```bash
# Run with example data
python -m automl_agent --data data/customer_churn.csv --target churn

# With custom config and output directory
python -m automl_agent \
  --data path/to/data.csv \
  --target target_column \
  --config automl_agent/config.yaml \
  --output automl_agent/output
```

#### 2. REST API (FastAPI)

```bash
# Start API server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Or with just Python
python api/main.py
```

**API Endpoints:**
- `POST /api/v1/jobs` - Create new AutoML job (upload CSV)
- `GET /api/v1/jobs/{job_id}/status` - Check job status and progress
- `GET /api/v1/jobs/{job_id}/results` - Get evaluation metrics
- `GET /api/v1/jobs/{job_id}/logs` - View execution logs
- `GET /api/v1/jobs/{job_id}/download-model` - Download trained model
- `GET /api/v1/jobs` - List all jobs (paginated)

**Features:**
- Async job execution with background workers
- SQLite database for job persistence (stored at `outputs/jobs.db`)
- Logfire integration for observability (OPTIONAL - API works without it, just skip token setup)
- CORS enabled for web UI access

#### 3. Web UI (Streamlit)

```bash
# Start Streamlit UI (requires API server running)
streamlit run ui/app.py
```

**UI Pages:**
- New Job: Upload datasets, select target, start training
- Job Monitor: Track running jobs with auto-refresh
- Results: View metrics, download models

### Testing

```bash
# Generate example dataset
python generate_example_data.py

# Run test script (interactive) - Unix/Git Bash
./test_automl.sh

# Or run directly (works on Windows)
python -m automl_agent --data data/customer_churn.csv --target churn
```

### Environment Setup

1. Install core dependencies: `pip install -r requirements.txt`
2. Install API/UI dependencies: `pip install -r requirements-api.txt`
3. Set up `.env` file with API credentials:
   - For OpenAI: `OPENAI_API_KEY`
   - For Azure OpenAI: `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`
4. Configure `automl_agent/config.yaml` for LLM provider (OpenAI or Azure)

## Architecture

### Core Orchestration Flow

The system follows a strict sequential workflow orchestrated by a PlannerAgent (LLM):

```
1. DataAgent → Analyzes dataset, proposes preprocessing (runs ONCE)
2. ModelingAgent → Generates model candidates (runs ONCE)
3. HPOAgent → Optimizes hyperparameters using FLAML/Optuna
4. EvaluationAgent → Evaluates and compares models
5. PlannerAgent → Decides to iterate or stop
```

### Key Components

**Core AutoML Engine:**
- **`automl_agent/main.py`**: Entry point and orchestration loop
- **`automl_agent/orchestrator/planner.py`**: LLM-based decision maker that chooses next action
- **`automl_agent/orchestrator/state.py`**: Global state management using `AutoMLState` dataclass
- **`automl_agent/agents/`**: Specialized agents for data, modeling, HPO, and evaluation
- **`automl_agent/tools/`**: Utilities for data processing, preprocessing, metrics, logging
- **`automl_agent/prompts/planner.txt`**: System prompt for the LLM orchestrator

**REST API:**
- **`api/main.py`**: FastAPI application with all endpoints
- **`api/models.py`**: Pydantic schemas for request/response validation
- **`api/database.py`**: SQLite database for job persistence
- **`api/jobs.py`**: Background job execution and file management

**Web UI:**
- **`ui/app.py`**: Streamlit application with three pages (New Job, Monitor, Results)

### State Management

The `AutoMLState` class (in `orchestrator/state.py`) maintains:
- Dataset analysis results and preprocessing plans
- Pipeline candidates and optimized models
- Evaluation results and best model tracking
- Iteration counts, timing, and execution history
- Workflow stage flags used by PlannerAgent

### Agent Communication

Agents communicate through structured state (JSON), NOT natural language:
- PlannerAgent reads state and returns action decisions as JSON
- Each specialized agent updates specific parts of the state
- The main loop in `main.py` coordinates agent execution based on planner decisions

### Critical Implementation Details

1. **Task Type Handling**: The system auto-detects classification vs regression tasks and adjusts:
   - Metrics (classification: accuracy/f1/precision/recall, regression: r2/rmse/mae)
   - Target type conversion (classification: int, regression: float)
   - Data splitting strategy (classification uses stratification)
   - See `automl_agent/main.py:113-128` and `automl_agent/tools/data_utils.py`

2. **Type Conversion**: FLAML requires correct target types:
   - Classification tasks: `y.astype(int)` before fitting
   - Regression tasks: `y.astype(float)` before fitting
   - See `automl_agent/agents/hpo_agent.py:55-61`

3. **LLM Provider Configuration**: The system supports both OpenAI and Azure OpenAI:
   - Set `llm.provider` in `config.yaml` to "openai" or "azure"
   - For Azure: specify `deployment_name` and `api_version`
   - For OpenAI: specify `model` name
   - See `automl_agent/orchestrator/planner.py:27-52`

4. **Workflow Stage Tracking**: PlannerAgent uses workflow flags to enforce sequential execution:
   - `data_analyzed`, `candidates_generated`, `models_optimized`, `models_evaluated`
   - The `next_required_action` field tells the planner what MUST happen next
   - See `automl_agent/orchestrator/planner.py:161-182`

## Configuration

Edit `automl_agent/config.yaml` to customize:

- **budget**: Max iterations, time limits, early stopping rounds
- **llm**: Provider (openai/azure), model/deployment name, temperature
- **metrics**: Primary and secondary metrics, task type
- **model_families**: Which algorithms to try (lightgbm, xgboost, catboost, etc.)
- **hpo**: Optimization engine (flaml/optuna), trials, time budget
- **data**: Train/test/validation split, random state, stratification
- **preprocessing**: Auto-detection, missing value handling, scaling, categorical encoding
- **logging**: MLflow settings
- **output**: What artifacts to save

## Output Artifacts

**CLI runs** save to `automl_agent/output/`:
- `model.joblib` - Best trained model pipeline
- `metrics.json` - Evaluation results
- `run_summary.json` - Complete execution trace with config, state, and history
- `config.json` - Configuration used for the run

**API jobs** save to `outputs/{job_id}/` with the same file structure

## Extending the System

### Adding a New Agent

1. Create file in `agents/` with an `execute()` method
2. Update `prompts/planner.txt` to describe when to use it
3. Add action handler in `main.py` orchestration loop
4. Update `AutoMLState` in `orchestrator/state.py` if new state fields are needed

### Adding Custom Metrics

1. Add to `tools/metrics.py` in `CLASSIFICATION_METRICS` or `REGRESSION_METRICS`
2. Update `config.yaml` to use the new metric as primary or secondary

### Changing HPO Engine

In `config.yaml`, set `hpo.engine` to "flaml" or "optuna"

## Important Notes

- The planner (LLM) orchestrates but does NOT train models, see raw data, or calculate metrics
- All ML operations are performed by specialized agents using standard ML libraries
- The system enforces a strict sequential workflow to prevent agent loops
- Results are saved incrementally in the state history for debugging
- MLflow tracking is optional but enabled by default
