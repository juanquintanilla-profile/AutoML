# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```bash
# === Docker (Recommended - No API keys needed) ===
docker-compose up              # First run downloads model (~4GB)
# Open http://localhost:8501 for UI, http://localhost:8000/docs for API

# === Local Development ===
# Run AutoML on dataset (CLI)
python -m automl_agent --data data/customer_churn.csv --target churn

# Run with custom config/output
python -m automl_agent --data data.csv --target target_col --config path/to/config.yaml --output path/to/output

# Start API server
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start Streamlit UI (requires API running)
streamlit run ui/app.py

# Generate test dataset
python generate_example_data.py
```

## Project Overview

AutoML Agent is a multi-agent system that uses LLM-based orchestration to automate machine learning workflows. Supports **Ollama** (local/free), **OpenAI**, and **Azure OpenAI** as LLM providers. The planner (LLM) orchestrates but does NOT train models, see raw data, or calculate metrics - all ML operations are performed by specialized agents using standard ML libraries.

## Architecture

### Core Orchestration Flow

```
DataAgent → ModelingAgent → HPOAgent → EvaluationAgent → PlannerAgent (loop or stop)
```

Each agent runs ONCE per stage. PlannerAgent decides whether to iterate (re-run HPO/Eval) or stop.

### Key Files

| File | Purpose |
|------|---------|
| `automl_agent/main.py` | Entry point, orchestration loop |
| `automl_agent/orchestrator/planner.py` | LLM-based decision maker |
| `automl_agent/orchestrator/state.py` | `AutoMLState` dataclass for global state |
| `automl_agent/prompts/planner.txt` | System prompt for LLM orchestrator |
| `api/main.py` | FastAPI application |
| `api/jobs.py` | Background job execution |
| `ui/app.py` | Streamlit UI |

### Agent Communication

Agents communicate through structured state (JSON), NOT natural language. Each agent updates specific parts of `AutoMLState`, and the main loop coordinates execution based on planner decisions.

### Workflow Stage Flags

PlannerAgent enforces sequential execution via flags in state:
- `data_analyzed`, `candidates_generated`, `models_optimized`, `models_evaluated`
- `next_required_action` field tells planner what MUST happen next
- See `automl_agent/orchestrator/planner.py:161-182`

## Critical Implementation Details

1. **Task Type Detection** (`automl_agent/tools/data_utils.py:infer_task_type`):
   - Auto-detects classification vs regression based on target column
   - Classification: accuracy/f1/precision/recall metrics, stratified splits, `y.astype(int)`
   - Regression: r2/rmse/mae metrics, `y.astype(float)`
   - Task type set in `main.py:157-172` after inferring from target

2. **FLAML Type Requirements** (`automl_agent/agents/hpo_agent.py`):
   - Classification: target must be `int`
   - Regression: target must be `float`
   - Target prepared via `prepare_target()` in `tools/data_utils.py`

3. **LLM Provider Configuration** (`automl_agent/orchestrator/planner.py:42-87`):
   - Set `llm.provider` in `config.yaml` to "ollama", "openai", or "azure"
   - Ollama (default): free, local - requires `ollama_model` and `ollama_host`
   - OpenAI: requires `OPENAI_API_KEY` env var and `model` name
   - Azure: requires `AZURE_OPENAI_*` env vars, `deployment_name` and `api_version`

4. **Planner Fallback** (`automl_agent/orchestrator/planner.py:252-303`):
   - If LLM fails to respond, rule-based `_fallback_decision()` ensures workflow continues
   - Follows sequential flow: data_agent → modeling_agent → hpo_agent → eval_agent → stop

## Output Locations

- **CLI runs**: `automl_agent/output/` (model.joblib, metrics.json, run_summary.json, config.json)
- **API jobs**: `outputs/{job_id}/` (same files)
- **Database**: `outputs/jobs.db` (SQLite for job persistence)

## Configuration

Edit `automl_agent/config.yaml`:
- `budget`: max iterations, time limits, early stopping
- `llm`: provider, model/deployment, temperature
- `metrics`: primary/secondary metrics
- `model_families`: algorithms to try (lightgbm, xgboost, catboost, etc.)
- `hpo.engine`: "flaml" or "optuna"

## Environment Setup

### Option 1: Docker (Recommended)
```bash
docker-compose up
```
No configuration needed. Uses Ollama with llama3.2 model (downloads on first run).

### Option 2: Local with Ollama
1. Install [Ollama](https://ollama.ai): `curl -fsSL https://ollama.ai/install.sh | sh`
2. Pull model: `ollama pull llama3.2`
3. `pip install -r requirements.txt && pip install -r requirements-api.txt`
4. Run (Ollama starts automatically)

### Option 3: Local with OpenAI/Azure
1. `pip install -r requirements.txt && pip install -r requirements-api.txt`
2. Create `.env` with:
   - OpenAI: `OPENAI_API_KEY`
   - Azure: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
3. Update `config.yaml`: set `llm.provider` to "openai" or "azure"

## Extending the System

### Adding a New Agent

1. Create file in `agents/` with `execute()` method
2. Update `prompts/planner.txt` to describe when to use it
3. Add action handler in `main.py` orchestration loop
4. Update `AutoMLState` in `orchestrator/state.py` if new state fields needed

### Adding Custom Metrics

1. Add to `tools/metrics.py` in `CLASSIFICATION_METRICS` or `REGRESSION_METRICS`
2. Update `config.yaml` to use new metric

## API Endpoints

The FastAPI server (`api/main.py`) exposes:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/jobs` | POST | Create new AutoML job (upload CSV + target column) |
| `/jobs/{job_id}` | GET | Get job status and results |
| `/jobs/{job_id}/model` | GET | Download trained model (.joblib) |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI documentation |

Jobs are stored in SQLite (`outputs/jobs.db`) with results in `outputs/{job_id}/`.
