# AutoML Agent

A multi-agent AutoML system that uses LLM-based orchestration to automate machine learning workflows.

## Architecture

```
automl_agent/
│
├── main.py                    # Entry point
├── config.yaml                # Configuration (budget, metrics, limits)
│
├── orchestrator/
│   ├── planner.py             # LLM-based decision maker
│   └── state.py               # Global system state
│
├── agents/
│   ├── data_agent.py          # Data analysis & preprocessing
│   ├── modeling_agent.py      # Pipeline generation
│   ├── hpo_agent.py           # Hyperparameter optimization (FLAML/Optuna)
│   └── eval_agent.py          # Model evaluation & comparison
│
├── tools/
│   ├── data_utils.py          # Data loading, splitting, validation
│   ├── preprocessing.py       # Encoders, scaling, imputation
│   ├── metrics.py             # Evaluation metrics
│   └── logging.py             # MLflow integration
│
├── memory/
│   └── runs.json              # Historical runs
│
├── prompts/
│   └── planner.txt            # LLM orchestrator prompt
│
└── output/
    ├── model.joblib           # Final model
    ├── metrics.json           # Results
    └── run_summary.json       # Execution trace
```

## Technology Stack

### LLM / Planner
- `openai` - GPT-4 for orchestration decisions

### AutoML / Search
- `flaml` - Fast AutoML optimization
- `optuna` - Hyperparameter tuning

### Modeling
- `scikit-learn` - ML pipelines
- `lightgbm` - Gradient boosting
- `xgboost` - Gradient boosting
- `catboost` - Gradient boosting

### Data
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `ydata-profiling` - Data profiling (optional)

### Evaluation
- `scikit-learn` - Metrics
- `mlflow` - Experiment tracking

### Infrastructure
- `joblib` - Model serialization
- `pyyaml` - Configuration

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Usage

### Basic Usage

```bash
python -m automl_agent.main \
  --data path/to/data.csv \
  --target target_column_name
```

### With Custom Configuration

```bash
python -m automl_agent.main \
  --data path/to/data.csv \
  --target target_column_name \
  --config path/to/custom_config.yaml \
  --output path/to/output_dir
```

## Configuration

Edit `automl_agent/config.yaml` to customize:

- **Budget**: Max iterations, time limits, early stopping
- **LLM**: Model selection, temperature, max tokens
- **Metrics**: Primary and secondary metrics
- **Model Families**: Which algorithms to try
- **HPO**: Optimization engine and parameters
- **Data**: Train/test split, preprocessing options
- **Logging**: MLflow settings

## How It Works

### Workflow

```
main.py
  ↓
Data Agent → Analyze dataset, propose preprocessing
  ↓
Planner (LLM) → Decide next action
  ↓
Modeling Agent → Generate model candidates
  ↓
HPO Agent → Optimize hyperparameters (FLAML/Optuna)
  ↓
Evaluation Agent → Compare models
  ↓
Planner → Decide: iterate or stop
```

### Agent Communication

Agents communicate via **structured state** (JSON), not natural language:

```json
{
  "action": "run_hpo",
  "parameters": {"model_family": "lightgbm"},
  "reason": "best trade-off so far"
}
```

### What the LLM Does

✅ Decides which agent to run next
✅ Prioritizes which models to optimize
✅ Decides when to stop

❌ Does NOT train models
❌ Does NOT see raw data
❌ Does NOT calculate metrics

### Stopping Conditions

- Budget exhausted (time or iterations)
- No improvement for N iterations
- Target score reached

## Output Artifacts

After completion, check `automl_agent/output/`:

- `model.joblib` - Best model pipeline
- `metrics.json` - Evaluation results
- `run_summary.json` - Complete execution trace
- `config.json` - Configuration used

## Example

```python
from automl_agent.main import run_automl

state = run_automl(
    data_path="data/train.csv",
    target_column="target",
    config_path="automl_agent/config.yaml",
    output_dir="automl_agent/output"
)

print(f"Best model: {state.best_model['model_name']}")
print(f"Best score: {state.best_score}")
```

## Extending

### Add a New Agent

1. Create `agents/new_agent.py`
2. Implement `execute()` method
3. Update `planner.txt` to describe when to use it
4. Add action handler in `main.py`

### Add Custom Metrics

1. Edit `tools/metrics.py`
2. Add to `CLASSIFICATION_METRICS` or `REGRESSION_METRICS`
3. Update `config.yaml` to use it

### Change HPO Engine

In `config.yaml`:
```yaml
hpo:
  engine: "optuna"  # or "flaml"
```

## License

MIT
