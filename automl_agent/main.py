"""
AutoML Agent - Main Entry Point
Orchestrates the entire AutoML workflow using specialized agents.
"""

import argparse
import yaml
from pathlib import Path
from datetime import datetime
import joblib
from typing import Dict, Any

from orchestrator.state import AutoMLState
from orchestrator.planner import PlannerAgent
from agents.data_agent import DataAgent
from agents.modeling_agent import ModelingAgent
from agents.hpo_agent import HPOAgent
from agents.eval_agent import EvaluationAgent
from tools.data_utils import load_data, split_data, validate_data, get_feature_target_split
from tools.preprocessing import apply_preprocessing
from tools.logging import setup_logger, log_config, log_metrics, log_run_summary


def load_config(config_path: str = "automl_agent/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_final_artifacts(state: AutoMLState, config: Dict[str, Any], output_dir: str = "automl_agent/output"):
    """
    Save final model and results.

    Args:
        state: Final AutoML state
        config: Configuration
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save best model
    if state.best_model and config["output"]["save_model"]:
        model_path = output_path / "model.joblib"
        joblib.dump(state.best_model, model_path)
        print(f"✓ Model saved to {model_path}")

    # Save metrics
    if config["output"]["save_metrics"]:
        metrics = {
            "best_model": state.best_model.get("model_name") if state.best_model else None,
            "best_score": state.best_score,
            "primary_metric": config["metrics"]["primary"],
            "all_results": state.evaluation_results,
        }
        log_metrics(metrics, output_dir)
        print(f"✓ Metrics saved to {output_path / 'metrics.json'}")

    # Save run summary
    if config["output"]["save_summary"]:
        summary = {
            "run_date": datetime.now().isoformat(),
            "config": config,
            "state": state.get_summary(),
            "history": state.history,
        }
        log_run_summary(summary, output_dir)
        print(f"✓ Summary saved to {output_path / 'run_summary.json'}")


def run_automl(
    data_path: str,
    target_column: str,
    config_path: str = "automl_agent/config.yaml",
    output_dir: str = "automl_agent/output",
) -> AutoMLState:
    """
    Run the complete AutoML workflow.

    Args:
        data_path: Path to input data file
        target_column: Name of target column
        config_path: Path to configuration file
        output_dir: Output directory for results

    Returns:
        Final AutoML state
    """
    # Setup
    logger = setup_logger("automl", level="INFO")
    logger.info("Starting AutoML Agent...")

    # Load configuration
    config = load_config(config_path)
    log_config(config, output_dir)
    logger.info(f"Configuration loaded from {config_path}")

    # Load and validate data
    logger.info(f"Loading data from {data_path}")
    data = load_data(data_path)

    validation = validate_data(data, target_column)
    if not validation["is_valid"]:
        logger.error(f"Data validation failed: {validation['issues']}")
        raise ValueError(f"Data validation failed: {validation['issues']}")

    logger.info(f"Data validated: {validation['n_samples']} samples, {validation['n_features']} features")

    # Split features and target
    X, y = get_feature_target_split(data, target_column)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=config["data"]["test_size"],
        validation_size=config["data"]["validation_size"],
        random_state=config["data"]["random_state"],
        stratify=config["data"]["stratify"],
    )

    logger.info(f"Data split: train={len(X_train)}, val={len(X_val) if X_val is not None else 0}, test={len(X_test)}")

    # Initialize state
    state = AutoMLState()
    state.status = "running"

    # Initialize agents
    logger.info("Initializing agents...")
    planner = PlannerAgent(config)
    data_agent = DataAgent(config)
    modeling_agent = ModelingAgent(config)
    hpo_agent = HPOAgent(config)
    eval_agent = EvaluationAgent(config)

    # Main orchestration loop
    logger.info("Starting orchestration loop...")
    while True:
        state.iteration += 1
        logger.info(f"\n{'='*50}")
        logger.info(f"Iteration {state.iteration}")
        logger.info(f"{'='*50}")

        # Get next action from planner
        decision = planner.decide_next_action(state.to_dict())
        action = decision["action"]
        reason = decision.get("reason", "")

        logger.info(f"Planner decision: {action}")
        logger.info(f"Reason: {reason}")

        # Execute action
        if action == "run_data_agent":
            logger.info("Running Data Agent...")
            result = data_agent.execute(data, target_column)
            state.dataset_summary = result["dataset_summary"]
            state.preprocessing_plan = result["preprocessing_plan"]
            state.add_to_history("run_data_agent", result)
            logger.info(f"✓ Data analysis complete: {state.dataset_summary['task_type']} task")

        elif action == "run_modeling_agent":
            logger.info("Running Modeling Agent...")
            candidates = modeling_agent.execute(state.dataset_summary, state.preprocessing_plan)
            state.pipeline_candidates = candidates
            state.add_to_history("run_modeling_agent", {"n_candidates": len(candidates)})
            logger.info(f"✓ Generated {len(candidates)} model candidates")

        elif action == "run_hpo_agent":
            logger.info("Running HPO Agent...")

            # Apply preprocessing
            X_train_processed, X_test_processed, preprocessing_pipeline = apply_preprocessing(
                X_train, X_test, state.preprocessing_plan
            )

            # Optimize models
            optimized = hpo_agent.execute(state.pipeline_candidates, X_train_processed, y_train.values)
            state.optimized_models = optimized
            state.total_trials += len(optimized)
            state.add_to_history("run_hpo_agent", {"n_optimized": len(optimized)})
            logger.info(f"✓ Optimized {len(optimized)} models")

        elif action == "run_eval_agent":
            logger.info("Running Evaluation Agent...")

            # Apply preprocessing
            X_train_processed, X_test_processed, preprocessing_pipeline = apply_preprocessing(
                X_train, X_test, state.preprocessing_plan
            )

            # Evaluate models
            eval_results = eval_agent.execute(
                state.optimized_models,
                X_train_processed, y_train.values,
                X_test_processed, y_test.values,
            )

            state.evaluation_results = eval_results["evaluation_results"]
            comparison = eval_results["comparison"]

            # Update best model
            best_model_name = comparison["best_model"]
            best_score = comparison["best_score"]

            best_model_info = next(
                (m for m in state.optimized_models if m["model_name"] == best_model_name),
                None
            )
            state.update_best_model(best_model_info, best_score)

            state.add_to_history("run_eval_agent", comparison)

            logger.info(f"✓ Evaluation complete")
            logger.info(f"  Best model: {best_model_name}")
            logger.info(f"  Best score: {best_score:.4f}")

        elif action == "stop":
            logger.info("Stopping: " + reason)
            state.status = "completed"
            break

        else:
            logger.warning(f"Unknown action: {action}")

        # Update time
        state.total_time = (datetime.now() - state.start_time).total_seconds()

        # Check hard limits
        if state.iteration >= config["budget"]["max_iterations"]:
            logger.info("Max iterations reached")
            state.status = "completed"
            break

        if state.total_time >= config["budget"]["max_time_seconds"]:
            logger.info("Max time reached")
            state.status = "completed"
            break

    # Save final artifacts
    logger.info("\nSaving final results...")
    save_final_artifacts(state, config, output_dir)

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("AutoML Run Complete!")
    logger.info("="*50)
    logger.info(f"Best model: {state.best_model.get('model_name') if state.best_model else 'None'}")
    logger.info(f"Best score ({config['metrics']['primary']}): {state.best_score:.4f}")
    logger.info(f"Total iterations: {state.iteration}")
    logger.info(f"Total time: {state.total_time:.2f}s")
    logger.info(f"Results saved to: {output_dir}")

    return state


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AutoML Agent")
    parser.add_argument("--data", type=str, required=True, help="Path to input data file")
    parser.add_argument("--target", type=str, required=True, help="Name of target column")
    parser.add_argument("--config", type=str, default="automl_agent/config.yaml", help="Path to config file")
    parser.add_argument("--output", type=str, default="automl_agent/output", help="Output directory")

    args = parser.parse_args()

    try:
        state = run_automl(
            data_path=args.data,
            target_column=args.target,
            config_path=args.config,
            output_dir=args.output,
        )
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
