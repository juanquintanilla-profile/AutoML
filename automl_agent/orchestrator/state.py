"""
System State Management
Maintains global state of the AutoML system.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class AutoMLState:
    """Global state for the AutoML system."""

    # Dataset information
    dataset_summary: Optional[Dict[str, Any]] = None
    preprocessing_plan: Optional[Dict[str, Any]] = None

    # Model candidates and results
    pipeline_candidates: List[Dict[str, Any]] = field(default_factory=list)
    optimized_models: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_results: List[Dict[str, Any]] = field(default_factory=list)

    # Best model tracking
    best_model: Optional[Dict[str, Any]] = None
    best_score: float = 0.0

    # Iteration tracking
    iteration: int = 0
    iterations_without_improvement: int = 0

    # Resource tracking
    start_time: datetime = field(default_factory=datetime.now)
    total_time: float = 0.0
    total_trials: int = 0

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Status
    status: str = "initialized"  # initialized, running, completed, failed

    def update_best_model(self, model_info: Dict[str, Any], score: float):
        """
        Update best model if score is better.

        Args:
            model_info: Model information
            score: Model score
        """
        if score > self.best_score:
            self.best_model = model_info
            self.best_score = score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1

    def add_to_history(self, action: str, result: Dict[str, Any]):
        """
        Add entry to history.

        Args:
            action: Action taken
            result: Result of action
        """
        self.history.append({
            "iteration": self.iteration,
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        })

    def get_summary(self) -> Dict[str, Any]:
        """
        Get state summary.

        Returns:
            Summary dictionary
        """
        return {
            "status": self.status,
            "iteration": self.iteration,
            "best_score": self.best_score,
            "best_model": self.best_model.get("model_name") if self.best_model else None,
            "total_time": self.total_time,
            "total_trials": self.total_trials,
            "iterations_without_improvement": self.iterations_without_improvement,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert state to dictionary.

        Returns:
            State as dictionary
        """
        return {
            "dataset_summary": self.dataset_summary,
            "preprocessing_plan": self.preprocessing_plan,
            "pipeline_candidates": self.pipeline_candidates,
            "optimized_models": self.optimized_models,
            "evaluation_results": self.evaluation_results,
            "best_model": self.best_model,
            "best_score": self.best_score,
            "iteration": self.iteration,
            "iterations_without_improvement": self.iterations_without_improvement,
            "total_time": self.total_time,
            "total_trials": self.total_trials,
            "status": self.status,
            "history": self.history,
        }

    def save(self, filepath: str):
        """
        Save state to file.

        Args:
            filepath: Path to save state
        """
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, filepath: str) -> 'AutoMLState':
        """
        Load state from file.

        Args:
            filepath: Path to load state from

        Returns:
            Loaded state
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        state = cls()
        for key, value in data.items():
            setattr(state, key, value)

        return state
