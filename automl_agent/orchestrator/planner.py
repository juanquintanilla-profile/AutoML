"""
Planner Agent
Uses LLM to decide the next action in the AutoML workflow.
"""

from typing import Dict, Any, Optional
import json
import os
from openai import OpenAI, AzureOpenAI
from pathlib import Path


class PlannerAgent:
    """Agent that uses LLM to plan and orchestrate the AutoML workflow."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Planner Agent.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.llm_config = config.get("llm", {})

        # Initialize OpenAI client (supports both OpenAI and Azure OpenAI)
        provider = self.llm_config.get("provider", "openai")

        if provider == "azure":
            # Azure OpenAI configuration
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = self.llm_config.get("api_version", "2024-02-15-preview")

            if not azure_endpoint or not api_key:
                raise ValueError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY environment variables must be set")

            self.client = AzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                api_version=api_version,
            )
            # For Azure, model is the deployment name
            self.model = self.llm_config.get("deployment_name", self.llm_config.get("model", "gpt-4"))
        else:
            # Standard OpenAI configuration
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)
            self.model = self.llm_config.get("model", "gpt-4-turbo-preview")

        self.temperature = self.llm_config.get("temperature", 0.7)
        self.max_tokens = self.llm_config.get("max_tokens", 2000)

        # Load system prompt
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load system prompt from file or use default."""
        prompt_file = Path("automl_agent/prompts/planner.txt")

        if prompt_file.exists():
            return prompt_file.read_text()

        # Default prompt
        return """You are an AutoML planner agent. Your role is to orchestrate the machine learning workflow by deciding which actions to take next.

You have access to the following agents:
1. data_agent: Analyzes datasets and proposes preprocessing
2. modeling_agent: Selects model families and builds pipelines
3. hpo_agent: Optimizes hyperparameters using FLAML/Optuna
4. eval_agent: Evaluates and compares models

Your task is to:
- Decide which agent to run next based on the current state
- Prioritize which models to optimize based on initial results
- Decide when to stop (budget exhausted, target reached, no improvement)

Respond ONLY with a valid JSON object with this structure:
{
  "action": "run_data_agent" | "run_modeling_agent" | "run_hpo_agent" | "run_eval_agent" | "stop",
  "parameters": {
    // action-specific parameters
  },
  "reason": "brief explanation of the decision"
}

Consider:
- Budget constraints (time, iterations, trials)
- Current best score and improvements
- Which models show promise
- When to stop iterating
"""

    def decide_next_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide the next action based on current state.

        Args:
            state: Current AutoML state

        Returns:
            Action decision as dictionary
        """
        # Prepare state summary for LLM
        state_summary = self._prepare_state_summary(state)

        # Create prompt
        user_prompt = f"""Current state of the AutoML system:

{json.dumps(state_summary, indent=2)}

What should be the next action?"""

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse response
            decision_text = response.choices[0].message.content.strip()

            # Extract JSON from response
            if "```json" in decision_text:
                decision_text = decision_text.split("```json")[1].split("```")[0].strip()
            elif "```" in decision_text:
                decision_text = decision_text.split("```")[1].split("```")[0].strip()

            decision = json.loads(decision_text)

            return decision

        except Exception as e:
            # Fallback to rule-based decision
            return self._fallback_decision(state)

    def _prepare_state_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a concise state summary for the LLM.

        Args:
            state: Full state dictionary

        Returns:
            Concise summary
        """
        summary = {
            "iteration": state.get("iteration", 0),
            "status": state.get("status", "unknown"),
            "budget": {
                "max_iterations": self.config["budget"]["max_iterations"],
                "max_time": self.config["budget"]["max_time_seconds"],
                "elapsed_time": state.get("total_time", 0),
            },
            "dataset": {
                "analyzed": state.get("dataset_summary") is not None,
                "n_samples": state.get("dataset_summary", {}).get("n_rows", 0),
                "n_features": state.get("dataset_summary", {}).get("n_features", 0),
                "task_type": state.get("dataset_summary", {}).get("task_type", "unknown"),
            },
            "models": {
                "candidates_generated": len(state.get("pipeline_candidates", [])),
                "optimized": len(state.get("optimized_models", [])),
                "evaluated": len(state.get("evaluation_results", [])),
            },
            "best_result": {
                "model": state.get("best_model", {}).get("model_name") if state.get("best_model") else None,
                "score": state.get("best_score", 0),
                "iterations_without_improvement": state.get("iterations_without_improvement", 0),
            },
            "target_metric": self.config["metrics"]["primary"],
        }

        return summary

    def _fallback_decision(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback decision when LLM fails.

        Args:
            state: Current state

        Returns:
            Fallback decision
        """
        # Check stopping conditions
        if self._should_stop(state):
            return {
                "action": "stop",
                "parameters": {},
                "reason": "Stopping condition met (fallback)",
            }

        # Simple sequential flow
        if not state.get("dataset_summary"):
            return {
                "action": "run_data_agent",
                "parameters": {},
                "reason": "Need to analyze dataset (fallback)",
            }

        if not state.get("pipeline_candidates"):
            return {
                "action": "run_modeling_agent",
                "parameters": {},
                "reason": "Need to generate model candidates (fallback)",
            }

        if not state.get("optimized_models"):
            return {
                "action": "run_hpo_agent",
                "parameters": {},
                "reason": "Need to optimize models (fallback)",
            }

        if not state.get("evaluation_results"):
            return {
                "action": "run_eval_agent",
                "parameters": {},
                "reason": "Need to evaluate models (fallback)",
            }

        return {
            "action": "stop",
            "parameters": {},
            "reason": "All steps completed (fallback)",
        }

    def _should_stop(self, state: Dict[str, Any]) -> bool:
        """
        Check if we should stop the workflow.

        Args:
            state: Current state

        Returns:
            True if should stop
        """
        budget = self.config["budget"]

        # Max iterations reached
        if state.get("iteration", 0) >= budget["max_iterations"]:
            return True

        # Max time reached
        if state.get("total_time", 0) >= budget["max_time_seconds"]:
            return True

        # No improvement for N iterations
        early_stopping = budget.get("early_stopping_rounds", 3)
        if state.get("iterations_without_improvement", 0) >= early_stopping:
            return True

        return False
