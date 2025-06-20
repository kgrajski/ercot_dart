"""Bootstrap Evaluation Module for ERCOT DART Price Classification.

This module provides the BootstrapEvaluator class which handles bootstrap resampling
evaluation with comprehensive classification metrics. This separates evaluation logic from model
training logic for better architecture and maintainability.

Key Features:
- Bootstrap resampling with out-of-bag (OOB) evaluation
- Comprehensive classification metrics via EvaluationMetrics integration
- Robust error handling for numerical issues
- Backward compatibility with existing result formats
- Extensible design for future evaluation strategies (e.g., CrossValidation)

Usage:
    evaluator = EvaluationMetrics()
    bootstrap_eval = BootstrapEvaluator(evaluator, random_state=42)
    results = bootstrap_eval.evaluate_model_with_bootstrap(model, X_train, y_train, X_validation, y_validation)
"""

from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from src.models.ercot.exp2.evaluation_metrics import EvaluationMetrics


class BootstrapEvaluator:
    """Handles bootstrap resampling evaluation with comprehensive metrics.

    This class encapsulates all bootstrap evaluation logic, making it reusable
    across different model types and easily testable in isolation.
    """

    def __init__(self, metrics_evaluator: EvaluationMetrics, random_state: int = 42):
        """Initialize bootstrap evaluator.

        Args:
            metrics_evaluator: EvaluationMetrics instance for comprehensive metrics
            random_state: Random seed for reproducible bootstrap sampling
        """
        self.evaluator = metrics_evaluator
        self.random_state = random_state

    def evaluate_model_with_bootstrap(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.Series] = None,
        bootstrap_iterations: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate model performance using bootstrap resampling + final validation with comprehensive metrics.

        CRITICAL EVALUATION WORKFLOW:
        This method performs TWO DISTINCT types of evaluation:

        1. Bootstrap performance estimation: Creates TEMPORARY models on bootstrap
           samples of training data to estimate performance variability and uncertainty.
           These bootstrap metrics give us confidence intervals but are NOT the final
           model performance.

        2. Final validation: Evaluates the MAIN MODEL (which was trained on ALL training data)
           against the holdout validation set. These validation metrics represent the
           TRUE performance of our deployed model on unseen data.

        Args:
            model: Trained model (ALREADY FITTED ON ALL TRAINING DATA - this is the main model)
            X_train: Training features (2024 data for this hour) - only used for bootstrap estimation
            y_train: Training target (2024 data for this hour) - only used for bootstrap estimation
            X_validation: Validation features (2025 data for this hour) - HOLDOUT DATA for final evaluation
            y_validation: Validation target (2025 data for this hour) - HOLDOUT DATA for final evaluation
            bootstrap_iterations: Number of bootstrap iterations for performance estimation

        Returns:
            Dictionary with evaluation metrics:
            - bootstrap_*_mean/std: Bootstrap performance estimates (from temporary models on training data)
            - validation_*: FINAL PERFORMANCE METRICS (main model evaluated on holdout validation data)
            - Comprehensive metrics from EvaluationMetrics class
        """
        results = {}

        # === PART 1: Bootstrap Performance Estimation ===
        # This creates temporary models to estimate uncertainty, NOT final performance
        bootstrap_results = self._run_bootstrap_evaluation(
            model, X_train, y_train, bootstrap_iterations
        )
        results.update(bootstrap_results)

        # === PART 2: Final Validation Evaluation ===
        # THIS IS THE REAL PERFORMANCE: Main model (trained on all training data) on holdout validation data
        if X_validation is not None and y_validation is not None:
            validation_results = self._run_validation_evaluation(
                model, X_validation, y_validation
            )
            results.update(validation_results)

        return results

    def _run_bootstrap_evaluation(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        bootstrap_iterations: int,
    ) -> Dict[str, Any]:
        """Run bootstrap performance estimation.

        Creates temporary models on bootstrap samples to estimate performance variability.
        """
        bootstrap_y_true_list = []
        bootstrap_y_pred_list = []
        bootstrap_timestamps_list = []
        failed_iterations = 0

        print(
            f"  Starting bootstrap evaluation with {bootstrap_iterations} iterations on {len(X_train)} training samples"
        )

        for i in range(bootstrap_iterations):
            try:
                # Create bootstrap sample (sample with replacement, same size as original)
                X_boot, y_boot = resample(
                    X_train,
                    y_train,
                    n_samples=len(X_train),
                    random_state=self.random_state
                    + i,  # Different seed for each iteration
                )

                # Identify out-of-bag (OOB) samples - samples not in bootstrap
                boot_indices = set(X_boot.index)
                oob_indices = [idx for idx in X_train.index if idx not in boot_indices]

                print(
                    f"    Bootstrap {i+1}: Bootstrap sample={len(X_boot)}, OOB samples={len(oob_indices)}"
                )

                # Determine test set (OOB or fallback)
                X_test, y_test = self._get_bootstrap_test_set(
                    X_train, y_train, oob_indices, i + 1
                )

                # Train and evaluate temporary model
                bootstrap_model = clone(model)  # Temporary model for evaluation only

                if self._train_bootstrap_model(bootstrap_model, X_boot, y_boot, i + 1):
                    if self._evaluate_bootstrap_model(
                        bootstrap_model,
                        X_test,
                        y_test,
                        i + 1,
                        bootstrap_y_true_list,
                        bootstrap_y_pred_list,
                        bootstrap_timestamps_list,
                    ):
                        continue  # Success - continue to next iteration

                failed_iterations += 1

            except Exception as e:
                print(f"    Bootstrap {i+1}: Unexpected error - {str(e)}")
                failed_iterations += 1
                continue

        # Calculate comprehensive bootstrap metrics
        return self._calculate_bootstrap_metrics(
            bootstrap_y_true_list,
            bootstrap_y_pred_list,
            bootstrap_timestamps_list,
            bootstrap_iterations,
            failed_iterations,
            len(X_train),
        )

    def _get_bootstrap_test_set(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        oob_indices: List,
        iteration: int,
    ) -> tuple:
        """Determine test set for bootstrap evaluation (OOB or fallback)."""
        if len(oob_indices) < 5:  # Need minimum samples for meaningful evaluation
            # Fallback: use a random 20% of original data as test set
            test_size = max(5, int(0.2 * len(X_train)))
            test_indices = np.random.choice(
                X_train.index, size=test_size, replace=False
            )
            X_test = X_train.loc[test_indices]
            y_test = y_train.loc[test_indices]
            print(
                f"    Bootstrap {iteration}: Using fallback random test set of {len(X_test)} samples (OOB too small)"
            )
        else:
            # Use out-of-bag samples as test set
            X_test = X_train.loc[oob_indices]
            y_test = y_train.loc[oob_indices]
            print(
                f"    Bootstrap {iteration}: Using OOB test set of {len(X_test)} samples"
            )

        return X_test, y_test

    def _train_bootstrap_model(
        self,
        bootstrap_model: Any,
        X_boot: pd.DataFrame,
        y_boot: pd.Series,
        iteration: int,
    ) -> bool:
        """Train bootstrap model with error handling."""
        try:
            bootstrap_model.fit(X_boot, y_boot)
            return True
        except (ValueError, RuntimeWarning, np.linalg.LinAlgError) as fit_error:
            print(f"    Bootstrap {iteration}: Model fitting failed - {str(fit_error)}")
            return False
        except Exception as e:
            print(f"    Bootstrap {iteration}: Unexpected training error - {str(e)}")
            return False

    def _evaluate_bootstrap_model(
        self,
        bootstrap_model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        iteration: int,
        y_true_list: List,
        y_pred_list: List,
        timestamps_list: List,
    ) -> bool:
        """Evaluate bootstrap model and store results."""
        try:
            y_pred = bootstrap_model.predict(X_test)

            # Validate predictions are finite
            if not np.all(np.isfinite(y_pred)):
                print(
                    f"    Bootstrap {iteration}: Predictions contain NaN/infinite values"
                )
                return False

            # Store for comprehensive metrics calculation
            y_true_list.append(y_test.values)
            y_pred_list.append(y_pred)
            # Use test indices as proxy for timestamps (ordered)
            timestamps_list.append(X_test.index.values)

            # Quick accuracy calculation for progress reporting
            score = accuracy_score(y_test, y_pred)
            print(f"    Bootstrap {iteration}: Accuracy = {score:.4f}")

            return True

        except (ValueError, RuntimeWarning, np.linalg.LinAlgError) as pred_error:
            print(
                f"    Bootstrap {iteration}: Prediction/scoring failed - {str(pred_error)}"
            )
            return False
        except Exception as e:
            print(f"    Bootstrap {iteration}: Unexpected evaluation error - {str(e)}")
            return False

    def _calculate_bootstrap_metrics(
        self,
        y_true_list: List,
        y_pred_list: List,
        timestamps_list: List,
        bootstrap_iterations: int,
        failed_iterations: int,
        n_train_samples: int,
    ) -> Dict[str, Any]:
        """Calculate comprehensive bootstrap metrics with error handling."""
        if len(y_true_list) == 0:
            # All bootstrap iterations failed - set metrics to NaN
            print(
                f"  WARNING: All {bootstrap_iterations} bootstrap iterations failed! Setting bootstrap metrics to NaN"
            )
            return {
                "bootstrap_accuracy_mean": np.nan,
                "bootstrap_accuracy_std": np.nan,
                "n_train_samples": n_train_samples,
                "bootstrap_iterations": bootstrap_iterations,
                "failed_iterations": failed_iterations,
                "success_rate": 0.0,
            }

        # Calculate comprehensive metrics across all bootstrap samples
        try:
            bootstrap_summary = self.evaluator.calculate_bootstrap_metrics(
                y_true_list, y_pred_list, timestamps_list
            )

            success_rate = len(y_true_list) / bootstrap_iterations
            if failed_iterations > 0:
                print(
                    f"  WARNING: {failed_iterations}/{bootstrap_iterations} bootstrap iterations failed (success rate: {success_rate:.1%})"
                )

            # Extract results and flatten for storage
            flattened_bootstrap = self.evaluator.format_for_storage(
                bootstrap_summary, prefix="bootstrap_"
            )

            # Add backward compatibility and summary fields
            result = {
                **flattened_bootstrap,
                "bootstrap_accuracy_mean": bootstrap_summary.get("accuracy", {}).get(
                    "mean", np.nan
                ),
                "bootstrap_accuracy_std": bootstrap_summary.get("accuracy", {}).get(
                    "std", np.nan
                ),
                "n_train_samples": n_train_samples,
                "bootstrap_iterations": bootstrap_iterations,
                "failed_iterations": failed_iterations,
                "success_rate": success_rate,
            }

            return result

        except Exception as e:
            print(f"  ERROR: Bootstrap metrics calculation failed - {str(e)}")
            return {
                "bootstrap_accuracy_mean": np.nan,
                "bootstrap_accuracy_std": np.nan,
                "n_train_samples": n_train_samples,
                "bootstrap_iterations": bootstrap_iterations,
                "failed_iterations": failed_iterations,
                "success_rate": len(y_true_list) / bootstrap_iterations,
                "bootstrap_calculation_error": str(e),
            }

    def _run_validation_evaluation(
        self, model: Any, X_validation: pd.DataFrame, y_validation: pd.Series
    ) -> Dict[str, Any]:
        """Run final validation evaluation on holdout data.

        CRITICAL: This method evaluates the MAIN MODEL (which was trained on ALL training data)
        against the holdout validation set. This gives us the TRUE performance metrics that
        represent how well our deployed model will perform on unseen data.

        The model passed here is NOT a bootstrap model - it's the main model that was fitted
        on the complete training dataset and will be used for actual predictions.
        """
        try:
            # MAIN MODEL PREDICTION ON HOLDOUT VALIDATION DATA
            # This is the real performance evaluation - not bootstrap estimation
            y_pred = model.predict(X_validation)  # Main model on holdout data

            # Calculate comprehensive validation metrics
            validation_metrics = self.evaluator.calculate_metrics(
                y_validation.values,
                y_pred,
                timestamps=X_validation.index.values,  # Use DataFrame index as timestamp proxy
            )

            # Format and add to results
            flattened_validation = self.evaluator.format_for_storage(
                validation_metrics, prefix="validation_"
            )

            # Add backward compatibility fields
            result = {
                **flattened_validation,
                "validation_accuracy": validation_metrics.get("accuracy", np.nan),
                "n_validation_samples": len(X_validation),
                "validation_failed": False,
            }

            return result

        except Exception as e:
            print(f"  ERROR: Validation evaluation failed - {str(e)}")
            return {
                "validation_accuracy": np.nan,
                "n_validation_samples": len(X_validation),
                "validation_failed": True,
                "validation_error": str(e),
            }

    def get_supported_strategies(self) -> List[str]:
        """Return list of supported evaluation strategies.

        Future extension point for adding CrossValidation, etc.
        """
        return ["bootstrap"]

    def set_random_state(self, random_state: int) -> None:
        """Update random state for reproducible results."""
        self.random_state = random_state
