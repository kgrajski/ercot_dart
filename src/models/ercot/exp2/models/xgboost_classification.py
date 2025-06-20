"""XGBoost Classification Model for Experiment 2."""

import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

from src.features.ercot.visualization import COLOR_SEQUENCE
from src.features.ercot.visualization import PROFESSIONAL_COLORS
from src.features.ercot.visualization import SEMANTIC_COLORS
from src.features.ercot.visualization import apply_professional_axis_styling
from src.features.ercot.visualization import get_professional_layout
from src.models.ercot.exp2.base_classification_model import BaseClassificationModel
from src.models.ercot.exp2.cross_hour_analyzer import CrossHourAnalyzer


class XGBoostClassificationModel(BaseClassificationModel):
    """XGBoost Classification implementation for Experiment 2.

    This class implements XGBoost gradient boosting for DART price classification
    with enhanced cross-hour analysis capabilities.

    XGBoost is particularly well-suited for electricity market classification because it can:
    - Learn non-linear decision boundaries between pricing regimes
    - Capture complex feature interactions (e.g., low wind + high load = price spike)
    - Handle mixed categorical and numerical features naturally
    - Provide feature importance rankings for market insight
    - Maintain temporal patterns through ensemble cross-hour analysis
    """

    def __init__(
        self,
        output_dir: str,
        settlement_point: str,
        random_state: int = 42,
        feature_scaling: str = "none",  # XGBoost works best without scaling
        use_synthetic_data: bool = False,
        use_dart_features: bool = True,
        # Classification-specific parameters
        classification_strategy: str = "sign_only",
        classification_config: Optional[Dict] = None,
        # XGBoost-specific parameters
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        min_child_weight: int = 1,
        # Cross-hour analysis parameters
        enable_cross_hour_analysis: bool = True,
        cross_hour_ensemble: bool = False,  # Use ensemble of nearby hours
        ensemble_window: int = 2,  # ±2 hours for ensemble
    ):
        """Initialize XGBoost Classification Model.

        Args:
            output_dir: Directory for saving outputs and artifacts
            settlement_point: Settlement point identifier
            random_state: Random seed for reproducibility
            feature_scaling: Feature scaling method (should be 'none' for XGBoost)
            use_synthetic_data: Whether to inject synthetic data for validation
            use_dart_features: Whether to include DART lag/rolling features

            classification_strategy: Strategy for target transformation
            classification_config: Configuration dictionary for classification strategy

            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            min_child_weight: Minimum sum of instance weight needed in a child

            enable_cross_hour_analysis: Whether to perform cross-hour pattern analysis
            cross_hour_ensemble: Whether to use ensemble of nearby hour models
            ensemble_window: Window size for ensemble (±hours)
        """
        super().__init__(
            model_type="xgboost_classification",
            output_dir=output_dir,
            settlement_point=settlement_point,
            random_state=random_state,
            feature_scaling=feature_scaling,
            use_synthetic_data=use_synthetic_data,
            use_dart_features=use_dart_features,
            classification_strategy=classification_strategy,
            classification_config=classification_config,
        )

        # XGBoost parameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight

        # Cross-hour analysis parameters
        self.enable_cross_hour_analysis = enable_cross_hour_analysis
        self.cross_hour_ensemble = cross_hour_ensemble
        self.ensemble_window = ensemble_window

        # Initialize cross-hour analyzer
        self.cross_hour_analyzer = None

    def _initialize_cross_hour_analyzer(self):
        """Initialize the cross-hour analyzer with current model configuration."""
        if self.enable_cross_hour_analysis and self.cross_hour_analyzer is None:
            self.cross_hour_analyzer = CrossHourAnalyzer(
                feature_names=self.feature_names,
                class_labels=self.class_labels,
                output_dir=self.model_dir,
                random_state=self.random_state,
            )

    def _train_model_for_hour(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_validation: Optional[pd.DataFrame] = None,
        y_validation: Optional[pd.Series] = None,
        bootstrap_iterations: int = 5,
        hour: int = 1,
    ) -> Tuple[Any, Dict]:
        """Train XGBoost model for a specific hour with cross-hour analysis."""

        print(f"   Training XGBoost for hour {hour}...")
        print(
            f"   Training samples: {len(X_train)}, Features: {len(self.feature_names)}"
        )

        # Check for class balance
        class_counts = y_train.value_counts().sort_index()
        print(f"   Class distribution: {dict(class_counts)}")

        # Handle class imbalance with scale_pos_weight for binary classification
        scale_pos_weight = None
        if len(self.class_labels) == 2:
            n_negative = class_counts.get(0, 1)
            n_positive = class_counts.get(1, 1)
            scale_pos_weight = n_negative / n_positive
            print(f"   Scale pos weight: {scale_pos_weight:.3f}")

        # Configure XGBoost parameters
        xgb_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Set objective based on number of classes
        if len(self.class_labels) == 2:
            xgb_params["objective"] = "binary:logistic"
            xgb_params["eval_metric"] = "logloss"
            if scale_pos_weight is not None:
                xgb_params["scale_pos_weight"] = scale_pos_weight
        else:
            xgb_params["objective"] = "multi:softprob"
            xgb_params["num_class"] = len(self.class_labels)
            xgb_params["eval_metric"] = "mlogloss"

        # Train model
        model = xgb.XGBClassifier(**xgb_params)

        # Simple fit without early stopping for now (version compatibility)
        model.fit(X_train, y_train, verbose=False)

        # Evaluate model
        results = self._evaluate_model(
            model, X_train, y_train, X_validation, y_validation, bootstrap_iterations
        )

        # Add XGBoost-specific results
        results["n_estimators_used"] = (
            model.best_iteration
            if hasattr(model, "best_iteration")
            else self.n_estimators
        )
        results["feature_importances"] = model.feature_importances_.tolist()
        results["feature_names"] = self.feature_names
        results["feature_importance_dict"] = dict(
            zip(self.feature_names, model.feature_importances_)
        )

        # Perform cross-hour analysis if enabled
        if self.enable_cross_hour_analysis:
            self._initialize_cross_hour_analyzer()
            cross_hour_results = self.cross_hour_analyzer.analyze_hour_patterns(
                model=model,
                X_train=X_train,
                y_train=y_train,
                hour=hour,
                model_type="xgboost",
            )
            results.update(cross_hour_results)

        print(f"   ✅ Hour {hour} training completed")
        print(
            f"   Accuracy: {results['train_accuracy']:.3f}, Precision: {results['train_precision']:.3f}"
        )

        return model, results

    def generate_cross_hour_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive cross-hour analysis report."""

        if not self.enable_cross_hour_analysis or self.cross_hour_analyzer is None:
            return {"error": "Cross-hour analysis not enabled or no analyzer available"}

        return self.cross_hour_analyzer.generate_cross_hour_analysis_report()

    def create_cross_hour_visualization_dashboard(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """Create comprehensive cross-hour analysis visualization dashboard."""

        if not self.enable_cross_hour_analysis or self.cross_hour_analyzer is None:
            raise ValueError("Cross-hour analysis not enabled or no analyzer available")

        return self.cross_hour_analyzer.create_cross_hour_visualization_dashboard(
            output_path
        )

    def get_cross_hour_patterns(self) -> Dict[int, Dict]:
        """Get stored cross-hour patterns for external analysis."""

        if not self.enable_cross_hour_analysis or self.cross_hour_analyzer is None:
            return {}

        return self.cross_hour_analyzer.cross_hour_patterns
