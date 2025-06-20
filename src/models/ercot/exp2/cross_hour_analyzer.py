"""Model-agnostic cross-hour analysis framework for classification models."""

import json
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Protocol
from typing import Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score

from src.features.ercot.visualization import COLOR_SEQUENCE
from src.features.ercot.visualization import PROFESSIONAL_COLORS
from src.features.ercot.visualization import SEMANTIC_COLORS
from src.features.ercot.visualization import apply_professional_axis_styling
from src.features.ercot.visualization import get_professional_layout


class ClassificationModel(Protocol):
    """Protocol defining the interface that classification models must implement for cross-hour analysis."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on input data."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        ...


class ModelComplexityExtractor(Protocol):
    """Protocol for extracting model-specific complexity metrics."""

    def extract_complexity_metrics(self, model: Any) -> Dict[str, Any]:
        """Extract complexity metrics specific to this model type."""
        ...


class XGBoostComplexityExtractor:
    """XGBoost-specific complexity extraction."""

    def extract_complexity_metrics(self, model: Any) -> Dict[str, Any]:
        """Extract XGBoost-specific complexity metrics."""
        try:
            # Number of trees
            n_trees = len(model.get_booster().get_dump())

            # Average tree depth (approximate)
            tree_dumps = model.get_booster().get_dump()
            total_depth = 0
            for tree_dump in tree_dumps[:10]:  # Sample first 10 trees for efficiency
                depth = tree_dump.count("\t")  # Simple depth estimation
                total_depth += depth
            avg_depth = total_depth / min(10, len(tree_dumps))

            return {
                "model_type": "xgboost",
                "n_trees": n_trees,
                "avg_tree_depth": avg_depth,
                "complexity_score": n_trees * avg_depth,  # Simple combined metric
            }

        except Exception as e:
            return {"complexity_analysis_error": str(e)}


class GenericComplexityExtractor:
    """Generic complexity extraction for models without specific extractors."""

    def extract_complexity_metrics(self, model: Any) -> Dict[str, Any]:
        """Extract generic complexity metrics."""
        try:
            # Count number of parameters if available
            n_params = 0
            if hasattr(model, "coef_"):
                # Linear models
                n_params = np.prod(model.coef_.shape)
            elif hasattr(model, "n_features_in_"):
                # Sklearn models with feature count
                n_params = model.n_features_in_

            return {
                "model_type": type(model).__name__.lower(),
                "n_parameters": n_params,
                "complexity_score": n_params,  # Simple metric
            }

        except Exception as e:
            return {"complexity_analysis_error": str(e)}


class CrossHourAnalyzer:
    """Model-agnostic cross-hour analysis framework for classification models.

    This class provides comprehensive analysis capabilities that work with any
    classification model implementing the ClassificationModel protocol.
    """

    def __init__(
        self,
        feature_names: List[str],
        class_labels: List[str],
        output_dir: str,
        random_state: int = 42,
    ):
        """Initialize the cross-hour analyzer.

        Args:
            feature_names: List of feature names used by models
            class_labels: List of class labels for classification
            output_dir: Directory for saving analysis outputs
            random_state: Random seed for reproducibility
        """
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.output_dir = output_dir
        self.random_state = random_state

        # Storage for cross-hour patterns
        self.cross_hour_patterns = {}
        self.hour_similarity_matrix = None
        self.ensemble_weights = {}
        self.temporal_consistency_scores = {}

        # Model complexity extractors
        self.complexity_extractors = {
            "xgboost": XGBoostComplexityExtractor(),
            "generic": GenericComplexityExtractor(),
        }

    def analyze_hour_patterns(
        self,
        model: ClassificationModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hour: int,
        model_type: str = "generic",
    ) -> Dict[str, Any]:
        """Analyze cross-hour patterns for a specific hour and model.

        Args:
            model: Trained classification model
            X_train: Training features
            y_train: Training targets
            hour: Hour being analyzed
            model_type: Type of model for complexity analysis

        Returns:
            Dictionary containing analysis results
        """
        cross_hour_results = {}

        try:
            # 1. Feature importance ranking
            feature_importance_ranking = self._get_feature_importance_ranking(model)
            cross_hour_results["feature_ranking"] = feature_importance_ranking

            # 2. Pattern stability analysis
            pattern_stability = self._analyze_pattern_stability(
                model, X_train, y_train, hour
            )
            cross_hour_results["pattern_stability"] = pattern_stability

            # 3. Decision boundary analysis
            boundary_analysis = self._analyze_decision_boundaries(
                model, X_train, y_train, hour
            )
            cross_hour_results["boundary_analysis"] = boundary_analysis

            # 4. Model complexity analysis
            complexity_metrics = self._calculate_model_complexity(model, model_type)
            cross_hour_results["model_complexity"] = complexity_metrics

            # Store for cross-hour comparison
            self.cross_hour_patterns[hour] = {
                "feature_ranking": feature_importance_ranking,
                "pattern_stability": pattern_stability,
                "boundary_analysis": boundary_analysis,
                "model_complexity": complexity_metrics,
            }

        except Exception as e:
            print(f"   ⚠️  Cross-hour analysis failed for hour {hour}: {e}")
            cross_hour_results["cross_hour_analysis_error"] = str(e)

        return cross_hour_results

    def _get_feature_importance_ranking(
        self, model: ClassificationModel
    ) -> Dict[str, Any]:
        """Get detailed feature importance analysis (model-agnostic)."""

        # Extract feature importance based on model type
        if hasattr(model, "feature_importances_"):
            # Tree-based models (XGBoost, Random Forest, etc.)
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            # Linear models (Logistic Regression, etc.)
            if len(model.coef_.shape) == 1:
                importances = np.abs(model.coef_)
            else:
                importances = np.abs(model.coef_).mean(axis=0)
        else:
            # Fallback: use permutation importance or uniform
            print(
                f"   ⚠️  No feature importance method found for {type(model).__name__}"
            )
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        features = self.feature_names

        # Create ranking
        importance_df = pd.DataFrame(
            {"feature": features, "importance": importances}
        ).sort_values("importance", ascending=False)

        # Categorize features by type
        feature_categories = {
            "temporal": [],
            "load_forecast": [],
            "wind_forecast": [],
            "solar_forecast": [],
            "dart_historical": [],
            "other": [],
        }

        for feature in features:
            if any(
                temp_keyword in feature.lower()
                for temp_keyword in [
                    "hour",
                    "dow",
                    "doy",
                    "month",
                    "week",
                    "holiday",
                    "weekend",
                ]
            ):
                feature_categories["temporal"].append(feature)
            elif "load" in feature.lower():
                feature_categories["load_forecast"].append(feature)
            elif "wind" in feature.lower():
                feature_categories["wind_forecast"].append(feature)
            elif "solar" in feature.lower():
                feature_categories["solar_forecast"].append(feature)
            elif "dart" in feature.lower():
                feature_categories["dart_historical"].append(feature)
            else:
                feature_categories["other"].append(feature)

        # Calculate category importance
        category_importance = {}
        for category, cat_features in feature_categories.items():
            cat_indices = [i for i, f in enumerate(features) if f in cat_features]
            category_importance[category] = sum(importances[i] for i in cat_indices)

        return {
            "top_10_features": importance_df.head(10).to_dict("records"),
            "category_importance": category_importance,
            "temporal_feature_rank": self._get_temporal_feature_ranking(importance_df),
        }

    def _get_temporal_feature_ranking(
        self, importance_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Get ranking of temporal features specifically."""

        temporal_keywords = [
            "hour",
            "dow",
            "doy",
            "month",
            "week",
            "holiday",
            "weekend",
        ]
        temporal_features = importance_df[
            importance_df["feature"]
            .str.lower()
            .str.contains("|".join(temporal_keywords))
        ]

        return temporal_features.set_index("feature")["importance"].to_dict()

    def _analyze_pattern_stability(
        self,
        model: ClassificationModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hour: int,
    ) -> Dict[str, Any]:
        """Analyze how stable the model's predictions are to small feature perturbations."""

        # Sample a subset for efficiency
        sample_size = min(1000, len(X_train))
        np.random.seed(self.random_state)
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train.iloc[sample_indices]

        # Get baseline predictions
        baseline_pred = model.predict(X_sample)
        baseline_proba = model.predict_proba(X_sample)

        # Add small noise and see how predictions change
        noise_levels = [0.01, 0.05, 0.1]  # 1%, 5%, 10% noise
        stability_scores = {}

        for noise_level in noise_levels:
            # Add noise to numerical features only
            X_noisy = X_sample.copy()
            numerical_cols = X_sample.select_dtypes(include=[np.number]).columns

            np.random.seed(self.random_state)
            for col in numerical_cols:
                if X_sample[col].std() > 0:  # Avoid division by zero
                    noise = np.random.normal(
                        0, noise_level * X_sample[col].std(), len(X_sample)
                    )
                    X_noisy[col] = X_sample[col] + noise

            # Get noisy predictions
            noisy_pred = model.predict(X_noisy)
            noisy_proba = model.predict_proba(X_noisy)

            # Calculate stability metrics
            prediction_stability = accuracy_score(baseline_pred, noisy_pred)
            probability_stability = np.mean(
                [
                    np.corrcoef(baseline_proba[:, i], noisy_proba[:, i])[0, 1]
                    for i in range(baseline_proba.shape[1])
                    if not np.isnan(
                        np.corrcoef(baseline_proba[:, i], noisy_proba[:, i])[0, 1]
                    )
                ]
            )

            stability_scores[f"noise_{int(noise_level*100)}pct"] = {
                "prediction_stability": prediction_stability,
                "probability_stability": probability_stability,
            }

        return stability_scores

    def _analyze_decision_boundaries(
        self,
        model: ClassificationModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        hour: int,
    ) -> Dict[str, Any]:
        """Analyze decision boundaries using feature importance and prediction patterns."""

        # Get top features for boundary analysis
        feature_ranking = self._get_feature_importance_ranking(model)
        top_features = [f["feature"] for f in feature_ranking["top_10_features"][:5]]

        boundary_analysis = {
            "top_features_for_boundaries": top_features,
            "class_separation_analysis": {},
        }

        # Analyze how well each feature separates classes
        for feature in top_features:
            if feature in X_train.columns:
                feature_values = X_train[feature]

                # Calculate class-wise statistics for this feature
                class_stats = {}
                for class_label in range(len(self.class_labels)):
                    class_mask = y_train == class_label
                    if class_mask.sum() > 0:
                        class_values = feature_values[class_mask]
                        class_stats[self.class_labels[class_label]] = {
                            "mean": float(class_values.mean()),
                            "std": float(class_values.std()),
                            "min": float(class_values.min()),
                            "max": float(class_values.max()),
                            "count": int(class_mask.sum()),
                        }

                boundary_analysis["class_separation_analysis"][feature] = class_stats

        return boundary_analysis

    def _calculate_model_complexity(
        self, model: Any, model_type: str = "generic"
    ) -> Dict[str, Any]:
        """Calculate various complexity metrics for the model."""

        # Use appropriate complexity extractor
        if model_type in self.complexity_extractors:
            extractor = self.complexity_extractors[model_type]
        else:
            extractor = self.complexity_extractors["generic"]

        return extractor.extract_complexity_metrics(model)

    def generate_cross_hour_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive cross-hour analysis report."""

        if not self.cross_hour_patterns:
            return {"error": "No cross-hour patterns available. Run analysis first."}

        report = {
            "summary": self._generate_cross_hour_summary(),
            "feature_consistency": self._analyze_feature_consistency_across_hours(),
            "temporal_patterns": self._identify_temporal_patterns(),
            "model_complexity_trends": self._analyze_complexity_trends(),
            "recommendations": self._generate_cross_hour_recommendations(),
        }

        # Save report
        report_path = os.path.join(self.output_dir, "cross_hour_analysis_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Cross-hour analysis report saved to: {report_path}")

        return report

    def _generate_cross_hour_summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all hours."""

        hours_analyzed = list(self.cross_hour_patterns.keys())

        return {
            "hours_analyzed": len(hours_analyzed),
            "hour_range": f"{min(hours_analyzed)}-{max(hours_analyzed)}"
            if hours_analyzed
            else "None",
            "analysis_components": [
                "feature_ranking",
                "pattern_stability",
                "boundary_analysis",
                "model_complexity",
            ],
        }

    def _analyze_feature_consistency_across_hours(self) -> Dict[str, Any]:
        """Analyze how consistent feature importance is across hours."""

        hours = list(self.cross_hour_patterns.keys())
        if len(hours) < 2:
            return {"error": "Need at least 2 hours for consistency analysis"}

        # Extract feature importance rankings for each hour
        hour_rankings = {}
        for hour in hours:
            patterns = self.cross_hour_patterns[hour]
            if (
                "feature_ranking" in patterns
                and "top_10_features" in patterns["feature_ranking"]
            ):
                top_features = patterns["feature_ranking"]["top_10_features"]
                hour_rankings[hour] = [f["feature"] for f in top_features]

        # Calculate consistency metrics
        consistency_analysis = {
            "most_consistent_features": self._find_most_consistent_features(
                hour_rankings
            ),
            "hour_similarity": self._calculate_hour_similarity(hour_rankings),
            "feature_stability_score": self._calculate_feature_stability(hour_rankings),
        }

        return consistency_analysis

    def _find_most_consistent_features(
        self, hour_rankings: Dict[int, List[str]]
    ) -> List[Dict]:
        """Find features that consistently rank high across hours."""

        feature_appearances = {}
        for hour, features in hour_rankings.items():
            for rank, feature in enumerate(features):
                if feature not in feature_appearances:
                    feature_appearances[feature] = []
                feature_appearances[feature].append(rank)

        # Calculate consistency score (lower average rank + high frequency = more consistent)
        consistent_features = []
        for feature, ranks in feature_appearances.items():
            if (
                len(ranks) >= len(hour_rankings) * 0.5
            ):  # Appears in at least 50% of hours
                avg_rank = np.mean(ranks)
                frequency = len(ranks) / len(hour_rankings)
                consistency_score = frequency / (avg_rank + 1)  # Higher is better

                consistent_features.append(
                    {
                        "feature": feature,
                        "average_rank": avg_rank,
                        "frequency": frequency,
                        "consistency_score": consistency_score,
                    }
                )

        # Sort by consistency score
        consistent_features.sort(key=lambda x: x["consistency_score"], reverse=True)

        return consistent_features[:10]  # Top 10 most consistent

    def _calculate_hour_similarity(
        self, hour_rankings: Dict[int, List[str]]
    ) -> Dict[str, float]:
        """Calculate similarity between different hours based on feature rankings."""

        hours = list(hour_rankings.keys())
        similarity_scores = {}

        for i, hour1 in enumerate(hours):
            for hour2 in hours[i + 1 :]:
                if hour1 in hour_rankings and hour2 in hour_rankings:
                    features1 = set(hour_rankings[hour1])
                    features2 = set(hour_rankings[hour2])

                    # Jaccard similarity
                    intersection = len(features1.intersection(features2))
                    union = len(features1.union(features2))
                    jaccard_similarity = intersection / union if union > 0 else 0

                    similarity_scores[f"hour_{hour1}_vs_{hour2}"] = jaccard_similarity

        return similarity_scores

    def _calculate_feature_stability(
        self, hour_rankings: Dict[int, List[str]]
    ) -> float:
        """Calculate overall feature stability score across all hours."""

        if len(hour_rankings) < 2:
            return 0.0

        # Get all unique features
        all_features = set()
        for features in hour_rankings.values():
            all_features.update(features)

        # Calculate variance in rankings for each feature
        stability_scores = []
        for feature in all_features:
            rankings = []
            for hour_features in hour_rankings.values():
                if feature in hour_features:
                    rankings.append(hour_features.index(feature))
                else:
                    rankings.append(10)  # Penalty for not being in top 10

            if len(rankings) > 1:
                rank_variance = np.var(rankings)
                stability_score = 1 / (
                    1 + rank_variance
                )  # Lower variance = higher stability
                stability_scores.append(stability_score)

        return np.mean(stability_scores) if stability_scores else 0.0

    def _identify_temporal_patterns(self) -> Dict[str, Any]:
        """Identify patterns in model behavior across different times."""

        # Analyze complexity by time periods
        complexity_by_period = {
            "morning_hours": [],  # 6-10
            "midday_hours": [],  # 11-15
            "evening_hours": [],  # 16-20
            "night_hours": [],  # 21-5
        }

        for hour, patterns in self.cross_hour_patterns.items():
            if "model_complexity" in patterns:
                complexity_score = patterns["model_complexity"].get(
                    "complexity_score", 0
                )

                if 6 <= hour <= 10:
                    complexity_by_period["morning_hours"].append(complexity_score)
                elif 11 <= hour <= 15:
                    complexity_by_period["midday_hours"].append(complexity_score)
                elif 16 <= hour <= 20:
                    complexity_by_period["evening_hours"].append(complexity_score)
                else:
                    complexity_by_period["night_hours"].append(complexity_score)

        # Calculate period statistics
        period_stats = {}
        for period, complexities in complexity_by_period.items():
            if complexities:
                period_stats[period] = {
                    "mean_complexity": np.mean(complexities),
                    "std_complexity": np.std(complexities),
                    "sample_count": len(complexities),
                }

        return {
            "complexity_by_time_period": period_stats,
            "most_complex_period": max(
                period_stats.items(), key=lambda x: x[1]["mean_complexity"]
            )[0]
            if period_stats
            else None,
            "most_stable_period": min(
                period_stats.items(), key=lambda x: x[1]["std_complexity"]
            )[0]
            if period_stats
            else None,
        }

    def _analyze_complexity_trends(self) -> Dict[str, Any]:
        """Analyze how model complexity varies across hours."""

        complexity_by_hour = {}
        for hour, patterns in self.cross_hour_patterns.items():
            if "model_complexity" in patterns:
                complexity_score = patterns["model_complexity"].get(
                    "complexity_score", 0
                )
                complexity_by_hour[hour] = complexity_score

        if not complexity_by_hour:
            return {"error": "No complexity data available"}

        hours = sorted(complexity_by_hour.keys())
        complexities = [complexity_by_hour[h] for h in hours]

        return {
            "complexity_trend": {
                "hours": hours,
                "complexity_scores": complexities,
                "trend_slope": np.polyfit(hours, complexities, 1)[0]
                if len(hours) > 1
                else 0,
            },
            "most_complex_hour": max(complexity_by_hour.items(), key=lambda x: x[1]),
            "least_complex_hour": min(complexity_by_hour.items(), key=lambda x: x[1]),
        }

    def _generate_cross_hour_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on cross-hour analysis."""

        recommendations = []

        # Check if we have sufficient data
        if len(self.cross_hour_patterns) < 5:
            recommendations.append(
                "Collect more hours of data for robust cross-hour analysis"
            )
            return recommendations

        # Feature consistency recommendations
        recommendations.append(
            "Focus on the most consistent features across hours for more robust models"
        )
        recommendations.append(
            "Consider separate feature sets for morning/evening vs. midday periods"
        )

        # Temporal modeling recommendations
        recommendations.append(
            "Enhanced temporal features are active - monitor their impact on cross-hour consistency"
        )
        recommendations.append(
            "Consider implementing ensemble methods that combine predictions from adjacent hours"
        )

        # Model complexity recommendations
        complexity_trend = self._analyze_complexity_trends()
        if (
            "complexity_trend" in complexity_trend
            and complexity_trend["complexity_trend"]["trend_slope"] > 0
        ):
            recommendations.append(
                "Model complexity increases throughout the day - consider time-specific regularization"
            )

        return recommendations

    def create_cross_hour_visualization_dashboard(
        self,
        output_path: Optional[str] = None,
    ) -> str:
        """Create comprehensive cross-hour analysis visualization dashboard.

        Args:
            output_path: Path to save HTML dashboard

        Returns:
            Path to saved dashboard
        """
        if not self.cross_hour_patterns:
            raise ValueError("No cross-hour patterns available. Run analysis first.")

        # Create 4-panel dashboard
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=[
                "Feature Importance Consistency Across Hours",
                "Pattern Stability by Hour",
                "Model Complexity Trends",
                "Hour-to-Hour Similarity Heatmap",
            ],
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # Panel 1: Feature importance consistency
        self._add_feature_consistency_panel(fig, row=1, col=1)

        # Panel 2: Pattern stability
        self._add_pattern_stability_panel(fig, row=1, col=2)

        # Panel 3: Model complexity trends
        self._add_complexity_trends_panel(fig, row=2, col=1)

        # Panel 4: Hour similarity heatmap
        self._add_hour_similarity_panel(fig, row=2, col=2)

        # Apply professional styling
        layout = get_professional_layout(
            title="Cross-Hour Analysis Dashboard",
            height=900,
            width=1600,
            showlegend=True,
            legend_position="external_right",
        )

        fig.update_layout(**layout)
        apply_professional_axis_styling(fig, rows=2, cols=2)

        # Save dashboard
        if output_path is None:
            output_path = os.path.join(
                self.output_dir, "cross_hour_analysis_dashboard.html"
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fig.write_html(
            output_path,
            include_plotlyjs=True,
            config={
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            },
        )

        print(f"📊 Cross-hour analysis dashboard saved to: {output_path}")
        return output_path

    def _add_feature_consistency_panel(self, fig, row: int, col: int) -> None:
        """Add feature importance consistency panel."""

        # Get most consistent features
        consistency_analysis = self._analyze_feature_consistency_across_hours()
        if "most_consistent_features" in consistency_analysis:
            consistent_features = consistency_analysis["most_consistent_features"][:10]

            if consistent_features:
                features = [f["feature"] for f in consistent_features]
                scores = [f["consistency_score"] for f in consistent_features]

                fig.add_trace(
                    go.Bar(
                        x=scores,
                        y=features,
                        orientation="h",
                        marker_color=PROFESSIONAL_COLORS["primary"],
                        hovertemplate="<b>%{y}</b><br>Consistency Score: %{x:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

    def _add_pattern_stability_panel(self, fig, row: int, col: int) -> None:
        """Add pattern stability analysis panel."""

        hours = sorted(self.cross_hour_patterns.keys())
        noise_levels = ["noise_1pct", "noise_5pct", "noise_10pct"]

        for i, noise_level in enumerate(noise_levels):
            stability_scores = []
            for hour in hours:
                patterns = self.cross_hour_patterns[hour]
                if (
                    "pattern_stability" in patterns
                    and noise_level in patterns["pattern_stability"]
                ):
                    score = patterns["pattern_stability"][noise_level][
                        "prediction_stability"
                    ]
                    stability_scores.append(score)
                else:
                    stability_scores.append(None)

            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=stability_scores,
                    name=f"{noise_level.replace('noise_', '').replace('pct', '%')} Noise",
                    line=dict(color=COLOR_SEQUENCE[i], width=2),
                    mode="lines+markers",
                    hovertemplate=f"<b>{noise_level}</b><br>Hour: %{{x}}<br>Stability: %{{y:.3f}}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def _add_complexity_trends_panel(self, fig, row: int, col: int) -> None:
        """Add model complexity trends panel."""

        complexity_trends = self._analyze_complexity_trends()
        if "complexity_trend" in complexity_trends:
            hours = complexity_trends["complexity_trend"]["hours"]
            complexities = complexity_trends["complexity_trend"]["complexity_scores"]

            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=complexities,
                    name="Model Complexity",
                    line=dict(color=PROFESSIONAL_COLORS["accent"], width=2),
                    mode="lines+markers",
                    hovertemplate="<b>Model Complexity</b><br>Hour: %{x}<br>Complexity Score: %{y:.1f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

    def _add_hour_similarity_panel(self, fig, row: int, col: int) -> None:
        """Add hour-to-hour similarity heatmap panel."""

        hours = sorted(self.cross_hour_patterns.keys())
        if len(hours) < 2:
            return

        # Create similarity matrix
        similarity_matrix = np.zeros((len(hours), len(hours)))
        hour_rankings = {}

        # Extract rankings
        for hour in hours:
            patterns = self.cross_hour_patterns[hour]
            if (
                "feature_ranking" in patterns
                and "top_10_features" in patterns["feature_ranking"]
            ):
                top_features = patterns["feature_ranking"]["top_10_features"]
                hour_rankings[hour] = [f["feature"] for f in top_features]

        # Calculate similarity
        for i, hour1 in enumerate(hours):
            for j, hour2 in enumerate(hours):
                if hour1 in hour_rankings and hour2 in hour_rankings:
                    features1 = set(hour_rankings[hour1])
                    features2 = set(hour_rankings[hour2])

                    intersection = len(features1.intersection(features2))
                    union = len(features1.union(features2))
                    similarity = intersection / union if union > 0 else 0
                    similarity_matrix[i, j] = similarity

        fig.add_trace(
            go.Heatmap(
                z=similarity_matrix,
                x=hours,
                y=hours,
                colorscale="Viridis",
                hovertemplate="Hour %{x} vs Hour %{y}<br>Similarity: %{z:.3f}<extra></extra>",
            ),
            row=row,
            col=col,
        )
