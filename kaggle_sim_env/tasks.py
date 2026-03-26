"""Task definitions for KaggleSimEnv v3.

Each task now includes:
  - DatasetProperties  → causal ground truth driving reward logic
  - failure_modes      → traps the agent can fall into
  - context_relevance  → per-action relevance to THIS dataset
  - strategy_combos    → synergy bonuses
  - score_modifiers    → deterministic CV/test deltas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kaggle_sim_env.models import (
    ColumnInfo,
    DatasetMetadata,
    DatasetProperties,
    FailureMode,
)


@dataclass(frozen=True)
class StrategyComboDef:
    name: str
    required: frozenset[str]
    cv_bonus: float
    test_bonus: float


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    title: str
    difficulty: str
    description: str
    dataset_metadata: DatasetMetadata
    dataset_properties: DatasetProperties
    base_cv_score: float
    base_test_score: float
    expected_strategies: list[str]
    ghost_scores: list[float]
    hints: list[str]
    score_modifiers: dict[str, dict[str, float]] = field(default_factory=dict)
    overfitting_risk: dict[str, float] = field(default_factory=dict)
    strategy_combos: list[StrategyComboDef] = field(default_factory=list)
    failure_modes: list[FailureMode] = field(default_factory=list)
    context_relevance: dict[str, float] = field(default_factory=dict)
    max_steps: int = 20


# =========================================================================
# Contextual relevance helpers
# =========================================================================
# context_relevance maps action tags → relevance score [-1.0 .. +1.0]
#   +1.0 = highly relevant for this dataset
#    0.0 = neutral
#   -1.0 = completely irrelevant / harmful


# =========================================================================
# Task 1 – Easy: Clean tabular classification
# =========================================================================

def _easy_task() -> TaskDefinition:
    columns = [
        ColumnInfo(name="id", dtype="int64", missing_pct=0.0, unique_count=10000),
        ColumnInfo(name="age", dtype="float64", missing_pct=0.0, unique_count=80),
        ColumnInfo(name="income", dtype="float64", missing_pct=0.0, unique_count=5000),
        ColumnInfo(name="credit_score", dtype="float64", missing_pct=0.0, unique_count=300),
        ColumnInfo(name="num_products", dtype="int64", missing_pct=0.0, unique_count=4),
        ColumnInfo(name="tenure", dtype="int64", missing_pct=0.0, unique_count=12),
        ColumnInfo(name="balance", dtype="float64", missing_pct=0.0, unique_count=6000),
        ColumnInfo(name="has_credit_card", dtype="int64", missing_pct=0.0, unique_count=2),
        ColumnInfo(name="is_active_member", dtype="int64", missing_pct=0.0, unique_count=2),
        ColumnInfo(name="target", dtype="int64", missing_pct=0.0, unique_count=2),
    ]

    props = DatasetProperties(
        has_imbalance=True, imbalance_ratio=0.20,
    )

    failure_modes = [
        FailureMode(
            name="unnecessary_shift_detection",
            trigger_tag="detect_shift:adversarial_validation",
            condition_field="has_shift", condition_value=False,
            cv_effect=0.0, test_effect=0.0,
            message="No distribution shift exists — adversarial validation wasted a step.",
        ),
        FailureMode(
            name="image_aug_on_tabular",
            trigger_tag="augmentation:geometric",
            condition_field="has_images", condition_value=False,
            cv_effect=0.0, test_effect=-0.01,
            message="Geometric augmentation is meaningless on tabular data.",
        ),
        FailureMode(
            name="target_encoding_without_group_cv",
            trigger_tag="feature_engineering:target_encoding",
            condition_field="has_group_column", condition_value=False,
            cv_effect=0.06, test_effect=-0.04,
            message="Target encoding without group-aware CV causes leakage — CV inflated but test hurt.",
        ),
    ]

    context_relevance = {
        # Highly relevant
        "set_cv:kfold": 1.0,
        "set_cv:repeated_kfold": 0.8,
        "feature_engineering:normalize": 0.8,
        "feature_engineering:domain_ratios": 1.0,
        "train_model:xgboost": 1.0,
        "train_model:lightgbm": 0.9,
        "handle_imbalance:scale_pos_weight": 1.0,
        "handle_imbalance:optimize_threshold": 0.8,
        "ensemble:weighted_average": 0.7,
        # Neutral
        "feature_engineering:log_transform": 0.3,
        "regularize:strong_regularization": 0.2,
        # Irrelevant
        "detect_shift:adversarial_validation": -0.5,
        "augmentation:geometric": -1.0,
        "augmentation:color_transform": -1.0,
        "augmentation:camera_simulation": -1.0,
        "feature_engineering:sin_cos_encoding": -0.8,
        "feature_engineering:relative_coordinates": -1.0,
        "feature_engineering:frequency_features": -0.8,
        "tune_loss:auxiliary_physics_loss": -1.0,
        "postprocess:physics_constraints": -1.0,
        "train_model:pretrained_backbone": -0.5,
        "train_model:temporal_cnn": -0.8,
        "train_model:transformer_encoder": -0.5,
    }

    return TaskDefinition(
        task_id="easy_churn",
        title="Customer Churn Prediction",
        difficulty="easy",
        description=(
            "Predict customer churn from clean tabular banking data. "
            "Mild class imbalance (80/20). No missing values, no shift."
        ),
        dataset_metadata=DatasetMetadata(
            num_rows=10000, num_features=9, columns=columns,
            target_column="target", task_type="classification",
            class_balance={"0": 0.80, "1": 0.20},
        ),
        dataset_properties=props,
        base_cv_score=0.50,
        base_test_score=0.50,
        expected_strategies=[
            "set_cv:kfold",
            "feature_engineering:normalize",
            "feature_engineering:domain_ratios",
            "train_model:xgboost",
            "handle_imbalance:scale_pos_weight",
        ],
        ghost_scores=[0.90, 0.88, 0.86, 0.84, 0.82, 0.79, 0.75, 0.72, 0.68],
        hints=[
            "This is a clean tabular dataset — standard k-fold CV works well.",
            "Create domain ratios like balance/income, products/tenure.",
            "XGBoost dominates clean tabular data.",
            "Tune scale_pos_weight for the 80/20 class split.",
            "No distribution shift exists — don't waste steps detecting it.",
        ],
        score_modifiers={
            "set_cv:kfold": {"cv": 0.05, "test": 0.05},
            "set_cv:repeated_kfold": {"cv": 0.06, "test": 0.06},
            "set_cv:group_kfold": {"cv": 0.03, "test": 0.03},
            "set_cv:time_split": {"cv": 0.02, "test": 0.01},
            "feature_engineering:normalize": {"cv": 0.04, "test": 0.04},
            "feature_engineering:log_transform": {"cv": 0.02, "test": 0.02},
            "feature_engineering:interaction_terms": {"cv": 0.03, "test": 0.02},
            "feature_engineering:domain_ratios": {"cv": 0.05, "test": 0.05},
            "feature_engineering:target_encoding": {"cv": 0.06, "test": -0.04},
            "feature_engineering:quantile_features": {"cv": 0.02, "test": 0.02},
            "train_model:xgboost": {"cv": 0.20, "test": 0.19},
            "train_model:lightgbm": {"cv": 0.19, "test": 0.18},
            "train_model:catboost": {"cv": 0.18, "test": 0.18},
            "train_model:random_forest": {"cv": 0.16, "test": 0.15},
            "train_model:linear": {"cv": 0.10, "test": 0.10},
            "train_model:neural_network": {"cv": 0.08, "test": 0.06},
            "handle_imbalance:scale_pos_weight": {"cv": 0.04, "test": 0.04},
            "handle_imbalance:calibrate_probabilities": {"cv": 0.02, "test": 0.03},
            "handle_imbalance:optimize_threshold": {"cv": 0.03, "test": 0.03},
            "ensemble:weighted_average": {"cv": 0.04, "test": 0.03},
            "ensemble:stacking": {"cv": 0.05, "test": 0.03},
            "ensemble:multi_seed_averaging": {"cv": 0.03, "test": 0.03},
            "regularize:strong_regularization": {"cv": 0.01, "test": 0.02},
            "pseudo_label:1": {"cv": 0.01, "test": 0.00},
            "postprocess:bias_correction": {"cv": 0.01, "test": 0.01},
            "inspect_top_solution": {"cv": 0.0, "test": 0.0},
        },
        overfitting_risk={
            "feature_engineering:target_encoding": 0.06,
            "feature_engineering:interaction_terms": 0.01,
            "ensemble:stacking": 0.02,
        },
        strategy_combos=[
            StrategyComboDef(
                name="clean_tabular_pipeline",
                required=frozenset({"set_cv:kfold", "train_model:xgboost", "feature_engineering:normalize"}),
                cv_bonus=0.02, test_bonus=0.02,
            ),
            StrategyComboDef(
                name="imbalance_aware",
                required=frozenset({"handle_imbalance:scale_pos_weight", "handle_imbalance:optimize_threshold"}),
                cv_bonus=0.02, test_bonus=0.03,
            ),
        ],
        failure_modes=failure_modes,
        context_relevance=context_relevance,
        max_steps=20,
    )


# =========================================================================
# Task 2 – Medium: Temporal fraud detection + distribution shift
# =========================================================================

def _medium_task() -> TaskDefinition:
    columns = [
        ColumnInfo(name="id", dtype="int64", missing_pct=0.0, unique_count=50000),
        ColumnInfo(name="timestamp", dtype="datetime64", missing_pct=0.0, unique_count=365),
        ColumnInfo(name="merchant_id", dtype="int64", missing_pct=0.0, unique_count=500),
        ColumnInfo(name="amount", dtype="float64", missing_pct=1.2, unique_count=15000),
        ColumnInfo(name="category", dtype="object", missing_pct=0.0, unique_count=15),
        ColumnInfo(name="customer_age", dtype="float64", missing_pct=3.5, unique_count=70),
        ColumnInfo(name="customer_region", dtype="object", missing_pct=0.5, unique_count=8),
        ColumnInfo(name="prev_transactions", dtype="int64", missing_pct=0.0, unique_count=200),
        ColumnInfo(name="is_weekend", dtype="int64", missing_pct=0.0, unique_count=2),
        ColumnInfo(name="device_type", dtype="object", missing_pct=2.0, unique_count=4),
        ColumnInfo(name="is_fraud", dtype="int64", missing_pct=0.0, unique_count=2),
    ]

    props = DatasetProperties(
        has_shift=True,
        has_imbalance=True, imbalance_ratio=0.05,
        has_time_column=True, has_group_column=True,
        is_safety_critical=True,
    )

    failure_modes = [
        FailureMode(
            name="kfold_on_temporal_data",
            trigger_tag="set_cv:kfold",
            condition_field="has_shift", condition_value=True,
            cv_effect=0.08, test_effect=-0.04,
            message="Random k-fold on shifted temporal data — CV is optimistic, test will be much worse.",
        ),
        FailureMode(
            name="ignoring_shift",
            trigger_tag="train_model:xgboost",
            condition_field="has_shift", condition_value=True,
            cv_effect=0.0, test_effect=-0.06,
            message="Training without addressing distribution shift — model memorises train distribution.",
        ),
        FailureMode(
            name="target_encoding_leakage",
            trigger_tag="feature_engineering:target_encoding",
            condition_field="has_shift", condition_value=True,
            cv_effect=0.05, test_effect=-0.06,
            message="Target encoding on shifted data leaks train-specific patterns.",
        ),
    ]

    context_relevance = {
        "detect_shift:adversarial_validation": 1.0,
        "detect_shift:remove_identifiers": 1.0,
        "detect_shift:domain_invariant_features": 0.9,
        "set_cv:time_split": 1.0,
        "set_cv:combined_group_time": 0.9,
        "set_cv:kfold": -0.8,
        "feature_engineering:log_transform": 0.8,
        "feature_engineering:domain_ratios": 0.9,
        "train_model:xgboost": 0.9,
        "handle_imbalance:scale_pos_weight": 1.0,
        "handle_imbalance:optimize_threshold": 0.9,
        "tune_loss:asymmetric_loss": 1.0,
        "regularize:strong_regularization": 0.8,
        "ensemble:weighted_average": 0.7,
        # Irrelevant
        "augmentation:geometric": -1.0,
        "augmentation:color_transform": -1.0,
        "feature_engineering:sin_cos_encoding": -0.5,
        "feature_engineering:relative_coordinates": -1.0,
        "tune_loss:auxiliary_physics_loss": -1.0,
        "postprocess:physics_constraints": -1.0,
        "train_model:pretrained_backbone": -0.5,
    }

    return TaskDefinition(
        task_id="medium_fraud",
        title="Fraud Detection with Distribution Shift",
        difficulty="medium",
        description=(
            "Detect fraudulent transactions. Test set comes from a later time "
            "period with distribution shift. Heavy imbalance (95/5). Safety-critical."
        ),
        dataset_metadata=DatasetMetadata(
            num_rows=50000, num_features=10, columns=columns,
            target_column="is_fraud", task_type="classification",
            has_time_column=True, has_group_column=True,
            class_balance={"0": 0.95, "1": 0.05},
        ),
        dataset_properties=props,
        base_cv_score=0.42,
        base_test_score=0.38,
        expected_strategies=[
            "detect_shift:adversarial_validation",
            "detect_shift:remove_identifiers",
            "set_cv:time_split",
            "feature_engineering:log_transform",
            "feature_engineering:domain_ratios",
            "train_model:xgboost",
            "handle_imbalance:scale_pos_weight",
            "handle_imbalance:optimize_threshold",
            "tune_loss:asymmetric_loss",
            "ensemble:weighted_average",
            "regularize:strong_regularization",
        ],
        ghost_scores=[0.92, 0.90, 0.87, 0.84, 0.80, 0.76, 0.72, 0.67, 0.60],
        hints=[
            "Adversarial validation AUC > 0.82 — significant distribution shift exists!",
            "Use time-based CV — random k-fold will give you a false sense of accuracy.",
            "Log-transform 'amount' — it has extreme skew.",
            "Remove merchant_id and device_type — they are identifiers causing shift.",
            "This is safety-critical: asymmetric loss prioritises fraud recall.",
            "Tune scale_pos_weight on OOF outputs, then optimize threshold.",
            "Strong regularisation prevents memorising train-specific patterns.",
        ],
        score_modifiers={
            "set_cv:kfold": {"cv": 0.08, "test": -0.04},
            "set_cv:group_kfold": {"cv": 0.06, "test": 0.05},
            "set_cv:time_split": {"cv": 0.05, "test": 0.10},
            "set_cv:combined_group_time": {"cv": 0.05, "test": 0.09},
            "detect_shift:adversarial_validation": {"cv": 0.02, "test": 0.08},
            "detect_shift:feature_importance_shift": {"cv": 0.01, "test": 0.04},
            "detect_shift:remove_identifiers": {"cv": -0.01, "test": 0.06},
            "detect_shift:domain_invariant_features": {"cv": 0.02, "test": 0.05},
            "feature_engineering:log_transform": {"cv": 0.05, "test": 0.06},
            "feature_engineering:normalize": {"cv": 0.02, "test": 0.02},
            "feature_engineering:domain_ratios": {"cv": 0.04, "test": 0.05},
            "feature_engineering:interaction_terms": {"cv": 0.04, "test": 0.01},
            "feature_engineering:target_encoding": {"cv": 0.05, "test": -0.06},
            "train_model:xgboost": {"cv": 0.18, "test": 0.16},
            "train_model:lightgbm": {"cv": 0.17, "test": 0.16},
            "train_model:catboost": {"cv": 0.16, "test": 0.15},
            "train_model:random_forest": {"cv": 0.14, "test": 0.12},
            "train_model:linear": {"cv": 0.08, "test": 0.07},
            "handle_imbalance:scale_pos_weight": {"cv": 0.04, "test": 0.05},
            "handle_imbalance:calibrate_probabilities": {"cv": 0.02, "test": 0.03},
            "handle_imbalance:optimize_threshold": {"cv": 0.03, "test": 0.04},
            "tune_loss:asymmetric_loss": {"cv": 0.03, "test": 0.05},
            "ensemble:weighted_average": {"cv": 0.05, "test": 0.05},
            "ensemble:stacking": {"cv": 0.06, "test": 0.04},
            "ensemble:multi_seed_averaging": {"cv": 0.03, "test": 0.04},
            "ensemble:heterogeneous": {"cv": 0.04, "test": 0.05},
            "regularize:strong_regularization": {"cv": -0.01, "test": 0.04},
            "regularize:ema": {"cv": 0.01, "test": 0.02},
            "pseudo_label:1": {"cv": 0.02, "test": 0.01},
            "pseudo_label:2": {"cv": 0.03, "test": -0.02},
            "postprocess:bias_correction": {"cv": 0.01, "test": 0.02},
            "clean_data:remove_outliers": {"cv": 0.01, "test": 0.02},
            "clean_data:remove_leaky_features": {"cv": -0.02, "test": 0.04},
            "inspect_top_solution": {"cv": 0.0, "test": 0.0},
        },
        overfitting_risk={
            "set_cv:kfold": 0.08,
            "pseudo_label:2": 0.05,
            "feature_engineering:target_encoding": 0.05,
            "feature_engineering:interaction_terms": 0.03,
        },
        strategy_combos=[
            StrategyComboDef(
                name="shift_aware_pipeline",
                required=frozenset({
                    "detect_shift:adversarial_validation",
                    "set_cv:time_split",
                    "detect_shift:remove_identifiers",
                }),
                cv_bonus=0.03, test_bonus=0.06,
            ),
            StrategyComboDef(
                name="safety_critical_fraud",
                required=frozenset({
                    "handle_imbalance:scale_pos_weight",
                    "tune_loss:asymmetric_loss",
                    "handle_imbalance:optimize_threshold",
                }),
                cv_bonus=0.02, test_bonus=0.04,
            ),
            StrategyComboDef(
                name="robust_generalisation",
                required=frozenset({
                    "regularize:strong_regularization",
                    "detect_shift:domain_invariant_features",
                }),
                cv_bonus=0.01, test_bonus=0.03,
            ),
        ],
        failure_modes=failure_modes,
        context_relevance=context_relevance,
        max_steps=25,
    )


# =========================================================================
# Task 3 – Hard: Leakage + noise + missing data + shift
# =========================================================================

def _hard_task() -> TaskDefinition:
    columns = [
        ColumnInfo(name="id", dtype="int64", missing_pct=0.0, unique_count=100000),
        ColumnInfo(name="timestamp", dtype="datetime64", missing_pct=0.0, unique_count=730),
        ColumnInfo(name="group_id", dtype="int64", missing_pct=0.0, unique_count=1000),
        ColumnInfo(name="feature_0", dtype="float64", missing_pct=8.5, unique_count=50000),
        ColumnInfo(name="feature_1", dtype="float64", missing_pct=12.0, unique_count=40000),
        ColumnInfo(name="feature_2", dtype="float64", missing_pct=0.0, unique_count=60000),
        ColumnInfo(name="feature_3", dtype="float64", missing_pct=5.0, unique_count=30000),
        ColumnInfo(name="feature_4", dtype="object", missing_pct=15.0, unique_count=20),
        ColumnInfo(name="feature_5_leaky", dtype="float64", missing_pct=0.0, unique_count=80000),
        ColumnInfo(name="noise_0", dtype="float64", missing_pct=0.0, unique_count=100000),
        ColumnInfo(name="noise_1", dtype="float64", missing_pct=0.0, unique_count=100000),
        ColumnInfo(name="noise_2", dtype="float64", missing_pct=0.0, unique_count=100000),
        ColumnInfo(name="target", dtype="int64", missing_pct=0.0, unique_count=2),
    ]

    props = DatasetProperties(
        has_shift=True, has_leakage=True, has_noise_features=True,
        has_missing_data=True, has_imbalance=True, imbalance_ratio=0.08,
        has_time_column=True, has_group_column=True,
    )

    failure_modes = [
        FailureMode(
            name="kfold_on_grouped_data",
            trigger_tag="set_cv:kfold",
            condition_field="has_group_column", condition_value=True,
            cv_effect=0.10, test_effect=-0.05,
            message="Random k-fold on grouped data leaks group structure — CV hugely inflated.",
        ),
        FailureMode(
            name="keeping_leaky_feature",
            trigger_tag="train_model:xgboost",
            condition_field="has_leakage", condition_value=True,
            cv_effect=0.0, test_effect=-0.08,
            message="Training on leaky features — model depends on data that won't exist at test time.",
        ),
        FailureMode(
            name="interaction_terms_on_noisy_data",
            trigger_tag="feature_engineering:interaction_terms",
            condition_field="has_noise_features", condition_value=True,
            cv_effect=0.05, test_effect=-0.04,
            message="Interaction terms on noisy features amplify noise — CV up, test down.",
        ),
        FailureMode(
            name="pseudo_label_overfit",
            trigger_tag="pseudo_label:2",
            condition_field="has_shift", condition_value=True,
            cv_effect=0.05, test_effect=-0.06,
            message="Iterative pseudo-labeling on shifted data compounds distribution errors.",
        ),
    ]

    context_relevance = {
        "detect_shift:adversarial_validation": 1.0,
        "detect_shift:remove_identifiers": 0.9,
        "set_cv:stratified_group_kfold": 1.0,
        "set_cv:kfold": -0.9,
        "clean_data:remove_leaky_features": 1.0,
        "clean_data:remove_outliers": 0.8,
        "clean_data:analytical_reconstruction": 1.0,
        "clean_data:nan_native_model": 0.8,
        "feature_engineering:log_transform": 0.7,
        "feature_engineering:domain_ratios": 0.8,
        "feature_engineering:interaction_terms": -0.7,
        "feature_engineering:target_encoding": -0.9,
        "train_model:xgboost": 0.9,
        "train_model:lightgbm": 0.9,
        "handle_imbalance:scale_pos_weight": 1.0,
        "handle_imbalance:calibrate_probabilities": 0.9,
        "ensemble:stacking": 0.8,
        "ensemble:diverse_features": 0.9,
        "regularize:strong_regularization": 0.9,
        "pseudo_label:1": 0.5,
        "pseudo_label:2": -0.8,
        # Irrelevant
        "augmentation:geometric": -1.0,
        "augmentation:camera_simulation": -1.0,
        "tune_loss:auxiliary_physics_loss": -1.0,
        "postprocess:physics_constraints": -1.0,
        "train_model:pretrained_backbone": -0.5,
    }

    return TaskDefinition(
        task_id="hard_leaky_noisy",
        title="Leaky Noisy Imbalanced Classification",
        difficulty="hard",
        description=(
            "High-dimensional classification with data leakage (feature_5_leaky), "
            "pure-noise columns (noise_0-2), structured missing data, class imbalance "
            "(92/8), and temporal distribution shift."
        ),
        dataset_metadata=DatasetMetadata(
            num_rows=100000, num_features=12, columns=columns,
            target_column="target", task_type="classification",
            has_time_column=True, has_group_column=True,
            class_balance={"0": 0.92, "1": 0.08},
        ),
        dataset_properties=props,
        base_cv_score=0.38,
        base_test_score=0.32,
        expected_strategies=[
            "detect_shift:adversarial_validation",
            "detect_shift:remove_identifiers",
            "set_cv:stratified_group_kfold",
            "clean_data:remove_leaky_features",
            "clean_data:remove_outliers",
            "clean_data:analytical_reconstruction",
            "feature_engineering:log_transform",
            "feature_engineering:domain_ratios",
            "train_model:xgboost",
            "train_model:lightgbm",
            "handle_imbalance:scale_pos_weight",
            "handle_imbalance:calibrate_probabilities",
            "ensemble:stacking",
            "ensemble:diverse_features",
            "pseudo_label:1",
            "regularize:strong_regularization",
        ],
        ghost_scores=[0.91, 0.88, 0.85, 0.82, 0.78, 0.74, 0.69, 0.63, 0.55],
        hints=[
            "'feature_5_leaky' is suspiciously predictive — it encodes target leakage. Remove it!",
            "noise_0/1/2 have zero signal; removing them reduces overfitting dramatically.",
            "Use stratified group k-fold — random k-fold will leak group structure.",
            "Adversarial validation AUC 0.85 — heavy distribution shift.",
            "Many rows with structured missing data can be analytically reconstructed.",
            "Calibrate probabilities with isotonic regression on OOF predictions.",
            "Stack XGBoost + LightGBM with diverse feature subsets.",
            "One iteration of pseudo-labeling helps; two iterations overfit badly on shifted data.",
        ],
        score_modifiers={
            "set_cv:kfold": {"cv": 0.10, "test": -0.05},
            "set_cv:group_kfold": {"cv": 0.06, "test": 0.08},
            "set_cv:stratified_group_kfold": {"cv": 0.05, "test": 0.10},
            "set_cv:time_split": {"cv": 0.05, "test": 0.07},
            "detect_shift:adversarial_validation": {"cv": 0.02, "test": 0.08},
            "detect_shift:remove_identifiers": {"cv": -0.02, "test": 0.07},
            "clean_data:remove_leaky_features": {"cv": -0.05, "test": 0.08},
            "clean_data:remove_outliers": {"cv": 0.02, "test": 0.03},
            "clean_data:analytical_reconstruction": {"cv": 0.03, "test": 0.04},
            "clean_data:nan_native_model": {"cv": 0.02, "test": 0.03},
            "feature_engineering:log_transform": {"cv": 0.04, "test": 0.05},
            "feature_engineering:domain_ratios": {"cv": 0.04, "test": 0.05},
            "feature_engineering:interaction_terms": {"cv": 0.05, "test": -0.04},
            "feature_engineering:target_encoding": {"cv": 0.06, "test": -0.05},
            "feature_engineering:normalize": {"cv": 0.03, "test": 0.03},
            "train_model:xgboost": {"cv": 0.16, "test": 0.14},
            "train_model:lightgbm": {"cv": 0.15, "test": 0.14},
            "train_model:catboost": {"cv": 0.14, "test": 0.13},
            "train_model:random_forest": {"cv": 0.12, "test": 0.10},
            "train_model:linear": {"cv": 0.06, "test": 0.06},
            "handle_imbalance:scale_pos_weight": {"cv": 0.04, "test": 0.05},
            "handle_imbalance:calibrate_probabilities": {"cv": 0.02, "test": 0.04},
            "handle_imbalance:optimize_threshold": {"cv": 0.03, "test": 0.03},
            "tune_loss:asymmetric_loss": {"cv": 0.02, "test": 0.04},
            "ensemble:stacking": {"cv": 0.06, "test": 0.05},
            "ensemble:diverse_features": {"cv": 0.04, "test": 0.05},
            "ensemble:weighted_average": {"cv": 0.04, "test": 0.04},
            "regularize:strong_regularization": {"cv": -0.02, "test": 0.05},
            "regularize:ema": {"cv": 0.01, "test": 0.02},
            "pseudo_label:1": {"cv": 0.03, "test": 0.03},
            "pseudo_label:2": {"cv": 0.05, "test": -0.06},
            "postprocess:bias_correction": {"cv": 0.01, "test": 0.02},
            "inspect_top_solution": {"cv": 0.0, "test": 0.0},
        },
        overfitting_risk={
            "set_cv:kfold": 0.10,
            "pseudo_label:2": 0.06,
            "feature_engineering:interaction_terms": 0.05,
            "feature_engineering:target_encoding": 0.06,
        },
        strategy_combos=[
            StrategyComboDef(
                name="data_hygiene",
                required=frozenset({
                    "clean_data:remove_leaky_features",
                    "clean_data:remove_outliers",
                    "detect_shift:adversarial_validation",
                }),
                cv_bonus=0.02, test_bonus=0.06,
            ),
            StrategyComboDef(
                name="missing_data_mastery",
                required=frozenset({
                    "clean_data:analytical_reconstruction",
                    "clean_data:nan_native_model",
                }),
                cv_bonus=0.02, test_bonus=0.03,
            ),
            StrategyComboDef(
                name="diverse_ensemble",
                required=frozenset({
                    "train_model:xgboost", "train_model:lightgbm",
                    "ensemble:diverse_features", "ensemble:stacking",
                }),
                cv_bonus=0.03, test_bonus=0.05,
            ),
            StrategyComboDef(
                name="calibrated_imbalance",
                required=frozenset({
                    "handle_imbalance:scale_pos_weight",
                    "handle_imbalance:calibrate_probabilities",
                    "handle_imbalance:optimize_threshold",
                }),
                cv_bonus=0.02, test_bonus=0.04,
            ),
        ],
        failure_modes=failure_modes,
        context_relevance=context_relevance,
        max_steps=30,
    )


# =========================================================================
# Task 4 – Image: Quality regression
# =========================================================================

def _image_task() -> TaskDefinition:
    columns = [
        ColumnInfo(name="image_id", dtype="int64", missing_pct=0.0, unique_count=25000),
        ColumnInfo(name="image_path", dtype="object", missing_pct=0.0, unique_count=25000),
        ColumnInfo(name="camera_model", dtype="object", missing_pct=0.0, unique_count=12),
        ColumnInfo(name="exposure", dtype="float64", missing_pct=2.0, unique_count=500),
        ColumnInfo(name="iso", dtype="int64", missing_pct=1.0, unique_count=20),
        ColumnInfo(name="focal_length", dtype="float64", missing_pct=0.5, unique_count=30),
        ColumnInfo(name="width", dtype="int64", missing_pct=0.0, unique_count=10),
        ColumnInfo(name="height", dtype="int64", missing_pct=0.0, unique_count=10),
        ColumnInfo(name="quality_score", dtype="float64", missing_pct=0.0, unique_count=1000),
    ]

    props = DatasetProperties(
        has_heavy_tails=True, has_group_column=True,
        has_images=True,
    )

    failure_modes = [
        FailureMode(
            name="no_augmentation_on_images",
            trigger_tag="submit",
            condition_field="has_images", condition_value=True,
            cv_effect=0.0, test_effect=-0.04,
            message="Submitting image model without augmentation — overfitting on small image data.",
        ),
        FailureMode(
            name="tree_model_on_images",
            trigger_tag="train_model:xgboost",
            condition_field="has_images", condition_value=True,
            cv_effect=0.04, test_effect=-0.02,
            message="Tree models on raw image metadata miss the visual signal entirely.",
        ),
    ]

    context_relevance = {
        "train_model:pretrained_backbone": 1.0,
        "regularize:freeze_backbone": 1.0,
        "augmentation:geometric": 1.0,
        "augmentation:color_transform": 1.0,
        "augmentation:camera_simulation": 1.0,
        "augmentation:clahe": 0.9,
        "postprocess:tta": 1.0,
        "postprocess:per_group_calibration": 0.9,
        "tune_loss:gaussian_nll": 1.0,
        "tune_loss:multi_task": 0.8,
        "regularize:ema": 0.8,
        "ensemble:multi_seed_averaging": 0.8,
        "ensemble:swa": 0.8,
        "feature_engineering:log_transform": 0.7,
        "set_cv:group_kfold": 1.0,
        "set_cv:kfold": -0.6,
        # Irrelevant
        "feature_engineering:tfidf_features": -1.0,
        "feature_engineering:relative_coordinates": -0.8,
        "tune_loss:auxiliary_physics_loss": -0.8,
        "detect_shift:adversarial_validation": 0.2,
        "train_model:xgboost": -0.3,
        "train_model:linear": -0.7,
    }

    return TaskDefinition(
        task_id="image_quality",
        title="Image Quality Assessment",
        difficulty="hard",
        description=(
            "Predict continuous quality scores from photographs. Images from "
            "multiple cameras with varying resolution. Heavy-tailed target "
            "and camera-specific biases."
        ),
        dataset_metadata=DatasetMetadata(
            num_rows=25000, num_features=8, columns=columns,
            target_column="quality_score", task_type="regression",
            has_image_data=True, has_group_column=True,
            target_distribution="heavy_tailed",
        ),
        dataset_properties=props,
        base_cv_score=0.35,
        base_test_score=0.30,
        expected_strategies=[
            "set_cv:group_kfold",
            "train_model:pretrained_backbone",
            "augmentation:geometric",
            "augmentation:color_transform",
            "augmentation:camera_simulation",
            "augmentation:clahe",
            "feature_engineering:log_transform",
            "tune_loss:gaussian_nll",
            "tune_loss:multi_task",
            "regularize:freeze_backbone",
            "regularize:ema",
            "ensemble:multi_seed_averaging",
            "ensemble:swa",
            "postprocess:tta",
            "postprocess:per_group_calibration",
        ],
        ghost_scores=[0.90, 0.87, 0.84, 0.81, 0.77, 0.73, 0.68, 0.62, 0.55],
        hints=[
            "Group k-fold by camera_model — camera bias leaks between folds.",
            "Pretrained ImageNet backbones are essential with only 25k images.",
            "Freeze backbone first, train head, then fine-tune.",
            "Geometric + color augmentations are a must for image tasks.",
            "Camera simulation augmentation handles focal length variance.",
            "CLAHE normalisation helps with exposure variation.",
            "Heavy-tailed target → log transform + Gaussian NLL loss.",
            "Multi-task: add classification head for quality bins alongside regression.",
            "EMA and SWA stabilise training; multi-seed averaging adds robustness.",
            "TTA with brightness + crops gives 1-2% boost at inference.",
        ],
        score_modifiers={
            "set_cv:kfold": {"cv": 0.06, "test": 0.00},
            "set_cv:group_kfold": {"cv": 0.04, "test": 0.08},
            "train_model:pretrained_backbone": {"cv": 0.22, "test": 0.20},
            "train_model:neural_network": {"cv": 0.14, "test": 0.11},
            "train_model:xgboost": {"cv": 0.04, "test": -0.02},
            "train_model:linear": {"cv": 0.02, "test": 0.02},
            "augmentation:geometric": {"cv": 0.04, "test": 0.05},
            "augmentation:color_transform": {"cv": 0.03, "test": 0.04},
            "augmentation:clahe": {"cv": 0.03, "test": 0.03},
            "augmentation:camera_simulation": {"cv": 0.03, "test": 0.04},
            "augmentation:gaussian_noise": {"cv": 0.02, "test": 0.02},
            "augmentation:image_rectification": {"cv": 0.02, "test": 0.03},
            "augmentation:multi_view_processing": {"cv": 0.03, "test": 0.03},
            "feature_engineering:log_transform": {"cv": 0.03, "test": 0.04},
            "feature_engineering:normalize": {"cv": 0.02, "test": 0.02},
            "tune_loss:gaussian_nll": {"cv": 0.04, "test": 0.05},
            "tune_loss:multi_task": {"cv": 0.03, "test": 0.04},
            "tune_loss:interval_regression": {"cv": 0.02, "test": 0.03},
            "tune_loss:quantile_regression": {"cv": 0.02, "test": 0.03},
            "regularize:freeze_backbone": {"cv": 0.03, "test": 0.05},
            "regularize:ema": {"cv": 0.02, "test": 0.03},
            "regularize:dropout": {"cv": 0.01, "test": 0.02},
            "ensemble:multi_seed_averaging": {"cv": 0.03, "test": 0.04},
            "ensemble:swa": {"cv": 0.03, "test": 0.04},
            "ensemble:weighted_average": {"cv": 0.03, "test": 0.03},
            "postprocess:tta": {"cv": 0.03, "test": 0.04},
            "postprocess:per_group_calibration": {"cv": 0.02, "test": 0.04},
            "postprocess:prediction_shrinkage": {"cv": 0.02, "test": 0.03},
            "pseudo_label:1": {"cv": 0.02, "test": 0.01},
            "inspect_top_solution": {"cv": 0.0, "test": 0.0},
        },
        overfitting_risk={
            "set_cv:kfold": 0.06,
            "train_model:xgboost": 0.06,
        },
        strategy_combos=[
            StrategyComboDef(
                name="vision_pipeline",
                required=frozenset({
                    "train_model:pretrained_backbone", "regularize:freeze_backbone",
                    "augmentation:geometric", "augmentation:color_transform",
                }),
                cv_bonus=0.03, test_bonus=0.05,
            ),
            StrategyComboDef(
                name="heavy_tail_handling",
                required=frozenset({
                    "feature_engineering:log_transform",
                    "tune_loss:gaussian_nll",
                    "postprocess:prediction_shrinkage",
                }),
                cv_bonus=0.02, test_bonus=0.04,
            ),
            StrategyComboDef(
                name="inference_boost",
                required=frozenset({
                    "postprocess:tta", "ensemble:multi_seed_averaging", "ensemble:swa",
                }),
                cv_bonus=0.02, test_bonus=0.04,
            ),
            StrategyComboDef(
                name="camera_robustness",
                required=frozenset({
                    "augmentation:camera_simulation", "augmentation:clahe",
                    "postprocess:per_group_calibration",
                }),
                cv_bonus=0.02, test_bonus=0.04,
            ),
        ],
        failure_modes=failure_modes,
        context_relevance=context_relevance,
        max_steps=30,
    )


# =========================================================================
# Task 5 – Trajectory: Multi-agent spatial-temporal prediction
# =========================================================================

def _trajectory_task() -> TaskDefinition:
    columns = [
        ColumnInfo(name="scene_id", dtype="int64", missing_pct=0.0, unique_count=5000),
        ColumnInfo(name="frame_id", dtype="int64", missing_pct=0.0, unique_count=50),
        ColumnInfo(name="agent_id", dtype="int64", missing_pct=0.0, unique_count=20),
        ColumnInfo(name="x", dtype="float64", missing_pct=0.0, unique_count=100000),
        ColumnInfo(name="y", dtype="float64", missing_pct=0.0, unique_count=100000),
        ColumnInfo(name="vx", dtype="float64", missing_pct=0.0, unique_count=50000),
        ColumnInfo(name="vy", dtype="float64", missing_pct=0.0, unique_count=50000),
        ColumnInfo(name="heading", dtype="float64", missing_pct=0.0, unique_count=3600),
        ColumnInfo(name="agent_type", dtype="object", missing_pct=0.0, unique_count=4),
        ColumnInfo(name="future_x", dtype="float64", missing_pct=0.0, unique_count=100000),
        ColumnInfo(name="future_y", dtype="float64", missing_pct=0.0, unique_count=100000),
    ]

    props = DatasetProperties(
        has_time_column=True, has_group_column=True,
        has_spatial_data=True, needs_physics=True,
    )

    failure_modes = [
        FailureMode(
            name="tree_model_on_trajectories",
            trigger_tag="train_model:xgboost",
            condition_field="has_spatial_data", condition_value=True,
            cv_effect=0.04, test_effect=-0.03,
            message="Tree models cannot capture spatial-temporal dynamics — use neural approaches.",
        ),
        FailureMode(
            name="raw_heading_without_sincos",
            trigger_tag="submit",
            condition_field="has_spatial_data", condition_value=True,
            cv_effect=0.0, test_effect=-0.03,
            message="Submitting without sin/cos encoding — raw heading angles wrap around and hurt predictions.",
        ),
    ]

    context_relevance = {
        "set_cv:group_kfold": 1.0,
        "feature_engineering:sin_cos_encoding": 1.0,
        "feature_engineering:relative_coordinates": 1.0,
        "feature_engineering:distance_features": 1.0,
        "train_model:transformer_encoder": 1.0,
        "train_model:temporal_cnn": 0.9,
        "augmentation:rotation_invariant": 1.0,
        "augmentation:symmetry_augmentation": 0.9,
        "augmentation:temporal_augmentation": 0.9,
        "tune_loss:gaussian_nll": 1.0,
        "tune_loss:auxiliary_physics_loss": 1.0,
        "regularize:ema": 0.9,
        "ensemble:multi_seed_averaging": 0.8,
        "postprocess:physics_constraints": 1.0,
        "clean_data:remove_corrupted": 0.8,
        "clean_data:remove_outliers": 0.7,
        "set_cv:kfold": -0.8,
        # Irrelevant
        "train_model:xgboost": -0.6,
        "train_model:linear": -0.9,
        "augmentation:color_transform": -1.0,
        "augmentation:clahe": -1.0,
        "augmentation:camera_simulation": -0.8,
        "feature_engineering:tfidf_features": -1.0,
        "feature_engineering:target_encoding": -0.7,
    }

    return TaskDefinition(
        task_id="trajectory_pred",
        title="Multi-Agent Trajectory Prediction",
        difficulty="hard",
        description=(
            "Predict future (x,y) positions of multiple interacting agents in "
            "driving scenes. Requires spatial-temporal encoding, rotation-invariant "
            "processing, multi-agent interaction modeling, and physics-aware predictions."
        ),
        dataset_metadata=DatasetMetadata(
            num_rows=500000, num_features=10, columns=columns,
            target_column="future_x,future_y", task_type="regression",
            has_time_column=True, has_group_column=True,
            has_spatial_data=True,
        ),
        dataset_properties=props,
        base_cv_score=0.32,
        base_test_score=0.28,
        expected_strategies=[
            "set_cv:group_kfold",
            "feature_engineering:sin_cos_encoding",
            "feature_engineering:relative_coordinates",
            "feature_engineering:distance_features",
            "clean_data:remove_corrupted",
            "clean_data:remove_outliers",
            "train_model:transformer_encoder",
            "train_model:temporal_cnn",
            "augmentation:rotation_invariant",
            "augmentation:symmetry_augmentation",
            "augmentation:temporal_augmentation",
            "tune_loss:gaussian_nll",
            "tune_loss:auxiliary_physics_loss",
            "regularize:ema",
            "ensemble:multi_seed_averaging",
            "postprocess:physics_constraints",
        ],
        ghost_scores=[0.89, 0.86, 0.83, 0.80, 0.76, 0.71, 0.66, 0.60, 0.52],
        hints=[
            "Group k-fold by scene_id — prevents leaking scene-specific patterns.",
            "Encode heading with sin/cos — raw angles wrap around and mislead models.",
            "Use relative coordinates instead of absolute positions.",
            "Distance features to nearby agents help interaction modeling.",
            "Remove scenes with abnormal sequence lengths.",
            "Transformer encoders capture multi-agent interactions best.",
            "Temporal CNN with gradual pooling works well for positional time series.",
            "Rotation-invariant augmentation: rotate scene, preserve relative positions.",
            "Gaussian NLL lets the model express uncertainty per prediction.",
            "Auxiliary physics loss: penalise unrealistic acceleration/jerk.",
            "Physics post-processing: clamp velocity, smooth jerk, enforce kinematic limits.",
        ],
        score_modifiers={
            "set_cv:kfold": {"cv": 0.06, "test": -0.02},
            "set_cv:group_kfold": {"cv": 0.04, "test": 0.08},
            "feature_engineering:sin_cos_encoding": {"cv": 0.04, "test": 0.05},
            "feature_engineering:relative_coordinates": {"cv": 0.05, "test": 0.06},
            "feature_engineering:distance_features": {"cv": 0.04, "test": 0.05},
            "feature_engineering:normalize": {"cv": 0.02, "test": 0.02},
            "feature_engineering:spatial_encoding": {"cv": 0.03, "test": 0.03},
            "clean_data:remove_corrupted": {"cv": 0.02, "test": 0.03},
            "clean_data:remove_outliers": {"cv": 0.02, "test": 0.03},
            "train_model:transformer_encoder": {"cv": 0.18, "test": 0.17},
            "train_model:temporal_cnn": {"cv": 0.16, "test": 0.15},
            "train_model:neural_network": {"cv": 0.12, "test": 0.10},
            "train_model:xgboost": {"cv": 0.04, "test": -0.03},
            "train_model:linear": {"cv": 0.02, "test": 0.02},
            "augmentation:rotation_invariant": {"cv": 0.04, "test": 0.05},
            "augmentation:symmetry_augmentation": {"cv": 0.03, "test": 0.04},
            "augmentation:temporal_augmentation": {"cv": 0.03, "test": 0.04},
            "tune_loss:gaussian_nll": {"cv": 0.04, "test": 0.05},
            "tune_loss:auxiliary_physics_loss": {"cv": 0.03, "test": 0.05},
            "tune_loss:multi_task": {"cv": 0.02, "test": 0.03},
            "regularize:ema": {"cv": 0.03, "test": 0.04},
            "regularize:dropout": {"cv": 0.01, "test": 0.02},
            "ensemble:multi_seed_averaging": {"cv": 0.03, "test": 0.04},
            "ensemble:swa": {"cv": 0.02, "test": 0.03},
            "ensemble:heterogeneous": {"cv": 0.04, "test": 0.04},
            "postprocess:physics_constraints": {"cv": 0.03, "test": 0.05},
            "postprocess:tta": {"cv": 0.02, "test": 0.02},
            "pseudo_label:1": {"cv": 0.02, "test": 0.01},
            "inspect_top_solution": {"cv": 0.0, "test": 0.0},
        },
        overfitting_risk={
            "set_cv:kfold": 0.08,
            "train_model:xgboost": 0.07,
        },
        strategy_combos=[
            StrategyComboDef(
                name="spatial_encoding_suite",
                required=frozenset({
                    "feature_engineering:sin_cos_encoding",
                    "feature_engineering:relative_coordinates",
                    "feature_engineering:distance_features",
                }),
                cv_bonus=0.03, test_bonus=0.05,
            ),
            StrategyComboDef(
                name="geometric_augmentation_suite",
                required=frozenset({
                    "augmentation:rotation_invariant",
                    "augmentation:symmetry_augmentation",
                    "augmentation:temporal_augmentation",
                }),
                cv_bonus=0.03, test_bonus=0.05,
            ),
            StrategyComboDef(
                name="physics_aware_training",
                required=frozenset({
                    "tune_loss:gaussian_nll",
                    "tune_loss:auxiliary_physics_loss",
                    "postprocess:physics_constraints",
                }),
                cv_bonus=0.03, test_bonus=0.06,
            ),
            StrategyComboDef(
                name="stable_training",
                required=frozenset({
                    "regularize:ema", "ensemble:multi_seed_averaging",
                }),
                cv_bonus=0.02, test_bonus=0.03,
            ),
        ],
        failure_modes=failure_modes,
        context_relevance=context_relevance,
        max_steps=30,
    )


# =========================================================================
# Registry
# =========================================================================

TASK_REGISTRY: dict[str, TaskDefinition] = {
    "easy_churn": _easy_task(),
    "medium_fraud": _medium_task(),
    "hard_leaky_noisy": _hard_task(),
    "image_quality": _image_task(),
    "trajectory_pred": _trajectory_task(),
}


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{task_id}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_id]
