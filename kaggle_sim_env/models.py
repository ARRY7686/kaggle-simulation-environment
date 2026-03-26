"""Pydantic models for KaggleSimEnv.

v3 — hierarchical categories, causal dataset properties, failure modes.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# =========================================================================
# Hierarchical category mappings
# technique → category (the single source of truth)
# =========================================================================

CATEGORY_MAP: dict[str, dict[str, str]] = {
    "set_cv": {
        "kfold": "standard",
        "repeated_kfold": "standard",
        "group_kfold": "group",
        "stratified_group_kfold": "group",
        "time_split": "temporal",
        "combined_group_time": "temporal",
    },
    "feature_engineering": {
        "log_transform": "distribution",
        "normalize": "distribution",
        "quantile_features": "distribution",
        "interaction_terms": "interaction",
        "domain_ratios": "interaction",
        "sin_cos_encoding": "encoding",
        "target_encoding": "encoding",
        "spatial_encoding": "encoding",
        "tfidf_features": "encoding",
        "relative_coordinates": "spatial",
        "distance_features": "spatial",
        "frequency_features": "signal",
        "multi_layer_features": "signal",
        "fourier_resampling": "signal",
    },
    "detect_shift": {
        "adversarial_validation": "detection",
        "feature_importance_shift": "detection",
        "remove_identifiers": "mitigation",
        "domain_invariant_features": "mitigation",
    },
    "train_model": {
        "xgboost": "tree",
        "lightgbm": "tree",
        "catboost": "tree",
        "random_forest": "tree",
        "linear": "linear",
        "neural_network": "neural",
        "pretrained_backbone": "neural",
        "temporal_cnn": "neural",
        "transformer_encoder": "neural",
    },
    "handle_imbalance": {
        "scale_pos_weight": "weighting",
        "class_weighted_loss": "weighting",
        "calibrate_probabilities": "calibration",
        "optimize_threshold": "calibration",
        "hierarchical_labels": "hierarchy",
        "lower_thresholds_recall": "hierarchy",
    },
    "clean_data": {
        "remove_corrupted": "removal",
        "remove_outliers": "removal",
        "remove_leaky_features": "removal",
        "analytical_reconstruction": "reconstruction",
        "nan_native_model": "reconstruction",
        "domain_augmentation": "reconstruction",
        "clean_subset_training": "reconstruction",
    },
    "augmentation": {
        "geometric": "geometric",
        "rotation_invariant": "geometric",
        "image_rectification": "geometric",
        "color_transform": "color",
        "clahe": "color",
        "gaussian_noise": "noise",
        "robustness_augmentation": "noise",
        "camera_simulation": "domain",
        "temporal_augmentation": "domain",
        "symmetry_augmentation": "domain",
        "multi_view_processing": "domain",
    },
    "ensemble": {
        "weighted_average": "averaging",
        "multi_seed_averaging": "averaging",
        "swa": "averaging",
        "stacking": "stacking",
        "diverse_features": "diversity",
        "heterogeneous": "diversity",
    },
    "postprocess": {
        "bias_correction": "calibration",
        "prediction_shrinkage": "calibration",
        "per_group_calibration": "calibration",
        "domain_rules": "domain",
        "physics_constraints": "domain",
        "tta": "inference",
    },
    "tune_loss": {
        "asymmetric_loss": "asymmetric",
        "epsilon_insensitive": "asymmetric",
        "gaussian_nll": "uncertainty",
        "multi_task": "multi_objective",
        "interval_regression": "multi_objective",
        "quantile_regression": "multi_objective",
        "sample_weighted": "weighting",
        "auxiliary_physics_loss": "weighting",
    },
    "regularize": {
        "strong_regularization": "weight",
        "ema": "weight",
        "dropout": "weight",
        "freeze_backbone": "transfer",
    },
}


def get_categories_for_action(action_type: str) -> dict[str, list[str]]:
    """Return {category: [techniques]} for an action type."""
    mapping = CATEGORY_MAP.get(action_type, {})
    result: dict[str, list[str]] = {}
    for technique, cat in mapping.items():
        result.setdefault(cat, []).append(technique)
    return result


def validate_category(action_type: str, category: str | None, technique: str) -> str | None:
    """Validate and return the correct category. Returns error message on failure."""
    mapping = CATEGORY_MAP.get(action_type)
    if mapping is None:
        return None
    expected_cat = mapping.get(technique)
    if expected_cat is None:
        return f"Unknown technique '{technique}' for {action_type}"
    if category is not None and category != expected_cat:
        return (
            f"Wrong category '{category}' for {action_type}:{technique}. "
            f"Expected '{expected_cat}'."
        )
    return None


def infer_category(action_type: str, technique: str) -> str | None:
    """Infer category from action_type + technique."""
    mapping = CATEGORY_MAP.get(action_type, {})
    return mapping.get(technique)


# =========================================================================
# Action types
# =========================================================================

class ActionType(str, Enum):
    SET_CV = "set_cv"
    FEATURE_ENGINEERING = "feature_engineering"
    DETECT_SHIFT = "detect_shift"
    TRAIN_MODEL = "train_model"
    HANDLE_IMBALANCE = "handle_imbalance"
    CLEAN_DATA = "clean_data"
    AUGMENTATION = "augmentation"
    ENSEMBLE = "ensemble"
    PSEUDO_LABEL = "pseudo_label"
    POSTPROCESS = "postprocess"
    TUNE_LOSS = "tune_loss"
    REGULARIZE = "regularize"
    INSPECT_TOP_SOLUTION = "inspect_top_solution"
    SUBMIT = "submit"


_PARAM_KEY_MAP: dict[str, str] = {
    "set_cv": "strategy",
    "feature_engineering": "technique",
    "detect_shift": "method",
    "train_model": "algorithm",
    "handle_imbalance": "method",
    "clean_data": "method",
    "augmentation": "method",
    "ensemble": "method",
    "postprocess": "method",
    "tune_loss": "method",
    "regularize": "method",
}


def get_param_key(action_type: str) -> str | None:
    return _PARAM_KEY_MAP.get(action_type)


def get_allowed_values(action_type: str) -> list[str]:
    mapping = CATEGORY_MAP.get(action_type, {})
    return list(mapping.keys())


# =========================================================================
# Action
# =========================================================================

class Action(BaseModel):
    """A single structured action with hierarchical category."""

    action_type: ActionType
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("parameters", mode="before")
    @classmethod
    def _coerce_parameters(cls, v: Any) -> dict[str, Any]:
        if v is None:
            return {}
        return v

    def technique_value(self) -> str | None:
        """Return the technique/strategy/method/algorithm value."""
        key = get_param_key(self.action_type.value)
        if key is None:
            return None
        return self.parameters.get(key)

    def category_value(self) -> str | None:
        """Return the declared category, or infer it."""
        declared = self.parameters.get("category")
        if declared:
            return declared
        tv = self.technique_value()
        if tv:
            return infer_category(self.action_type.value, tv)
        return None

    def tag(self) -> str:
        """Concise string for history tracking (without category)."""
        at = self.action_type
        if at == ActionType.PSEUDO_LABEL:
            return f"pseudo_label:{self.parameters.get('iterations', 1)}"
        if at in (ActionType.INSPECT_TOP_SOLUTION, ActionType.SUBMIT):
            return at.value
        tv = self.technique_value()
        if tv:
            return f"{at.value}:{tv}"
        return at.value

    def full_tag(self) -> str:
        """Verbose string including category."""
        cat = self.category_value()
        base = self.tag()
        if cat:
            parts = base.split(":", 1)
            if len(parts) == 2:
                return f"{parts[0]}:{cat}:{parts[1]}"
        return base


# =========================================================================
# Causal dataset properties
# =========================================================================

class DatasetProperties(BaseModel):
    """Ground-truth properties that govern causal reward logic."""

    has_shift: bool = False
    has_leakage: bool = False
    has_noise_features: bool = False
    has_missing_data: bool = False
    has_imbalance: bool = False
    imbalance_ratio: float = 0.5
    has_heavy_tails: bool = False
    has_time_column: bool = False
    has_group_column: bool = False
    has_images: bool = False
    has_spatial_data: bool = False
    has_text: bool = False
    is_safety_critical: bool = False
    needs_physics: bool = False


# =========================================================================
# Failure mode definition
# =========================================================================

class FailureMode(BaseModel):
    """A trap the agent can fall into."""

    name: str
    trigger_tag: str
    condition_field: str
    condition_value: bool
    cv_effect: float
    test_effect: float
    message: str


# =========================================================================
# Dataset metadata (observation-facing)
# =========================================================================

class ColumnInfo(BaseModel):
    name: str
    dtype: str
    missing_pct: float = Field(ge=0.0, le=100.0)
    unique_count: int = Field(ge=0)


class DatasetMetadata(BaseModel):
    num_rows: int
    num_features: int
    columns: list[ColumnInfo]
    target_column: str
    task_type: str = "classification"
    has_time_column: bool = False
    has_group_column: bool = False
    has_image_data: bool = False
    has_text_data: bool = False
    has_spatial_data: bool = False
    class_balance: dict[str, float] = Field(default_factory=dict)
    target_distribution: str = "normal"


# =========================================================================
# Observation
# =========================================================================

class Observation(BaseModel):
    dataset_metadata: DatasetMetadata
    applied_strategies: list[str]
    current_cv_score: float
    leaderboard_rank: int
    step_count: int
    max_steps: int
    done: bool
    message: str = ""


# =========================================================================
# Reward
# =========================================================================

class RewardBreakdown(BaseModel):
    cv_improvement: float = 0.0
    strategy_bonus: float = 0.0
    context_bonus: float = 0.0
    combo_bonus: float = 0.0
    redundancy_penalty: float = 0.0
    irrelevant_penalty: float = 0.0
    trap_penalty: float = 0.0
    overfitting_penalty: float = 0.0
    submission_bonus: float = 0.0


class Reward(BaseModel):
    total: float
    breakdown: RewardBreakdown


# =========================================================================
# Full environment state
# =========================================================================

class EnvState(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    done: bool
    cv_score: float
    test_score: float
    applied_strategies: list[str]
    strategy_history: list[str]
    leaderboard_rank: int
    leaderboard: list[dict[str, Any]]
    submitted: bool
    hint_count: int
    active_combos: list[str] = Field(default_factory=list)
    traps_triggered: list[str] = Field(default_factory=list)


# =========================================================================
# Step response
# =========================================================================

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
