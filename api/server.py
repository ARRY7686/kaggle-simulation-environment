"""FastAPI server for KaggleSimEnv v3.

Endpoints:
    POST /reset, /step, /grader, /baseline
    GET  /state, /tasks, /actions, /health
"""

from __future__ import annotations

from typing import Any

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, Field

from kaggle_sim_env.environment import KaggleSimEnv
from kaggle_sim_env.grader import GradeResult, Grader
from kaggle_sim_env.models import (
    Action,
    ActionType,
    CATEGORY_MAP,
    EnvState,
    Observation,
    StepResponse,
    get_categories_for_action,
)
from kaggle_sim_env.tasks import TASK_REGISTRY, get_task

app = FastAPI(
    title="KaggleSimEnv – OpenEnv API",
    description=(
        "RL environment simulating Kaggle competitions with hierarchical actions, "
        "causal dataset properties, failure-mode traps, and contextual scoring."
    ),
    version="3.0.0",
)

env = KaggleSimEnv()
grader = Grader()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_churn"

class StepRequest(BaseModel):
    action_type: str
    parameters: dict[str, Any] = Field(default_factory=dict)

class TaskSummary(BaseModel):
    task_id: str
    title: str
    difficulty: str
    description: str
    max_steps: int
    num_expected_strategies: int
    num_strategy_combos: int
    num_failure_modes: int

class BaselineRequest(BaseModel):
    task_id: str = "easy_churn"

class BaselineResult(BaseModel):
    task_id: str
    actions_taken: list[dict[str, Any]]
    grade: GradeResult

class ActionCategoryEntry(BaseModel):
    action_type: str
    parameter_key: str | None = None
    categories: dict[str, list[str]] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    try:
        return env.reset(task_id=req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    try:
        action = Action(action_type=req.action_type, parameters=req.parameters)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid action: {exc}")
    try:
        return env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/state", response_model=EnvState)
def state() -> EnvState:
    try:
        return env.state()
    except (AssertionError, RuntimeError):
        raise HTTPException(status_code=400, detail="No active environment.")

@app.get("/tasks", response_model=list[TaskSummary])
def list_tasks() -> list[TaskSummary]:
    return [
        TaskSummary(
            task_id=t.task_id, title=t.title, difficulty=t.difficulty,
            description=t.description, max_steps=t.max_steps,
            num_expected_strategies=len(t.expected_strategies),
            num_strategy_combos=len(t.strategy_combos),
            num_failure_modes=len(t.failure_modes),
        )
        for t in TASK_REGISTRY.values()
    ]

@app.post("/grader", response_model=GradeResult)
def grade() -> GradeResult:
    try:
        s = env.state()
    except (AssertionError, RuntimeError):
        raise HTTPException(status_code=400, detail="No active environment.")
    return grader.grade(s, get_task(s.task_id))

@app.get("/actions", response_model=list[ActionCategoryEntry])
def action_space() -> list[ActionCategoryEntry]:
    """Hierarchical action space with categories."""
    from kaggle_sim_env.models import _PARAM_KEY_MAP
    entries: list[ActionCategoryEntry] = []
    for at, key in _PARAM_KEY_MAP.items():
        cats = get_categories_for_action(at)
        entries.append(ActionCategoryEntry(action_type=at, parameter_key=key, categories=cats))
    entries.append(ActionCategoryEntry(
        action_type="pseudo_label", parameter_key="iterations",
        categories={"iterations": ["1", "2", "3"]},
    ))
    entries.append(ActionCategoryEntry(action_type="inspect_top_solution"))
    entries.append(ActionCategoryEntry(action_type="submit"))
    return entries


# ---------------------------------------------------------------------------
# Baseline: structured, hint-using, phase-based agent
# ---------------------------------------------------------------------------

@app.post("/baseline", response_model=BaselineResult)
def run_baseline(req: BaselineRequest) -> BaselineResult:
    task = get_task(req.task_id)
    bl_env = KaggleSimEnv()
    bl_env.reset(task_id=req.task_id)

    actions_log: list[dict[str, Any]] = []
    plan = _baseline_plan(task.task_id)

    for action in plan:
        result = bl_env.step(action)
        actions_log.append({
            "action": action.model_dump(),
            "cv_score": result.observation.current_cv_score,
            "reward": result.reward.total,
            "traps": result.info.get("traps", []),
            "combos": result.info.get("combos_completed", []),
        })
        if result.done:
            break

    s = bl_env.state()
    g = grader.grade(s, task)
    return BaselineResult(task_id=req.task_id, actions_taken=actions_log, grade=g)


def _baseline_plan(task_id: str) -> list[Action]:
    """Structured, phase-based expert plans that use hints first."""
    A = Action

    if task_id == "easy_churn":
        return [
            # Phase 1: Inspect
            A(action_type=ActionType.INSPECT_TOP_SOLUTION, parameters={}),
            # Phase 2: CV + Features
            A(action_type=ActionType.SET_CV, parameters={"category": "standard", "strategy": "kfold"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "distribution", "technique": "normalize"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "interaction", "technique": "domain_ratios"}),
            # Phase 3: Train
            A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "tree", "algorithm": "xgboost"}),
            # Phase 4: Imbalance
            A(action_type=ActionType.HANDLE_IMBALANCE, parameters={"category": "weighting", "method": "scale_pos_weight"}),
            A(action_type=ActionType.HANDLE_IMBALANCE, parameters={"category": "calibration", "method": "optimize_threshold"}),
            # Phase 5: Submit
            A(action_type=ActionType.SUBMIT, parameters={}),
        ]

    if task_id == "medium_fraud":
        return [
            # Phase 1: Inspect + Diagnose
            A(action_type=ActionType.INSPECT_TOP_SOLUTION, parameters={}),
            A(action_type=ActionType.INSPECT_TOP_SOLUTION, parameters={}),
            A(action_type=ActionType.DETECT_SHIFT, parameters={"category": "detection", "method": "adversarial_validation"}),
            A(action_type=ActionType.DETECT_SHIFT, parameters={"category": "mitigation", "method": "remove_identifiers"}),
            A(action_type=ActionType.DETECT_SHIFT, parameters={"category": "mitigation", "method": "domain_invariant_features"}),
            # Phase 2: CV + Features
            A(action_type=ActionType.SET_CV, parameters={"category": "temporal", "strategy": "time_split"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "distribution", "technique": "log_transform"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "interaction", "technique": "domain_ratios"}),
            # Phase 3: Train + Imbalance
            A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "tree", "algorithm": "xgboost"}),
            A(action_type=ActionType.HANDLE_IMBALANCE, parameters={"category": "weighting", "method": "scale_pos_weight"}),
            A(action_type=ActionType.HANDLE_IMBALANCE, parameters={"category": "calibration", "method": "optimize_threshold"}),
            A(action_type=ActionType.TUNE_LOSS, parameters={"category": "asymmetric", "method": "asymmetric_loss"}),
            # Phase 4: Regularize + Ensemble
            A(action_type=ActionType.REGULARIZE, parameters={"category": "weight", "method": "strong_regularization"}),
            A(action_type=ActionType.ENSEMBLE, parameters={"category": "averaging", "method": "weighted_average"}),
            A(action_type=ActionType.SUBMIT, parameters={}),
        ]

    if task_id == "hard_leaky_noisy":
        return [
            # Phase 1: Inspect + Diagnose + Clean
            A(action_type=ActionType.INSPECT_TOP_SOLUTION, parameters={}),
            A(action_type=ActionType.DETECT_SHIFT, parameters={"category": "detection", "method": "adversarial_validation"}),
            A(action_type=ActionType.DETECT_SHIFT, parameters={"category": "mitigation", "method": "remove_identifiers"}),
            A(action_type=ActionType.CLEAN_DATA, parameters={"category": "removal", "method": "remove_leaky_features"}),
            A(action_type=ActionType.CLEAN_DATA, parameters={"category": "removal", "method": "remove_outliers"}),
            A(action_type=ActionType.CLEAN_DATA, parameters={"category": "reconstruction", "method": "analytical_reconstruction"}),
            A(action_type=ActionType.CLEAN_DATA, parameters={"category": "reconstruction", "method": "nan_native_model"}),
            # Phase 2: CV + Features
            A(action_type=ActionType.SET_CV, parameters={"category": "group", "strategy": "stratified_group_kfold"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "distribution", "technique": "log_transform"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "interaction", "technique": "domain_ratios"}),
            # Phase 3: Train + Imbalance
            A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "tree", "algorithm": "xgboost"}),
            A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "tree", "algorithm": "lightgbm"}),
            A(action_type=ActionType.HANDLE_IMBALANCE, parameters={"category": "weighting", "method": "scale_pos_weight"}),
            A(action_type=ActionType.HANDLE_IMBALANCE, parameters={"category": "calibration", "method": "calibrate_probabilities"}),
            A(action_type=ActionType.HANDLE_IMBALANCE, parameters={"category": "calibration", "method": "optimize_threshold"}),
            # Phase 4: Ensemble + Regularize
            A(action_type=ActionType.ENSEMBLE, parameters={"category": "stacking", "method": "stacking"}),
            A(action_type=ActionType.ENSEMBLE, parameters={"category": "diversity", "method": "diverse_features"}),
            A(action_type=ActionType.REGULARIZE, parameters={"category": "weight", "method": "strong_regularization"}),
            A(action_type=ActionType.PSEUDO_LABEL, parameters={"iterations": 1}),
            A(action_type=ActionType.SUBMIT, parameters={}),
        ]

    if task_id == "image_quality":
        return [
            # Phase 1: Inspect
            A(action_type=ActionType.INSPECT_TOP_SOLUTION, parameters={}),
            A(action_type=ActionType.INSPECT_TOP_SOLUTION, parameters={}),
            # Phase 2: CV + Backbone
            A(action_type=ActionType.SET_CV, parameters={"category": "group", "strategy": "group_kfold"}),
            A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "neural", "algorithm": "pretrained_backbone"}),
            A(action_type=ActionType.REGULARIZE, parameters={"category": "transfer", "method": "freeze_backbone"}),
            # Phase 3: Augmentation
            A(action_type=ActionType.AUGMENTATION, parameters={"category": "geometric", "method": "geometric"}),
            A(action_type=ActionType.AUGMENTATION, parameters={"category": "color", "method": "color_transform"}),
            A(action_type=ActionType.AUGMENTATION, parameters={"category": "domain", "method": "camera_simulation"}),
            A(action_type=ActionType.AUGMENTATION, parameters={"category": "color", "method": "clahe"}),
            # Phase 4: Loss + Features
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "distribution", "technique": "log_transform"}),
            A(action_type=ActionType.TUNE_LOSS, parameters={"category": "uncertainty", "method": "gaussian_nll"}),
            A(action_type=ActionType.TUNE_LOSS, parameters={"category": "multi_objective", "method": "multi_task"}),
            # Phase 5: Stabilize + Ensemble
            A(action_type=ActionType.REGULARIZE, parameters={"category": "weight", "method": "ema"}),
            A(action_type=ActionType.ENSEMBLE, parameters={"category": "averaging", "method": "multi_seed_averaging"}),
            A(action_type=ActionType.ENSEMBLE, parameters={"category": "averaging", "method": "swa"}),
            # Phase 6: Postprocess
            A(action_type=ActionType.POSTPROCESS, parameters={"category": "inference", "method": "tta"}),
            A(action_type=ActionType.POSTPROCESS, parameters={"category": "calibration", "method": "per_group_calibration"}),
            A(action_type=ActionType.POSTPROCESS, parameters={"category": "calibration", "method": "prediction_shrinkage"}),
            A(action_type=ActionType.SUBMIT, parameters={}),
        ]

    if task_id == "trajectory_pred":
        return [
            # Phase 1: Inspect + Clean
            A(action_type=ActionType.INSPECT_TOP_SOLUTION, parameters={}),
            A(action_type=ActionType.SET_CV, parameters={"category": "group", "strategy": "group_kfold"}),
            A(action_type=ActionType.CLEAN_DATA, parameters={"category": "removal", "method": "remove_corrupted"}),
            A(action_type=ActionType.CLEAN_DATA, parameters={"category": "removal", "method": "remove_outliers"}),
            # Phase 2: Spatial features
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "encoding", "technique": "sin_cos_encoding"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "spatial", "technique": "relative_coordinates"}),
            A(action_type=ActionType.FEATURE_ENGINEERING, parameters={"category": "spatial", "technique": "distance_features"}),
            # Phase 3: Train
            A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "neural", "algorithm": "transformer_encoder"}),
            A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "neural", "algorithm": "temporal_cnn"}),
            # Phase 4: Augmentation
            A(action_type=ActionType.AUGMENTATION, parameters={"category": "geometric", "method": "rotation_invariant"}),
            A(action_type=ActionType.AUGMENTATION, parameters={"category": "domain", "method": "symmetry_augmentation"}),
            A(action_type=ActionType.AUGMENTATION, parameters={"category": "domain", "method": "temporal_augmentation"}),
            # Phase 5: Loss + Regularize
            A(action_type=ActionType.TUNE_LOSS, parameters={"category": "uncertainty", "method": "gaussian_nll"}),
            A(action_type=ActionType.TUNE_LOSS, parameters={"category": "weighting", "method": "auxiliary_physics_loss"}),
            A(action_type=ActionType.REGULARIZE, parameters={"category": "weight", "method": "ema"}),
            # Phase 6: Ensemble + Postprocess
            A(action_type=ActionType.ENSEMBLE, parameters={"category": "averaging", "method": "multi_seed_averaging"}),
            A(action_type=ActionType.POSTPROCESS, parameters={"category": "domain", "method": "physics_constraints"}),
            A(action_type=ActionType.SUBMIT, parameters={}),
        ]

    return [
        A(action_type=ActionType.TRAIN_MODEL, parameters={"category": "tree", "algorithm": "xgboost"}),
        A(action_type=ActionType.SUBMIT, parameters={}),
    ]


@app.get("/health")
def health() -> dict[str, str]:
    """OpenEnv runtime check expects status == 'healthy'."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict[str, str]:
    """OpenEnv standard metadata endpoint."""
    return {
        "name": "KaggleSimEnv",
        "description": (
            "RL environment simulating Kaggle competitions with hierarchical actions, "
            "causal dataset properties, failure-mode traps, and contextual scoring."
        ),
    }


@app.get("/schema")
def schema_endpoint() -> dict[str, Any]:
    """OpenEnv combined JSON Schema for action, observation, and state."""
    return {
        "action": Action.model_json_schema(),
        "observation": Observation.model_json_schema(),
        "state": EnvState.model_json_schema(),
    }


@app.post("/mcp")
def mcp_stub(payload: dict[str, Any] = Body(default_factory=dict)) -> dict[str, Any]:
    """Minimal JSON-RPC envelope for OpenEnv runtime validation."""
    return {
        "jsonrpc": "2.0",
        "id": payload.get("id"),
        "result": {"ok": True},
    }
