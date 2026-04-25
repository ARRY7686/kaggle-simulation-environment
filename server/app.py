"""
OpenEnv-compatible server for KaggleSimEnv.

Wraps the KaggleSimEnv in the openenv Environment interface and exposes
it via ``create_app``.  Additional custom endpoints (tasks, grader,
baseline, actions) are mounted on the same FastAPI app.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import Field

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import (
    Action as OEAction,
    EnvironmentMetadata,
    Observation as OEObservation,
    State as OEState,
)

from kaggle_sim_env.environment import KaggleSimEnv
from kaggle_sim_env.grader import GradeResult, Grader
from kaggle_sim_env.models import (
    Action as KSAction,
    ActionType,
    CATEGORY_MAP,
    get_categories_for_action,
)
from kaggle_sim_env.tasks import TASK_REGISTRY, get_task


# ── OpenEnv-compatible Pydantic models ───────────────────────────────────

class KaggleAction(OEAction):
    action_type: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class KaggleObservation(OEObservation):
    dataset_metadata: Dict[str, Any] = Field(default_factory=dict)
    applied_strategies: List[str] = Field(default_factory=list)
    current_cv_score: float = 0.0
    leaderboard_rank: int = 0
    step_count: int = 0
    max_steps: int = 10
    message: str = ""


class KaggleState(OEState):
    task_id: str = ""
    max_steps: int = 10
    done: bool = False
    cv_score: float = 0.0
    test_score: float = 0.0
    applied_strategies: List[str] = Field(default_factory=list)
    strategy_history: List[str] = Field(default_factory=list)
    leaderboard_rank: int = 0
    leaderboard: List[Dict[str, Any]] = Field(default_factory=list)
    submitted: bool = False
    hint_count: int = 0
    active_combos: List[str] = Field(default_factory=list)
    traps_triggered: List[str] = Field(default_factory=list)


# ── Module-level singleton so custom endpoints can access the live env ──

_active_env: Optional["KaggleSimEnvironment"] = None


# ── OpenEnv Environment adapter ──────────────────────────────────────────

class KaggleSimEnvironment(Environment[KaggleAction, KaggleObservation, KaggleState]):
    """Bridges KaggleSimEnv to the openenv ``Environment`` ABC."""

    def __init__(self) -> None:
        super().__init__()
        self._env = KaggleSimEnv()
        self._task_id = "easy_churn"

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        *,
        task_id: str = "easy_churn",
        **kwargs: Any,
    ) -> KaggleObservation:
        global _active_env
        _active_env = self
        self._task_id = task_id
        obs = self._env.reset(task_id=task_id)
        return KaggleObservation(
            done=obs.done,
            reward=0.0,
            dataset_metadata=obs.dataset_metadata.model_dump(),
            applied_strategies=obs.applied_strategies,
            current_cv_score=obs.current_cv_score,
            leaderboard_rank=obs.leaderboard_rank,
            step_count=obs.step_count,
            max_steps=obs.max_steps,
            message=obs.message,
        )

    def step(
        self,
        action: KaggleAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> KaggleObservation:
        ks_action = KSAction(
            action_type=action.action_type,
            parameters=action.parameters,
        )
        result = self._env.step(ks_action)
        obs = result.observation
        return KaggleObservation(
            done=obs.done,
            reward=result.reward.total,
            metadata={
                "info": result.info,
                "breakdown": result.reward.breakdown.model_dump(),
            },
            dataset_metadata=obs.dataset_metadata.model_dump(),
            applied_strategies=obs.applied_strategies,
            current_cv_score=obs.current_cv_score,
            leaderboard_rank=obs.leaderboard_rank,
            step_count=obs.step_count,
            max_steps=obs.max_steps,
            message=obs.message,
        )

    @property
    def state(self) -> KaggleState:
        s = self._env.state()
        return KaggleState(
            episode_id=s.task_id,
            step_count=s.step_count,
            task_id=s.task_id,
            max_steps=s.max_steps,
            done=s.done,
            cv_score=s.cv_score,
            test_score=s.test_score,
            applied_strategies=s.applied_strategies,
            strategy_history=s.strategy_history,
            leaderboard_rank=s.leaderboard_rank,
            leaderboard=s.leaderboard,
            submitted=s.submitted,
            hint_count=s.hint_count,
            active_combos=s.active_combos,
            traps_triggered=s.traps_triggered,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="KaggleSimEnv",
            description=(
                "RL environment simulating Kaggle competitions with hierarchical "
                "actions, causal dataset properties, failure-mode traps, and "
                "contextual scoring."
            ),
            version="3.0.0",
        )


# ── Create the OpenEnv app ───────────────────────────────────────────────

# Pre-create a singleton so /reset and /step share state across HTTP requests.
# create_app calls _env_factory() on every request; returning the same instance
# keeps episode state intact between reset and step calls.
_singleton_env = KaggleSimEnvironment()

app = create_app(
    lambda: _singleton_env,
    KaggleAction,
    KaggleObservation,
    env_name="kaggle_sim_env",
    max_concurrent_envs=1,
)


# ── Custom endpoints (tasks, grader, baseline, actions) ──────────────────

from pydantic import BaseModel

_grader = Grader()


class _TaskSummary(BaseModel):
    task_id: str
    title: str
    difficulty: str
    description: str
    max_steps: int
    num_expected_strategies: int
    num_strategy_combos: int
    num_failure_modes: int


class _BaselineRequest(BaseModel):
    task_id: str = "easy_churn"


class _ActionCategoryEntry(BaseModel):
    action_type: str
    parameter_key: Optional[str] = None
    categories: Dict[str, List[str]] = Field(default_factory=dict)


@app.get("/tasks", response_model=List[_TaskSummary], tags=["Custom"])
def list_tasks() -> list[_TaskSummary]:
    return [
        _TaskSummary(
            task_id=t.task_id,
            title=t.title,
            difficulty=t.difficulty,
            description=t.description,
            max_steps=t.max_steps,
            num_expected_strategies=len(t.expected_strategies),
            num_strategy_combos=len(t.strategy_combos),
            num_failure_modes=len(t.failure_modes),
        )
        for t in TASK_REGISTRY.values()
    ]


@app.post("/grader", response_model=GradeResult, tags=["Custom"])
def grade() -> GradeResult:
    if _active_env is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No active environment. Call /reset first.")
    s = _active_env._env.state()
    return _grader.grade(s, get_task(s.task_id))


@app.get("/health", tags=["Custom"])
def health() -> dict:
    return {"status": "healthy"}


@app.get("/actions", response_model=List[_ActionCategoryEntry], tags=["Custom"])
def action_space() -> list[_ActionCategoryEntry]:
    from kaggle_sim_env.models import _PARAM_KEY_MAP

    entries: list[_ActionCategoryEntry] = []
    for at, key in _PARAM_KEY_MAP.items():
        cats = get_categories_for_action(at)
        entries.append(_ActionCategoryEntry(action_type=at, parameter_key=key, categories=cats))
    entries.append(
        _ActionCategoryEntry(
            action_type="pseudo_label",
            parameter_key="iterations",
            categories={"iterations": ["1", "2", "3"]},
        )
    )
    entries.append(_ActionCategoryEntry(action_type="inspect_top_solution"))
    entries.append(_ActionCategoryEntry(action_type="submit"))
    return entries


# ── Entry points ─────────────────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
