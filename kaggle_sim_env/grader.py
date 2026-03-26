"""Deterministic grader for KaggleSimEnv v3.

Score [0.0 – 1.0] combining:
  performance_score  : test score normalised vs ghost competitors
  strategy_score     : contextual — only credit strategies relevant to THIS dataset
  combo_score        : fraction of strategy combos activated
  trap_penalty       : deduction for falling into failure-mode traps

final = 0.40×perf + 0.25×strategy + 0.20×combo + 0.15×(1 - trap_rate)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kaggle_sim_env.models import EnvState
from kaggle_sim_env.tasks import TaskDefinition


class GradeResult(BaseModel):
    task_id: str
    performance_score: float = Field(ge=0.0, le=1.0)
    strategy_score: float = Field(ge=0.0, le=1.0)
    combo_score: float = Field(ge=0.0, le=1.0)
    trap_score: float = Field(ge=0.0, le=1.0)
    final_score: float = Field(ge=0.0, le=1.0)
    details: dict[str, object] = Field(default_factory=dict)


class Grader:

    PERF_W = 0.40
    STRAT_W = 0.25
    COMBO_W = 0.20
    TRAP_W = 0.15

    def grade(self, state: EnvState, task: TaskDefinition) -> GradeResult:
        perf = self._performance_score(state, task)
        strat = self._contextual_strategy_score(state, task)
        combo = self._combo_score(state, task)
        trap = self._trap_score(state, task)
        final = round(
            self.PERF_W * perf + self.STRAT_W * strat
            + self.COMBO_W * combo + self.TRAP_W * trap,
            4,
        )

        return GradeResult(
            task_id=task.task_id,
            performance_score=round(perf, 4),
            strategy_score=round(strat, 4),
            combo_score=round(combo, 4),
            trap_score=round(trap, 4),
            final_score=final,
            details={
                "test_score": state.test_score,
                "cv_score": state.cv_score,
                "cv_test_gap": round(abs(state.cv_score - state.test_score), 4),
                "steps_used": state.step_count,
                "max_steps": state.max_steps,
                "submitted": state.submitted,
                "expected_strategies": task.expected_strategies,
                "applied_strategies": state.applied_strategies,
                "matched_strategies": self._matched(state, task),
                "missing_strategies": self._missing(state, task),
                "irrelevant_strategies_used": self._irrelevant_used(state, task),
                "total_combos": len(task.strategy_combos),
                "active_combos": state.active_combos,
                "traps_triggered": state.traps_triggered,
                "total_failure_modes": len(task.failure_modes),
            },
        )

    # --- Performance ---

    @staticmethod
    def _performance_score(state: EnvState, task: TaskDefinition) -> float:
        if not state.submitted:
            return 0.0
        ghost_max = max(task.ghost_scores) if task.ghost_scores else 1.0
        ghost_min = min(task.ghost_scores) if task.ghost_scores else 0.0
        rng = ghost_max - ghost_min
        if rng < 1e-9:
            return float(state.test_score >= ghost_max)
        raw = (state.test_score - ghost_min) / rng
        return max(0.0, min(1.0, raw))

    # --- Contextual strategy score ---

    @staticmethod
    def _contextual_strategy_score(state: EnvState, task: TaskDefinition) -> float:
        """Credit for relevant strategies, penalise irrelevant ones."""
        expected = set(task.expected_strategies)
        if not expected:
            return 1.0

        matched = expected.intersection(state.applied_strategies)
        base = len(matched) / len(expected)

        irrelevant_count = 0
        for strat in state.applied_strategies:
            rel = task.context_relevance.get(strat)
            if rel is not None and rel <= -0.5:
                irrelevant_count += 1

        penalty = min(irrelevant_count * 0.05, 0.3)
        return max(0.0, round(base - penalty, 4))

    # --- Combo ---

    @staticmethod
    def _combo_score(state: EnvState, task: TaskDefinition) -> float:
        total = len(task.strategy_combos)
        if total == 0:
            return 1.0
        return len(state.active_combos) / total

    # --- Trap score (1.0 = no traps, 0.0 = all traps triggered) ---

    @staticmethod
    def _trap_score(state: EnvState, task: TaskDefinition) -> float:
        total = len(task.failure_modes)
        if total == 0:
            return 1.0
        triggered = len(state.traps_triggered)
        return max(0.0, 1.0 - triggered / total)

    # --- Helpers ---

    @staticmethod
    def _matched(state: EnvState, task: TaskDefinition) -> list[str]:
        return sorted(set(task.expected_strategies) & set(state.applied_strategies))

    @staticmethod
    def _missing(state: EnvState, task: TaskDefinition) -> list[str]:
        return sorted(set(task.expected_strategies) - set(state.applied_strategies))

    @staticmethod
    def _irrelevant_used(state: EnvState, task: TaskDefinition) -> list[str]:
        result = []
        for strat in state.applied_strategies:
            rel = task.context_relevance.get(strat)
            if rel is not None and rel <= -0.5:
                result.append(strat)
        return result
