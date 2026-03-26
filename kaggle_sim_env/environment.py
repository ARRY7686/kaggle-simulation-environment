"""Core KaggleSimEnv environment v3.

Causal logic, hierarchical categories, failure-mode traps, contextual rewards.
"""

from __future__ import annotations

from typing import Any

from kaggle_sim_env.hints import HintProvider
from kaggle_sim_env.leaderboard import Leaderboard
from kaggle_sim_env.models import (
    Action,
    ActionType,
    EnvState,
    FailureMode,
    Observation,
    Reward,
    RewardBreakdown,
    StepResponse,
    get_allowed_values,
    get_param_key,
    validate_category,
)
from kaggle_sim_env.rewards import compute_reward
from kaggle_sim_env.tasks import TaskDefinition, get_task


class KaggleSimEnv:
    """RL environment simulating a Kaggle competition with causal dataset logic."""

    def __init__(self) -> None:
        self._task: TaskDefinition | None = None
        self._cv_score: float = 0.0
        self._test_score: float = 0.0
        self._applied: list[str] = []
        self._history: list[str] = []
        self._step_count: int = 0
        self._done: bool = True
        self._submitted: bool = False
        self._leaderboard: Leaderboard | None = None
        self._hints: HintProvider | None = None
        self._overfitting_accum: float = 0.0
        self._active_combos: list[str] = []
        self._traps_triggered: list[str] = []
        self._mitigated_traps: set[str] = set()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "easy_churn") -> Observation:
        self._task = get_task(task_id)
        self._cv_score = self._task.base_cv_score
        self._test_score = self._task.base_test_score
        self._applied = []
        self._history = []
        self._step_count = 0
        self._max_steps = 10
        self._done = False
        self._submitted = False
        self._overfitting_accum = 0.0
        self._active_combos = []
        self._traps_triggered = []
        self._mitigated_traps = set()
        self._leaderboard = Leaderboard(task_id, list(self._task.ghost_scores))
        self._hints = HintProvider(list(self._task.hints))
        return self._observation(message="Environment reset. Choose your first action.")

    def step(self, action: Action) -> StepResponse:
        if self._task is None or self._done:
            raise RuntimeError("Environment not active. Call reset() first.")

        self._step_count += 1
        info: dict[str, Any] = {}
        prev_cv = self._cv_score
        tag = action.tag()

        # --- Validate ---
        valid, msg = self._validate_action(action)
        if not valid:
            reward = Reward(total=-0.05, breakdown=RewardBreakdown(redundancy_penalty=-0.05))
            info["error"] = msg
            obs = self._observation(message=f"Invalid action: {msg}")
            done = self._check_done()
            return StepResponse(observation=obs, reward=reward, done=done, info=info)

        # --- Apply action (mutates scores) ---
        self._apply_action(action, info)

        # --- Check failure-mode traps ---
        traps_this_step = self._check_traps(action, info)

        # --- Check strategy combos ---
        newly_completed = self._check_combos()
        if newly_completed:
            info["combos_completed"] = newly_completed

        # --- Contextual relevance ---
        relevance = self._task.context_relevance.get(tag)

        is_submit = action.action_type == ActionType.SUBMIT
        reward = compute_reward(
            prev_cv=prev_cv,
            new_cv=self._cv_score,
            new_test=self._test_score,
            action_tag=tag,
            expected_strategies=self._task.expected_strategies,
            already_applied=self._applied[:-1] if tag in self._applied else self._applied,
            is_submit=is_submit,
            submitted_test_score=self._test_score if is_submit else None,
            newly_completed_combos=newly_completed,
            context_relevance=relevance,
            traps_triggered_this_step=traps_this_step,
        )

        self._history.append(tag)
        done = self._check_done()
        obs = self._observation(message=info.get("message", ""))
        return StepResponse(observation=obs, reward=reward, done=done, info=info)

    def state(self) -> EnvState:
        assert self._task is not None
        assert self._leaderboard is not None
        assert self._hints is not None
        return EnvState(
            task_id=self._task.task_id,
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            done=self._done,
            cv_score=round(self._cv_score, 4),
            test_score=round(self._test_score, 4),
            applied_strategies=list(self._applied),
            strategy_history=list(self._history),
            leaderboard_rank=self._leaderboard.agent_rank(self._test_score),
            leaderboard=self._leaderboard.full_board(self._test_score),
            submitted=self._submitted,
            hint_count=self._hints.hints_given,
            active_combos=list(self._active_combos),
            traps_triggered=list(self._traps_triggered),
        )

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _observation(self, message: str = "") -> Observation:
        assert self._task is not None and self._leaderboard is not None
        return Observation(
            dataset_metadata=self._task.dataset_metadata,
            applied_strategies=list(self._applied),
            current_cv_score=round(self._cv_score, 4),
            leaderboard_rank=self._leaderboard.agent_rank(self._test_score),
            step_count=self._step_count,
            max_steps=self._task.max_steps,
            done=self._done,
            message=message,
        )

    # ------------------------------------------------------------------
    # Validation (hierarchical categories)
    # ------------------------------------------------------------------

    def _validate_action(self, action: Action) -> tuple[bool, str]:
        at = action.action_type
        p = action.parameters

        if at == ActionType.PSEUDO_LABEL:
            iters = p.get("iterations")
            if iters is None or not isinstance(iters, int) or iters < 1:
                return False, "pseudo_label requires 'iterations' (int >= 1)"
            return True, ""

        if at == ActionType.SUBMIT:
            if self._submitted:
                return False, "Already submitted."
            if not any(s.startswith("train_model:") for s in self._applied):
                return False, "Cannot submit without training a model first."
            return True, ""

        if at == ActionType.INSPECT_TOP_SOLUTION:
            return True, ""

        key = get_param_key(at.value)
        if key:
            value = p.get(key)
            allowed = get_allowed_values(at.value)
            if value is None or value not in allowed:
                return False, f"{at.value} requires '{key}' in {allowed}"
            cat_err = validate_category(at.value, p.get("category"), value)
            if cat_err:
                return False, cat_err

        return True, ""

    # ------------------------------------------------------------------
    # Action application with causal logic
    # ------------------------------------------------------------------

    def _apply_action(self, action: Action, info: dict[str, Any]) -> None:
        assert self._task is not None and self._hints is not None
        tag = action.tag()
        modifiers = self._task.score_modifiers
        overfit_risk = self._task.overfitting_risk
        props = self._task.dataset_properties

        if action.action_type == ActionType.INSPECT_TOP_SOLUTION:
            hint = self._hints.next_hint()
            info["hint"] = hint
            info["message"] = f"Hint: {hint}"
            if tag not in self._applied:
                self._applied.append(tag)
            return

        if action.action_type == ActionType.SUBMIT:
            self._submitted = True
            self._done = True
            info["message"] = (
                f"Submitted! Final test score: {self._test_score:.4f}  "
                f"CV score: {self._cv_score:.4f}"
            )
            return

        is_repeat = tag in self._applied

        # --- Track mitigation actions ---
        self._update_mitigations(tag, props)

        if tag in modifiers:
            cv_delta = modifiers[tag]["cv"]
            test_delta = modifiers[tag]["test"]

            if is_repeat:
                cv_delta *= 0.1
                test_delta *= 0.1
            else:
                self._applied.append(tag)

            overfit_extra = overfit_risk.get(tag, 0.0)
            self._overfitting_accum += overfit_extra
            cv_delta += overfit_extra

            self._cv_score = min(1.0, self._cv_score + cv_delta)
            self._test_score = min(1.0, self._test_score + test_delta)

            rank = self._leaderboard.agent_rank(self._test_score) if self._leaderboard else "?"
            info["message"] = f"Applied {action.full_tag()}. CV: {self._cv_score:.4f}, Rank: {rank}"
        else:
            if tag not in self._applied:
                self._applied.append(tag)
            info["message"] = f"Applied {action.full_tag()} (no modifier for this task)."

    def _update_mitigations(self, tag: str, props: Any) -> None:
        """Track which failure modes the agent has pre-emptively mitigated."""
        if tag == "detect_shift:adversarial_validation" and props.has_shift:
            self._mitigated_traps.add("ignoring_shift")
        if tag == "detect_shift:remove_identifiers" and props.has_shift:
            self._mitigated_traps.add("ignoring_shift")
        if tag == "clean_data:remove_leaky_features" and props.has_leakage:
            self._mitigated_traps.add("keeping_leaky_feature")
        if tag == "feature_engineering:sin_cos_encoding" and props.has_spatial_data:
            self._mitigated_traps.add("raw_heading_without_sincos")
        if tag.startswith("augmentation:") and props.has_images:
            self._mitigated_traps.add("no_augmentation_on_images")

    # ------------------------------------------------------------------
    # Failure-mode trap detection
    # ------------------------------------------------------------------

    def _check_traps(self, action: Action, info: dict[str, Any]) -> list[str]:
        assert self._task is not None
        tag = action.tag()
        props = self._task.dataset_properties
        triggered: list[str] = []

        for fm in self._task.failure_modes:
            if fm.name in self._traps_triggered:
                continue
            if fm.name in self._mitigated_traps:
                continue
            if tag != fm.trigger_tag:
                continue

            prop_val = getattr(props, fm.condition_field, None)
            if prop_val == fm.condition_value:
                self._cv_score = min(1.0, self._cv_score + fm.cv_effect)
                self._test_score = max(0.0, self._test_score + fm.test_effect)
                self._traps_triggered.append(fm.name)
                triggered.append(fm.name)

                prev_msg = info.get("message", "")
                info["message"] = f"{prev_msg} TRAP: {fm.message}"
                info.setdefault("traps", []).append({
                    "name": fm.name,
                    "message": fm.message,
                    "cv_effect": fm.cv_effect,
                    "test_effect": fm.test_effect,
                })

        return triggered

    # ------------------------------------------------------------------
    # Combo detection
    # ------------------------------------------------------------------

    def _check_combos(self) -> list[str]:
        assert self._task is not None
        newly_completed: list[str] = []
        applied_set = set(self._applied)
        for combo in self._task.strategy_combos:
            if combo.name in self._active_combos:
                continue
            if combo.required.issubset(applied_set):
                self._active_combos.append(combo.name)
                self._cv_score = min(1.0, self._cv_score + combo.cv_bonus)
                self._test_score = min(1.0, self._test_score + combo.test_bonus)
                newly_completed.append(combo.name)
        return newly_completed

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def _check_done(self) -> bool:
        if self._done:
            return True
        assert self._task is not None
        if self._step_count >= self._task.max_steps:
            self._done = True
        return self._done
