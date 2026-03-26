"""Reward calculation engine for KaggleSimEnv v3.

Nine-component dense reward:
  cv_improvement       Δ CV score
  strategy_bonus       +0.05 for expected strategy (first use)
  context_bonus        +/- based on action relevance to this dataset
  combo_bonus          +0.08 per strategy combo completed
  redundancy_penalty   -0.03 × repeat count
  irrelevant_penalty   penalty for actions irrelevant to this domain
  trap_penalty         penalty for falling into a failure mode
  overfitting_penalty  penalty when CV-test gap > threshold
  submission_bonus     0.5 × test_score on submit
"""

from __future__ import annotations

from kaggle_sim_env.models import Reward, RewardBreakdown


def compute_reward(
    *,
    prev_cv: float,
    new_cv: float,
    new_test: float,
    action_tag: str,
    expected_strategies: list[str],
    already_applied: list[str],
    is_submit: bool,
    submitted_test_score: float | None = None,
    newly_completed_combos: list[str] | None = None,
    context_relevance: float | None = None,
    traps_triggered_this_step: list[str] | None = None,
) -> Reward:
    cv_improvement = round(new_cv - prev_cv, 6)

    # --- Expected strategy bonus ---
    strategy_bonus = 0.0
    if action_tag in expected_strategies and action_tag not in already_applied:
        strategy_bonus = 0.05

    # --- Contextual relevance bonus/penalty ---
    context_bonus = 0.0
    if context_relevance is not None:
        if context_relevance > 0:
            context_bonus = 0.03 * context_relevance
        elif context_relevance < 0:
            context_bonus = 0.04 * context_relevance  # harsher penalty

    # --- Combo bonus ---
    combo_bonus = 0.0
    if newly_completed_combos:
        combo_bonus = 0.08 * len(newly_completed_combos)

    # --- Redundancy penalty ---
    redundancy_penalty = 0.0
    if action_tag in already_applied:
        count = already_applied.count(action_tag)
        redundancy_penalty = -0.03 * min(count, 3)

    # --- Irrelevant action penalty ---
    irrelevant_penalty = 0.0
    if context_relevance is not None and context_relevance <= -0.8:
        irrelevant_penalty = -0.05

    # --- Trap penalty ---
    trap_penalty = 0.0
    if traps_triggered_this_step:
        trap_penalty = -0.08 * len(traps_triggered_this_step)

    # --- Overfitting penalty ---
    gap = abs(new_cv - new_test)
    overfitting_penalty = -gap * 0.5 if gap > 0.05 else 0.0

    # --- Submission bonus ---
    submission_bonus = 0.0
    if is_submit and submitted_test_score is not None:
        submission_bonus = submitted_test_score * 0.5

    breakdown = RewardBreakdown(
        cv_improvement=round(cv_improvement, 6),
        strategy_bonus=round(strategy_bonus, 6),
        context_bonus=round(context_bonus, 6),
        combo_bonus=round(combo_bonus, 6),
        redundancy_penalty=round(redundancy_penalty, 6),
        irrelevant_penalty=round(irrelevant_penalty, 6),
        trap_penalty=round(trap_penalty, 6),
        overfitting_penalty=round(overfitting_penalty, 6),
        submission_bonus=round(submission_bonus, 6),
    )

    total = sum([
        cv_improvement, strategy_bonus, context_bonus, combo_bonus,
        redundancy_penalty, irrelevant_penalty, trap_penalty,
        overfitting_penalty, submission_bonus,
    ])

    return Reward(total=round(total, 6), breakdown=breakdown)
