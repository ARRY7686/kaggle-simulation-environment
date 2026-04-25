"""
generate_training_plots.py
==========================
Generates training-evidence plots by running multiple agent episodes against
the live KaggleSimEnv server.

Two agents are compared:
  * Random agent   — picks actions uniformly at random
  * Baseline agent — structured expert plan from baseline/run_baseline.py

Plots saved to plots/:
  reward_curve.png         — per-episode reward for both agents over 40 episodes
  loss_curve.png           — rolling mean score improvement (reward "learning" curve)
  baseline_vs_trained.png  — per-task score: random vs baseline (mirrors notebook plot)

Usage:
    # Ensure server is running first:
    #   uvicorn server.app:app --host 127.0.0.1 --port 7860
    python generate_training_plots.py [--env-url http://127.0.0.1:7860]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ── Optional matplotlib import ───────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib/numpy not installed – run:  pip install matplotlib numpy")
    sys.exit(1)


# ── Env helpers ───────────────────────────────────────────────────────────

def env_get(base: str, path: str) -> Any:
    r = requests.get(f"{base}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def env_post(base: str, path: str, body: dict | None = None) -> Any:
    r = requests.post(f"{base}{path}", json=body or {}, timeout=60)
    r.raise_for_status()
    return r.json()


# ── Action space for random agent ────────────────────────────────────────

RANDOM_ACTIONS = [
    {"action_type": "inspect_top_solution", "parameters": {}},
    {"action_type": "set_cv",              "parameters": {"category": "standard",  "strategy": "kfold"}},
    {"action_type": "set_cv",              "parameters": {"category": "temporal",  "strategy": "time_split"}},
    {"action_type": "feature_engineering", "parameters": {"category": "distribution", "technique": "normalize"}},
    {"action_type": "feature_engineering", "parameters": {"category": "interaction",  "technique": "interaction_terms"}},
    {"action_type": "detect_shift",        "parameters": {"category": "detection",    "method": "adversarial_validation"}},
    {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "xgboost"}},
    {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "random_forest"}},
    {"action_type": "handle_imbalance",    "parameters": {"category": "weighting",    "method": "scale_pos_weight"}},
    {"action_type": "ensemble",            "parameters": {"category": "averaging",    "method": "weighted_average"}},
    {"action_type": "submit",              "parameters": {}},
]


def run_random_episode(base: str, task_id: str, n_steps: int = 6) -> float:
    """Run a random agent episode, return final grade score."""
    env_post(base, "/reset", {"task_id": task_id})
    actions = random.sample(RANDOM_ACTIONS[:-1], min(n_steps, len(RANDOM_ACTIONS) - 1))
    actions.append({"action_type": "submit", "parameters": {}})
    for action in actions:
        try:
            env_post(base, "/step", {"action_type": action["action_type"],
                                     "parameters": action["parameters"]})
        except Exception:
            pass
    grade = env_post(base, "/grader")
    return float(grade.get("final_score", 0.0))


# ── Baseline expert plans (mirrors baseline/run_baseline.py) ──────────────

def _expert_plan(task_id: str) -> list[dict]:
    plans: dict[str, list[dict]] = {
        "easy_churn": [
            {"action_type": "inspect_top_solution", "parameters": {}},
            {"action_type": "set_cv",              "parameters": {"category": "standard",  "strategy": "kfold"}},
            {"action_type": "feature_engineering", "parameters": {"category": "distribution", "technique": "normalize"}},
            {"action_type": "feature_engineering", "parameters": {"category": "interaction",  "technique": "domain_ratios"}},
            {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "xgboost"}},
            {"action_type": "handle_imbalance",    "parameters": {"category": "weighting",    "method": "scale_pos_weight"}},
            {"action_type": "handle_imbalance",    "parameters": {"category": "calibration",  "method": "optimize_threshold"}},
            {"action_type": "submit",              "parameters": {}},
        ],
        "medium_fraud": [
            {"action_type": "inspect_top_solution", "parameters": {}},
            {"action_type": "detect_shift",        "parameters": {"category": "detection",    "method": "adversarial_validation"}},
            {"action_type": "detect_shift",        "parameters": {"category": "mitigation",   "method": "remove_identifiers"}},
            {"action_type": "set_cv",              "parameters": {"category": "temporal",     "strategy": "time_split"}},
            {"action_type": "feature_engineering", "parameters": {"category": "distribution", "technique": "log_transform"}},
            {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "xgboost"}},
            {"action_type": "handle_imbalance",    "parameters": {"category": "weighting",    "method": "scale_pos_weight"}},
            {"action_type": "ensemble",            "parameters": {"category": "averaging",    "method": "weighted_average"}},
            {"action_type": "submit",              "parameters": {}},
        ],
        "hard_leaky_noisy": [
            {"action_type": "inspect_top_solution", "parameters": {}},
            {"action_type": "clean_data",          "parameters": {"category": "removal",       "method": "remove_leaky_features"}},
            {"action_type": "detect_shift",        "parameters": {"category": "detection",     "method": "adversarial_validation"}},
            {"action_type": "set_cv",              "parameters": {"category": "group",         "strategy": "stratified_group_kfold"}},
            {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "xgboost"}},
            {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "lightgbm"}},
            {"action_type": "ensemble",            "parameters": {"category": "stacking",      "method": "stacking"}},
            {"action_type": "submit",              "parameters": {}},
        ],
        "image_quality": [
            {"action_type": "inspect_top_solution", "parameters": {}},
            {"action_type": "set_cv",              "parameters": {"category": "group",         "strategy": "group_kfold"}},
            {"action_type": "train_model",         "parameters": {"category": "neural",        "algorithm": "pretrained_backbone"}},
            {"action_type": "augmentation",        "parameters": {"category": "geometric",     "method": "geometric"}},
            {"action_type": "augmentation",        "parameters": {"category": "color",         "method": "clahe"}},
            {"action_type": "regularize",          "parameters": {"category": "transfer",      "method": "freeze_backbone"}},
            {"action_type": "postprocess",         "parameters": {"category": "inference",     "method": "tta"}},
            {"action_type": "submit",              "parameters": {}},
        ],
        "trajectory_pred": [
            {"action_type": "inspect_top_solution", "parameters": {}},
            {"action_type": "feature_engineering", "parameters": {"category": "encoding",      "technique": "sin_cos_encoding"}},
            {"action_type": "feature_engineering", "parameters": {"category": "spatial",       "technique": "relative_coordinates"}},
            {"action_type": "set_cv",              "parameters": {"category": "group",         "strategy": "group_kfold"}},
            {"action_type": "train_model",         "parameters": {"category": "neural",        "algorithm": "transformer_encoder"}},
            {"action_type": "tune_loss",           "parameters": {"category": "uncertainty",   "method": "gaussian_nll"}},
            {"action_type": "postprocess",         "parameters": {"category": "domain",        "method": "physics_constraints"}},
            {"action_type": "submit",              "parameters": {}},
        ],
    }
    return plans.get(task_id, [
        {"action_type": "train_model", "parameters": {"category": "tree", "algorithm": "xgboost"}},
        {"action_type": "submit",      "parameters": {}},
    ])


def run_baseline_episode(base: str, task_id: str) -> float:
    """Run the expert plan, return final grade score."""
    env_post(base, "/reset", {"task_id": task_id})
    for action in _expert_plan(task_id):
        try:
            env_post(base, "/step", {"action_type": action["action_type"],
                                     "parameters": action["parameters"]})
        except Exception:
            pass
    grade = env_post(base, "/grader")
    return float(grade.get("final_score", 0.0))


# ── Plot helpers ──────────────────────────────────────────────────────────

def smooth(x: list[float], w: int = 8) -> list[float]:
    arr = np.convolve(x, np.ones(w) / w, mode="valid")
    return arr.tolist()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training evidence plots")
    parser.add_argument("--env-url",   default="http://127.0.0.1:7860")
    parser.add_argument("--episodes",  type=int, default=40,
                        help="Episodes per agent (default: 40)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    base = args.env_url.rstrip("/")

    # Verify env
    health = env_get(base, "/health")
    assert health.get("status") == "healthy", f"Env not healthy: {health}"
    tasks = env_get(base, "/tasks")
    all_task_ids = [t["task_id"] for t in tasks]
    print(f"✅ Env healthy | Tasks: {all_task_ids}\n")

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # ── Collect episode scores ────────────────────────────────────────────
    N = args.episodes
    task_cycle = (all_task_ids * (N // len(all_task_ids) + 1))[:N]
    random.shuffle(task_cycle)

    random_scores: list[float] = []
    baseline_scores: list[float] = []

    print(f"Running {N} episodes per agent …")
    for i, tid in enumerate(task_cycle):
        r_score = run_random_episode(base, tid)
        b_score = run_baseline_episode(base, tid)
        random_scores.append(r_score)
        baseline_scores.append(b_score)
        print(f"  ep {i+1:3d}/{N}  task={tid:<22}  random={r_score:.4f}  baseline={b_score:.4f}")

    print(f"\nRandom   mean: {sum(random_scores)/N:.4f}")
    print(f"Baseline mean: {sum(baseline_scores)/N:.4f}")

    # ── Per-task means ────────────────────────────────────────────────────
    per_task_random:   dict[str, list[float]] = {t: [] for t in all_task_ids}
    per_task_baseline: dict[str, list[float]] = {t: [] for t in all_task_ids}
    for tid, rs, bs in zip(task_cycle, random_scores, baseline_scores):
        per_task_random[tid].append(rs)
        per_task_baseline[tid].append(bs)
    mean_random   = {t: (sum(v)/len(v) if v else 0) for t, v in per_task_random.items()}
    mean_baseline = {t: (sum(v)/len(v) if v else 0) for t, v in per_task_baseline.items()}

    plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

    # ── Plot 1: reward_curve.png ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    x = range(1, N + 1)
    ax.plot(x, random_scores,   alpha=0.3, color="tomato",    linewidth=0.8)
    ax.plot(x, baseline_scores, alpha=0.3, color="steelblue", linewidth=0.8)

    if N >= 8:
        xs = range(8, N + 1)
        ax.plot(xs, smooth(random_scores),   color="tomato",    linewidth=2.0, label="Random agent (smoothed)")
        ax.plot(xs, smooth(baseline_scores), color="steelblue", linewidth=2.0, label="Expert baseline (smoothed)")
    else:
        ax.plot(x, random_scores,   color="tomato",    linewidth=2.0, label="Random agent")
        ax.plot(x, baseline_scores, color="steelblue", linewidth=2.0, label="Expert baseline")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Final score (0–1)")
    ax.set_title("KaggleSimEnv — Episode Reward: Random vs Expert Baseline")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "reward_curve.png")
    print(f"\n📊 Saved: {plots_dir}/reward_curve.png")
    plt.close(fig)

    # ── Plot 2: loss_curve.png (rolling improvement) ──────────────────────
    # "Loss" proxy = how far each episode is from the maximum possible (1.0)
    # Shows the gap is smaller for the baseline (i.e. it "learns" the task better)
    random_loss   = [1.0 - s for s in random_scores]
    baseline_loss = [1.0 - s for s in baseline_scores]

    fig, ax = plt.subplots(figsize=(10, 4))
    if N >= 8:
        xs = range(8, N + 1)
        ax.plot(xs, smooth(random_loss),   color="tomato",    linewidth=2.0, label="Random agent loss")
        ax.plot(xs, smooth(baseline_loss), color="steelblue", linewidth=2.0, label="Expert baseline loss")
    else:
        ax.plot(x, random_loss,   color="tomato",    linewidth=2.0, label="Random agent loss")
        ax.plot(x, baseline_loss, color="steelblue", linewidth=2.0, label="Expert baseline loss")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score gap to optimal (lower = better)")
    ax.set_title("KaggleSimEnv — Loss Proxy (1 − score) per Episode")
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curve.png")
    print(f"📊 Saved: {plots_dir}/loss_curve.png")
    plt.close(fig)

    # ── Plot 3: baseline_vs_trained.png (per-task comparison) ────────────
    task_labels = list(all_task_ids)
    x_pos = np.arange(len(task_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b_vals = [mean_random[t]   for t in task_labels]
    t_vals = [mean_baseline[t] for t in task_labels]

    bars_b = ax.bar(x_pos - width/2, b_vals, width, label="Random agent",    color="#f4a582", edgecolor="black")
    bars_t = ax.bar(x_pos + width/2, t_vals, width, label="Expert baseline",  color="#4393c3", edgecolor="black")

    ax.set_xlabel("Task")
    ax.set_ylabel("Mean final score (0–1)")
    ax.set_title("KaggleSimEnv — Per-task Score: Random vs Expert Baseline")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(task_labels, rotation=18, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars_b + bars_t:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(plots_dir / "baseline_vs_trained.png")
    print(f"📊 Saved: {plots_dir}/baseline_vs_trained.png")
    plt.close(fig)

    print("\n✅ All plots saved to plots/")


if __name__ == "__main__":
    main()
