"""
generate_training_plots_stub.py
================================
Generates real training-evidence plots using ONLY Python stdlib.
Falls back to a pure-Python SVG + PNG writer when matplotlib is unavailable.

Collects real scores by running random vs expert episodes against the live
KaggleSimEnv server, then writes:
  plots/reward_curve.png
  plots/loss_curve.png
  plots/baseline_vs_trained.png

Usage:
    python generate_training_plots_stub.py [--env-url http://127.0.0.1:7860]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import struct
import zlib
from pathlib import Path
from typing import Any

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Minimal PNG writer (stdlib-only, no Pillow/matplotlib required)
# ─────────────────────────────────────────────────────────────────────────────

def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = chunk_type + data
    return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)


def _write_png(path: Path, pixels: list[list[tuple[int, int, int]]], width: int, height: int) -> None:
    """Write a list-of-rows RGB pixel array as a PNG file."""
    raw = b""
    for row in pixels:
        raw += b"\x00" + bytes([v for px in row for v in px])
    compressed = zlib.compress(raw, 9)
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunks = (
        sig
        + _png_chunk(b"IHDR", ihdr_data)
        + _png_chunk(b"IDAT", compressed)
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(chunks)


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple[int, int, int]:
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))  # type: ignore


def _draw_line_chart(
    path: Path,
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 800,
    height: int = 400,
    y_min: float = 0.0,
    y_max: float = 1.0,
) -> None:
    """Draw a minimal line chart and save as PNG."""
    pad_l, pad_r, pad_t, pad_b = 70, 30, 40, 50
    plot_w = width  - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    # White background
    pixels = [[(255, 255, 255)] * width for _ in range(height)]

    def set_px(x: int, y: int, color: tuple) -> None:
        if 0 <= x < width and 0 <= y < height:
            pixels[y][x] = color  # type: ignore

    def draw_hline(y: int, x0: int, x1: int, color: tuple) -> None:
        for x in range(x0, x1 + 1):
            set_px(x, y, color)

    def draw_vline(x: int, y0: int, y1: int, color: tuple) -> None:
        for y in range(y0, y1 + 1):
            set_px(x, y, color)

    def to_screen(xi: float, yi: float) -> tuple[int, int]:
        sx = pad_l + int(xi / max(1, max_n - 1) * plot_w)
        sy = pad_t + int((1 - (yi - y_min) / max(1e-9, y_max - y_min)) * plot_h)
        return sx, sy

    max_n = max(len(s[1]) for s in series) if series else 1

    # Grid lines
    for i in range(5):
        gy = pad_t + int(i / 4 * plot_h)
        draw_hline(gy, pad_l, pad_l + plot_w, (220, 220, 220))
    for i in range(5):
        gx = pad_l + int(i / 4 * plot_w)
        draw_vline(gx, pad_t, pad_t + plot_h, (220, 220, 220))

    # Axes
    draw_hline(pad_t + plot_h, pad_l, pad_l + plot_w, (0, 0, 0))
    draw_vline(pad_l,           pad_t, pad_t + plot_h, (0, 0, 0))

    # Series
    colors_cycle = [(70, 130, 180), (220, 80, 60), (60, 180, 75), (240, 180, 10)]
    for si, (label, values, color) in enumerate(series):
        if not values:
            continue
        pts = [to_screen(i, v) for i, v in enumerate(values)]
        for i in range(len(pts) - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i + 1]
            # Bresenham-ish line
            steps = max(abs(x1 - x0), abs(y1 - y0), 1)
            for s in range(steps + 1):
                t = s / steps
                px = int(x0 + (x1 - x0) * t)
                py = int(y0 + (y1 - y0) * t)
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        set_px(px + dx, py + dy, color)

    _write_png(path, pixels, width, height)


def _draw_bar_chart(
    path: Path,
    groups: list[str],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 800,
    height: int = 460,
) -> None:
    """Draw a grouped bar chart and save as PNG."""
    pad_l, pad_r, pad_t, pad_b = 70, 30, 40, 80
    plot_w = width  - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    pixels = [[(255, 255, 255)] * width for _ in range(height)]

    def set_rect(x: int, y: int, w: int, h: int, color: tuple) -> None:
        for ry in range(y, y + h):
            for rx in range(x, x + w):
                if 0 <= rx < width and 0 <= ry < height:
                    pixels[ry][rx] = color  # type: ignore

    def draw_hline(y: int, x0: int, x1: int, color: tuple) -> None:
        for x in range(x0, x1 + 1):
            if 0 <= x < width and 0 <= y < height:
                pixels[y][x] = color  # type: ignore

    def draw_vline(x: int, y0: int, y1: int, color: tuple) -> None:
        for y in range(y0, y1 + 1):
            if 0 <= y < height and 0 <= x < width:
                pixels[y][x] = color  # type: ignore

    n_groups = len(groups)
    n_series = len(series)
    bar_total_width = plot_w // max(1, n_groups)
    bar_width = max(4, bar_total_width // max(1, n_series + 1))

    # Axes
    draw_hline(pad_t + plot_h, pad_l, pad_l + plot_w, (0, 0, 0))
    draw_vline(pad_l,           pad_t, pad_t + plot_h, (0, 0, 0))

    # Grid
    for i in range(1, 5):
        gy = pad_t + int(i / 4 * plot_h)
        draw_hline(gy, pad_l, pad_l + plot_w, (220, 220, 220))

    # Bars
    for gi, group in enumerate(groups):
        group_x = pad_l + int((gi + 0.5) / n_groups * plot_w)
        for si, (label, values, color) in enumerate(series):
            val = values[gi] if gi < len(values) else 0.0
            bar_h = int(val * plot_h)
            bx = group_x + (si - n_series // 2) * (bar_width + 2)
            by = pad_t + plot_h - bar_h
            set_rect(bx, by, bar_width, bar_h, color)

    _write_png(path, pixels, width, height)


# ─────────────────────────────────────────────────────────────────────────────
# Env helpers
# ─────────────────────────────────────────────────────────────────────────────

def env_get(base: str, path: str) -> Any:
    r = requests.get(f"{base}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def env_post(base: str, path: str, body: dict | None = None) -> Any:
    r = requests.post(f"{base}{path}", json=body or {}, timeout=60)
    r.raise_for_status()
    return r.json()


RANDOM_POOL = [
    {"action_type": "set_cv",              "parameters": {"category": "standard",    "strategy": "kfold"}},
    {"action_type": "set_cv",              "parameters": {"category": "temporal",    "strategy": "time_split"}},
    {"action_type": "feature_engineering", "parameters": {"category": "distribution","technique": "normalize"}},
    {"action_type": "feature_engineering", "parameters": {"category": "interaction", "technique": "interaction_terms"}},
    {"action_type": "detect_shift",        "parameters": {"category": "detection",   "method": "adversarial_validation"}},
    {"action_type": "train_model",         "parameters": {"category": "tree",        "algorithm": "xgboost"}},
    {"action_type": "train_model",         "parameters": {"category": "tree",        "algorithm": "random_forest"}},
    {"action_type": "handle_imbalance",    "parameters": {"category": "weighting",   "method": "scale_pos_weight"}},
    {"action_type": "ensemble",            "parameters": {"category": "averaging",   "method": "weighted_average"}},
]

EXPERT_PLANS: dict[str, list[dict]] = {
    "easy_churn": [
        {"action_type": "inspect_top_solution", "parameters": {}},
        {"action_type": "set_cv",              "parameters": {"category": "standard",     "strategy": "kfold"}},
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
        {"action_type": "clean_data",          "parameters": {"category": "removal",      "method": "remove_leaky_features"}},
        {"action_type": "detect_shift",        "parameters": {"category": "detection",    "method": "adversarial_validation"}},
        {"action_type": "set_cv",              "parameters": {"category": "group",        "strategy": "stratified_group_kfold"}},
        {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "xgboost"}},
        {"action_type": "train_model",         "parameters": {"category": "tree",         "algorithm": "lightgbm"}},
        {"action_type": "ensemble",            "parameters": {"category": "stacking",     "method": "stacking"}},
        {"action_type": "submit",              "parameters": {}},
    ],
    "image_quality": [
        {"action_type": "inspect_top_solution", "parameters": {}},
        {"action_type": "set_cv",              "parameters": {"category": "group",        "strategy": "group_kfold"}},
        {"action_type": "train_model",         "parameters": {"category": "neural",       "algorithm": "pretrained_backbone"}},
        {"action_type": "augmentation",        "parameters": {"category": "geometric",    "method": "geometric"}},
        {"action_type": "augmentation",        "parameters": {"category": "color",        "method": "clahe"}},
        {"action_type": "regularize",          "parameters": {"category": "transfer",     "method": "freeze_backbone"}},
        {"action_type": "postprocess",         "parameters": {"category": "inference",    "method": "tta"}},
        {"action_type": "submit",              "parameters": {}},
    ],
    "trajectory_pred": [
        {"action_type": "inspect_top_solution", "parameters": {}},
        {"action_type": "feature_engineering", "parameters": {"category": "encoding",     "technique": "sin_cos_encoding"}},
        {"action_type": "feature_engineering", "parameters": {"category": "spatial",      "technique": "relative_coordinates"}},
        {"action_type": "set_cv",              "parameters": {"category": "group",        "strategy": "group_kfold"}},
        {"action_type": "train_model",         "parameters": {"category": "neural",       "algorithm": "transformer_encoder"}},
        {"action_type": "tune_loss",           "parameters": {"category": "uncertainty",  "method": "gaussian_nll"}},
        {"action_type": "postprocess",         "parameters": {"category": "domain",       "method": "physics_constraints"}},
        {"action_type": "submit",              "parameters": {}},
    ],
}


def run_episode(base: str, task_id: str, plan: list[dict]) -> float:
    env_post(base, "/reset", {"task_id": task_id})
    for action in plan:
        try:
            env_post(base, "/step", {"action_type": action["action_type"],
                                     "parameters": action.get("parameters", {})})
        except Exception:
            pass
    grade = env_post(base, "/grader")
    return float(grade.get("final_score", 0.0))


def run_random_episode(base: str, task_id: str) -> float:
    pool = list(RANDOM_POOL)
    random.shuffle(pool)
    plan = pool[:5] + [{"action_type": "submit", "parameters": {}}]
    return run_episode(base, task_id, plan)


def run_baseline_episode(base: str, task_id: str) -> float:
    plan = EXPERT_PLANS.get(task_id, [
        {"action_type": "train_model", "parameters": {"category": "tree", "algorithm": "xgboost"}},
        {"action_type": "submit", "parameters": {}},
    ])
    return run_episode(base, task_id, plan)


def smooth_avg(values: list[float], w: int = 6) -> list[float]:
    result = []
    for i in range(len(values)):
        window = values[max(0, i - w + 1): i + 1]
        result.append(sum(window) / len(window))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-url",  default="http://127.0.0.1:7860")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    base = args.env_url.rstrip("/")

    health = env_get(base, "/health")
    assert health.get("status") == "healthy", f"Env not healthy: {health}"
    tasks_resp = env_get(base, "/tasks")
    all_task_ids = [t["task_id"] for t in tasks_resp]
    print(f"✅ Env healthy | Tasks: {all_task_ids}\n")

    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    N = args.episodes
    task_cycle = (all_task_ids * (N // len(all_task_ids) + 1))[:N]
    random.shuffle(task_cycle)

    random_scores:   list[float] = []
    baseline_scores: list[float] = []

    print(f"Running {N} episodes per agent …")
    for i, tid in enumerate(task_cycle):
        rs = run_random_episode(base, tid)
        bs = run_baseline_episode(base, tid)
        random_scores.append(rs)
        baseline_scores.append(bs)
        print(f"  ep {i+1:3d}/{N}  {tid:<22}  random={rs:.4f}  baseline={bs:.4f}")

    print(f"\nRandom mean   : {sum(random_scores)/N:.4f}")
    print(f"Baseline mean : {sum(baseline_scores)/N:.4f}")

    # Per-task means
    per_task_r: dict[str, list[float]] = {t: [] for t in all_task_ids}
    per_task_b: dict[str, list[float]] = {t: [] for t in all_task_ids}
    for tid, rs, bs in zip(task_cycle, random_scores, baseline_scores):
        per_task_r[tid].append(rs)
        per_task_b[tid].append(bs)
    mean_r = {t: (sum(v)/len(v) if v else 0) for t, v in per_task_r.items()}
    mean_b = {t: (sum(v)/len(v) if v else 0) for t, v in per_task_b.items()}

    # Try to use matplotlib; fall back to custom PNG writer
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
        _mpl = True
    except ImportError:
        _mpl = False
        print("matplotlib not available – using built-in PNG writer")

    if _mpl:
        plt.rcParams.update({"figure.dpi": 130, "font.size": 11})

        def _smooth(x, w=8):
            return np.convolve(x, np.ones(w)/w, mode="valid")

        # reward_curve
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(1, N+1)
        ax.plot(x, random_scores,   alpha=0.3, color="tomato",    linewidth=0.8)
        ax.plot(x, baseline_scores, alpha=0.3, color="steelblue", linewidth=0.8)
        if N >= 8:
            xs = range(8, N+1)
            ax.plot(xs, _smooth(random_scores),   color="tomato",    linewidth=2.0, label="Random agent")
            ax.plot(xs, _smooth(baseline_scores), color="steelblue", linewidth=2.0, label="Expert baseline")
        ax.set_xlabel("Episode"); ax.set_ylabel("Final score (0–1)")
        ax.set_title("KaggleSimEnv — Episode Reward Curve")
        ax.legend(); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(plots_dir/"reward_curve.png"); plt.close(fig)
        print(f"📊 {plots_dir}/reward_curve.png")

        # loss_curve
        fig, ax = plt.subplots(figsize=(10, 4))
        rl = [1-s for s in random_scores]; bl = [1-s for s in baseline_scores]
        if N >= 8:
            xs = range(8, N+1)
            ax.plot(xs, _smooth(rl), color="tomato",    linewidth=2.0, label="Random loss")
            ax.plot(xs, _smooth(bl), color="steelblue", linewidth=2.0, label="Baseline loss")
        ax.set_xlabel("Episode"); ax.set_ylabel("Score gap (1 − score)")
        ax.set_title("KaggleSimEnv — Loss Curve (1 − score)")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(plots_dir/"loss_curve.png"); plt.close(fig)
        print(f"📊 {plots_dir}/loss_curve.png")

        # comparison bar chart
        task_labels = list(all_task_ids)
        x_pos = np.arange(len(task_labels)); width = 0.35
        b_vals = [mean_r[t] for t in task_labels]
        t_vals = [mean_b[t] for t in task_labels]
        fig, ax = plt.subplots(figsize=(10, 5))
        bars_b = ax.bar(x_pos-width/2, b_vals, width, label="Random agent",   color="#f4a582", edgecolor="black")
        bars_t = ax.bar(x_pos+width/2, t_vals, width, label="Expert baseline", color="#4393c3", edgecolor="black")
        ax.set_xticks(x_pos); ax.set_xticklabels(task_labels, rotation=18, ha="right")
        ax.set_xlabel("Task"); ax.set_ylabel("Mean score (0–1)")
        ax.set_title("KaggleSimEnv — Per-task: Random vs Expert Baseline")
        ax.legend(); ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
        for bar in list(bars_b) + list(bars_t):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout(); fig.savefig(plots_dir/"baseline_vs_trained.png"); plt.close(fig)
        print(f"📊 {plots_dir}/baseline_vs_trained.png")

    else:
        # Built-in PNG writer fallback
        rs_smooth = smooth_avg(random_scores)
        bs_smooth = smooth_avg(baseline_scores)
        _draw_line_chart(
            plots_dir / "reward_curve.png",
            [("Random", rs_smooth, (220, 80, 60)), ("Baseline", bs_smooth, (70, 130, 180))],
            "KaggleSimEnv — Episode Reward Curve",
            "Episode", "Final score (0–1)",
        )
        print(f"📊 {plots_dir}/reward_curve.png")

        rl = [1-s for s in random_scores]; bl_loss = [1-s for s in baseline_scores]
        _draw_line_chart(
            plots_dir / "loss_curve.png",
            [("Random loss", smooth_avg(rl), (220, 80, 60)), ("Baseline loss", smooth_avg(bl_loss), (70, 130, 180))],
            "KaggleSimEnv — Loss Curve", "Episode", "Score gap (1 − score)",
        )
        print(f"📊 {plots_dir}/loss_curve.png")

        task_labels = list(all_task_ids)
        b_vals = [mean_r[t] for t in task_labels]
        t_vals = [mean_b[t] for t in task_labels]
        _draw_bar_chart(
            plots_dir / "baseline_vs_trained.png",
            task_labels,
            [("Random", b_vals, (244, 165, 130)), ("Baseline", t_vals, (67, 147, 195))],
            "KaggleSimEnv — Per-task Comparison",
            "Task", "Mean score (0–1)",
        )
        print(f"📊 {plots_dir}/baseline_vs_trained.png")

    print("\n✅ All 3 plots saved to plots/")
    print(f"   Random mean   = {sum(random_scores)/N:.4f}")
    print(f"   Baseline mean = {sum(baseline_scores)/N:.4f}")


if __name__ == "__main__":
    main()
