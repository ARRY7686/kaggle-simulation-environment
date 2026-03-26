#!/usr/bin/env python3
"""Baseline agent for KaggleSimEnv v3.

Structured multi-phase approach:
  Phase 1: Inspect hints (1-2 hints)
  Phase 2: Diagnose dataset (shift detection, cleaning)
  Phase 3: Set CV + feature engineering
  Phase 4: Train model + handle imbalance
  Phase 5: Ensemble + regularize + postprocess
  Phase 6: Submit

Usage:
    python -m baseline.run_baseline --mode local
    python -m baseline.run_baseline --mode api --base-url http://localhost:7860
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests
from openai import OpenAI

from kaggle_sim_env.environment import KaggleSimEnv
from kaggle_sim_env.grader import Grader
from kaggle_sim_env.models import Action
from kaggle_sim_env.tasks import TASK_REGISTRY, get_task

SYSTEM_PROMPT = """\
You are an expert Kaggle Grand Master agent. You work in structured phases.

## PHASES (follow in order)
1. INSPECT: Use inspect_top_solution to get 1-2 hints before acting.
2. DIAGNOSE: Check dataset properties. If has_time_column → detect shift. If columns look leaky → clean data.
3. CV: Choose CV based on dataset properties (group/time/standard).
4. FEATURES: Engineer domain-appropriate features only. Don't use image/spatial features on tabular data.
5. TRAIN: Pick the right model family for the domain.
6. TUNE: Handle imbalance, tune loss if needed.
7. STABILIZE: Regularize + ensemble (keep it to 1-2 techniques).
8. POSTPROCESS: Only if domain-relevant (TTA for images, physics for trajectories).
9. SUBMIT: When confident or running low on steps.

## HIERARCHICAL ACTIONS (use category field!)

Example:
{"action_type": "feature_engineering", "parameters": {"category": "distribution", "technique": "log_transform"}}

Categories per action type:
  set_cv:               standard(kfold,repeated_kfold) | group(group_kfold,stratified_group_kfold) | temporal(time_split,combined_group_time)
  feature_engineering:  distribution(log_transform,normalize,quantile_features) | interaction(interaction_terms,domain_ratios) | encoding(sin_cos_encoding,target_encoding,spatial_encoding,tfidf_features) | spatial(relative_coordinates,distance_features) | signal(frequency_features,multi_layer_features,fourier_resampling)
  detect_shift:         detection(adversarial_validation,feature_importance_shift) | mitigation(remove_identifiers,domain_invariant_features)
  train_model:          tree(xgboost,lightgbm,catboost,random_forest) | linear(linear) | neural(neural_network,pretrained_backbone,temporal_cnn,transformer_encoder)
  handle_imbalance:     weighting(scale_pos_weight,class_weighted_loss) | calibration(calibrate_probabilities,optimize_threshold) | hierarchy(hierarchical_labels,lower_thresholds_recall)
  clean_data:           removal(remove_corrupted,remove_outliers,remove_leaky_features) | reconstruction(analytical_reconstruction,nan_native_model,domain_augmentation,clean_subset_training)
  augmentation:         geometric(geometric,rotation_invariant,image_rectification) | color(color_transform,clahe) | noise(gaussian_noise,robustness_augmentation) | domain(camera_simulation,temporal_augmentation,symmetry_augmentation,multi_view_processing)
  ensemble:             averaging(weighted_average,multi_seed_averaging,swa) | stacking(stacking) | diversity(diverse_features,heterogeneous)
  postprocess:          calibration(bias_correction,prediction_shrinkage,per_group_calibration) | domain(domain_rules,physics_constraints) | inference(tta)
  tune_loss:            asymmetric(asymmetric_loss,epsilon_insensitive) | uncertainty(gaussian_nll) | multi_objective(multi_task,interval_regression,quantile_regression) | weighting(sample_weighted,auxiliary_physics_loss)
  regularize:           weight(strong_regularization,ema,dropout) | transfer(freeze_backbone)

Also available:
  {"action_type": "pseudo_label", "parameters": {"iterations": 1}}
  {"action_type": "inspect_top_solution", "parameters": {}}
  {"action_type": "submit", "parameters": {}}

## FAILURE MODE AWARENESS
- Using kfold on temporal/grouped data → CV inflated, test destroyed
- Image augmentation on tabular data → wasted step + penalty
- Target encoding without group CV → leakage trap
- Training without addressing shift → model memorises wrong distribution
- Using tree models on image/trajectory data → poor fit

## CRITICAL RULES
- Start with 1-2 hints to understand the dataset
- NEVER use more than 8-10 substantive actions before submitting
- Only use strategies relevant to THIS dataset
- Respond with ONLY valid JSON, nothing else
"""


def build_user_message(observation: dict[str, Any]) -> str:
    meta = observation["dataset_metadata"]
    return (
        f"Step {observation['step_count']}/{observation['max_steps']}  "
        f"CV: {observation['current_cv_score']:.4f}  "
        f"Rank: {observation['leaderboard_rank']}\n"
        f"Applied: {observation['applied_strategies']}\n"
        f"Message: {observation.get('message', '')}\n\n"
        f"Dataset: rows={meta['num_rows']} features={meta['num_features']} "
        f"target={meta['target_column']} type={meta['task_type']}\n"
        f"  has_time={meta.get('has_time_column',False)} "
        f"has_group={meta.get('has_group_column',False)} "
        f"has_image={meta.get('has_image_data',False)} "
        f"has_spatial={meta.get('has_spatial_data',False)}\n"
        f"  target_dist={meta.get('target_distribution','normal')} "
        f"class_balance={meta.get('class_balance',{})}\n"
        f"  columns: {', '.join(c['name']+'('+c['dtype']+',miss='+str(c['missing_pct'])+'%)' for c in meta['columns'][:8])}"
        f"{'...' if len(meta['columns'])>8 else ''}\n\n"
        "Choose your next action (JSON only):"
    )


def parse_llm_action(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return json.loads(text)


def run_local(client: OpenAI, task_id: str) -> dict[str, Any]:
    env = KaggleSimEnv()
    grader_inst = Grader()
    task = get_task(task_id)

    obs = env.reset(task_id=task_id)
    obs_dict = obs.model_dump()
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    actions_log: list[dict[str, Any]] = []

    while not obs_dict.get("done", False):
        messages.append({"role": "user", "content": build_user_message(obs_dict)})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or "{}"
        messages.append({"role": "assistant", "content": raw})

        try:
            action_dict = parse_llm_action(raw)
            action = Action(**action_dict)
        except Exception as exc:
            print(f"  [!] Parse error: {exc}. Submitting.")
            action = Action(action_type="submit", parameters={})

        step_result = env.step(action)
        obs_dict = step_result.observation.model_dump()
        actions_log.append({
            "action": action.model_dump(),
            "cv_score": step_result.observation.current_cv_score,
            "reward": step_result.reward.total,
        })

        trap_info = ""
        if step_result.info.get("traps"):
            trap_info = f"  TRAP!"
        combo_info = ""
        if step_result.info.get("combos_completed"):
            combo_info = f"  COMBO: {step_result.info['combos_completed']}"

        print(
            f"  Step {obs_dict['step_count']:2d}: {action.full_tag():50s} "
            f"CV={obs_dict['current_cv_score']:.4f} R={step_result.reward.total:+.4f}"
            f"{trap_info}{combo_info}"
        )

    state = env.state()
    grade = grader_inst.grade(state, task)
    return {"task_id": task_id, "actions": actions_log, "grade": grade.model_dump()}


def run_api(client: OpenAI, task_id: str, base_url: str) -> dict[str, Any]:
    resp = requests.post(f"{base_url}/reset", json={"task_id": task_id}, timeout=30)
    resp.raise_for_status()
    obs_dict = resp.json()
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    actions_log: list[dict[str, Any]] = []

    while not obs_dict.get("done", False):
        messages.append({"role": "user", "content": build_user_message(obs_dict)})
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages,
            temperature=0.0, max_tokens=256,
        )
        raw = response.choices[0].message.content or "{}"
        messages.append({"role": "assistant", "content": raw})

        try:
            action_dict = parse_llm_action(raw)
        except Exception:
            action_dict = {"action_type": "submit", "parameters": {}}

        resp = requests.post(f"{base_url}/step", json=action_dict, timeout=30)
        resp.raise_for_status()
        step_data = resp.json()
        obs_dict = step_data["observation"]
        actions_log.append({
            "action": action_dict,
            "cv_score": obs_dict["current_cv_score"],
            "reward": step_data["reward"]["total"],
        })

    resp = requests.post(f"{base_url}/grader", timeout=30)
    resp.raise_for_status()
    return {"task_id": task_id, "actions": actions_log, "grade": resp.json()}


def main() -> None:
    parser = argparse.ArgumentParser(description="KaggleSimEnv baseline agent")
    parser.add_argument("--mode", choices=["local", "api"], default="local")
    parser.add_argument("--base-url", default="http://localhost:7860")
    parser.add_argument("--tasks", nargs="*", default=list(TASK_REGISTRY.keys()))
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    results: list[dict[str, Any]] = []

    for task_id in args.tasks:
        print(f"\n{'='*70}")
        print(f"  Task: {task_id}")
        print(f"{'='*70}")
        result = run_local(client, task_id) if args.mode == "local" else run_api(client, task_id, args.base_url)
        results.append(result)
        g = result["grade"]
        print(f"\n  Grade: perf={g['performance_score']:.4f} strat={g['strategy_score']:.4f} "
              f"combo={g['combo_score']:.4f} trap={g['trap_score']:.4f} final={g['final_score']:.4f}")

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for r in results:
        g = r["grade"]
        print(f"  {r['task_id']:25s} final={g['final_score']:.4f}")
    avg = sum(r["grade"]["final_score"] for r in results) / max(len(results), 1)
    print(f"\n  Average: {avg:.4f}")


if __name__ == "__main__":
    main()
