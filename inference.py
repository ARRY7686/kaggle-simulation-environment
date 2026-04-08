"""
Inference script — KaggleSimEnv (submission entrypoint)
======================================================

MANDATORY (environment configuration)
-------------------------------------
- API_BASE_URL   : LLM API base URL (e.g. https://router.huggingface.co/v1)
- MODEL_NAME     : Model id for chat completions
- HF_TOKEN       : Hugging Face / API key (passed to OpenAI client as api_key)

Optional
--------
- ENV_URL        : Base URL of the running KaggleSimEnv Space or server (default: http://127.0.0.1:7860)

Requirements
------------
- Named ``inference.py`` in the project root
- All LLM calls go through ``openai.OpenAI`` using the variables above
- Completes all tasks from ``GET /tasks``, runs each episode, then ``POST /grader``; prints scores in [0.0, 1.0]
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests
from openai import OpenAI

from baseline.run_baseline import SYSTEM_PROMPT, build_user_message, parse_llm_action
from kaggle_sim_env.models import Action

# Tunables (keep total runtime modest on 2 vCPU / 8GB)
REQUEST_TIMEOUT_S = 60
MAX_LLM_STEPS_PER_TASK = 14
TEMPERATURE = 0.0
MAX_TOKENS = 256


def _require_env(name: str) -> str:
    v = os.getenv(name, "").strip()
    if not v:
        print(f"Warning: environment variable {name} is not set, using default.", file=sys.stderr)
        defaults = {
            "API_BASE_URL": "https://router.huggingface.co/v1",
            "MODEL_NAME": "Qwen/Qwen2.5-72B-Instruct",
        }
        v = defaults.get(name, "")
        if not v:
            print(f"Error: required environment variable {name} is not set.", file=sys.stderr)
            sys.exit(1)
    return v


def _client() -> OpenAI:
    base_url = _require_env("API_BASE_URL")
    _require_env("MODEL_NAME")
    # Spec: HF_TOKEN; allow OPENAI_API_KEY for local OpenAI-only runs
    api_key = os.getenv("HF_TOKEN", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "Error: set HF_TOKEN (or OPENAI_API_KEY for local OpenAI).",
            file=sys.stderr,
        )
        sys.exit(1)
    return OpenAI(base_url=base_url, api_key=api_key)


def _env_base() -> str:
    return os.getenv("ENV_URL", "http://127.0.0.1:7860").rstrip("/")


def list_task_ids(base: str) -> list[str]:
    r = requests.get(f"{base}/tasks", timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    return [t["task_id"] for t in data]


def coerce_action_for_step(raw: Any) -> dict[str, Any]:
    """Normalize LLM output so POST /step matches api.server StepRequest + Action."""
    fallback = {"action_type": "submit", "parameters": {}}
    if not isinstance(raw, dict):
        return fallback
    d = raw
    if "action" in d and isinstance(d["action"], dict):
        d = d["action"]
    at = d.get("action_type")
    params = d.get("parameters")
    if params is None:
        params = {}
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except json.JSONDecodeError:
            params = {}
    if not isinstance(params, dict):
        params = {}
    if not at or not isinstance(at, str):
        return fallback
    at = at.strip()
    try:
        Action(action_type=at, parameters=params)
    except Exception:
        return fallback
    return {"action_type": at, "parameters": params}


def run_episode(client: OpenAI, model: str, base: str, task_id: str) -> dict[str, Any]:
    print(f"[START] task={task_id}", flush=True)
    r = requests.post(f"{base}/reset", json={"task_id": task_id}, timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    obs_dict: dict[str, Any] = r.json()

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    steps = 0

    while not obs_dict.get("done", False) and steps < MAX_LLM_STEPS_PER_TASK:
        messages.append({"role": "user", "content": build_user_message(obs_dict)})
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = response.choices[0].message.content or "{}"
        messages.append({"role": "assistant", "content": raw})

        try:
            parsed = parse_llm_action(raw)
        except Exception:
            parsed = {"action_type": "submit", "parameters": {}}

        action_dict = coerce_action_for_step(parsed)

        r = requests.post(f"{base}/step", json=action_dict, timeout=REQUEST_TIMEOUT_S)
        if not r.ok:
            detail = (r.text or "")[:1000]
            raise RuntimeError(f"POST /step HTTP {r.status_code}: {detail}")
        step_data = r.json()
        obs_dict = step_data["observation"]
        reward = step_data.get("reward", {}).get("total", 0.0) if isinstance(step_data.get("reward"), dict) else step_data.get("reward", 0.0)
        steps += 1
        print(f"[STEP] step={steps} reward={reward}", flush=True)

    r = requests.post(f"{base}/grader", timeout=REQUEST_TIMEOUT_S)
    r.raise_for_status()
    grade = r.json()
    final_score = grade.get("final_score", 0.0)
    print(f"[END] task={task_id} score={final_score} steps={steps}", flush=True)
    return grade


def _assert_score_range(grade: dict[str, Any], task_id: str) -> None:
    for key in (
        "final_score",
        "performance_score",
        "strategy_score",
        "combo_score",
        "trap_score",
    ):
        v = float(grade[key])
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{task_id}: {key}={v} not in [0, 1]")


def main() -> None:
    client = _client()
    model = _require_env("MODEL_NAME")
    base = _env_base()

    task_ids = list_task_ids(base)
    if len(task_ids) < 3:
        raise RuntimeError(f"Expected at least 3 tasks from /tasks, got {len(task_ids)}")

    print(f"ENV_URL={base}")
    print(f"Tasks: {task_ids}\n")

    all_grades: list[dict[str, Any]] = []
    for tid in task_ids:
        print("=" * 60)
        print(f"Task: {tid}")
        print("=" * 60)
        grade = run_episode(client, model, base, tid)
        _assert_score_range(grade, tid)
        all_grades.append(grade)
        print(
            f"  final={grade['final_score']:.4f}  perf={grade['performance_score']:.4f}  "
            f"strat={grade['strategy_score']:.4f}  combo={grade['combo_score']:.4f}  "
            f"trap={grade['trap_score']:.4f}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY (all scores in [0.0, 1.0])")
    print("=" * 60)
    for g in all_grades:
        print(f"  {g['task_id']:<22} final={float(g['final_score']):.4f}")
    avg = sum(float(g["final_score"]) for g in all_grades) / len(all_grades)
    print(f"\n  Mean final score: {avg:.4f}")


if __name__ == "__main__":
    main()
