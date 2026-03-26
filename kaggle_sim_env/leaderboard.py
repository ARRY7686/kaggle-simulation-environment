"""Simulated leaderboard with ghost competitors.

Ghost competitors have fixed scores. The agent's rank is computed by
inserting its current test score into the sorted ghost list.
"""

from __future__ import annotations

import hashlib
from typing import Any


_GHOST_NAMES = [
    "GrandMaster_42",
    "DataWizard",
    "FeatureKing",
    "BoostQueen",
    "StackMaster",
    "NeuralNinja",
    "EnsembleGuru",
    "CVExpert",
    "KagglePro",
]


def _deterministic_name(index: int, task_id: str) -> str:
    """Generate a stable ghost name seeded by task + index."""
    digest = hashlib.md5(f"{task_id}:{index}".encode()).hexdigest()[:4]
    base = _GHOST_NAMES[index % len(_GHOST_NAMES)]
    return f"{base}_{digest}"


class Leaderboard:
    """Maintains a leaderboard of ghost competitors plus the agent."""

    def __init__(self, task_id: str, ghost_scores: list[float]) -> None:
        self.task_id = task_id
        self.ghosts: list[dict[str, Any]] = []
        for i, score in enumerate(sorted(ghost_scores, reverse=True)):
            self.ghosts.append({
                "name": _deterministic_name(i, task_id),
                "score": round(score, 4),
                "rank": i + 1,
            })

    def agent_rank(self, agent_score: float) -> int:
        """Return 1-indexed rank of the agent among ghosts."""
        rank = 1
        for ghost in self.ghosts:
            if ghost["score"] > agent_score:
                rank += 1
        return rank

    def full_board(self, agent_score: float) -> list[dict[str, Any]]:
        """Return complete leaderboard with agent inserted."""
        entries = list(self.ghosts) + [
            {"name": "Agent", "score": round(agent_score, 4), "rank": 0}
        ]
        entries.sort(key=lambda e: e["score"], reverse=True)
        for i, entry in enumerate(entries):
            entry["rank"] = i + 1
        return entries
