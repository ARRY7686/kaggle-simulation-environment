"""Hints system for KaggleSimEnv.

When the agent takes the `inspect_top_solution` action it receives a hint
from the task's hint list. Hints are dispensed in order; once all hints are
exhausted the agent receives a generic message.
"""

from __future__ import annotations


class HintProvider:
    """Dispenses task-specific hints one at a time."""

    def __init__(self, hints: list[str]) -> None:
        self._hints = list(hints)
        self._index = 0

    @property
    def hints_given(self) -> int:
        return self._index

    def next_hint(self) -> str:
        if self._index < len(self._hints):
            hint = self._hints[self._index]
            self._index += 1
            return hint
        return "No more hints available. Try combining strategies you've learned so far."

    def reset(self) -> None:
        self._index = 0
