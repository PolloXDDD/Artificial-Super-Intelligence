"""Bounded mutations of the training recipe; the proposer learns from trials."""
from dataclasses import dataclass, asdict
import math
import numpy as np
from .data import TASKS


@dataclass
class Recipe:
    learning_rate: float = 0.001
    focus: float = 0.0
    jepa_weight: float = 0.02

    def as_dict(self):
        return asdict(self)


class MetaController:
    OPERATORS = ("continue", "slower", "faster", "focus_weakness", "balance", "representation")

    def __init__(self):
        self.counts = {key: 0 for key in self.OPERATORS}
        self.values = {key: 0.0 for key in self.OPERATORS}

    def propose(self, incumbent, n, generation):
        result = [("continue", Recipe(**incumbent.as_dict()))]
        remaining = [op for op in self.OPERATORS if op != "continue"]
        total = sum(self.counts.values()) + 1
        def ucb(op):
            return self.values[op] + math.sqrt(2 * math.log(total + 1) / (self.counts[op] + 1))
        # Round-robin tie-breaking gives all mutation families a trial.
        remaining = remaining[generation % len(remaining):] + remaining[:generation % len(remaining)]
        remaining.sort(key=ucb, reverse=True)
        for op in remaining[:max(0, n - 1)]:
            recipe = Recipe(**incumbent.as_dict())
            if op == "slower": recipe.learning_rate *= 0.5
            if op == "faster": recipe.learning_rate *= 1.6
            if op == "focus_weakness": recipe.focus += 0.3
            if op == "balance": recipe.focus = 0.0
            if op == "representation": recipe.jepa_weight *= 1.5
            recipe.learning_rate = min(0.005, max(0.00005, recipe.learning_rate))
            recipe.focus = min(0.75, max(0.0, recipe.focus))
            recipe.jepa_weight = min(0.1, max(0.005, recipe.jepa_weight))
            result.append((op, recipe))
        return result

    def update(self, operator, relative_gain):
        gain = float(np.clip(relative_gain, -1, 1))
        if not math.isfinite(gain): gain = -1.0
        self.counts[operator] += 1
        self.values[operator] += (gain - self.values[operator]) / self.counts[operator]

    def state_dict(self):
        return {"counts": self.counts.copy(), "values": self.values.copy()}

    def load_state_dict(self, state):
        self.counts, self.values = state["counts"].copy(), state["values"].copy()


def curriculum(validation, focus):
    errors = np.array([validation.per_task[task] for task in TASKS], dtype=float)
    errors = np.maximum(errors, 1e-9)
    probabilities = (1 - focus) / len(TASKS) + focus * errors / errors.sum()
    return probabilities.tolist()
