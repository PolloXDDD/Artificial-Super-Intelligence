"""Frozen ground-truth scoring and paired promotion gates."""
from dataclasses import dataclass, asdict
import math
import torch
from .data import TASKS


@dataclass
class Evaluation:
    mse: float
    raw_mse: float
    persistence_mse: float
    per_task: dict
    episode_errors: list
    episodes: list
    n_transitions: int

    def as_dict(self):
        return asdict(self)


@torch.no_grad()
def evaluate(agent, data, batch_size=256):
    """No gradients, fitting, optimizer steps, EMA updates or buffer writes."""
    modes = [(module, module.training) for module in agent.modules()]
    agent.eval()
    errors, raw_errors, persistence = [], [], []
    try:
        for start in range(0, len(data), batch_size):
            obs, act, nxt = (getattr(data, key)[start:start+batch_size].to(agent.device)
                             for key in ("obs", "actions", "next_obs"))
            pred = agent.predict_next(obs, act)
            if not bool(torch.isfinite(pred).all()):
                raise FloatingPointError("Non-finite evaluation predictions")
            errors.append(((pred - nxt) / agent.dynamics.delta_scale).square().mean(-1).cpu())
            raw_errors.append((pred - nxt).square().mean(-1).cpu())
            persistence.append(((obs - nxt) / agent.dynamics.delta_scale).square().mean(-1).cpu())
    finally:
        for module, mode in modes:
            module.training = mode
    error = torch.cat(errors)
    ids = data.episodes.unique(sorted=True)
    return Evaluation(float(error.mean()), float(torch.cat(raw_errors).mean()),
                      float(torch.cat(persistence).mean()),
                      {task: float(error[data.tasks == i].mean()) for i, task in enumerate(TASKS)},
                      [float(error[data.episodes == i].mean()) for i in ids],
                      ids.tolist(), len(data))


def promotion_gate(parent, candidate, min_improvement=0.005, max_task_regression=0.02):
    """A paired normal-approximation bound is a heuristic, not a proof of gain.

    No selection rule can guarantee improvement on every future distribution.
    This gate is applied to validation only; the final test is report-only.
    """
    if parent.episodes != candidate.episodes or len(parent.episodes) < 2:
        raise ValueError("Promotion requires the same independent evaluation episodes")
    values = [candidate.mse, candidate.raw_mse, *candidate.per_task.values(), *candidate.episode_errors]
    if not all(math.isfinite(v) for v in values):
        return False, {"reason": "non_finite"}
    if set(parent.per_task) != set(candidate.per_task):
        raise ValueError("Task sets differ")
    delta = torch.tensor(parent.episode_errors, dtype=torch.float64) - torch.tensor(candidate.episode_errors, dtype=torch.float64)
    lower = float(delta.mean() - 1.96 * delta.std(correction=1) / math.sqrt(len(delta)))
    relative = (parent.mse - candidate.mse) / max(parent.mse, 1e-12)
    regressions = [name for name, value in candidate.per_task.items()
                   if value > parent.per_task[name] * (1 + max_task_regression) + 1e-8]
    passed = relative >= min_improvement and lower > 0 and not regressions
    return passed, {"reason": "promote" if passed else "insufficient_gain_or_regression",
                    "relative_gain": relative, "paired_lower_95": lower,
                    "regressed_tasks": regressions}
