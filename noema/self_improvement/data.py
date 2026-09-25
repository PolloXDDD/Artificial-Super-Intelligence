"""Seeded simulator experience, immutable split identities and bounded replay."""
from dataclasses import dataclass
import numpy as np
import torch
from ..environments.playground import PhysicsPlayground

TASKS = ("free_fall", "impacts", "control")
SPLITS = {"train": 0, "validation": 1, "test": 2}


@dataclass
class Transitions:
    obs: torch.Tensor
    actions: torch.Tensor
    next_obs: torch.Tensor
    tasks: torch.Tensor
    episodes: torch.Tensor
    split: str

    def __post_init__(self):
        if self.split not in SPLITS:
            raise ValueError("Unknown data split")
        n = len(self.obs)
        if n == 0 or self.obs.ndim != 2 or self.obs.shape != self.next_obs.shape:
            raise ValueError("Invalid observation batch")
        if self.actions.ndim != 2 or len(self.actions) != n:
            raise ValueError("Invalid action batch")
        if self.tasks.shape != (n,) or self.episodes.shape != (n,):
            raise ValueError("Invalid metadata")
        if not all(bool(torch.isfinite(t).all()) for t in (self.obs, self.actions, self.next_obs)):
            raise ValueError("Non-finite experience")
        if not bool(((self.tasks >= 0) & (self.tasks < len(TASKS))).all()):
            raise ValueError("Unknown task")
        if not bool((self.episodes // (1 << 50) == SPLITS[self.split]).all()):
            raise ValueError("Episode provenance disagrees with split")

    def __len__(self):
        return len(self.obs)

    def state_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def collect(*, seed, split, episodes_per_task=8, horizon=20, round_index=0, agent=None):
    """Collect real transitions. Validation/test always use the fixed random policy.

    Each episode is an independent reset, and paired evaluation aggregates by
    episode. Different splits use separate SeedSequence namespaces.
    """
    if split not in SPLITS or episodes_per_task < 1 or horizon < 1:
        raise ValueError("Invalid collection arguments")
    if not 0 <= round_index < (1 << 20) or episodes_per_task >= (1 << 20):
        raise ValueError("Episode ID capacity exceeded")
    if split != "train" and agent is not None:
        raise ValueError("Evaluation collection must not depend on the agent")
    rows = []
    for task in range(len(TASKS)):
        for episode in range(episodes_per_task):
            seq = np.random.SeedSequence([seed, SPLITS[split], round_index, task, episode])
            rng = np.random.default_rng(seq)
            env = PhysicsPlayground(seed=int(rng.integers(0, 2**63)))
            if task == 0:
                env.objects[1:, 1] = torch.tensor(rng.uniform(1.5, 3, 3), dtype=torch.float32)
                env.objects[1:, 3] = 0
            elif task == 1:
                env.objects[1:, 1] = env.objects[1:, 5] - 0.5 + 0.1
                env.objects[1:, 3] = torch.tensor(rng.uniform(-3, -1, 3), dtype=torch.float32)
            else:
                env.objects[1:, 2:4] = torch.tensor(rng.uniform(-2, 2, (3, 2)), dtype=torch.float32)
            state = env._make_state()
            episode_id = (SPLITS[split] << 50) | (round_index << 30) | (task << 20) | episode
            for _ in range(horizon):
                obs = state.observation.clone()
                action = torch.tensor(rng.uniform(-1, 1, 4), dtype=torch.float32)
                if agent is not None and int(agent.updates) > 0 and rng.random() < 0.2:
                    action = agent.plan_action(obs)[0][0].cpu()
                state = env.step(action)
                rows.append((obs, action.clone(), state.observation.clone(), task, episode_id))
    columns = list(zip(*rows))
    return Transitions(*(torch.stack(columns[i]) for i in range(3)),
                       torch.tensor(columns[3]), torch.tensor(columns[4]), split)


class ReplayBuffer:
    def __init__(self, capacity=12000):
        if capacity < len(TASKS):
            raise ValueError("Replay capacity must cover every task")
        self.capacity = capacity
        self.data = None

    def add(self, data):
        if data.split != "train":
            raise ValueError("Only training data can enter replay")
        if self.data is not None:
            args = [torch.cat([getattr(self.data, key), getattr(data, key)])
                    for key in ("obs", "actions", "next_obs", "tasks", "episodes")]
            data = Transitions(*args, "train")
        # Keep a bounded, balanced history so one task cannot evict all others.
        indices = [torch.where(data.tasks == task)[0][-(self.capacity // len(TASKS)):]
                   for task in range(len(TASKS))]
        keep = torch.cat(indices)
        self.data = Transitions(*(getattr(data, key)[keep].clone() for key in
                                 ("obs", "actions", "next_obs", "tasks", "episodes")), "train")

    def sample(self, batch_size, probabilities, generator):
        if self.data is None:
            raise ValueError("Empty replay")
        task_choices = torch.multinomial(torch.tensor(probabilities), batch_size,
                                        replacement=True, generator=generator)
        indices = []
        for task in task_choices.tolist():
            pool = torch.where(self.data.tasks == task)[0]
            if len(pool) == 0:
                raise ValueError("Replay is missing a task")
            indices.append(int(pool[torch.randint(len(pool), (1,), generator=generator)]))
        return [getattr(self.data, key)[indices] for key in ("obs", "actions", "next_obs")]
