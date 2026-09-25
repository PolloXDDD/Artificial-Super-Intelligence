"""
Ontogeny: A Curriculum Without a Teacher.

Historical developmental heuristic, separate from the measured RSI trainer.
Its proposed curriculum is motivated by EFE: epistemic value peaks at the frontier of competence
(learning-progress dynamics, Oudeyer & Kaplan), so the agent
self-schedules its own development:

  Phase 0: Body babbling
  Phase 1: Object permanence and contact physics
  Phase 2: Tool use and first analogies
  Phase 3: Other minds
  Phase 4: Symbols and culture

Invariant I4: Competence emerges through a developmental trajectory
in which earlier structures scaffold later ones. Intelligence is grown,
not assembled.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .agent import NOEMAAgent


class OntogenyScheduler:
    """
    Manages the developmental trajectory of a NOEMA agent.

    Phases are self-scheduled: the agent transitions when its
    competence (measured by prediction error reduction rate)
    plateaus, indicating the current frontier has been exhausted.
    """

    def __init__(
        self,
        agent: NOEMAAgent,
        phase_threshold: float = 0.1,  # Plateau detection threshold
        window_size: int = 100,
    ):
        self.agent = agent
        self.phase_threshold = phase_threshold
        self.window_size = window_size

        # Learning progress tracking
        self.free_energy_history: list[float] = []
        self.learning_progress: list[float] = []
        self.phase_transitions: list[dict] = []

    def compute_learning_progress(self) -> float:
        """
        Learning progress = rate of free energy reduction.
        Peaks at the frontier of competence (Oudeyer & Kaplan).
        """
        if len(self.free_energy_history) < self.window_size:
            return 0.0

        recent = self.free_energy_history[-self.window_size:]
        first_half = np.mean(recent[: len(recent) // 2])
        second_half = np.mean(recent[len(recent) // 2 :])

        progress = first_half - second_half
        self.learning_progress.append(progress)
        return progress

    def should_transition(self) -> bool:
        """
        Check if the agent has plateaued and should advance phase.
        Transition when learning progress drops below threshold.
        """
        if len(self.learning_progress) < self.window_size:
            return False

        recent_progress = self.learning_progress[-self.window_size:]
        return np.mean(recent_progress) < self.phase_threshold

    def advance_phase(self):
        """Advance to next developmental phase."""
        old_phase = self.agent.phase
        self.agent.phase = min(old_phase + 1, 4)

        self.phase_transitions.append({
            "from_phase": old_phase,
            "to_phase": self.agent.phase,
            "step": self.agent.step_count,
            "free_energy": self.free_energy_history[-1] if self.free_energy_history else 0,
        })

    def step(self, free_energy: float):
        """Record one step and check for phase transition."""
        self.free_energy_history.append(free_energy)
        self.compute_learning_progress()

        if self.should_transition() and self.agent.phase < 4:
            self.advance_phase()

    def get_phase_name(self, phase: Optional[int] = None) -> str:
        """Get human-readable phase name."""
        phase = phase if phase is not None else self.agent.phase
        names = {
            0: "Body Babbling",
            1: "Object Permanence & Contact Physics",
            2: "Tool Use & First Analogies",
            3: "Other Minds",
            4: "Symbols & Culture",
        }
        return names.get(phase, "Unknown")

    def get_phase_description(self, phase: Optional[int] = None) -> str:
        """Get detailed phase description."""
        phase = phase if phase is not None else self.agent.phase
        descriptions = {
            0: (
                "Phase 0 — Body Babbling: Random motor exploration collapses "
                "into forward and inverse models of the agent's own embodiment. "
                "The agent learns what its body can do."
            ),
            1: (
                "Phase 1 — Object Permanence & Contact Physics: S2 factorizes "
                "the world into persistent entities; the first relations "
                "(ABOVE, PUSHES, CONTAINS) crystallize in S4."
            ),
            2: (
                "Phase 2 — Tool Use & First Analogies: S5 begins projecting "
                "schemas: a stick EXTENDS the arm as the arm EXTENDS the will."
            ),
            3: (
                "Phase 3 — Other Minds: The cheapest generative model of another "
                "agent is an analogical projection of the self-model. Empathy is "
                "predicted by the architecture, not added to it."
            ),
            4: (
                "Phase 4 — Symbols & Culture: Communicative signs are learned as "
                "actions that minimize joint free energy between agents. Language "
                "grounds out in S4 schemas; I2 is preserved all the way up."
            ),
        }
        return descriptions.get(phase, "Unknown phase")
