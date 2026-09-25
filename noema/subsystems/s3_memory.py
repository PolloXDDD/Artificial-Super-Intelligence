"""
S3: Complementary Memory — the "hippocampus-cortex" pair.

Episodic store: one-shot, non-parametric storage of surprising trajectories.
Consolidation: offline replay through world model into parametric weights
and relational graph (S4). Replay prioritized by residual prediction error.

Resolves the stability-plasticity dilemma (catastrophic forgetting).
Provides the offline phase in which extrapolation operates without risk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from collections import deque
import random


class EpisodicTrace:
    """A single episodic memory trace."""

    def __init__(
        self,
        z_trajectory: torch.Tensor,
        a_trajectory: torch.Tensor,
        surprise: float,
        timestamp: int,
    ):
        self.z = z_trajectory.detach().cpu()  # Latent trajectory
        self.a = a_trajectory.detach().cpu()  # Action trajectory
        self.surprise = surprise  # Free energy spike
        self.timestamp = timestamp
        self.replay_count = 0
        self.residual_error = surprise  # Updated during consolidation

    def priority(self) -> float:
        """Replay priority: high residual error, low replay count."""
        return self.residual_error / (1 + self.replay_count)


class EpisodicStore(nn.Module):
    """
    One-shot, non-parametric episodic memory.

    Writes are triggered by surprise (spike in free energy).
    Content-indexed for retrieval.
    """

    def __init__(self, capacity: int = 10000, surprise_threshold: float = 2.0):
        super().__init__()
        self.capacity = capacity
        self.surprise_threshold = surprise_threshold
        self.traces: deque[EpisodicTrace] = deque(maxlen=capacity)
        self.global_timestamp = 0

    def try_write(
        self,
        z_trajectory: torch.Tensor,
        a_trajectory: torch.Tensor,
        free_energy: float,
    ) -> bool:
        """
        Write to episodic store if surprise exceeds threshold.

        The agent remembers exactly what it failed to predict.
        """
        self.global_timestamp += 1

        if free_energy > self.surprise_threshold:
            trace = EpisodicTrace(
                z_trajectory=z_trajectory,
                a_trajectory=a_trajectory,
                surprise=free_energy,
                timestamp=self.global_timestamp,
            )
            self.traces.append(trace)
            return True
        return False

    def retrieve_similar(
        self, query_z: torch.Tensor, top_k: int = 5
    ) -> list[EpisodicTrace]:
        """Retrieve traces most similar to query (content-addressable)."""
        if not self.traces:
            return []

        query = query_z.detach().cpu()
        similarities = []
        for trace in self.traces:
            # Cosine similarity with first latent of trajectory
            sim = F.cosine_similarity(
                query.flatten().unsqueeze(0),
                trace.z[0].flatten().unsqueeze(0),
                dim=-1,
            ).item()
            similarities.append((sim, trace))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in similarities[:top_k]]

    def sample_for_replay(self, n: int = 8) -> list[EpisodicTrace]:
        """
        Sample traces for consolidation, prioritized by residual error.
        Mirrors hippocampal replay in rodents.
        """
        if not self.traces:
            return []

        priorities = np.array([t.priority() for t in self.traces])
        priorities = priorities / (priorities.sum() + 1e-8)

        n = min(n, len(self.traces))
        indices = np.random.choice(len(self.traces), size=n, replace=False, p=priorities)
        return [self.traces[i] for i in indices]


class ComplementaryMemory(nn.Module):
    """
    S3: Complementary Memory System.

    Combines:
    1. Episodic store (fast, one-shot, hippocampal-like)
    2. Consolidation mechanism (offline replay → cortical/slow learning)

    Resolves stability-plasticity dilemma through interleaved replay.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        action_dim: int = 4,
        episodic_capacity: int = 5000,
        surprise_threshold: float = 1.5,
        replay_batch_size: int = 8,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.replay_batch_size = replay_batch_size

        # Episodic store
        self.episodic = EpisodicStore(
            capacity=episodic_capacity,
            surprise_threshold=surprise_threshold,
        )

        # Consolidation replay buffer (parametric "cortical" memory)
        self.consolidation_buffer = deque(maxlen=50000)

    def forward(
        self,
        z_trajectory: torch.Tensor,
        a_trajectory: torch.Tensor,
        free_energy: float,
    ) -> dict:
        """
        Process a new experience.

        Args:
            z_trajectory: latent states (seq_len, latent_dim)
            a_trajectory: actions (seq_len, action_dim)
            free_energy: current free energy (surprise measure)

        Returns:
            Dictionary with write status and memory statistics
        """
        written = self.episodic.try_write(z_trajectory, a_trajectory, free_energy)

        # Always add to consolidation buffer
        for i in range(len(z_trajectory) - 1):
            self.consolidation_buffer.append(
                (z_trajectory[i].detach(), a_trajectory[i].detach(),
                 z_trajectory[i + 1].detach())
            )

        return {
            "written_to_episodic": written,
            "episodic_size": len(self.episodic.traces),
            "consolidation_size": len(self.consolidation_buffer),
            "surprise": free_energy,
        }

    def consolidate(
        self, world_model: nn.Module, n_steps: int = 4
    ) -> dict:
        """
        Legacy diagnostic: measure distances between stored latent states.

        This method does not optimize model parameters. For actual replay
        training on observed transitions, use NOEMAAgent.consolidate().
        """
        if not self.episodic.traces:
            return {"consolidation_loss": 0.0, "n_replayed": 0}

        traces = self.episodic.sample_for_replay(self.replay_batch_size)
        total_loss = 0.0

        for trace in traces:
            trace.replay_count += 1
            z = trace.z.to(next(world_model.parameters()).device)
            a = trace.a.to(next(world_model.parameters()).device)

            # Replay through world model and compute prediction error
            for i in range(min(n_steps, len(z) - 1)):
                try:
                    # Use world model to predict next state
                    pred_loss = F.mse_loss(z[i + 1], z[i])  # Simplified
                    total_loss += pred_loss.item()

                    # Update residual error for priority
                    trace.residual_error = 0.9 * trace.residual_error + 0.1 * pred_loss.item()
                except Exception:
                    pass

        avg_loss = total_loss / max(len(traces), 1)

        return {
            "consolidation_loss": avg_loss,
            "n_replayed": len(traces),
        }

    def retrieve(
        self, query: torch.Tensor, top_k: int = 5
    ) -> list[EpisodicTrace]:
        """Retrieve relevant episodic memories for a given query."""
        return self.episodic.retrieve_similar(query, top_k)

    def memory_stats(self) -> dict:
        """Return statistics about memory state."""
        if not self.episodic.traces:
            return {"size": 0, "avg_surprise": 0, "avg_replay": 0,
                    "consolidation_buffer_size": len(self.consolidation_buffer)}

        surprises = [t.surprise for t in self.episodic.traces]
        replays = [t.replay_count for t in self.episodic.traces]

        return {
            "size": len(self.episodic.traces),
            "avg_surprise": np.mean(surprises),
            "max_surprise": np.max(surprises),
            "avg_replay": np.mean(replays),
            "consolidation_buffer_size": len(self.consolidation_buffer),
        }
