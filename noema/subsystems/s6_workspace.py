"""
S6: Global Workspace — coordination.

A limited-capacity broadcast buffer (Baars; Dehaene) through which
S2 predictions, S3 recalls, and S5 hypotheses compete for access,
with PRECISION — inverse expected uncertainty — as the bidding currency.

The winning content becomes globally available to all subsystems,
implementing serial, reportable, deliberate cognition atop massively
parallel substrates. Language is a late, learned compression code
over workspace contents — never the foundation of thought.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass
from enum import Enum


class ContentType(Enum):
    PREDICTION = "prediction"      # From S2 (world model)
    RECALL = "recall"             # From S3 (memory)
    HYPOTHESIS = "hypothesis"     # From S5 (analogy engine)
    REFLEX = "reflex"             # From S1 (sensorimotor)


@dataclass
class WorkspaceContent:
    """A piece of content competing for workspace access."""
    content_type: ContentType
    data: torch.Tensor           # The actual content embedding
    precision: float             # Inverse uncertainty → bidding currency
    source: str                  # Which subsystem generated this
    metadata: dict               # Additional info


class GlobalWorkspace(nn.Module):
    """
    S6: Global Workspace.

    Limited-capacity broadcast buffer. Contents from S1-S5 compete
    for access. Precision (inverse uncertainty) is the bidding currency.
    Winner is broadcast globally to all subsystems.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        capacity: int = 7,         # Miller's law: 7±2
        competition_temperature: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.capacity = capacity
        self.temperature = competition_temperature

        # Input projection: maps arbitrary input to workspace embedding_dim
        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
        )

        # Competition mechanism: attention-style precision weighting
        self.competition = nn.Sequential(
            nn.Linear(embedding_dim + 1, 64),  # +1 for precision
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Broadcast encoder
        self.broadcast_encoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        # Current workspace contents
        self.current_contents: list[WorkspaceContent] = []
        self.broadcast: Optional[torch.Tensor] = None

    def submit(
        self,
        content_type: ContentType,
        data: torch.Tensor,
        precision: float,
        source: str,
        metadata: Optional[dict] = None,
    ):
        """
        Submit content to the workspace for competition.
        """
        entry = WorkspaceContent(
            content_type=content_type,
            data=data.detach(),
            precision=precision,
            source=source,
            metadata=metadata or {},
        )
        self.current_contents.append(entry)

    def competition_step(self) -> dict:
        """
        Run one competition cycle.

        Contents compete based on precision (inverse uncertainty).
        Winners are broadcast globally.

        Returns the competition results.
        """
        if not self.current_contents:
            return {
                "winners": [],
                "broadcast": None,
                "n_submitted": 0,
            }

        # Score each content
        scores = []
        for content in self.current_contents:
            # Precision is the bidding currency (Dehaene's global workspace)
            score = content.precision

            # Project data to workspace embedding_dim
            data_flat = content.data.flatten()
            # Pad or truncate to embedding_dim
            if data_flat.shape[0] < self.embedding_dim:
                data_proj = F.pad(data_flat, (0, self.embedding_dim - data_flat.shape[0]))
            elif data_flat.shape[0] > self.embedding_dim:
                data_proj = data_flat[:self.embedding_dim]
            else:
                data_proj = data_flat

            data_proj = self.input_proj(data_proj.unsqueeze(0)).squeeze(0)

            # Additional learned scoring
            input_feat = torch.cat([
                data_proj,
                data_proj.new_tensor([content.precision]),
            ])
            learned_score = self.competition(input_feat.unsqueeze(0)).item()
            scores.append(score + learned_score)

        # Select top-k winners (limited capacity)
        scores_tensor = next(self.parameters()).new_tensor(scores)
        scores_softmax = F.softmax(scores_tensor / self.temperature, dim=0)

        n_winners = min(self.capacity, len(self.current_contents))
        winner_indices = scores_tensor.topk(n_winners).indices.tolist()

        winners = [self.current_contents[i] for i in winner_indices]

        # Create broadcast: weighted combination of winners
        winner_data_list = []
        for w in winners:
            data_flat = w.data.flatten()
            if data_flat.shape[0] < self.embedding_dim:
                data_proj = F.pad(data_flat, (0, self.embedding_dim - data_flat.shape[0]))
            elif data_flat.shape[0] > self.embedding_dim:
                data_proj = data_flat[:self.embedding_dim]
            else:
                data_proj = data_flat
            winner_data_list.append(self.broadcast_encoder(data_proj.unsqueeze(0)).squeeze(0))
        winner_data = torch.stack(winner_data_list)
        winner_weights = scores_softmax[winner_indices]
        self.broadcast = (winner_data * winner_weights.unsqueeze(-1)).sum(dim=0)

        # Clear workspace for next cycle
        results = {
            "winners": [
                {
                    "type": w.content_type.value,
                    "source": w.source,
                    "precision": w.precision,
                    "metadata": w.metadata,
                }
                for w in winners
            ],
            "broadcast": self.broadcast.detach(),
            "n_submitted": len(self.current_contents),
            "n_winners": n_winners,
        }

        self.current_contents = []
        return results

    def get_broadcast(self) -> Optional[torch.Tensor]:
        """Get the current broadcast signal."""
        return self.broadcast

    def clear(self):
        """Clear workspace."""
        self.current_contents = []
        self.broadcast = None
