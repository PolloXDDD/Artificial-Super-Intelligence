"""
Slot Attention — Factorizes experience into entities and relations.

Section 4.2 (S2): The world model is forced to factorize experience
into objects, properties, and relations between them. This factorization
is what S4 (Relational Knowledge Graph) will later detach and transport.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    Slot Attention module (Locatello et al., 2020).

    Maps a set of latent features to a set of slots (entities),
    enabling object-centric representations.

    Args:
        num_slots: number of entity slots
        input_dim: dimensionality of input features
        slot_dim: dimensionality of each slot
        n_iterations: number of attention refinement iterations
    """

    def __init__(
        self,
        num_slots: int = 8,
        input_dim: int = 64,
        slot_dim: int = 64,
        n_iterations: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.n_iterations = n_iterations

        # Projections
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # Slot update GRU
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Layer norm
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_input = nn.LayerNorm(input_dim)
        self.norm_mu = nn.LayerNorm(slot_dim)

        # MLP for slot refinement
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.GELU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )
        self.norm_mlp = nn.LayerNorm(slot_dim)

        # Learnable slot initialization
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim))
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)

    def forward(
        self, inputs: torch.Tensor, n_slots: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: (batch, n_features, input_dim)
            n_slots: override number of slots

        Returns:
            slots: (batch, num_slots, slot_dim) — entity representations
            attn_weights: (batch, num_slots, n_features) — attention masks
        """
        B, N, _ = inputs.shape
        n_s = n_slots or self.num_slots

        inputs = self.norm_input(inputs)

        # Initialize slots
        mu = self.slots_mu.expand(B, -1, -1)[:, :n_s, :]
        sigma = self.slots_log_sigma.exp().expand(B, -1, -1)[:, :n_s, :]
        slots = mu
        if self.training:
            slots = mu + sigma * torch.randn(
                B, n_s, self.slot_dim, device=inputs.device
            )

        k = self.project_k(inputs)  # (B, N, slot_dim)
        v = self.project_v(inputs)  # (B, N, slot_dim)

        for _ in range(self.n_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = self.project_q(slots)  # (B, n_s, slot_dim)

            # Attention: softmax over slots for each input position
            # (B, n_s, slot_dim) x (B, slot_dim, N) → (B, n_s, N)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) / (self.slot_dim ** 0.5)
            attn = F.softmax(attn_logits, dim=1)  # Normalize over slots
            attn_weights = attn  # (B, n_s, N)

            # Weighted mean
            updates = torch.bmm(attn_weights, v)  # (B, n_s, slot_dim)

            # GRU update
            slots = self.gru(
                updates.reshape(B * n_s, self.slot_dim),
                slots_prev.reshape(B * n_s, self.slot_dim),
            ).reshape(B, n_s, self.slot_dim)

            # MLP residual
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn_weights
