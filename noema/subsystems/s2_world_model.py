"""
S2: Hierarchical World Model — the "cortex".

A stack of JEPA modules with increasing temporal receptive fields:
  z^(1): ~100ms ahead (textures, contacts)
  z^(2): seconds ahead (object trajectories, agent motion)
  z^(3): minutes-to-hours ahead (tasks, intentions, narratives)

Each level predicts the level below, implementing predictive coding
across timescales. Level k represents in terms of entities and relations
extracted from level k-1 through slot-attention bottleneck.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.jepa import JEPAModule
from ..core.slot_attention import SlotAttention


class JEPALevel(nn.Module):
    """
    One level of the hierarchical world model.

    Each level:
    1. Receives entity slots from the level below (or sensorimotor input)
    2. Runs slot attention to further factorize
    3. Predicts next entity state conditioned on action
    4. Computes JEPA loss (predict in latent space)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        action_dim: int = 4,
        num_slots: int = 8,
        slot_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Slot attention for entity factorization
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            input_dim=latent_dim,
            slot_dim=slot_dim,
            n_iterations=3,
        )

        # Flatten slots for JEPA
        self.flat_dim = num_slots * slot_dim

        # JEPA module for this level
        self.jepa = JEPAModule(
            obs_dim=self.flat_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )

        # Relation extraction: pairwise relations between slots
        self.relation_encoder = nn.Sequential(
            nn.Linear(slot_dim * 4, 64),  # (slot_i, slot_j, |slot_i-slot_j|, slot_i*slot_j)
            nn.GELU(),
            nn.Linear(64, 32),  # relation type embedding
        )

    def forward(
        self,
        obs_t: torch.Tensor,
        obs_next: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict:
        """
        Process one timestep through this level.

        Returns entity slots, relations, and JEPA loss.
        """
        batch = obs_t.shape[0]

        # Project input
        x_t = self.input_proj(obs_t)
        x_next = self.input_proj(obs_next)

        # Reshape for slot attention (batch, n_features, dim)
        # Treat input_dim features as "pixels"
        n_chunks = max(1, x_t.shape[-1] // 8)
        x_t_chunked = x_t.reshape(batch, n_chunks, -1)
        x_next_chunked = x_next.reshape(batch, n_chunks, -1)

        # Pad if needed for slot attention
        feat_dim = x_t_chunked.shape[-1]

        # Slot attention → entity factorization
        if feat_dim < self.latent_dim:
            pad_size = self.latent_dim - feat_dim
            x_t_padded = F.pad(x_t_chunked, (0, pad_size))
            x_next_padded = F.pad(x_next_chunked, (0, pad_size))
        else:
            x_t_padded = x_t_chunked
            x_next_padded = x_next_chunked

        slots_t, attn_t = self.slot_attention(x_t_padded)
        slots_next, attn_next = self.slot_attention(x_next_padded)

        # Flatten for JEPA
        flat_t = slots_t.reshape(batch, -1)
        flat_next = slots_next.reshape(batch, -1)

        # Pad if needed for JEPA
        if flat_t.shape[-1] < self.flat_dim:
            pad_size = self.flat_dim - flat_t.shape[-1]
            flat_t = F.pad(flat_t, (0, pad_size))
            flat_next = F.pad(flat_next, (0, pad_size))
        elif flat_t.shape[-1] > self.flat_dim:
            flat_t = flat_t[..., :self.flat_dim]
            flat_next = flat_next[..., :self.flat_dim]

        # JEPA prediction
        jepa_out = self.jepa(flat_t, flat_next, action_t)

        # Extract relations between entity slots
        relations = self._extract_relations(slots_t)

        return {
            "slots": slots_t,
            "attn_weights": attn_t,
            "relations": relations,
            "jepa_loss": jepa_out["total_loss"],
            "prediction_loss": jepa_out["prediction_loss"],
            "z_t": jepa_out["z_t"],
            "z_next": jepa_out["z_next_target"],
        }

    def _extract_relations(self, slots: torch.Tensor) -> dict:
        """
        Extract pairwise relations between entity slots.

        Returns relation embeddings for all slot pairs.
        """
        batch, n_slots, slot_dim = slots.shape

        # All pairs
        si = slots.unsqueeze(2).expand(-1, -1, n_slots, -1)
        sj = slots.unsqueeze(1).expand(-1, n_slots, -1, -1)

        # Relation features
        rel_features = torch.cat([
            si, sj,
            (si - sj).abs(),
            si * sj,
        ], dim=-1)  # (batch, n_slots, n_slots, slot_dim*4)

        # Encode relations
        rel_embeddings = self.relation_encoder(rel_features)

        return {
            "embeddings": rel_embeddings,  # (batch, n_slots, n_slots, 32)
            "n_entities": n_slots,
        }

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent at this level."""
        return self.perceive(obs)["z"]

    def perceive(self, obs: torch.Tensor) -> dict:
        """Encode the present scene without inventing a future observation."""
        x = self.input_proj(obs)
        n_chunks = max(1, x.shape[-1] // 8)
        x_chunked = x.reshape(x.shape[0], n_chunks, -1)
        feat_dim = x_chunked.shape[-1]
        if feat_dim < self.latent_dim:
            x_chunked = F.pad(x_chunked, (0, self.latent_dim - feat_dim))
        slots, _ = self.slot_attention(x_chunked)
        return {"z": self.jepa.encode(slots.reshape(x.shape[0], -1)),
                "slots": slots, "relations": self._extract_relations(slots)}


class HierarchicalWorldModel(nn.Module):
    """
    S2: Hierarchical World Model.

    Stack of JEPA levels with increasing temporal receptive fields.
    Each level predicts the level below through slot-attention bottleneck.
    """

    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 4,
        n_levels: int = 3,
        latent_dim: int = 64,
        num_slots: int = 8,
        slot_dim: int = 32,
    ):
        super().__init__()
        self.n_levels = n_levels

        # Level 1: fast (~100ms), direct sensorimotor input
        # Level 2: medium (~seconds), operates on level 1 entities
        # Level 3: slow (~minutes), operates on level 2 entities
        self.levels = nn.ModuleList()
        prev_dim = obs_dim
        for i in range(n_levels):
            level = JEPALevel(
                input_dim=prev_dim,
                latent_dim=latent_dim,
                action_dim=action_dim,
                num_slots=num_slots,
                slot_dim=slot_dim,
                hidden_dim=128,
            )
            self.levels.append(level)
            prev_dim = latent_dim

        # Cross-level prediction: level k predicts level k-1
        self.cross_level_predictors = nn.ModuleList()
        for i in range(1, n_levels):
            self.cross_level_predictors.append(
                nn.Linear(latent_dim, latent_dim)
            )

    def forward(
        self,
        obs_sequence: list[torch.Tensor],
        action_sequence: list[torch.Tensor],
    ) -> dict:
        """
        Process observation sequence through hierarchical model.

        Args:
            obs_sequence: list of (batch, obs_dim) tensors
            action_sequence: list of (batch, action_dim) tensors

        Returns:
            Dictionary with losses, slots, and relations from all levels.
        """
        total_loss = 0.0
        level_outputs = []
        all_relations = []

        for level_idx in range(self.n_levels):
            # Input for this level
            if level_idx == 0:
                # Level 0: direct observations
                inp_t = obs_sequence[0]
                inp_next = obs_sequence[1] if len(obs_sequence) > 1 else obs_sequence[0]
            else:
                # Higher levels: entity representations from level below
                prev_out = level_outputs[-1]
                inp_t = prev_out["z_t"]
                inp_next = prev_out["z_next"].detach()

            act_t = action_sequence[min(level_idx, len(action_sequence) - 1)]

            out = self.levels[level_idx](inp_t, inp_next, act_t)
            level_outputs.append(out)

            total_loss = total_loss + out["jepa_loss"]
            all_relations.append(out["relations"])

        # Cross-level prediction losses
        cross_loss = 0.0
        for i, pred in enumerate(self.cross_level_predictors):
            upper_z = level_outputs[i + 1]["z_t"].detach()
            lower_z = level_outputs[i]["z_t"].detach()
            cross_loss = cross_loss + F.mse_loss(pred(upper_z), lower_z)

        total_loss = total_loss + cross_loss

        # NaN protection
        if not torch.isfinite(total_loss):
            raise FloatingPointError("Non-finite world-model loss")

        return {
            "total_loss": total_loss,
            "level_losses": [o["jepa_loss"].item() for o in level_outputs],
            "cross_level_loss": float(cross_loss.detach()) if isinstance(cross_loss, torch.Tensor) else cross_loss,
            "slots": [o["slots"] for o in level_outputs],
            "relations": all_relations,
            "z_t": level_outputs[-1]["z_t"],  # Top-level representation
        }

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation through all levels to top-level representation."""
        return self.perceive(obs)["z"]

    def perceive(self, obs: torch.Tensor) -> dict:
        z = obs
        scenes = []
        for level in self.levels:
            scene = level.perceive(z)
            z = scene["z"]
            scenes.append(scene)
        return {"z": z, "levels": scenes}

    @torch.no_grad()
    def update_targets(self):
        """Advance EMA only after an actual optimizer update."""
        for level in self.levels:
            level.jepa.update_target_encoder()
