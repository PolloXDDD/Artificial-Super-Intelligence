"""Grounded ensemble: every target is an observed environment transition."""

import torch
from torch import nn


class GroundedDynamics(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim, members=3, hidden_dim=64):
        super().__init__()
        self.members = nn.ModuleList([
            nn.Sequential(nn.Linear(obs_dim + action_dim + latent_dim, hidden_dim),
                          nn.GELU(), nn.Linear(hidden_dim, hidden_dim),
                          nn.GELU(), nn.Linear(hidden_dim, obs_dim))
            for _ in range(members)
        ])
        for member in self.members:
            nn.init.zeros_(member[-1].weight)
            nn.init.zeros_(member[-1].bias)
        self.register_buffer("obs_mean", torch.zeros(obs_dim))
        self.register_buffer("obs_scale", torch.ones(obs_dim))
        self.register_buffer("delta_scale", torch.ones(obs_dim))
        self.register_buffer("calibrated", torch.tensor(False))

    @torch.no_grad()
    def calibrate(self, obs, next_obs):
        """Fit once on training data; freeze for comparable generation scores."""
        if bool(self.calibrated):
            raise RuntimeError("Normalization already fitted; start a new run to refit.")
        self.obs_mean.copy_(obs.mean(0))
        self.obs_scale.copy_(obs.std(0, correction=0).clamp_min(0.1))
        self.delta_scale.copy_((next_obs - obs).std(0, correction=0).clamp_min(0.02))
        self.calibrated.fill_(True)

    def normalize(self, obs):
        return (obs - self.obs_mean) / self.obs_scale

    def forward(self, obs, action, latent):
        features = torch.cat([self.normalize(obs), action, latent], dim=-1)
        deltas = torch.stack([member(features) for member in self.members])
        return obs.unsqueeze(0) + deltas * self.delta_scale
