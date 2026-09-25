"""
JEPA Module — Joint-Embedding Predictive Architecture.

Implements Section 3.2 of the paper:
  L_JEPA = ||P_ψ(z_t, a_t) - sg[E_φ(o_{t+1})]||^2 + λ * R_anti-collapse(z)

Prediction in latent space, not pixel space. Discards unpredictable noise,
retains only causally relevant structure.

Claim 1 (Unification): Under Gaussian likelihood, L_JEPA is a special case
of the free energy F.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import copy


class EMA:
    """Exponential moving average for target encoder (slow-moving)."""

    def __init__(self, model: nn.Module, decay: float = 0.996):
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        for p_shadow, p_model in zip(
            self.shadow.parameters(), model.parameters()
        ):
            p_shadow.data.mul_(self.decay).add_(p_model.data, alpha=1 - self.decay)

    def module(self) -> nn.Module:
        return self.shadow


class JEPAModule(nn.Module):
    """
    Joint-Embedding Predictive Architecture.

    Encoder E_φ maps observations → embeddings z.
    Predictor P_ψ forecasts future embeddings conditioned on action.
    Target encoder Ē_φ is a slow EMA copy (stop-gradient).

    Args:
        obs_dim: dimensionality of observation space
        latent_dim: dimensionality of latent embedding space
        action_dim: dimensionality of action space
        hidden_dim: hidden layer width
        anti_collapse_lambda: weight for anti-collapse regularizer
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 64,
        action_dim: int = 4,
        hidden_dim: int = 128,
        anti_collapse_lambda: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.anti_collapse_lambda = anti_collapse_lambda

        # Online encoder E_φ
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Target encoder Ē_φ (EMA of online)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.eval()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Predictor P_ψ: (z_t, a_t) → z_{t+1}
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Anti-collapse: variance regularization to prevent
        # all embeddings collapsing to a single point
        self.register_buffer("running_mean", torch.zeros(latent_dim))
        self.register_buffer("running_var", torch.ones(latent_dim))
        self.register_buffer("n_samples", torch.tensor(0.0))

    @torch.no_grad()
    def update_target_encoder(self, decay: float = 0.996):
        """EMA update of target encoder."""
        for p_target, p_online in zip(
            self.target_encoder.parameters(), self.encoder.parameters()
        ):
            p_target.data.mul_(decay).add_(p_online.data, alpha=1 - decay)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation to latent embedding z = E_φ(o)."""
        return self.encoder(obs)

    @torch.no_grad()
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observation with target encoder (stop-gradient)."""
        return self.target_encoder(obs)

    def predict(self, z_t: torch.Tensor, a_t: torch.Tensor) -> torch.Tensor:
        """Predict next latent embedding: P_ψ(z_t, a_t)."""
        return self.predictor(torch.cat([z_t, a_t], dim=-1))

    def anti_collapse_regularizer(self, z: torch.Tensor) -> torch.Tensor:
        """
        Prevent representation collapse by maintaining variance
        across the batch dimension in each latent dimension.
        Uses off-diagonal covariance penalty (VICReg-style).
        """
        batch_size = z.shape[0]

        # Variance regularization: keep std above threshold
        if batch_size < 2:
            return z.sum() * 0.0
        std_z = torch.sqrt(z.var(dim=0, correction=0) + 1e-4)
        var_loss = F.relu(1.0 - std_z).mean()

        # Covariance regularization: decorrelate dimensions
        z_centered = z - z.mean(dim=0)
        cov = (z_centered.T @ z_centered) / (batch_size - 1)
        diag = torch.eye(self.latent_dim, device=z.device)
        cov_loss = cov[~diag.bool()].pow(2).sum() / self.latent_dim

        return var_loss + cov_loss

    def forward(
        self,
        obs_t: torch.Tensor,
        obs_next: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute JEPA loss.

        L = ||P_ψ(z_t, a_t) - sg[Ē_φ(o_{t+1})]||^2 + λ * R(z)

        Returns:
            Dictionary with total_loss, prediction_loss, anti_collapse_loss
        """
        # Online encoding
        z_t = self.encode(obs_t)

        # Target encoding (stop-gradient)
        z_next_target = self.encode_target(obs_next)

        # Prediction
        z_next_pred = self.predict(z_t, action_t)

        # Prediction loss
        pred_loss = F.mse_loss(z_next_pred, z_next_target.detach())

        # Anti-collapse regularizer
        anti_collapse = self.anti_collapse_regularizer(z_t)

        # Total loss
        total_loss = pred_loss + self.anti_collapse_lambda * anti_collapse

        # Update running statistics
        if self.training:
            with torch.no_grad():
                self.n_samples += obs_t.shape[0]
                momentum = min(0.1, 1.0 / (1 + float(self.n_samples) / 1000))
                self.running_mean.mul_(1 - momentum).add_(
                    z_t.mean(dim=0).detach(), alpha=momentum
                )
                self.running_var.mul_(1 - momentum).add_(
                    z_t.var(dim=0, correction=0).detach(), alpha=momentum
                )

        return {
            "total_loss": total_loss,
            "prediction_loss": pred_loss,
            "anti_collapse_loss": anti_collapse,
            "z_t": z_t,
            "z_next_pred": z_next_pred,
            "z_next_target": z_next_target,
        }

    def compute_free_energy_equivalent(
        self,
        obs_t: torch.Tensor,
        obs_next: torch.Tensor,
        action_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Claim 1 (Unification): Under Gaussian likelihood in latent space,
        L_JEPA is a special case of variational free energy F.
        The JEPA prediction error IS the precision-weighted prediction error
        of active inference.

        This method returns the free energy interpretation.
        """
        z_t = self.encode(obs_t)
        z_next_target = self.encode_target(obs_next)
        z_next_pred = self.predict(z_t, action_t)

        # Precision = 1/variance (from running statistics)
        precision = 1.0 / (self.running_var + 1e-6)

        # Precision-weighted prediction error
        delta = z_next_target - z_next_pred
        fep = 0.5 * (delta * precision * delta).sum(dim=-1) + 0.5 * torch.log(
            self.running_var + 1e-6
        ).sum(dim=-1)

        return fep.mean()
