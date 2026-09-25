"""
Core objective functions: Variational Free Energy and Expected Free Energy.

Implements the unified objective from the paper:
  F = E_q(s)[ln q(s) - ln p(o,s)]           (variational free energy)
  G(π) = -E[dynamic KL] - E[pragmatic]       (expected free energy)

Section 3 of the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VariationalFreeEnergy(nn.Module):
    """
    Variational Free Energy:
      F = E_{q(s)}[ln q(s) - ln p_θ(o, s)]
        = D_KL[q(s) || p(s|o)] - ln p(o)
    
    Under Gaussian assumptions this reduces to a precision-weighted
    prediction error (Claim 1 in the paper).
    """

    def __init__(self, latent_dim: int, obs_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

        # Generative model components
        # p(s) = N(0, I)
        # p(o|s) = N(decoder(s), Σ_obs)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, obs_dim),
        )
        self.log_sigma_obs = nn.Parameter(torch.zeros(obs_dim))

    def forward(
        self,
        obs: torch.Tensor,
        q_mu: torch.Tensor,
        q_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute variational free energy.

        Args:
            obs: observations (batch, obs_dim)
            q_mu: posterior mean (batch, latent_dim)
            q_logvar: posterior log-variance (batch, latent_dim)

        Returns:
            F: variational free energy scalar (averaged over batch)
        """
        # KL divergence: D_KL[q(s) || p(s)] under diagonal Gaussians
        # = 0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl = -0.5 * torch.sum(
            1 + q_logvar - q_mu.pow(2) - q_logvar.exp(), dim=-1
        )

        # Reconstruction: -log p(o|s) = precision-weighted prediction error
        s_sample = self._reparameterize(q_mu, q_logvar)
        obs_pred = self.decoder(s_sample)
        sigma_obs = self.log_sigma_obs.exp()
        recon = 0.5 * torch.sum(
            ((obs - obs_pred) / sigma_obs).pow(2) + 2 * self.log_sigma_obs,
            dim=-1,
        )

        return (kl + recon).mean()

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std


class ExpectedFreeEnergy(nn.Module):
    """
    Expected Free Energy (EFE) for policy selection:
      G(π) = -E_{q(o,s|π)}[D_KL[q(s|o,π) || q(s|π)]]     (epistemic value)
             -E_{q(o|π)}[ln p(o|C)]                          (pragmatic value)

    The epistemic term makes exploration mathematically obligatory.
    The agent acts to resolve its own uncertainty. No dataset needed.
    Invariant I1 is satisfied by construction.

    Equation (2) in the paper.
    """

    def __init__(self, latent_dim: int, obs_dim: int, n_actions: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        # Transition model: p(s_{t+1} | s_t, a_t)
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + n_actions, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim * 2),  # mu and logvar
        )

        # Likelihood: p(o|s)
        self.likelihood = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.Linear(64, obs_dim),
        )

        # Preference model: p(o|C) — prior preferences (pragmatic)
        # Initially uniform; can be shaped
        self.preference_log_sigma = nn.Parameter(torch.zeros(obs_dim))

    def transition_posterior(
        self, s_t: torch.Tensor, a_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict next latent state distribution given current state and action."""
        x = torch.cat([s_t, a_t], dim=-1)
        params = self.transition(x)
        mu, logvar = params.chunk(2, dim=-1)
        return mu, logvar

    def compute_efe(
        self,
        s_t: torch.Tensor,
        actions: torch.Tensor,
        preferred_obs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute EFE for a batch of action candidates.

        Args:
            s_t: current latent state (batch, latent_dim)
            actions: action candidates (batch, n_actions)
            preferred_obs: preferred observations for pragmatic term (batch, obs_dim)

        Returns:
            G: expected free energy for each action (batch,) — lower is better
        """
        batch = s_t.shape[0]

        # Predict next state
        s_next_mu, s_next_logvar = self.transition_posterior(s_t, actions)

        # Sample from q(s_{t+1}|s_t, a_t) — posterior over states given action
        std = (0.5 * s_next_logvar).exp()
        s_samples = s_next_mu.unsqueeze(0) + torch.randn(
            16, batch, self.latent_dim, device=s_t.device
        ) * std.unsqueeze(0)

        # Predicted observations
        o_pred = self.likelihood(s_samples)  # (n_samples, batch, obs_dim)

        # --- Epistemic value: -E[D_KL[q(s|o,π) || q(s|π)]]
        # Approximate via mutual information: entropy of prior - avg entropy of posterior
        # Higher entropy of prior = more uncertainty to resolve = more epistemic value
        prior_entropy = 0.5 * self.latent_dim * (
            1 + torch.log(torch.tensor(2 * torch.pi, device=s_t.device))
        ) + 0.5 * s_next_logvar.sum(dim=-1)

        # Approximate epistemic value as negative of posterior entropy
        # (simplification for tractability — full form requires marginal over o)
        epistemic = -s_next_logvar.sum(dim=-1) * 0.5  # Higher entropy → higher epistemic value

        # --- Pragmatic value: -E[ln p(o|C)]
        if preferred_obs is not None:
            sigma_pref = self.preference_log_sigma.exp()
            pragmatic = -0.5 * ((o_pred.mean(0) - preferred_obs) / sigma_pref).pow(2).sum(
                dim=-1
            )
        else:
            # Without specific preferences, pragmatic term is neutral
            pragmatic = torch.zeros(batch, device=s_t.device)

        # G = -epistemic - pragmatic (lower G = better action)
        # We return G so that argmin gives the best action
        G = -epistemic - pragmatic

        return G

    def select_action(
        self,
        s_t: torch.Tensor,
        preferred_obs: Optional[torch.Tensor] = None,
        action_candidates: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Select action by minimizing expected free energy.

        Returns:
            action: selected action (1, n_actions)
            efe_values: EFE for all candidates (n_candidates,)
        """
        if action_candidates is None:
            # Generate candidates: one-hot per action
            action_candidates = torch.eye(self.n_actions, device=s_t.device)

        n_candidates = action_candidates.shape[0]
        s_t_expanded = s_t.expand(n_candidates, -1)
        efe_values = self.compute_efe(s_t_expanded, action_candidates, preferred_obs)

        best_idx = efe_values.argmin(dim=0)
        return action_candidates[best_idx : best_idx + 1], efe_values

    def compute_loss(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training loss: negative log-likelihood of transition + KL to prior.
        """
        mu, logvar = self.transition_posterior(s_t, a_t)
        # NLL under predicted distribution
        nll = 0.5 * (
            (s_next - mu).pow(2) / logvar.exp()
            + logvar
            + torch.log(torch.tensor(2 * torch.pi, device=s_t.device))
        ).sum(dim=-1)
        # KL to standard normal prior
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return (nll + kl).mean()
