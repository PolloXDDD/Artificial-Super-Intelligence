"""NOEMA integration. Inference never trains; learning consumes real transitions."""
from collections import deque
import copy
import torch
from torch import nn
from torch.nn import functional as F
from .core.dynamics import GroundedDynamics
from .core.free_energy import ExpectedFreeEnergy
from .subsystems.s1_sensorimotor import SensorimotorCore
from .subsystems.s2_world_model import HierarchicalWorldModel
from .subsystems.s3_memory import ComplementaryMemory
from .subsystems.s4_knowledge import RelationalKnowledgeGraph
from .subsystems.s5_analogy import AnalogyEngine
from .subsystems.s6_workspace import GlobalWorkspace, ContentType


class NOEMAAgent(nn.Module):
    """Research agent with grounded prediction, planning and explicit learning."""
    def __init__(self, obs_dim=32, proprio_dim=12, extero_dim=16, action_dim=4,
                 latent_dim=64, n_world_model_levels=3, num_slots=8, slot_dim=32,
                 embedding_dim=32, episodic_capacity=5000, surprise_threshold=1.5,
                 ensemble_members=3, dynamics_hidden=64, device="cpu"):
        super().__init__()
        self.config = dict(obs_dim=obs_dim, proprio_dim=proprio_dim,
                           extero_dim=extero_dim, action_dim=action_dim,
                           latent_dim=latent_dim, n_world_model_levels=n_world_model_levels,
                           num_slots=num_slots, slot_dim=slot_dim, embedding_dim=embedding_dim,
                           episodic_capacity=episodic_capacity,
                           surprise_threshold=surprise_threshold,
                           ensemble_members=ensemble_members, dynamics_hidden=dynamics_hidden)
        self.obs_dim, self.action_dim, self.latent_dim = obs_dim, action_dim, latent_dim
        self.s1 = SensorimotorCore(proprio_dim=proprio_dim, extero_dim=extero_dim,
                                   motor_dim=action_dim)
        self.s2 = HierarchicalWorldModel(obs_dim=obs_dim, action_dim=action_dim,
                                        n_levels=n_world_model_levels, latent_dim=latent_dim,
                                        num_slots=num_slots, slot_dim=slot_dim)
        self.s3 = ComplementaryMemory(latent_dim=latent_dim, action_dim=action_dim,
                                      episodic_capacity=episodic_capacity,
                                      surprise_threshold=surprise_threshold)
        self.s4 = RelationalKnowledgeGraph(embedding_dim=embedding_dim,
                                           relation_embedding_dim=32)
        self.s5 = AnalogyEngine(knowledge_graph=self.s4, embedding_dim=embedding_dim,
                                relation_dim=32)
        self.s6 = GlobalWorkspace(embedding_dim=latent_dim, capacity=7)
        self.efe = ExpectedFreeEnergy(latent_dim=latent_dim, obs_dim=obs_dim,
                                      n_actions=action_dim)
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, 128), nn.GELU(),
                                          nn.Linear(128, latent_dim))
        self.action_decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(),
                                             nn.Linear(64, action_dim), nn.Tanh())
        self.dynamics = GroundedDynamics(obs_dim, action_dim, latent_dim,
                                         ensemble_members, dynamics_hidden)
        self.step_count = 0
        self.phase = 0
        self.loss_history = deque(maxlen=2000)
        self.transition_memory = deque(maxlen=episodic_capacity)
        self.register_buffer("updates", torch.tensor(0, dtype=torch.long))
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    @property
    def device(self):
        return next(self.parameters()).device

    def _batch(self, value, dim):
        value = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2 or value.shape[-1] != dim or value.shape[0] == 0:
            raise ValueError(f"Expected a nonempty (batch, {dim}) tensor")
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Inputs must be finite")
        return value

    def predict_next(self, obs, action, return_members=False):
        obs, action = self._batch(obs, self.obs_dim), self._batch(action, self.action_dim)
        if obs.shape[0] != action.shape[0]:
            raise ValueError("Observation/action batch lengths differ")
        z = self.obs_encoder(self.dynamics.normalize(obs))
        predictions = self.dynamics(obs, action, z)
        return predictions if return_members else predictions.mean(0)

    def learn_transitions(self, obs, action, next_obs, *, jepa_weight=0.02,
                          auxiliary=True, bootstrap=True):
        """Update weights on measured (o_t, a_t, o_next), including S2 and EFE.

        JEPA/EFE are auxiliary objectives. Grounded prediction error, rather than
        a moving latent target, is the independently measurable primary objective.
        """
        if not self.training:
            raise RuntimeError("Call agent.train() before learning")
        obs = self._batch(obs, self.obs_dim)
        action = self._batch(action, self.action_dim)
        next_obs = self._batch(next_obs, self.obs_dim)
        if obs.shape[0] != action.shape[0] or obs.shape != next_obs.shape:
            raise ValueError("Transition batch shapes differ")
        predictions = self.predict_next(obs, action, return_members=True)
        errors = ((predictions - next_obs) / self.dynamics.delta_scale).square().mean(-1)
        if bootstrap and obs.shape[0] > 1:
            mask = (torch.rand_like(errors) > 0.2).to(errors.dtype)
            mask[:, 0] = 1
            primary = ((errors * mask).sum(-1) / mask.sum(-1)).mean()
        else:
            primary = errors.mean()
        loss = primary
        jepa_loss = primary.new_zeros(())
        if auxiliary:
            normalized = self.dynamics.normalize(obs)
            normalized_next = self.dynamics.normalize(next_obs)
            jepa_loss = self.s2([normalized, normalized_next], [action])["total_loss"]
            z = self.obs_encoder(normalized).detach()
            z_next = self.obs_encoder(normalized_next).detach()
            # EFE transition model learns on real transitions; no invented labels.
            efe_loss = self.efe.compute_loss(z, action, z_next) / self.latent_dim
            loss = loss + jepa_weight * jepa_loss + 0.005 * efe_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0, error_if_nonfinite=True)
        self.optimizer.step()
        if auxiliary:
            self.s2.update_targets()
        self.updates.add_(1)
        self.loss_history.append(float(primary.detach()))
        return {"prediction_loss": float(primary.detach()),
                "jepa_loss": float(jepa_loss.detach()), "total_loss": float(loss.detach())}

    def observe_transition(self, obs, action, next_obs, *, learn=True):
        """Store actual experience, then optionally learn. Reset is not a transition."""
        obs, action = self._batch(obs, self.obs_dim), self._batch(action, self.action_dim)
        next_obs = self._batch(next_obs, self.obs_dim)
        if obs.shape != next_obs.shape or len(obs) != len(action):
            raise ValueError("Transition batch shapes differ")
        with torch.no_grad():
            surprise = float(((self.predict_next(obs, action) - next_obs)
                              / self.dynamics.delta_scale).square().mean())
            z = self.obs_encoder(self.dynamics.normalize(obs))
            zn = self.obs_encoder(self.dynamics.normalize(next_obs))
            for o, a, n, zi, zni in zip(obs, action, next_obs, z, zn):
                self.transition_memory.append(tuple(t.detach().cpu().clone() for t in (o, a, n)))
                self.s3(torch.stack([zi, zni]), torch.stack([a, a]), surprise)
        result = {"surprise": surprise}
        if learn:
            result.update(self.learn_transitions(obs, action, next_obs))
        return result

    @torch.no_grad()
    def plan_action(self, obs, preferred_obs=None, *, exploration=0.1):
        """One-step model-based control, or ensemble-disagreement exploration.

        Multi-step planning and task-success guarantees are outside this version.
        """
        obs = self._batch(obs, self.obs_dim)
        if len(obs) != 1:
            raise ValueError("plan_action accepts one observation")
        eye = torch.eye(self.action_dim, device=self.device)
        candidates = torch.cat([torch.zeros_like(eye[:1]), eye, -eye], 0)
        predictions = self.predict_next(obs.expand(len(candidates), -1), candidates, True)
        uncertainty = (predictions / self.dynamics.delta_scale).var(0, correction=0).mean(-1)
        scores = -float(exploration) * uncertainty + 0.0001 * candidates.square().mean(-1)
        if preferred_obs is not None:
            target = self._batch(preferred_obs, self.obs_dim)
            if len(target) != 1:
                raise ValueError("preferred_obs must contain one target")
            scores += ((predictions.mean(0) - target) / self.dynamics.obs_scale).square().mean(-1)
        best = int(scores.argmin())
        return candidates[best:best+1], scores

    @torch.no_grad()
    def forward(self, obs, proprio=None, extero=None, preferred_obs=None):
        """Inference only. Call observe_transition after the environment step."""
        obs = self._batch(obs, self.obs_dim)
        if len(obs) != 1:
            raise ValueError("Interaction uses one observation; predict_next supports batches")
        proprio = torch.zeros(1, self.config["proprio_dim"], device=self.device) if proprio is None else self._batch(proprio, self.config["proprio_dim"])
        extero = torch.zeros(1, self.config["extero_dim"], device=self.device) if extero is None else self._batch(extero, self.config["extero_dim"])
        # Restore each submodule's original mode; no EMA or running-stat updates.
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            z = self.obs_encoder(self.dynamics.normalize(obs))
            s1 = self.s1(proprio, extero, torch.zeros(1, self.action_dim, device=self.device))
            world_z = self.s2.encode(self.dynamics.normalize(obs))
            self.s6.clear()
            self.s6.submit(ContentType.PREDICTION, world_z, 1.0, "S2")
            self.s6.submit(ContentType.REFLEX, s1["hidden_state"], 0.5, "S1")
            workspace = self.s6.competition_step()
            action, scores = self.plan_action(obs, preferred_obs)
            self.step_count += 1
            return {"action": action, "action_scores": scores,
                    "free_energy": self.loss_history[-1] if self.loss_history else None,
                    "s1": {"sparsity": float(s1["sparsity"])},
                    "s2": {"representation": world_z}, "s3": self.s3.memory_stats(),
                    "s4": self.s4.graph_stats(), "s6": workspace, "latent": z,
                    "phase": self.phase}
        finally:
            for module, mode in modes:
                module.training = mode

    def consolidate(self, n_steps=4, batch_size=32):
        """Sleep: optimize on actual replayed transitions, not a diagnostic distance."""
        if not self.transition_memory:
            return {"consolidation_loss": 0.0, "n_replayed": 0}
        if not self.training:
            raise RuntimeError("Call train() before consolidation")
        if n_steps < 1 or batch_size < 1:
            raise ValueError("Positive consolidation steps and batch size required")
        losses = []
        for _ in range(n_steps):
            indices = torch.randint(len(self.transition_memory), (batch_size,)).tolist()
            batch = [torch.stack([self.transition_memory[i][j] for i in indices]) for j in range(3)]
            losses.append(self.learn_transitions(*batch)["prediction_loss"])
        return {"consolidation_loss": sum(losses) / len(losses),
                "n_replayed": n_steps * batch_size}

    @torch.no_grad()
    def build_knowledge(self, obs, domain="interaction", capacity=128):
        """Explicit S2 -> S4 -> S5 pathway with bounded schema memory.

        The physics RSI score does not measure semantic or analogical quality.
        """
        if capacity < 2:
            raise ValueError("Knowledge capacity must be at least two")
        obs = self._batch(obs, self.obs_dim)
        if len(obs) != 1:
            raise ValueError("Knowledge construction accepts one observation")
        modes = [(module, module.training) for module in self.modules()]
        self.eval()
        try:
            scene = self.s2.perceive(self.dynamics.normalize(obs))["levels"][0]
            slots = scene["slots"]
            width = self.config["embedding_dim"]
            slots = F.pad(slots, (0, max(0, width - slots.shape[-1])))[..., :width]
            while len(self.s4.schemas) >= capacity:
                self.s4.remove_schema(next(iter(self.s4.schemas)))
            schema = self.s4.build_schema_from_slots(slots, scene["relations"]["embeddings"], domain)
            return {"schema_id": schema.id, "analogy": self.s5(schema),
                    "graph": self.s4.graph_stats()}
        finally:
            for module, mode in modes:
                module.training = mode

    def clone_for_training(self):
        """Fork weights and Adam state; ephemeral interaction memory stays separate."""
        clone = type(self)(**self.config, device=str(self.device))
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        clone.optimizer.load_state_dict(copy.deepcopy(self.optimizer.state_dict()))
        return clone

    def get_diagnostics(self):
        return {"step_count": self.step_count, "phase": self.phase,
                "updates": int(self.updates), "real_transitions": len(self.transition_memory),
                "s3_memory": self.s3.memory_stats(), "s4_graph": self.s4.graph_stats()}
