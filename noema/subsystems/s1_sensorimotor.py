"""
S1: Sensorimotor Core — the "brainstem".

Event-driven layer of spiking predictive-coding units.
Units transmit only prediction *errors*, yielding sparsity and
millisecond reactivity of biological circuits.

Reflexive loops (balance, saccades, grasp correction) close locally
here, inside the latency budget that deliberation can never meet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikingPredictiveUnit(nn.Module):
    """
    A single spiking predictive-coding unit.

    Predicts its input from context. Transmits only the prediction error
    (difference between predicted and actual input). Spikes when error
    exceeds threshold, implementing event-driven communication.
    """

    def __init__(self, input_dim: int, context_dim: int, threshold: float = 1.0):
        super().__init__()
        self.predictor = nn.Linear(context_dim, input_dim)
        self.threshold = threshold
        self.register_buffer("spike_count", torch.tensor(0.0))

    def forward(
        self, x: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: input signal (batch, input_dim)
            context: context for prediction (batch, context_dim)

        Returns:
            error: prediction error (only transmitted quantity)
            spike: binary spike (1 where |error| > threshold)
        """
        prediction = self.predictor(context)
        error = x - prediction
        spike = (error.abs() > self.threshold).float()
        self.spike_count += spike.sum().detach()
        return error * spike, spike  # Sparse error signal


class SensorimotorCore(nn.Module):
    """
    S1: Sensorimotor Core.

    Event-driven spiking predictive-coding layer. Implements:
    - Sparse, event-driven processing
    - Local reflexive loops (balance, saccades, grasp)
    - Neuromorphic-compatible prediction error transmission

    This is the lowest level of the hierarchy. It handles:
    - Proprioceptive prediction (body state)
    - Basic reflex circuits
    - Forward/inverse kinematics at reflex speed
    """

    def __init__(
        self,
        proprio_dim: int = 12,
        extero_dim: int = 16,
        motor_dim: int = 6,
        hidden_dim: int = 64,
        n_units: int = 8,
        spike_threshold: float = 0.5,
    ):
        super().__init__()
        self.proprio_dim = proprio_dim
        self.extero_dim = extero_dim
        self.motor_dim = motor_dim

        total_input = proprio_dim + extero_dim

        self.chunk_dim = total_input // n_units

        # Context encoder: motor command + hidden state → context
        self.context_encoder = nn.Sequential(
            nn.Linear(motor_dim + hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Spiking predictive units — context is the encoded context (hidden_dim)
        self.units = nn.ModuleList(
            [
                SpikingPredictiveUnit(
                    input_dim=self.chunk_dim,
                    context_dim=hidden_dim,
                    threshold=spike_threshold,
                )
                for _ in range(n_units)
            ]
        )

        # Hidden state (recurrent)
        self.hidden_layer = nn.GRUCell(total_input, hidden_dim)

        # Reflex circuits: direct sensorimotor loops
        self.reflex_balance = nn.Linear(proprio_dim, motor_dim)
        self.reflex_grasp = nn.Linear(extero_dim, motor_dim)

        # Forward model: (proprio, motor) → next proprio
        self.forward_model = nn.Sequential(
            nn.Linear(proprio_dim + motor_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proprio_dim),
        )

        # Inverse model: (proprio, target_proprio) → motor
        self.inverse_model = nn.Sequential(
            nn.Linear(proprio_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, motor_dim),
        )

        self.hidden_dim = hidden_dim
        self.register_buffer("hidden_state", torch.zeros(1, hidden_dim))

    def forward(
        self,
        proprio: torch.Tensor,
        extero: torch.Tensor,
        motor: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Process sensorimotor input through spiking predictive coding.

        Args:
            proprio: proprioceptive input (batch, proprio_dim)
            extero: exteroceptive input (batch, extero_dim)
            motor: motor command (batch, motor_dim)

        Returns:
            Dictionary with:
                - prediction_errors: sparse error signals
                - sparsity: fraction of active (spiking) units
                - reflex_motor: reflex-corrected motor output
                - forward_prediction: predicted next proprioceptive state
        """
        batch = proprio.shape[0]

        # Combine inputs
        sensor_input = torch.cat([proprio, extero], dim=-1)

        # Expand hidden state if needed for batch size
        if self.hidden_state.shape[0] < batch:
            self.hidden_state = self.hidden_state[:1].expand(batch, -1).clone()

        # Update hidden state
        h = self.hidden_layer(sensor_input, self.hidden_state[:batch])
        self.hidden_state = h.detach()

        # Context for prediction
        context = self.context_encoder(torch.cat([motor, h], dim=-1))

        # Spiking prediction errors
        errors = []
        spikes = []
        chunk_size = sensor_input.shape[-1] // len(self.units)
        for i, unit in enumerate(self.units):
            start = i * chunk_size
            end = start + chunk_size
            chunk = sensor_input[..., start:end]
            error, spike = unit(chunk, context)
            errors.append(error)
            spikes.append(spike)

        prediction_errors = torch.cat(errors, dim=-1)
        spike_tensor = torch.cat(spikes, dim=-1)
        sparsity = spike_tensor.mean()

        # Reflex circuits
        reflex_motor = torch.tanh(
            self.reflex_balance(proprio) + self.reflex_grasp(extero)
        )

        # Forward prediction
        forward_pred = self.forward_model(torch.cat([proprio, motor], dim=-1))

        return {
            "prediction_errors": prediction_errors,
            "sparsity": sparsity,
            "spike_mask": spike_tensor,
            "reflex_motor": reflex_motor,
            "forward_prediction": forward_pred,
            "hidden_state": h,
        }

    def inverse_kinematics(
        self, proprio: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute motor command to reach target proprioceptive state."""
        return self.inverse_model(torch.cat([proprio, target], dim=-1))

    def compute_reflexive_correction(
        self, proprio: torch.Tensor, extero: torch.Tensor
    ) -> torch.Tensor:
        """Pure reflexive motor correction (sub-deliberation latency)."""
        return torch.tanh(self.reflex_balance(proprio) + self.reflex_grasp(extero))

    def prediction_error_loss(
        self,
        proprio: torch.Tensor,
        extero: torch.Tensor,
        motor: torch.Tensor,
    ) -> torch.Tensor:
        """Compute prediction error as loss for learning."""
        result = self.forward(proprio, extero, motor)
        errors = result["prediction_errors"]
        return (errors ** 2).sum(dim=-1).mean()
