"""
Physics Playground for NOEMA experiments.

Implements environments for testing the four decisive experiments:
  1. Zero-dataset acquisition (intuitive physics)
  2. Far transfer (fluid → heat → circuits)
  3. Self-designed experiments (hypothesis testing)
  4. Continual learning (catastrophic forgetting)

These environments are simple but capture the essential structure
needed to test each invariant.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnvState:
    """State of the physics playground."""
    objects: torch.Tensor       # (n_objects, state_dim) positions, velocities, properties
    relations: torch.Tensor     # (n_objects, n_objects) contact/containment
    observation: torch.Tensor   # (obs_dim,) agent observation
    proprio: torch.Tensor       # (proprio_dim,) proprioceptive
    extero: torch.Tensor        # (extero_dim,) exteroceptive
    reward: float = 0.0
    done: bool = False
    info: dict = None

    def __post_init__(self):
        if self.info is None:
            self.info = {}


class PhysicsPlayground:
    """
    Simple 2D physics playground with objects.
    
    Tests intuitive physics: object permanence, gravity, collisions,
    containment, support.
    """

    def __init__(
        self,
        n_objects: int = 4,
        obs_dim: int = 32,
        proprio_dim: int = 12,
        extero_dim: int = 16,
        action_dim: int = 4,
        dt: float = 0.02,
        seed: Optional[int] = None,
    ):
        self.n_objects = n_objects
        self.obs_dim = obs_dim
        self.proprio_dim = proprio_dim
        self.extero_dim = extero_dim
        self.action_dim = action_dim
        self.dt = dt
        self.gravity = 9.8
        self.rng = np.random.default_rng(seed)

        # Object properties: [x, y, vx, vy, mass, radius, is_fixed, color]
        self.state_dim = 8
        self.reset()

    def reset(self, seed: Optional[int] = None) -> EnvState:
        """Reset environment with random objects."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.objects = torch.zeros(self.n_objects, self.state_dim)

        for i in range(self.n_objects):
            self.objects[i, 0] = self.rng.uniform(-2, 2)      # x
            self.objects[i, 1] = self.rng.uniform(0, 3)       # y
            self.objects[i, 2] = self.rng.uniform(-1, 1)      # vx
            self.objects[i, 3] = self.rng.uniform(-0.5, 0.5)  # vy
            self.objects[i, 4] = self.rng.uniform(0.5, 2.0)   # mass
            self.objects[i, 5] = self.rng.uniform(0.1, 0.3)   # radius
            self.objects[i, 6] = 0.0                            # not fixed
            self.objects[i, 7] = self.rng.uniform(0, 1)        # color

        # Ground is fixed
        self.objects[0, 1] = -0.5
        self.objects[0, 4] = 100.0  # Very heavy
        self.objects[0, 5] = 5.0    # Very large
        self.objects[0, 6] = 1.0    # Fixed

        return self._make_state()

    def step(self, action: torch.Tensor) -> EnvState:
        """Step physics simulation."""
        # Apply action as force to agent-controlled object (last object)
        agent_obj = self.objects[-1]
        agent_obj[2] += action[0] * self.dt * 10  # Force x
        agent_obj[3] += action[1] * self.dt * 10  # Force y

        # Physics simulation
        for i in range(self.n_objects):
            if self.objects[i, 6] > 0.5:  # Fixed object
                continue

            # Gravity
            self.objects[i, 3] -= self.gravity * self.dt

            # Air resistance
            self.objects[i, 2] *= 0.99
            self.objects[i, 3] *= 0.99

            # Update position
            self.objects[i, 0] += self.objects[i, 2] * self.dt
            self.objects[i, 1] += self.objects[i, 3] * self.dt

            # Floor collision
            if self.objects[i, 1] < -0.5 + self.objects[i, 5]:
                self.objects[i, 1] = -0.5 + self.objects[i, 5]
                self.objects[i, 3] = -0.6 * self.objects[i, 3]  # Bounce

        # Object-object collisions
        for i in range(self.n_objects):
            for j in range(i + 1, self.n_objects):
                dx = self.objects[j, 0] - self.objects[i, 0]
                dy = self.objects[j, 1] - self.objects[i, 1]
                dist = torch.sqrt(dx * dx + dy * dy + 1e-8)
                min_dist = self.objects[i, 5] + self.objects[j, 5]

                if dist < min_dist:
                    # Elastic collision
                    nx = dx / dist
                    ny = dy / dist

                    dvx = self.objects[i, 2] - self.objects[j, 2]
                    dvy = self.objects[i, 3] - self.objects[j, 3]
                    dvn = dvx * nx + dvy * ny

                    if dvn > 0:  # Objects approaching
                        m1 = self.objects[i, 4]
                        m2 = self.objects[j, 4]
                        if self.objects[i, 6] < 0.5:
                            self.objects[i, 2] -= (2 * m2 / (m1 + m2)) * dvn * nx
                            self.objects[i, 3] -= (2 * m2 / (m1 + m2)) * dvn * ny
                        if self.objects[j, 6] < 0.5:
                            self.objects[j, 2] += (2 * m1 / (m1 + m2)) * dvn * nx
                            self.objects[j, 3] += (2 * m1 / (m1 + m2)) * dvn * ny

        return self._make_state()

    def _make_state(self) -> EnvState:
        """Create EnvState from current simulation state."""
        # Flatten objects as observation
        obs = self.objects.flatten()[:self.obs_dim].clone()
        if obs.shape[0] < self.obs_dim:
            obs = torch.cat([obs, torch.zeros(self.obs_dim - obs.shape[0])])

        # Proprioceptive: agent object state
        proprio = torch.zeros(self.proprio_dim)
        proprio[:self.state_dim] = self.objects[-1, :self.proprio_dim] \
            if self.state_dim >= self.proprio_dim else self.objects[-1]

        # Exteroceptive: nearest objects
        extero = torch.zeros(self.extero_dim)
        for i in range(min(self.n_objects - 1, self.extero_dim // 4)):
            extero[i * 4] = self.objects[i, 0]
            extero[i * 4 + 1] = self.objects[i, 1]
            extero[i * 4 + 2] = self.objects[i, 2]
            extero[i * 4 + 3] = self.objects[i, 3]

        return EnvState(
            objects=self.objects.clone(),
            relations=torch.zeros(self.n_objects, self.n_objects),
            observation=obs,
            proprio=proprio,
            extero=extero,
        )

    def violation_of_expectation_test(self, perturbed_state: torch.Tensor) -> float:
        """
        IntPhys-style violation of expectation test.
        
        Show agent normal physics, then a perturbed scene.
        If agent has intuitive physics, free energy should spike on perturbation.
        """
        # Normal prediction
        normal_state = self._make_state()
        return (perturbed_state - normal_state.observation).norm().item()


class FluidEnvironment:
    """
    Fluid dynamics environment for testing far transfer.
    
    Learns: pressure differential → flow, container → holds fluid,
    pipe → channels flow.
    """

    def __init__(
        self,
        grid_size: int = 8,
        obs_dim: int = 32,
        action_dim: int = 4,
    ):
        self.grid_size = grid_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Fluid state: pressure at each grid cell
        self.pressure = torch.zeros(grid_size, grid_size)
        self.temperature = torch.ones(grid_size, grid_size) * 300  # Kelvin

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset with random pressure configuration."""
        self.pressure = torch.rand(self.grid_size, self.grid_size) * 10
        return self._get_obs(), {"domain": "fluid"}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, dict]:
        """
        Action: apply pressure change at a location.
        
        Fluid flows from high to low pressure.
        This is the physics that should transfer to heat and circuits.
        """
        # Action: (x, y, delta_pressure, 0)
        x = int(torch.clamp(action[0], 0, self.grid_size - 1).item())
        y = int(torch.clamp(action[1], 0, self.grid_size - 1).item())
        delta = action[2].item()

        self.pressure[x, y] += delta

        # Diffusion: pressure flows from high to low
        new_pressure = self.pressure.clone()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbors = 0.0
                count = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                        neighbors += self.pressure[ni, nj]
                        count += 1
                if count > 0:
                    # Flow proportional to pressure gradient
                    new_pressure[i, j] += 0.1 * (neighbors / count - self.pressure[i, j])

        self.pressure = new_pressure
        obs = self._get_obs()
        reward = -self.pressure.var().item()  # Reward for equalizing pressure

        return obs, reward, {"domain": "fluid", "pressure_var": self.pressure.var().item()}

    def _get_obs(self) -> torch.Tensor:
        """Get flattened pressure as observation."""
        flat = self.pressure.flatten()
        if flat.shape[0] < self.obs_dim:
            flat = torch.cat([flat, torch.zeros(self.obs_dim - flat.shape[0])])
        return flat[:self.obs_dim]


class HeatEnvironment:
    """
    Heat transfer environment for testing far transfer.
    
    Heat flows from hot to cold — same relational structure as
    fluid pressure, but different surface features.
    """

    def __init__(
        self,
        grid_size: int = 8,
        obs_dim: int = 32,
        action_dim: int = 4,
    ):
        self.grid_size = grid_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.temperature = torch.ones(grid_size, grid_size) * 300

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset with random temperature configuration."""
        self.temperature = torch.rand(self.grid_size, self.grid_size) * 200 + 200
        return self._get_obs(), {"domain": "heat"}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, dict]:
        """
        Action: apply heat at a location.
        
        Heat flows from hot to cold — isomorphic to pressure → flow.
        """
        x = int(torch.clamp(action[0], 0, self.grid_size - 1).item())
        y = int(torch.clamp(action[1], 0, self.grid_size - 1).item())
        delta = action[2].item() * 50  # Scale to temperature

        self.temperature[x, y] += delta

        # Heat diffusion (Fourier's law)
        new_temp = self.temperature.clone()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbors = 0.0
                count = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                        neighbors += self.temperature[ni, nj]
                        count += 1
                if count > 0:
                    new_temp[i, j] += 0.1 * (neighbors / count - self.temperature[i, j])

        self.temperature = new_temp
        obs = self._get_obs()
        reward = -self.temperature.var().item()

        return obs, reward, {"domain": "heat", "temp_var": self.temperature.var().item()}

    def _get_obs(self) -> torch.Tensor:
        flat = self.temperature.flatten()
        if flat.shape[0] < self.obs_dim:
            flat = torch.cat([flat, torch.zeros(self.obs_dim - flat.shape[0])])
        return flat[:self.obs_dim]


class CircuitEnvironment:
    """
    Electrical circuit environment for testing far transfer.
    
    Current flows from high voltage to low — isomorphic to
    pressure → flow and temperature → heat transfer.
    """

    def __init__(
        self,
        grid_size: int = 8,
        obs_dim: int = 32,
        action_dim: int = 4,
    ):
        self.grid_size = grid_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.voltage = torch.zeros(grid_size, grid_size)
        self.resistance = torch.ones(grid_size, grid_size)

    def reset(self) -> Tuple[torch.Tensor, dict]:
        """Reset with random voltage configuration."""
        self.voltage = torch.rand(self.grid_size, self.grid_size) * 12
        self.resistance = torch.ones(self.grid_size, grid_size) + torch.rand(
            self.grid_size, self.grid_size
        ) * 0.5
        return self._get_obs(), {"domain": "circuit"}

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, dict]:
        """
        Action: apply voltage at a location.
        
        Current flows from high to low voltage (Ohm's law).
        Isomorphic to pressure → flow.
        """
        x = int(torch.clamp(action[0], 0, self.grid_size - 1).item())
        y = int(torch.clamp(action[1], 0, self.grid_size - 1).item())
        delta = action[2].item() * 5

        self.voltage[x, y] += delta

        # Current flow (Ohm's law: I = V/R, current flows high → low)
        new_voltage = self.voltage.clone()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                neighbors = 0.0
                count = 0
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size and 0 <= nj < self.grid_size:
                        neighbors += self.voltage[ni, nj] / self.resistance[ni, nj]
                        count += 1
                if count > 0:
                    avg_neighbor = neighbors / count
                    new_voltage[i, j] += 0.1 * (avg_neighbor - self.voltage[i, j] / self.resistance[i, j])

        self.voltage = new_voltage
        obs = self._get_obs()
        reward = -self.voltage.var().item()

        return obs, reward, {"domain": "circuit", "voltage_var": self.voltage.var().item()}

    def _get_obs(self) -> torch.Tensor:
        flat = self.voltage.flatten()
        if flat.shape[0] < self.obs_dim:
            flat = torch.cat([flat, torch.zeros(self.obs_dim - flat.shape[0])])
        return flat[:self.obs_dim]
