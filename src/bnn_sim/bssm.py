"""BSSM: Biophysically Structured State-Space Model.

Core module implementing a 6-state biophysical state-space model:
  [I_fast_E, I_slow_E, I_fast_I, I_slow_I, V_m, adaptation]

Timescales are learnable per state. State evolution follows
biophysical priors: exponential decay + input-driven accumulation.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class BiophysicalStateSpace(nn.Module):
    """6-state BSSM core.

    State vector z_t = [I_fast_E, I_slow_E, I_fast_I, I_slow_I, V_m, adaptation]

    Evolution (discretized Euler):
      I_fast_E[t]  = alpha_fast_E * I_fast_E[t-1] + (1-alpha_fast_E) * w_fast_E * E_input[t]
      I_slow_E[t]  = alpha_slow_E * I_slow_E[t-1] + (1-alpha_slow_E) * w_slow_E * E_input[t]
      I_fast_I[t]  = alpha_fast_I * I_fast_I[t-1] + (1-alpha_fast_I) * w_fast_I * I_input[t]
      I_slow_I[t]  = alpha_slow_I * I_slow_I[t-1] + (1-alpha_slow_I) * w_slow_I * I_input[t]
      V_m[t]       = f_V(V_m[t-1], I_fast_E, I_slow_E, I_fast_I, I_slow_I, adaptation)
      adaptation[t]= alpha_adapt * adaptation[t-1] + beta_adapt * spike[t-1]

    where alphas are derived from learnable taus:
      alpha = exp(-dt / tau)
    """

    def __init__(
        self,
        n_branches: int = 45,
        dt_ms: float = 1.0,
        init_tau_fast_ms: float = 3.0,
        init_tau_slow_ms: float = 50.0,
        init_tau_adapt_ms: float = 200.0,
        init_tau_membrane_ms: float = 20.0,
        tau_min_ms: float = 0.5,
        tau_max_ms: float = 1000.0,
    ):
        super().__init__()
        self.n_branches = n_branches
        self.dt_ms = dt_ms
        self.tau_min = tau_min_ms
        self.tau_max = tau_max_ms

        # Learnable timescales (one per state variable, shared across branches)
        # We parameterize as softplus to keep positive, then clamp
        self._log_tau_fast_E = nn.Parameter(torch.tensor(math.log(init_tau_fast_ms)))
        self._log_tau_slow_E = nn.Parameter(torch.tensor(math.log(init_tau_slow_ms)))
        self._log_tau_fast_I = nn.Parameter(torch.tensor(math.log(init_tau_fast_ms)))
        self._log_tau_slow_I = nn.Parameter(torch.tensor(math.log(init_tau_slow_ms)))
        self._log_tau_membrane = nn.Parameter(torch.tensor(math.log(init_tau_membrane_ms)))
        self._log_tau_adapt = nn.Parameter(torch.tensor(math.log(init_tau_adapt_ms)))

        # Learnable gains for current inputs (per branch)
        self.w_fast_E = nn.Parameter(torch.ones(n_branches) * 0.1)
        self.w_slow_E = nn.Parameter(torch.ones(n_branches) * 0.1)
        self.w_fast_I = nn.Parameter(torch.ones(n_branches) * 0.1)
        self.w_slow_I = nn.Parameter(torch.ones(n_branches) * 0.1)
        self.w_adapt = nn.Parameter(torch.ones(1) * 0.5)

        # Membrane integration parameters
        self.E_leak = nn.Parameter(torch.tensor(-65.0))  # mV
        self.E_exc = nn.Parameter(torch.tensor(0.0))     # mV
        self.E_inh = nn.Parameter(torch.tensor(-75.0))   # mV
        self.g_leak = nn.Parameter(torch.tensor(0.05))   # 1/MOhm

        # Spike threshold for adaptation
        self.spike_threshold = nn.Parameter(torch.tensor(-50.0))

    def _get_taus(self) -> dict[str, torch.Tensor]:
        """Get clipped positive timescales."""
        return {
            "fast_E": torch.clamp(self._log_tau_fast_E.exp(), self.tau_min, self.tau_max),
            "slow_E": torch.clamp(self._log_tau_slow_E.exp(), self.tau_min, self.tau_max),
            "fast_I": torch.clamp(self._log_tau_fast_I.exp(), self.tau_min, self.tau_max),
            "slow_I": torch.clamp(self._log_tau_slow_I.exp(), self.tau_min, self.tau_max),
            "membrane": torch.clamp(self._log_tau_membrane.exp(), self.tau_min, self.tau_max),
            "adapt": torch.clamp(self._log_tau_adapt.exp(), self.tau_min, self.tau_max),
        }

    def _get_alphas(self, taus: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute decay alphas from timescales."""
        dt = self.dt_ms
        return {k: torch.exp(-dt / v) for k, v in taus.items()}

    def forward(
        self,
        E_input: torch.Tensor,  # (batch, time, n_branches)
        I_input: torch.Tensor,  # (batch, time, n_branches)
    ) -> dict[str, torch.Tensor]:
        """Forward pass through BSSM.

        Returns dict with state trajectories and readout variables.
        """
        batch, time, n_branches = E_input.shape
        assert n_branches == self.n_branches

        taus = self._get_taus()
        alphas = self._get_alphas(taus)

        # Initialize states
        I_fast_E = torch.zeros(batch, n_branches, device=E_input.device, dtype=E_input.dtype)
        I_slow_E = torch.zeros(batch, n_branches, device=E_input.device, dtype=E_input.dtype)
        I_fast_I = torch.zeros(batch, n_branches, device=E_input.device, dtype=E_input.dtype)
        I_slow_I = torch.zeros(batch, n_branches, device=E_input.device, dtype=E_input.dtype)
        V_m = torch.full((batch,), self.E_leak.item(), device=E_input.device, dtype=E_input.dtype)
        adaptation = torch.zeros(batch, device=E_input.device, dtype=E_input.dtype)

        # Storage for trajectories
        I_fast_E_traj = []
        I_slow_E_traj = []
        I_fast_I_traj = []
        I_slow_I_traj = []
        V_m_traj = []
        adapt_traj = []
        spike_traj = []

        for t in range(time):
            # Current states evolve
            I_fast_E = alphas["fast_E"] * I_fast_E + (1 - alphas["fast_E"]) * self.w_fast_E * E_input[:, t, :]
            I_slow_E = alphas["slow_E"] * I_slow_E + (1 - alphas["slow_E"]) * self.w_slow_E * E_input[:, t, :]
            I_fast_I = alphas["fast_I"] * I_fast_I + (1 - alphas["fast_I"]) * self.w_fast_I * I_input[:, t, :]
            I_slow_I = alphas["slow_I"] * I_slow_I + (1 - alphas["slow_I"]) * self.w_slow_I * I_input[:, t, :]

            # Sum currents across branches
            total_E = I_fast_E.sum(dim=-1) + I_slow_E.sum(dim=-1)
            total_I = I_fast_I.sum(dim=-1) + I_slow_I.sum(dim=-1)

            # Membrane dynamics (simplified LIF-like with conductances)
            g_exc = torch.relu(total_E)  # excitatory conductance
            g_inh = torch.relu(total_I)  # inhibitory conductance

            I_leak = self.g_leak * (self.E_leak - V_m)
            I_exc = g_exc * (self.E_exc - V_m)
            I_inh = g_inh * (self.E_inh - V_m)
            I_adapt = adaptation * (self.E_leak - V_m)  # adaptation current

            tau_m = taus["membrane"]
            dV = (I_leak + I_exc + I_inh + I_adapt) / tau_m * self.dt_ms
            V_m_new = V_m + dV

            # Spike detection (for adaptation)
            spike = (V_m_new > self.spike_threshold).float()

            # Adaptation evolves
            adaptation_new = alphas["adapt"] * adaptation + (1 - alphas["adapt"]) * self.w_adapt * spike
            
            V_m = V_m_new
            adaptation = adaptation_new

            # Store
            I_fast_E_traj.append(I_fast_E)
            I_slow_E_traj.append(I_slow_E)
            I_fast_I_traj.append(I_fast_I)
            I_slow_I_traj.append(I_slow_I)
            V_m_traj.append(V_m)
            adapt_traj.append(adaptation)
            spike_traj.append(spike)

        return {
            "I_fast_E": torch.stack(I_fast_E_traj, dim=1),  # (batch, time, n_branches)
            "I_slow_E": torch.stack(I_slow_E_traj, dim=1),
            "I_fast_I": torch.stack(I_fast_I_traj, dim=1),
            "I_slow_I": torch.stack(I_slow_I_traj, dim=1),
            "V_m": torch.stack(V_m_traj, dim=1),  # (batch, time)
            "adaptation": torch.stack(adapt_traj, dim=1),
            "spike": torch.stack(spike_traj, dim=1),
        }


class BSSMNeuron(nn.Module):
    """Full BSSM neuron with branch routing and readout.

    Integrates BiophysicalStateSpace with:
    - Branch routing (like Branch-ELM)
    - Readout from V_m and adaptation to soma voltage
    - Spike probability head
    """

    def __init__(
        self,
        input_dim: int = 1278,
        n_branches: int = 45,
        branch_size: int | None = None,
        dt_ms: float = 1.0,
        use_branch_trace: bool = True,
        hidden_readout: int = 32,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_branches = n_branches
        self.branch_size = branch_size or (input_dim // n_branches)
        self.dt_ms = dt_ms
        self.use_branch_trace = use_branch_trace

        # Branch routing matrix (fixed, like Branch-ELM)
        self.register_buffer(
            "routing_matrix",
            self._create_routing_matrix(),
        )

        # Optional branch trace (exponential synaptic trace)
        if use_branch_trace:
            self.branch_trace_decay = nn.Parameter(torch.ones(n_branches) * 0.85)

        # BSSM core
        self.bssm = BiophysicalStateSpace(n_branches=n_branches, dt_ms=dt_ms)

        # Readout network
        self.readout = nn.Sequential(
            nn.Linear(2 + n_branches * 4, hidden_readout),  # V_m, adaptation + 4 current types
            nn.Tanh(),
            nn.Linear(hidden_readout, 2),  # [spike_logit, soma_voltage]
        )

        # Direct soma voltage readout (affine transform of V_m)
        self.soma_scale = nn.Parameter(torch.tensor(0.1))
        self.soma_bias = nn.Parameter(torch.tensor(-6.77))

    def _create_routing_matrix(self) -> torch.Tensor:
        """Create fixed branch routing matrix."""
        n_branches = self.n_branches
        branch_size = self.branch_size
        mat = torch.zeros(n_branches, self.input_dim)
        for b in range(n_branches):
            start = b * branch_size
            end = min(start + branch_size, self.input_dim)
            mat[b, start:end] = 1.0
        return mat

    def _route_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Route raw inputs to E/I branch currents.

        x: (batch, time, input_dim) with signed events (+1 excitatory, -1 inhibitory)
        Returns E_input, I_input: (batch, time, n_branches)
        """
        # Branch sum: (batch, time, n_branches)
        branch_sum = torch.einsum("bti,ni->btn", x, self.routing_matrix)

        # Separate E/I by sign
        E_input = torch.relu(branch_sum)   # positive inputs = excitatory
        I_input = torch.relu(-branch_sum)  # negative inputs = inhibitory

        return E_input, I_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        x: (batch, time, input_dim) with signed inputs
        Returns: (batch, time, 2) with [spike_logit, soma_voltage]
        """
        batch, time, _ = x.shape

        # Route to branches
        E_input, I_input = self._route_inputs(x)

        # Optional branch trace
        if self.use_branch_trace:
            # Simple exponential trace per branch
            E_traced = []
            I_traced = []
            e_trace = torch.zeros(batch, self.n_branches, device=x.device, dtype=x.dtype)
            i_trace = torch.zeros(batch, self.n_branches, device=x.device, dtype=x.dtype)
            for t in range(time):
                decay = torch.sigmoid(self.branch_trace_decay)
                e_trace = decay * e_trace + (1 - decay) * E_input[:, t, :]
                i_trace = decay * i_trace + (1 - decay) * I_input[:, t, :]
                E_traced.append(e_trace)
                I_traced.append(i_trace)
            E_input = torch.stack(E_traced, dim=1)
            I_input = torch.stack(I_traced, dim=1)

        # Run BSSM core
        states = self.bssm(E_input, I_input)

        # Flatten currents for readout
        currents = torch.cat([
            states["I_fast_E"].view(batch, time, -1),
            states["I_slow_E"].view(batch, time, -1),
            states["I_fast_I"].view(batch, time, -1),
            states["I_slow_I"].view(batch, time, -1),
        ], dim=-1)

        # Combine with V_m and adaptation
        readout_input = torch.cat([
            states["V_m"].unsqueeze(-1),
            states["adaptation"].unsqueeze(-1),
            currents,
        ], dim=-1)

        # Readout
        output = self.readout(readout_input)

        # Scale soma voltage to normalized units
        soma_out = output[:, :, 1] * self.soma_scale + self.soma_bias
        output = torch.stack([output[:, :, 0], soma_out], dim=-1)

        return output


if __name__ == "__main__":
    # Smoke test
    model = BSSMNeuron(input_dim=1278, n_branches=45)
    x = torch.randn(2, 100, 1278) * 0.1
    y = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Output range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"Learnable taus:")
    for name, tau in model.bssm._get_taus().items():
        print(f"  {name}: {tau.item():.2f} ms")
