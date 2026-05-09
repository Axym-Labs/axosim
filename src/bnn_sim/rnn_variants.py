from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn as nn
from torch import jit

from bnn_sim.official.modeling_utils import (
    MLP,
    create_interlocking_indices,
    create_overlapping_window_indices,
    custom_tanh,
    inverse_scaled_sigmoid,
    scaled_sigmoid,
)


@dataclass(frozen=True)
class BranchTraceRNNConfig:
    num_input: int = 1278
    num_output: int = 2
    num_branch: int = 45
    num_synapse_per_branch: int = 100
    num_memory: int = 30
    lambda_value: float = 5.0
    mlp_num_layers: int = 1
    mlp_hidden_size: int | None = None
    mlp_activation: str = "relu"
    memory_tau_min: float = 1.0
    memory_tau_max: float = 150.0
    trace_tau_min: float = 2.0
    trace_tau_max: float = 80.0
    trace_banks: int = 3
    conductance_cell: bool = False
    membrane_tau: float = 40.0
    readout_hidden_size: int = 64
    w_s_value: float = 0.5
    delta_t: float = 1.0
    branch_gain: bool = False
    branch_nonlinearity: str = "none"
    branch_nonlinearity_scale_init: float = 0.1
    soma_multiplicative_gate: bool = False
    soma_multiplicative_scale_init: float = 0.1
    branch_readout_skip: bool = False
    branch_readout_skip_scale_init: float = 0.1


class BranchTraceRNN(jit.ScriptModule):
    """Branch-structured RNN variants for cheap state-construction probes."""

    __constants__ = [
        "num_input",
        "num_output",
        "num_branch",
        "num_synapse_per_branch",
        "num_synapse",
        "num_memory",
        "lambda_value",
        "mlp_num_layers",
        "mlp_hidden_size",
        "mlp_activation",
        "memory_tau_min",
        "memory_tau_max",
        "trace_tau_min",
        "trace_tau_max",
        "trace_banks",
        "conductance_cell",
        "membrane_tau",
        "readout_hidden_size",
        "w_s_value",
        "delta_t",
        "branch_gain_enabled",
        "branch_nonlinearity",
        "soma_multiplicative_gate_enabled",
        "branch_readout_skip_enabled",
    ]

    def __init__(self, config: BranchTraceRNNConfig):
        super().__init__()
        self.num_input = config.num_input
        self.num_output = config.num_output
        self.num_branch = config.num_branch
        self.num_synapse_per_branch = config.num_synapse_per_branch
        self.num_synapse = config.num_branch * config.num_synapse_per_branch
        self.num_memory = config.num_memory
        self.lambda_value = config.lambda_value
        self.mlp_num_layers = config.mlp_num_layers
        self.mlp_hidden_size = config.mlp_hidden_size or 2 * config.num_memory
        self.mlp_activation = config.mlp_activation
        self.memory_tau_min = config.memory_tau_min
        self.memory_tau_max = config.memory_tau_max
        self.trace_tau_min = config.trace_tau_min
        self.trace_tau_max = config.trace_tau_max
        self.trace_banks = 4 if config.conductance_cell else config.trace_banks
        self.conductance_cell = config.conductance_cell
        self.membrane_tau = config.membrane_tau
        self.readout_hidden_size = config.readout_hidden_size
        self.w_s_value = config.w_s_value
        self.delta_t = config.delta_t
        self.branch_gain_enabled = config.branch_gain
        self.branch_nonlinearity = config.branch_nonlinearity
        self.soma_multiplicative_gate_enabled = config.soma_multiplicative_gate
        self.branch_readout_skip_enabled = config.branch_readout_skip

        if self.num_synapse < self.num_input:
            raise ValueError("num_branch * num_synapse_per_branch must cover the routed inputs")
        if self.trace_banks <= 0:
            raise ValueError("trace_banks must be positive")
        if self.branch_nonlinearity not in {"none", "tanh"}:
            raise ValueError("branch_nonlinearity must be 'none' or 'tanh'")
        if config.branch_nonlinearity_scale_init <= 0:
            raise ValueError("branch_nonlinearity_scale_init must be positive")
        if config.soma_multiplicative_scale_init < 0:
            raise ValueError("soma_multiplicative_scale_init must be non-negative")
        if config.branch_readout_skip_scale_init < 0:
            raise ValueError("branch_readout_skip_scale_init must be non-negative")

        mlp_input_size = self.num_branch * self.trace_banks + self.num_memory
        self.mlp = MLP(
            mlp_input_size,
            self.mlp_hidden_size,
            self.num_memory,
            self.mlp_num_layers,
            self.mlp_activation,
        )
        self._proto_w_s = nn.Parameter(torch.full((self.trace_banks, self.num_synapse), self.w_s_value))

        readout_input_size = 2 * self.num_memory if self.conductance_cell else self.num_memory
        if self.conductance_cell:
            self.membrane_update = nn.Linear(mlp_input_size + self.num_memory, self.num_memory)
        else:
            self.membrane_update = nn.Identity()
        if self.readout_hidden_size > 0:
            self.readout_hidden = nn.Linear(readout_input_size, self.readout_hidden_size)
            self.w_y = nn.Linear(self.readout_hidden_size, self.num_output)
        else:
            self.readout_hidden = nn.Identity()
            self.w_y = nn.Linear(readout_input_size, self.num_output)
        self.soma_gate_readout = nn.Linear(readout_input_size, 1)
        self.soma_mod_readout = nn.Linear(readout_input_size, 1)
        nn.init.zeros_(self.soma_mod_readout.weight)
        nn.init.zeros_(self.soma_mod_readout.bias)
        self.soma_multiplicative_scale = nn.Parameter(torch.tensor(float(config.soma_multiplicative_scale_init)))
        self.branch_skip_readout = nn.Linear(self.num_branch * self.trace_banks, self.num_output)
        nn.init.zeros_(self.branch_skip_readout.weight)
        nn.init.zeros_(self.branch_skip_readout.bias)
        self.branch_readout_skip_scale = nn.Parameter(torch.tensor(float(config.branch_readout_skip_scale_init)))
        self.branch_gain = nn.Parameter(torch.ones(self.num_branch))
        self.branch_nonlinearity_log_scale = nn.Parameter(
            torch.full((self.num_branch,), math.log(config.branch_nonlinearity_scale_init))
        )
        self.branch_nonlinearity_bias = nn.Parameter(torch.zeros(self.num_branch))

        proto_tau_m = torch.logspace(
            math.log10(self.memory_tau_min + 1e-6),
            math.log10(self.memory_tau_max - 1e-6),
            self.num_memory,
        )
        proto_tau_m = inverse_scaled_sigmoid(proto_tau_m, self.memory_tau_min, self.memory_tau_max)
        self._proto_tau_m = nn.Parameter(proto_tau_m, requires_grad=False)

        trace_tau = torch.logspace(
            math.log10(self.trace_tau_min),
            math.log10(self.trace_tau_max),
            self.trace_banks,
        )
        self.trace_tau = nn.Parameter(trace_tau, requires_grad=False)
        self._kappa_membrane = nn.Parameter(
            torch.tensor(math.exp(-self.delta_t / max(self.membrane_tau, 1e-6))),
            requires_grad=False,
        )

        routing_indices, valid_mask = self._create_input_to_synapse_indices()
        self.input_to_synapse_indices = nn.Parameter(routing_indices, requires_grad=False)
        self.valid_indices_mask = nn.Parameter(valid_mask, requires_grad=False)
        self.bank_synapse_mask = nn.Parameter(self._create_bank_synapse_mask(routing_indices), requires_grad=False)
        self.feature_sign = nn.Parameter(self._create_feature_sign(), requires_grad=False)

    def _create_input_to_synapse_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        interlocking_indices = create_interlocking_indices(self.num_input)
        overlapping_indices, valid_indices_mask = create_overlapping_window_indices(
            self.num_input, self.num_branch, self.num_synapse_per_branch
        )
        return interlocking_indices[overlapping_indices], valid_indices_mask.to(torch.float32)

    def _create_bank_synapse_mask(self, routing_indices: torch.Tensor) -> torch.Tensor:
        mask = torch.ones(self.trace_banks, self.num_synapse)
        if self.conductance_cell:
            excitatory = (routing_indices < self.num_input // 2).to(torch.float32)
            inhibitory = 1.0 - excitatory
            mask[0] = excitatory
            mask[1] = excitatory
            mask[2] = inhibitory
            mask[3] = inhibitory
        return mask

    def _create_feature_sign(self) -> torch.Tensor:
        sign = torch.ones(self.trace_banks)
        if self.conductance_cell:
            sign[2:] = -1.0
        return sign

    @property
    def tau_m(self):
        return scaled_sigmoid(self._proto_tau_m, self.memory_tau_min, self.memory_tau_max)

    @property
    def kappa_m(self):
        return torch.exp(-self.delta_t / torch.clamp(self.tau_m, min=1e-6))

    @property
    def kappa_trace(self):
        return torch.exp(-self.delta_t / torch.clamp(self.trace_tau, min=1e-6))

    @property
    def kappa_membrane(self):
        return self._kappa_membrane

    @property
    def w_s(self):
        return torch.relu(self._proto_w_s) * self.bank_synapse_mask

    @jit.script_method
    def dynamics(self, x, trace_prev, m_prev, v_prev, w_s, kappa_trace, kappa_m, kappa_membrane):
        batch_size, _ = x.shape
        routed = torch.index_select(x, 1, self.input_to_synapse_indices)
        routed = routed * self.valid_indices_mask
        weighted = routed.unsqueeze(1) * w_s.unsqueeze(0)
        drive = weighted.view(batch_size, self.trace_banks, self.num_branch, self.num_synapse_per_branch).sum(dim=-1)
        drive = drive.transpose(1, 2)
        trace_t = kappa_trace.view(1, 1, self.trace_banks) * trace_prev + drive
        signed_branch_features = trace_t * self.feature_sign.view(1, 1, self.trace_banks)
        if self.branch_gain_enabled:
            signed_branch_features = signed_branch_features * self.branch_gain.view(1, self.num_branch, 1)
        if self.branch_nonlinearity == "tanh":
            scale = torch.exp(self.branch_nonlinearity_log_scale).view(1, self.num_branch, 1)
            bias = self.branch_nonlinearity_bias.view(1, self.num_branch, 1)
            signed_branch_features = torch.tanh(scale * (signed_branch_features + bias)) / torch.clamp(scale, min=1e-6)
        branch_features = signed_branch_features.flatten(1)
        memory_input = torch.cat([branch_features, kappa_m * m_prev], dim=-1)
        delta_m_t = custom_tanh(self.mlp(memory_input))
        m_t = kappa_m * m_prev + self.lambda_value * (1 - kappa_m) * delta_m_t

        if self.conductance_cell:
            v_candidate = custom_tanh(self.membrane_update(torch.cat([branch_features, kappa_m * m_prev, m_t], dim=-1)))
            v_t = kappa_membrane * v_prev + (1 - kappa_membrane) * v_candidate
            readout_state = torch.cat([m_t, v_t], dim=-1)
        else:
            v_t = v_prev
            readout_state = m_t

        if self.readout_hidden_size > 0:
            y_t = self.w_y(torch.nn.functional.silu(self.readout_hidden(readout_state)))
        else:
            y_t = self.w_y(readout_state)
        if self.branch_readout_skip_enabled:
            y_t = y_t + self.branch_readout_skip_scale * self.branch_skip_readout(branch_features)
        if self.soma_multiplicative_gate_enabled:
            soma = y_t[:, 1:2]
            gate = torch.sigmoid(self.soma_gate_readout(readout_state))
            modulation = torch.tanh(self.soma_mod_readout(readout_state))
            soma = soma + self.soma_multiplicative_scale * gate * soma * modulation
            y_t = torch.cat([y_t[:, 0:1], soma], dim=1)
        return y_t, trace_t, m_t, v_t

    @jit.script_method
    def forward(self, X):
        batch_size, T, _ = X.shape
        w_s = self.w_s
        kappa_trace = self.kappa_trace
        kappa_m = self.kappa_m
        kappa_membrane = self.kappa_membrane
        trace_prev = torch.zeros(batch_size, self.num_branch, self.trace_banks, device=X.device)
        m_prev = torch.zeros(batch_size, self.num_memory, device=X.device)
        v_prev = torch.zeros(batch_size, self.num_memory, device=X.device)
        outputs = torch.empty(batch_size, T, self.num_output, device=X.device)
        for t in range(T):
            y_t, trace_prev, m_prev, v_prev = self.dynamics(
                X[:, t], trace_prev, m_prev, v_prev, w_s, kappa_trace, kappa_m, kappa_membrane
            )
            outputs[:, t, :] = y_t
        return outputs
