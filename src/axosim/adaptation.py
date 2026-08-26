"""Reusable fixed-shape CUDA Graph adaptation steps."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn


TensorLoss = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
GradientTransform = Callable[[], None]


def _clone_optimizer_state(
    optimizer: torch.optim.Optimizer,
) -> dict[nn.Parameter, dict[str, object]]:
    return {
        parameter: {
            key: (
                value.detach().clone()
                if isinstance(value, torch.Tensor)
                else deepcopy(value)
            )
            for key, value in state.items()
        }
        for parameter, state in optimizer.state.items()
    }


def _restore_optimizer_state(
    optimizer: torch.optim.Optimizer,
    snapshot: dict[nn.Parameter, dict[str, object]],
) -> None:
    for parameter, state in optimizer.state.items():
        expected = snapshot.get(parameter, {})
        for key, value in state.items():
            if key not in expected:
                if isinstance(value, torch.Tensor):
                    value.zero_()
                elif isinstance(value, (int, float)):
                    state[key] = type(value)(0)
                continue
            source = expected[key]
            if isinstance(value, torch.Tensor):
                assert isinstance(source, torch.Tensor)
                value.copy_(source)
            else:
                state[key] = deepcopy(source)


@dataclass
class CudaGraphAdaptationStep:
    """A captured full-forward, backward, and optimizer update."""

    module: nn.Module
    optimizer: torch.optim.Optimizer
    static_inputs: torch.Tensor
    static_targets: torch.Tensor
    graph: torch.cuda.CUDAGraph
    static_loss: torch.Tensor

    @classmethod
    def capture(
        cls,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        example_inputs: torch.Tensor,
        example_targets: torch.Tensor,
        loss_function: TensorLoss,
        *,
        warmup_steps: int = 3,
        gradient_transform: GradientTransform | None = None,
    ) -> "CudaGraphAdaptationStep":
        """Capture a fixed-shape adaptation update without consuming it."""

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA Graph adaptation requires CUDA")
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive")
        if not example_inputs.is_cuda or not example_targets.is_cuda:
            raise ValueError("example inputs and targets must be on CUDA")
        parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        if not parameters:
            raise ValueError("optimizer has no adaptation parameters")
        if any(not parameter.is_cuda for parameter in parameters):
            raise ValueError("all adaptation parameters must be on CUDA")

        module_state = {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }
        optimizer_state = _clone_optimizer_state(optimizer)
        static_inputs = torch.empty_like(example_inputs)
        static_targets = torch.empty_like(example_targets)
        static_inputs.copy_(example_inputs)
        static_targets.copy_(example_targets)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmup_steps):
                optimizer.zero_grad(set_to_none=False)
                prediction = module(static_inputs)
                loss = loss_function(prediction, static_targets)
                loss.backward()
                if gradient_transform is not None:
                    gradient_transform()
                optimizer.step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            optimizer.zero_grad(set_to_none=False)
            prediction = module(static_inputs)
            static_loss = loss_function(prediction, static_targets)
            static_loss.backward()
            if gradient_transform is not None:
                gradient_transform()
            optimizer.step()
        torch.cuda.synchronize()

        module.load_state_dict(module_state, strict=True)
        _restore_optimizer_state(optimizer, optimizer_state)
        optimizer.zero_grad(set_to_none=False)
        torch.cuda.synchronize()
        return cls(
            module=module,
            optimizer=optimizer,
            static_inputs=static_inputs,
            static_targets=static_targets,
            graph=graph,
            static_loss=static_loss,
        )

    def __call__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        synchronize: bool = False,
    ) -> torch.Tensor:
        expected_inputs = tuple(self.static_inputs.shape)
        expected_targets = tuple(self.static_targets.shape)
        if tuple(inputs.shape) != expected_inputs:
            raise ValueError(f"inputs must have shape {expected_inputs}")
        if tuple(targets.shape) != expected_targets:
            raise ValueError(f"targets must have shape {expected_targets}")
        if inputs.dtype != self.static_inputs.dtype:
            raise ValueError(
                f"inputs must have dtype {self.static_inputs.dtype}"
            )
        if targets.dtype != self.static_targets.dtype:
            raise ValueError(
                f"targets must have dtype {self.static_targets.dtype}"
            )
        self.static_inputs.copy_(inputs, non_blocking=True)
        self.static_targets.copy_(targets, non_blocking=True)
        self.graph.replay()
        if synchronize:
            torch.cuda.synchronize()
        return self.static_loss.detach().clone()
