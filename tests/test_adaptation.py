import pytest
import torch
from torch import nn
from torch.nn import functional as F

from axosim.adaptation import CudaGraphAdaptationStep


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA Graph adaptation requires CUDA",
)


def _update(model, optimizer, inputs, targets, *, clip_norm=None):
    prediction = model(inputs)
    loss = F.mse_loss(prediction, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    return loss


def test_cuda_graph_adaptation_matches_repeated_clipped_adam_updates() -> None:
    torch.manual_seed(19)
    captured_model = nn.Linear(4, 2).cuda()
    eager_model = nn.Linear(4, 2).cuda()
    eager_model.load_state_dict(captured_model.state_dict())
    initial = {
        key: value.detach().clone()
        for key, value in captured_model.state_dict().items()
    }
    batches = [
        (
            torch.randn(8, 4, device="cuda"),
            torch.randn(8, 2, device="cuda"),
        )
        for _ in range(3)
    ]
    captured_optimizer = torch.optim.Adam(
        captured_model.parameters(),
        lr=1e-3,
        capturable=True,
    )
    eager_optimizer = torch.optim.Adam(
        eager_model.parameters(),
        lr=1e-3,
    )

    step = CudaGraphAdaptationStep.capture(
        captured_model,
        captured_optimizer,
        batches[0][0],
        batches[0][1],
        F.mse_loss,
        gradient_transform=lambda: torch.nn.utils.clip_grad_norm_(
            captured_model.parameters(),
            0.5,
        ),
    )

    for key, value in captured_model.state_dict().items():
        torch.testing.assert_close(value, initial[key], rtol=0, atol=0)

    for inputs, targets in batches:
        eager_loss = _update(
            eager_model,
            eager_optimizer,
            inputs,
            targets,
            clip_norm=0.5,
        )
        captured_loss = step(inputs, targets, synchronize=True)

        torch.testing.assert_close(captured_loss, eager_loss)
    for captured, eager in zip(
        captured_model.parameters(),
        eager_model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(captured, eager, rtol=1e-5, atol=1e-7)


def test_cuda_graph_adaptation_rejects_shape_changes() -> None:
    model = nn.Linear(4, 2).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        capturable=True,
    )
    inputs = torch.randn(8, 4, device="cuda")
    targets = torch.randn(8, 2, device="cuda")
    step = CudaGraphAdaptationStep.capture(
        model,
        optimizer,
        inputs,
        targets,
        F.mse_loss,
    )

    with pytest.raises(ValueError, match="inputs must have shape"):
        step(torch.randn(7, 4, device="cuda"), targets)
