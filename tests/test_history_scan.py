"""Consumer contracts for independently supplied Mamba histories."""

import copy

import pytest
import torch

import axosim
from axosim.axomamba import AxoMamba, AxoMambaConfig, AxoPyTorchMamba


def _config(*, conditioned=True):
    return AxoMambaConfig(
        num_input=8,
        num_output=2,
        num_branch=2,
        num_synapse_per_branch=4,
        model_dim=4,
        num_layers=1,
        state_dim=2,
        conv_kernel=2,
        expansion=2,
        dropout=0.0,
        morphology_ids=["a", "b"] if conditioned else None,
        morphology_embedding_scale=0.3 if conditioned else None,
    )


def _model(*, conditioned=True):
    torch.manual_seed(41)
    return AxoPyTorchMamba(_config(conditioned=conditioned))


def test_one_neuron_history_matches_native_forward_and_reports_sequential_backend():
    model = _model()
    inputs = torch.randn(1, 11, 8)
    morphology = torch.tensor([1])
    expected = model(inputs, morphology_indices=morphology)

    result = axosim.scan_histories(
        model, inputs, morphology_indices=morphology, require_parallel=False
    )

    assert isinstance(result, axosim.HistoryScanResult)
    assert isinstance(result.backend, axosim.HistoryScanBackend)
    assert result.prediction.shape == (1, 11, 2)
    assert result.prediction.dtype == inputs.dtype
    assert result.prediction.device == inputs.device
    torch.testing.assert_close(result.prediction, expected)
    assert result.backend.name == "pytorch-sequential-reference"
    assert result.backend.parallel is False
    assert "sequential" in result.backend.reason.lower()


def test_neuron_batching_preserves_order_morphology_weights_and_caller_mode():
    model = _model()
    inputs = torch.randn(5, 13, 8)
    morphology = torch.tensor([1, 0, 0, 1, 0])
    expected = model(inputs, morphology_indices=morphology)
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    assert model.training

    result = axosim.scan_histories(
        model,
        inputs,
        morphology_indices=morphology,
        neuron_batch_size=2,
        require_parallel=False,
    )

    torch.testing.assert_close(result.prediction, expected)
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)


def test_batched_scan_preserves_full_history_input_and_shared_weight_gradients():
    reference = _model()
    scanned = copy.deepcopy(reference)
    original = torch.randn(5, 17, 8)
    direct_inputs = original.clone().requires_grad_()
    scan_inputs = original.clone().requires_grad_()
    morphology = torch.tensor([0, 1, 1, 0, 1])
    direct = reference(direct_inputs, morphology_indices=morphology)
    direct[:, -1].square().sum().backward()

    result = axosim.scan_histories(
        scanned,
        scan_inputs,
        morphology_indices=morphology,
        neuron_batch_size=2,
        require_parallel=False,
    )
    result.prediction[:, -1].square().sum().backward()

    torch.testing.assert_close(result.prediction, direct)
    torch.testing.assert_close(scan_inputs.grad, direct_inputs.grad, rtol=2e-5, atol=2e-7)
    assert scan_inputs.grad[:, :3].abs().sum() > 0
    for (name, parameter), (_, expected_parameter) in zip(
        scanned.named_parameters(), reference.named_parameters()
    ):
        assert (parameter.grad is None) == (expected_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, expected_parameter.grad, rtol=2e-5, atol=2e-7, msg=name
            )


def test_parallel_is_required_by_default_instead_of_silently_using_reference():
    model = _model(conditioned=False)
    with pytest.raises(RuntimeError, match="require_parallel.*sequential"):
        axosim.scan_histories(model, torch.randn(1, 8, 8))


@pytest.mark.parametrize(
    "inputs, message",
    [
        (torch.zeros(7, 8), "shape"),
        (torch.zeros(0, 7, 8), "non-empty"),
        (torch.zeros(1, 0, 8), "non-empty"),
        (torch.zeros(1, 7, 9), "num_input"),
        (torch.ones(1, 7, 8, dtype=torch.long), "floating"),
    ],
)
def test_invalid_supplied_history_shape_or_dtype_fails_clearly(inputs, message):
    with pytest.raises(ValueError, match=message):
        axosim.scan_histories(_model(conditioned=False), inputs, require_parallel=False)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_neuron_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match="neuron_batch_size.*positive integer"):
        axosim.scan_histories(
            _model(conditioned=False),
            torch.zeros(1, 7, 8),
            neuron_batch_size=batch_size,
            require_parallel=False,
        )


@pytest.mark.parametrize(
    "morphology, message",
    [
        (None, "morphology_indices.*required"),
        (torch.tensor([[0], [1]]), "morphology_indices.*shape"),
        (torch.tensor([0]), "morphology_indices.*shape"),
        (torch.tensor([0.0, 1.0]), "morphology_indices.*integer"),
        (torch.tensor([-1, 0]), "morphology_indices.*range"),
        (torch.tensor([0, 2]), "morphology_indices.*range"),
    ],
)
def test_morphology_indices_cannot_be_dropped_truncated_or_misaligned(morphology, message):
    with pytest.raises(ValueError, match=message):
        axosim.scan_histories(
            _model(),
            torch.randn(2, 9, 8),
            morphology_indices=morphology,
            require_parallel=False,
        )


def test_scan_rejects_models_without_a_mamba_history_backend():
    with pytest.raises(TypeError, match="Mamba"):
        axosim.scan_histories(torch.nn.Linear(8, 2), torch.zeros(1, 7, 8))


def test_unconditioned_model_rejects_morphology_instead_of_ignoring_it():
    with pytest.raises(ValueError, match="unconditioned"):
        axosim.scan_histories(
            _model(conditioned=False),
            torch.zeros(1, 7, 8),
            morphology_indices=torch.tensor([0]),
            require_parallel=False,
        )


def test_repeated_calls_start_independent_histories_without_streaming_state():
    model = _model(conditioned=False).eval()
    inputs = torch.randn(2, 19, 8)
    first = axosim.scan_histories(model, inputs, require_parallel=False)
    axosim.scan_histories(model, inputs.flip(1), require_parallel=False)
    repeated = axosim.scan_histories(model, inputs, require_parallel=False)
    torch.testing.assert_close(first.prediction, repeated.prediction, rtol=0, atol=0)
    assert not model.training


@pytest.mark.skipif(not torch.cuda.is_available(), reason="official fused scan requires CUDA")
def test_official_cuda_history_scan_preserves_output_gradient_and_weights():
    pytest.importorskip("mamba_ssm")
    torch.manual_seed(41)
    reference = AxoMamba(_config()).cuda().eval()
    scanned = copy.deepcopy(reference)
    inputs = torch.randn(3, 37, 8, device="cuda")
    direct_inputs = inputs.clone().requires_grad_()
    scan_inputs = inputs.clone().requires_grad_()
    morphology = torch.tensor([1, 0, 1], device="cuda")
    before = {key: value.clone() for key, value in scanned.state_dict().items()}
    direct = reference(direct_inputs, morphology_indices=morphology)
    direct.square().sum().backward()

    result = axosim.scan_histories(
        scanned, scan_inputs, morphology_indices=morphology, neuron_batch_size=2
    )
    result.prediction.square().sum().backward()

    assert result.backend.name == "official-mamba1-fused"
    assert result.backend.parallel is True
    torch.testing.assert_close(result.prediction, direct, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(scan_inputs.grad, direct_inputs.grad, rtol=5e-4, atol=2e-5)
    for (name, parameter), (_, expected_parameter) in zip(
        scanned.named_parameters(), reference.named_parameters()
    ):
        assert (parameter.grad is None) == (expected_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, expected_parameter.grad, rtol=5e-4, atol=2e-5, msg=name
            )
    for key, value in scanned.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="official fused scan requires CUDA")
def test_official_cuda_disabled_fast_path_and_portable_cuda_fail_closed():
    pytest.importorskip("mamba_ssm")
    inputs = torch.randn(1, 9, 8, device="cuda")
    model = AxoMamba(_config(conditioned=False)).cuda()
    model.blocks[0].mamba.use_fast_path = False
    with pytest.raises(RuntimeError, match="use_fast_path"):
        axosim.scan_histories(model, inputs)
    portable = _model(conditioned=False).cuda()
    with pytest.raises(RuntimeError, match="require_parallel.*sequential"):
        axosim.scan_histories(portable, inputs)
