from __future__ import annotations

import pytest
import torch

from axosim.architecture_frontier import (
    ArchitectureSpec,
    build_architecture_candidate,
)
from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
from axosim.block_forecast import (
    BlockForecastConfig,
    CausalBlockForecastModel,
)
from axosim.checkpoint import load_checkpoint, save_checkpoint


def _source_model() -> AxoPyTorchMamba:
    return AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=8,
            num_output=2,
            num_branch=4,
            num_synapse_per_branch=2,
            input_to_synapse_routing="neuronio_routing",
            model_dim=6,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
            expansion=2,
            morphology_ids=["morph-a", "morph-b"],
            morphology_embedding_scale=0.2,
            morphology_synapse_gain_scale=1.0,
            branch_nonlinearity="tanh",
            separate_heads=True,
            spike_voltage_coupling_scale=1.5,
            local_tcn_scale=1.0,
            local_tcn_mode="undilated",
            local_tcn_position="post",
        )
    )


@pytest.mark.parametrize("core_kind", ["gru", "mamba"])
def test_block_forecaster_predicts_native_shape_and_trains(
    core_kind,
):
    torch.manual_seed(4)
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind=core_kind,
            patch_size=4,
            keep_local_tcn=core_kind == "mamba",
        ),
    )
    x = torch.randn(2, 18, 8)
    morphology = torch.tensor([0, 1])

    output = model(x, morphology_indices=morphology)
    assert output.shape == (2, 18, 2)
    assert model.base_soma_prediction is not None
    assert model.base_soma_prediction.shape == (2, 18, 1)

    output[:, 4:].square().mean().backward()
    assert model.patch_input.weight.grad is not None
    assert model.trajectory_head.weight.grad is not None
    if core_kind == "gru":
        assert model.temporal_core is not None
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.temporal_core.parameters()
        )
    else:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.backbone.blocks.parameters()
        )


@pytest.mark.parametrize("core_kind", ["gru", "mamba"])
def test_block_forecaster_never_reads_target_patch(core_kind):
    torch.manual_seed(7)
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind=core_kind,
            patch_size=4,
            keep_local_tcn=False,
        ),
    ).eval()
    morphology = torch.tensor([0])
    control = torch.randn(1, 16, 8)
    changed = control.clone()
    changed[:, 4:8] += 100.0

    with torch.inference_mode():
        control_output = model(
            control,
            morphology_indices=morphology,
        )
        changed_output = model(
            changed,
            morphology_indices=morphology,
        )

    torch.testing.assert_close(
        control_output[:, 4:8],
        changed_output[:, 4:8],
    )
    assert not torch.allclose(
        control_output[:, 8:12],
        changed_output[:, 8:12],
    )


def test_block_forecaster_can_emit_distinct_values_inside_patch():
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind="gru",
            patch_size=4,
            keep_local_tcn=False,
        ),
    ).eval()
    with torch.no_grad():
        model.trajectory_head.weight.zero_()
        model.trajectory_head.bias.copy_(
            torch.arange(
                model.trajectory_head.bias.numel(),
                dtype=model.trajectory_head.bias.dtype,
            )
        )
    x = torch.zeros(1, 12, 8)

    with torch.inference_mode():
        output = model(x, morphology_indices=torch.tensor([0]))

    assert torch.unique(output[0, 4:8, 0]).numel() > 1
    assert torch.unique(output[0, 4:8, 1]).numel() > 1


def test_high_patch_sizes_cap_parameter_growth():
    source = _source_model()
    patch8 = CausalBlockForecastModel(
        source,
        BlockForecastConfig(
            core_kind="gru",
            patch_size=8,
            max_patch_rank=8,
            max_trajectory_rank=8,
            keep_local_tcn=False,
        ),
    )
    patch32 = CausalBlockForecastModel(
        source,
        BlockForecastConfig(
            core_kind="gru",
            patch_size=32,
            max_patch_rank=8,
            max_trajectory_rank=8,
            keep_local_tcn=False,
        ),
    )
    count8 = sum(parameter.numel() for parameter in patch8.parameters())
    count32 = sum(parameter.numel() for parameter in patch32.parameters())

    assert patch8.patch_rank == patch32.patch_rank == 8
    assert patch8.trajectory_rank == patch32.trajectory_rank == 8
    assert count32 - count8 == 24 * 8 * 2


def test_capped_patch_input_starts_from_latest_rank_reconstruction():
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind="gru",
            patch_size=16,
            max_patch_rank=8,
        ),
    )
    coefficients = torch.randn(2, model.patch_rank, model.config.model_dim)
    hidden = torch.einsum(
        "brw,pr->bpw",
        coefficients,
        model.patch_basis.detach(),
    )

    token = model._pack_hidden(hidden)

    torch.testing.assert_close(token[:, 0], hidden[:, -1])


def test_sparse_voltage_decoder_preserves_anchors_and_trains():
    template = tuple(torch.linspace(-0.2, 0.2, 14).tolist())
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind="gru",
            patch_size=4,
            voltage_anchor_count=2,
            event_template=template,
        ),
    )
    x = torch.randn(2, 16, 8)

    output = model(x, morphology_indices=torch.tensor([0, 1]))

    assert output.shape == (2, 16, 2)
    output[:, 4:].square().mean().backward()
    assert model.voltage_anchor_head is not None
    assert model.voltage_anchor_head.weight.grad is not None
    assert model.event_template_kernel is not None
    assert model.event_template_center is not None
    assert model.event_template_center.grad is not None
    assert model.trajectory_head.weight.grad is not None


def test_sparse_voltage_materialization_copies_predicted_anchors():
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind="gru",
            patch_size=4,
            voltage_anchor_count=2,
            event_template=tuple(torch.zeros(14).tolist()),
        ),
    )
    spike_logits = torch.randn(2, 16)
    anchors = torch.randn(2, 3, 2)

    voltage = model._materialize_sparse_voltage(
        spike_logits,
        anchors,
    )

    actual = voltage.index_select(2, model.voltage_anchor_offsets)
    torch.testing.assert_close(actual, anchors)


def test_hard_event_template_uses_binary_forward_and_soft_gradient():
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind="gru",
            patch_size=4,
            voltage_anchor_count=2,
            event_template=(0.0, 1.0),
            event_template_center_init=0.0,
            event_template_hard=True,
        ),
    ).train()
    spike_logits = torch.tensor([[-1.0, 1.0]], requires_grad=True)

    voltage = model._event_template_voltage(spike_logits)

    torch.testing.assert_close(voltage.detach(), torch.tensor([[0.0, 1.0]]))
    voltage.sum().backward()
    assert spike_logits.grad is not None
    assert torch.all(spike_logits.grad > 0)


def test_sparse_voltage_head_reduces_parameters():
    source = _source_model()
    dense = CausalBlockForecastModel(
        source,
        BlockForecastConfig(core_kind="gru", patch_size=4),
    )
    sparse = CausalBlockForecastModel(
        source,
        BlockForecastConfig(
            core_kind="gru",
            patch_size=4,
            voltage_anchor_count=2,
            event_template=tuple(torch.zeros(14).tolist()),
        ),
    )

    dense_count = sum(parameter.numel() for parameter in dense.parameters())
    sparse_count = sum(parameter.numel() for parameter in sparse.parameters())

    assert sparse_count < dense_count


@pytest.mark.parametrize("core_kind", ["gru", "mamba"])
def test_architecture_frontier_builds_block_forecaster(core_kind):
    candidate = build_architecture_candidate(
        _source_model(),
        ArchitectureSpec(
            candidate_id=f"{core_kind}-block4",
            task="regression",
            temporal_core=core_kind,
            patch_size=4,
            block_forecast=True,
            keep_local_tcn=core_kind == "mamba",
        ),
    )

    assert isinstance(candidate, CausalBlockForecastModel)
    assert candidate.patch_size == 4
    assert candidate.forecast_config.core_kind == core_kind


@pytest.mark.parametrize("core_kind", ["gru", "mamba"])
def test_block_forecast_checkpoint_round_trips(tmp_path, core_kind):
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind=core_kind,
            patch_size=4,
            keep_local_tcn=core_kind == "mamba",
        ),
    )
    path = tmp_path / f"{core_kind}.pt"
    x = torch.randn(2, 12, 8)
    morphology = torch.tensor([0, 1])
    with torch.inference_mode():
        expected = model(x, morphology_indices=morphology)

    save_checkpoint(model, path, metadata={"experiment": "block"})
    loaded, metadata = load_checkpoint(path)
    with torch.inference_mode():
        actual = loaded(x, morphology_indices=morphology)

    assert isinstance(loaded, CausalBlockForecastModel)
    assert loaded.forecast_config == model.forecast_config
    assert metadata["experiment"] == "block"
    torch.testing.assert_close(actual, expected)


def test_block_forecast_checkpoint_ignores_retired_config_fields(
    tmp_path,
):
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind="mamba",
            patch_size=2,
            keep_local_tcn=True,
        ),
    )
    path = tmp_path / "historical.pt"
    save_checkpoint(model, path)
    payload = torch.load(path, weights_only=False)
    payload["config"]["forecast"].update(
        {
            "context_trajectory_skip": False,
            "mamba_time_scale": 1.0,
        }
    )
    torch.save(payload, path)

    loaded, _ = load_checkpoint(path)

    assert isinstance(loaded, CausalBlockForecastModel)
    assert loaded.forecast_config == model.forecast_config


def test_sparse_voltage_checkpoint_round_trips(tmp_path):
    model = CausalBlockForecastModel(
        _source_model(),
        BlockForecastConfig(
            core_kind="gru",
            patch_size=4,
            voltage_anchor_count=2,
            event_template=tuple(torch.linspace(-0.2, 0.2, 14).tolist()),
        ),
    ).eval()
    path = tmp_path / "sparse.pt"
    inputs = torch.randn(2, 16, 8)
    morphology = torch.tensor([0, 1])
    with torch.inference_mode():
        expected = model(inputs, morphology_indices=morphology)
    save_checkpoint(model, path)

    loaded, _ = load_checkpoint(path)
    with torch.inference_mode():
        actual = loaded(inputs, morphology_indices=morphology)

    assert isinstance(loaded, CausalBlockForecastModel)
    assert loaded.forecast_config == model.forecast_config
    torch.testing.assert_close(actual, expected)
