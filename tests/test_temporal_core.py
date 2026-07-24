from __future__ import annotations

import pytest
import torch

from axosim.architecture_frontier import (
    ArchitectureSpec,
    architecture_evidence,
    build_architecture_candidate,
    load_legacy_gru_candidate,
)
from axosim.autoresearch import model_population_memory_metrics
from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
from axosim.checkpoint import load_checkpoint, save_checkpoint
from axosim.temporal_core import (
    AxoTemporalCoreConfig,
    AxoTemporalModel,
    create_temporal_model,
    migrate_legacy_gru_state_dict,
)


def _source_model() -> AxoPyTorchMamba:
    config = AxoMambaConfig(
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
    return AxoPyTorchMamba(config)


@pytest.mark.parametrize("core_kind", ["gru", "branch_rnn"])
def test_temporal_model_reuses_selected_front_end_and_heads(core_kind):
    torch.manual_seed(7)
    source = _source_model()
    model = create_temporal_model(
        source,
        AxoTemporalCoreConfig(
            kind=core_kind,
            keep_local_tcn=True,
        ),
    )
    x = torch.randn(2, 9, 8)
    morphology = torch.tensor([0, 1])

    output = model(x, morphology_indices=morphology)
    assert output.shape == (2, 9, 2)
    assert model.num_input == source.num_input
    assert model.num_output == source.num_output
    assert not any(name.startswith("backbone.blocks.") for name, _ in model.named_parameters())
    assert model.backbone.morphology_embedding is not None
    assert model.backbone.local_tcn_out is not None

    output.square().mean().backward()
    assert model.backbone.branch_projection.weight.grad is not None
    assert model.backbone.spike_readout is not None
    assert model.backbone.spike_readout.weight.grad is not None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in model.temporal_core.parameters()
    )


def test_temporal_model_reports_core_specific_state_and_manifest():
    source = _source_model()
    gru = create_temporal_model(
        source,
        AxoTemporalCoreConfig(kind="gru"),
    )
    branch = create_temporal_model(
        source,
        AxoTemporalCoreConfig(
            kind="branch_rnn",
            branch_memory_units=5,
            branch_hidden_units=7,
        ),
    )

    assert gru.recurrent_state_bytes(dtype=torch.float32) == 6 * 4
    assert branch.recurrent_state_bytes(dtype=torch.float32) == (6 + 5) * 4
    assert gru.component_manifest()["temporal_core"] == "gru"
    assert branch.component_manifest()["temporal_core"] == "branch_rnn"
    assert branch.component_manifest()["shared_axomamba_front_end"]
    assert branch.component_manifest()["shared_axomamba_heads"]
    assert not branch.component_manifest()["local_tcn_retained"]

    memory = model_population_memory_metrics(gru)
    assert memory["per_neuron_streaming_state_bytes"] == 6 * 4
    assert memory["streaming_state_components_bytes"] == {
        "temporal_core": 6 * 4,
    }


@pytest.mark.parametrize("core_kind", ["gru", "branch_rnn"])
def test_temporal_model_checkpoint_round_trips(tmp_path, core_kind):
    torch.manual_seed(11)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(
            kind=core_kind,
            branch_memory_units=5,
            branch_hidden_units=7,
        ),
    ).eval()
    x = torch.randn(2, 7, 8)
    morphology = torch.tensor([0, 1])
    expected = model(x, morphology_indices=morphology)
    path = tmp_path / f"{core_kind}.pt"

    save_checkpoint(model, path, metadata={"iteration": 184})
    loaded, metadata = load_checkpoint(path)
    actual = loaded(x, morphology_indices=morphology)

    assert isinstance(loaded, AxoTemporalModel)
    assert loaded.temporal_config.kind == core_kind
    assert metadata["iteration"] == 184
    torch.testing.assert_close(actual, expected)


def test_mamba_factory_keeps_existing_model_and_rejects_unknown_core():
    source = _source_model()

    assert create_temporal_model(
        source,
        AxoTemporalCoreConfig(kind="mamba"),
    ) is source
    with pytest.raises(ValueError, match="temporal core"):
        AxoTemporalCoreConfig(kind="lstm")


def test_legacy_gru_state_migration_is_exact():
    torch.manual_seed(13)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru"),
    )
    unified = model.state_dict()
    legacy = {}
    for name, value in unified.items():
        if name == "temporal_core.residual_scale":
            target = "residual_scale"
        elif name.startswith("temporal_core.pre_norm."):
            target = "pre_gru_norm." + name.removeprefix(
                "temporal_core.pre_norm."
            )
        elif name.startswith("temporal_core.gru."):
            target = "gru." + name.removeprefix("temporal_core.gru.")
        else:
            target = name
        legacy[target] = value.clone()

    migrated = migrate_legacy_gru_state_dict(legacy)
    assert migrated.keys() == unified.keys()
    for name in unified:
        torch.testing.assert_close(migrated[name], unified[name])


def test_architecture_frontier_loads_legacy_gru_and_records_evidence(
    tmp_path,
):
    torch.manual_seed(17)
    source = _source_model()
    spec = ArchitectureSpec(
        candidate_id="spike_gru",
        task="spike",
        temporal_core="gru",
    )
    model = build_architecture_candidate(source, spec)
    assert isinstance(model, AxoTemporalModel)
    unified = model.state_dict()
    legacy = {}
    for name, value in unified.items():
        if name == "temporal_core.residual_scale":
            target = "residual_scale"
        elif name.startswith("temporal_core.pre_norm."):
            target = "pre_gru_norm." + name.removeprefix(
                "temporal_core.pre_norm."
            )
        elif name.startswith("temporal_core.gru."):
            target = "gru." + name.removeprefix("temporal_core.gru.")
        else:
            target = name
        legacy[target] = value.clone()
    checkpoint = tmp_path / "legacy-gru.pt"
    torch.save(
        {
            "state_dict": legacy,
            "candidate_id": "spike_gru",
        },
        checkpoint,
    )

    loaded, metadata = load_legacy_gru_candidate(
        source,
        checkpoint,
        candidate_id=spec.candidate_id,
        task=spec.task,
    )
    evidence = architecture_evidence(loaded, spec)

    assert metadata["candidate_id"] == "spike_gru"
    assert evidence["component_manifest"]["temporal_core"] == "gru"
    assert (
        evidence["population_memory"][
            "per_neuron_streaming_state_bytes"
        ]
        == 6 * 4
    )
