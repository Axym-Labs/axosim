from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

from axosim import temporal_core
from axosim.architecture_frontier import (
    ArchitectureSpec,
    architecture_evidence,
    build_architecture_candidate,
    load_legacy_gru_candidate,
    transfer_compatible_state,
)
from axosim.autoresearch import (
    model_population_memory_metrics,
    set_trainable_parameter_names,
)
from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
from axosim.checkpoint import load_checkpoint, save_checkpoint
from axosim.deployment import NeuronBehaviorAdaptation
from axosim.population import (
    AdaptedGRUBranchPopulationRunner,
    GroupedGRUPopulationRunner,
    GRUBranchPopulationRunner,
    GRUPopulationRunner,
    pack_neuron_behavior_adapter,
)
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


def test_gru_behavior_adapter_is_target_specific_and_keeps_memory_shared():
    torch.manual_seed(12)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(
            kind="gru",
            behavior_adapter_morphology_id="morph-b",
        ),
    ).eval()
    adapter = model.behavior_adapter
    assert adapter is not None
    assert adapter.parameter_counts == {
        "branch_gain": 4,
        "branch_bias": 4,
        "latent_gain": 6,
        "latent_offset": 6,
        "recurrent_output_gain": 6,
        "readout": 14,
    }
    assert adapter.parameter_count == 40

    x = torch.randn(2, 9, 8)
    morphology = torch.tensor([0, 1])
    baseline = model(x, morphology_indices=morphology)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    adapted = model(x, morphology_indices=morphology)

    torch.testing.assert_close(adapted[0], baseline[0])
    assert not torch.allclose(adapted[1], baseline[1])

    trainable_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("behavior_adapter.")
    ]
    set_trainable_parameter_names(model, trainable_names)
    adapted.square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in adapter.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model._unwrapped_temporal_core().gru.parameters()
    )


def test_gru_low_rank_behavior_adapter_updates_shared_matrix_subspaces():
    torch.manual_seed(14)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(
            kind="gru",
            patch_size=2,
            behavior_adapter_morphology_id="morph-b",
            behavior_adapter_rank=2,
        ),
    ).eval()
    adapter = model.behavior_adapter
    assert adapter is not None
    assert adapter.parameter_counts == {
        "branch_gain": 4,
        "branch_bias": 4,
        "latent_gain": 6,
        "latent_offset": 6,
        "recurrent_output_gain": 6,
        "readout": 14,
        "branch_projection_low_rank": 20,
        "gru_input_low_rank": 48,
        "gru_recurrent_low_rank": 48,
    }
    assert adapter.parameter_count == 156

    x = torch.randn(2, 8, 8)
    morphology = torch.tensor([0, 1])
    baseline = model(x, morphology_indices=morphology)
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    adapted = model(x, morphology_indices=morphology)

    torch.testing.assert_close(adapted[0], baseline[0])
    assert not torch.allclose(adapted[1], baseline[1])

    trainable_names = [
        name
        for name, _ in model.named_parameters()
        if name.startswith("behavior_adapter.")
    ]
    set_trainable_parameter_names(model, trainable_names)
    adapted.square().mean().backward()
    assert all(
        parameter.grad is not None
        for parameter in adapter.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in model._unwrapped_temporal_core().gru.parameters()
    )


def test_gru_behavior_adapter_checkpoint_round_trips(tmp_path):
    torch.manual_seed(13)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(
            kind="gru",
            patch_size=2,
            behavior_adapter_morphology_id="morph-b",
            behavior_adapter_rank=2,
        ),
    ).eval()
    assert model.behavior_adapter is not None
    with torch.no_grad():
        for parameter in model.behavior_adapter.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    x = torch.randn(2, 8, 8)
    morphology = torch.tensor([0, 1])
    expected = model(x, morphology_indices=morphology)
    path = tmp_path / "adapted-gru.pt"

    save_checkpoint(model, path)
    loaded, _ = load_checkpoint(path)
    actual = loaded(x, morphology_indices=morphology)

    assert loaded.temporal_config.behavior_adapter_morphology_id == "morph-b"
    assert loaded.temporal_config.behavior_adapter_rank == 2
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


def test_compatible_transfer_preserves_target_only_local_path():
    torch.manual_seed(19)
    source = _source_model()
    control = build_architecture_candidate(
        source,
        ArchitectureSpec(
            candidate_id="spike_gru_control",
            task="spike",
            temporal_core="gru",
            keep_local_tcn=False,
        ),
    )
    torch.manual_seed(23)
    local = build_architecture_candidate(
        source,
        ArchitectureSpec(
            candidate_id="spike_gru_local",
            task="spike",
            temporal_core="gru",
            keep_local_tcn=True,
        ),
    )
    local_tcn_before = {
        name: value.clone()
        for name, value in local.state_dict().items()
        if "local_tcn" in name
    }

    transfer = transfer_compatible_state(local, control.state_dict())

    assert transfer["target_only"]
    for name, value in control.state_dict().items():
        torch.testing.assert_close(local.state_dict()[name], value)
    for name, value in local_tcn_before.items():
        torch.testing.assert_close(local.state_dict()[name], value)


def test_causal_patch_wrapper_delays_correction_and_counts_online_state():
    class TwiceInputCorrection(nn.Module):
        recurrent_state_elements = 3

        def forward(self, hidden):
            return hidden + 2.0 * hidden

    hidden = torch.tensor(
        [[[1.0, 3.0], [3.0, 5.0], [7.0, 9.0], [9.0, 11.0]]]
    )
    patched = temporal_core.CausalPatchedTemporalCore(
        TwiceInputCorrection(),
        width=2,
        patch_size=2,
    )

    actual = patched(hidden)

    expected = hidden.clone()
    expected[:, 2:] += torch.tensor([[[4.0, 8.0], [4.0, 8.0]]])
    torch.testing.assert_close(actual, expected)
    assert patched.recurrent_state_elements == 3 + 2 * 2


def test_causal_patch_wrapper_handles_trailing_partial_patch():
    class TwiceInputCorrection(nn.Module):
        recurrent_state_elements = 3

        def temporal_correction(self, hidden):
            return 2.0 * hidden

    hidden = torch.tensor(
        [[[1.0], [3.0], [5.0], [7.0], [11.0]]]
    )
    patched = temporal_core.CausalPatchedTemporalCore(
        TwiceInputCorrection(),
        width=1,
        patch_size=2,
    )

    actual = patched(hidden)

    expected = torch.tensor(
        [[[1.0], [3.0], [9.0], [11.0], [23.0]]]
    )
    torch.testing.assert_close(actual, expected)


def test_causal_patch_wrapper_uses_gru_correction_without_cancellation():
    torch.manual_seed(37)
    hidden = torch.randn(3, 20, 6)
    core = temporal_core.ResidualGRUTemporalCore(
        6,
        residual_scale_init=0.1,
    )
    patched = temporal_core.CausalPatchedTemporalCore(
        core,
        width=6,
        patch_size=2,
    )
    coarse = hidden.reshape(3, 10, 2, 6).mean(dim=2)
    recurrent, _ = core.gru(core.pre_norm(coarse))
    correction = core.residual_scale.view(1, 1, -1) * recurrent
    delayed = torch.zeros_like(correction)
    delayed[:, 1:] = correction[:, :-1]
    expected = hidden + delayed.repeat_interleave(2, dim=1)

    actual = patched(hidden)

    assert torch.equal(actual, expected)


def test_temporal_model_builds_patched_core_and_reports_full_state():
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=2),
    )

    assert isinstance(
        model.temporal_core,
        temporal_core.CausalPatchedTemporalCore,
    )
    assert model.recurrent_state_bytes(dtype=torch.float32) == 3 * 6 * 4
    manifest = model.component_manifest()
    assert manifest["temporal_patch_size"] == 2


def test_patched_gru_can_reallocate_compute_to_wider_hidden_state():
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(
            kind="gru",
            gru_hidden_units=9,
            patch_size=2,
        ),
    )
    core = model.temporal_core

    assert core.core.gru.hidden_size == 9
    output = model(
        torch.randn(2, 8, 8),
        morphology_indices=torch.tensor([0, 1]),
    )
    assert output.shape == (2, 8, 2)
    assert model.recurrent_state_bytes(dtype=torch.float32) == (9 + 12) * 4


def test_architecture_spec_carries_gru_hidden_units_into_candidate():
    spec = ArchitectureSpec(
        candidate_id="regression_gru_patch2_wide",
        task="regression",
        temporal_core="gru",
        gru_hidden_units=9,
        patch_size=2,
    )

    model = build_architecture_candidate(_source_model(), spec)

    assert model.temporal_config.gru_hidden_units == 9
    assert model.temporal_core.core.gru.hidden_size == 9


def test_architecture_spec_carries_patch_size_into_candidate():
    spec = ArchitectureSpec(
        candidate_id="regression_branch_patch2",
        task="regression",
        temporal_core="branch_rnn",
        patch_size=2,
    )

    model = build_architecture_candidate(_source_model(), spec)

    assert model.temporal_config.patch_size == 2
    assert isinstance(
        model.temporal_core,
        temporal_core.CausalPatchedTemporalCore,
    )


def test_compatible_transfer_maps_native_core_into_patch_wrapper():
    torch.manual_seed(29)
    source = _source_model()
    native = build_architecture_candidate(
        source,
        ArchitectureSpec(
            candidate_id="regression_gru_native",
            task="regression",
            temporal_core="gru",
        ),
    )
    torch.manual_seed(31)
    patched = build_architecture_candidate(
        source,
        ArchitectureSpec(
            candidate_id="regression_gru_patch2",
            task="regression",
            temporal_core="gru",
            patch_size=2,
        ),
    )

    transfer = transfer_compatible_state(patched, native.state_dict())

    assert transfer["aliases"]
    for name, value in native.temporal_core.state_dict().items():
        torch.testing.assert_close(
            patched.temporal_core.core.state_dict()[name],
            value,
        )


@pytest.mark.parametrize(
    ("core_kind", "patch_size"),
    [
        ("gru", 1),
        ("branch_rnn", 1),
        ("gru", 2),
        ("branch_rnn", 2),
    ],
)
def test_temporal_model_streaming_matches_full_sequence_with_local_tcn(
    core_kind,
    patch_size,
):
    torch.manual_seed(41)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(
            kind=core_kind,
            branch_memory_units=5,
            branch_hidden_units=7,
            keep_local_tcn=True,
            patch_size=patch_size,
        ),
    ).eval()
    inputs = torch.randn(2, 7, model.num_input)
    morphology = torch.tensor([0, 1])

    expected = model(inputs, morphology_indices=morphology)
    state = model.allocate_streaming_state(inputs.shape[0])
    actual = torch.cat(
        [
            model.streaming_step(
                inputs[:, step],
                state,
                morphology_indices=morphology,
            )
            for step in range(inputs.shape[1])
        ],
        dim=1,
    )

    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)


def test_temporal_model_sparse_event_step_matches_dense_step():
    torch.manual_seed(43)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=2),
    ).eval()
    morphology = torch.tensor([0, 1])
    event_indices = torch.tensor([[0, 3, 3], [1, 5, -1]])
    event_values = torch.tensor([[0.4, 0.2, 0.1], [0.7, 0.5, 0.0]])
    dense = torch.zeros(2, model.num_input)
    dense.scatter_add_(
        1,
        event_indices.clamp_min(0),
        event_values * (event_indices >= 0),
    )
    dense_state = model.allocate_streaming_state(2)
    sparse_state = model.allocate_streaming_state(2)

    dense_output = model.streaming_step(
        dense,
        dense_state,
        morphology_indices=morphology,
    )
    sparse_output = model.streaming_step_events(
        event_indices,
        event_values,
        sparse_state,
        morphology_indices=morphology,
    )

    torch.testing.assert_close(sparse_output, dense_output)


@pytest.mark.parametrize("core_kind", ["gru", "branch_rnn"])
def test_temporal_sparse_streaming_chunks_preserve_outputs_and_state(
    core_kind,
):
    torch.manual_seed(47)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(
            kind=core_kind,
            branch_memory_units=5,
            branch_hidden_units=7,
            keep_local_tcn=True,
            patch_size=2,
        ),
    ).eval()
    morphology = torch.tensor([0, 1, 0, 1, 0])
    full_state = model.allocate_streaming_state(5)
    chunked_state = model.allocate_streaming_state(5)

    for _ in range(3):
        event_indices = torch.randint(0, model.num_input, (5, 3))
        event_values = torch.randn(5, 3)
        expected = model.streaming_step_events(
            event_indices,
            event_values,
            full_state,
            morphology_indices=morphology,
            chunk_size=5,
        )
        output_buffer = torch.empty(5, 1, model.num_output)
        actual = model.streaming_step_events(
            event_indices,
            event_values,
            chunked_state,
            morphology_indices=morphology,
            chunk_size=2,
            output_buffer=output_buffer,
            retain_base_soma_prediction=False,
        )

        assert actual is output_buffer
        torch.testing.assert_close(actual, expected)

    assert model.base_soma_prediction is None
    assert chunked_state.patch_position == full_state.patch_position
    for actual, expected in zip(
        _streaming_state_tensors(chunked_state),
        _streaming_state_tensors(full_state),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("patch_size", [1, 2])
def test_gru_population_runner_matches_sparse_streaming(patch_size):
    torch.manual_seed(53)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=patch_size),
    ).eval()
    population = 6
    morphology = torch.tensor([0, 1, 0, 1, 0, 1])
    reference_state = model.allocate_streaming_state(population)
    runner = GRUPopulationRunner(
        model,
        population=population,
        chunk_size=3,
        compile_step=False,
    )

    for step in range(7):
        event_indices = torch.randint(0, model.num_input, (population, 3))
        event_values = torch.randn(population, 3)
        if step % 3 == 0:
            event_values.neg_()
        expected = model.streaming_step_events(
            event_indices,
            event_values,
            reference_state,
            morphology_indices=morphology,
            retain_base_soma_prediction=False,
        )
        actual = runner.step(
            event_indices,
            event_values,
            morphology,
        )
        torch.testing.assert_close(actual, expected)

    assert runner.patch_position == reference_state.patch_position
    torch.testing.assert_close(
        runner.recurrent_state,
        reference_state.core_state,
    )
    if patch_size > 1:
        torch.testing.assert_close(
            runner.patch_sum,
            reference_state.patch_sum,
        )
        torch.testing.assert_close(
            runner.patch_correction,
            reference_state.patch_correction,
        )


def test_gru_population_runner_requires_even_chunks():
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru"),
    ).eval()

    with pytest.raises(ValueError, match="divisible"):
        GRUPopulationRunner(
            model,
            population=5,
            chunk_size=2,
            compile_step=False,
        )


def test_gru_population_runner_does_not_retain_cudnn_gru_module():
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru"),
    ).eval()
    runner = GRUPopulationRunner(
        model,
        population=4,
        chunk_size=4,
        compile_step=False,
    )

    assert runner._native_step is not None
    assert not any(
        isinstance(module, nn.GRU)
        for module in runner._native_step.modules()
    )


@pytest.mark.parametrize("patch_size", [1, 4])
def test_gru_population_runner_applies_per_neuron_embedding_adapters(
    patch_size,
):
    torch.manual_seed(57)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=patch_size),
    ).eval()
    population = 4
    morphology = torch.tensor([0, 1, 0, 1])
    adapters = torch.randn(population, model.config.model_dim) * 0.1
    adapted = GRUPopulationRunner(
        model,
        population=population,
        chunk_size=2,
        compile_step=False,
        population_embedding_adapters=adapters,
    )
    references = []
    for neuron in range(population):
        reference_model = deepcopy(model)
        assert reference_model.backbone.morphology_embedding is not None
        with torch.no_grad():
            reference_model.backbone.morphology_embedding.weight[
                morphology[neuron]
            ].add_(adapters[neuron])
        references.append(
            GRUPopulationRunner(
                reference_model,
                population=1,
                chunk_size=1,
                compile_step=False,
            )
        )

    for _ in range(5):
        event_indices = torch.randint(0, model.num_input, (population, 3))
        event_values = torch.randn(population, 3)
        actual = adapted.step(
            event_indices,
            event_values,
            morphology,
        )
        expected = torch.cat(
            [
                references[neuron].step(
                    event_indices[neuron : neuron + 1],
                    event_values[neuron : neuron + 1],
                    morphology[neuron : neuron + 1],
                )
                for neuron in range(population)
            ]
        )
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("patch_size", [1, 4])
def test_gru_branch_population_runner_matches_sparse_events(patch_size):
    torch.manual_seed(58)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=patch_size),
    ).eval()
    population = 6
    morphology = torch.tensor([0, 1, 0, 1, 0, 1])
    adapters = torch.randn(population, model.config.model_dim) * 0.1
    event_runner = GRUPopulationRunner(
        model,
        population=population,
        chunk_size=3,
        compile_step=False,
        population_embedding_adapters=adapters,
    )
    branch_runner = GRUBranchPopulationRunner(
        model,
        population=population,
        chunk_size=3,
        compile_step=False,
        population_embedding_adapters=adapters,
    )

    for _ in range(5):
        event_indices = torch.randint(0, model.num_input, (population, 3))
        event_values = torch.randn(population, 3)
        raw_branches = _raw_sparse_branch_inputs(
            model,
            event_indices,
            event_values,
            morphology,
        )
        expected = event_runner.step(
            event_indices,
            event_values,
            morphology,
        )
        actual = branch_runner.step(raw_branches, morphology)
        torch.testing.assert_close(actual, expected)

    torch.testing.assert_close(
        branch_runner.recurrent_state,
        event_runner.recurrent_state,
    )
    if patch_size > 1:
        torch.testing.assert_close(
            branch_runner.patch_sum,
            event_runner.patch_sum,
        )
        torch.testing.assert_close(
            branch_runner.patch_correction,
            event_runner.patch_correction,
        )


@pytest.mark.parametrize("patch_size", [1, 4])
def test_adapted_gru_branch_runner_matches_individual_models(
    patch_size,
):
    torch.manual_seed(61)
    base = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=patch_size),
    ).eval()
    adaptation = NeuronBehaviorAdaptation(
        width=base.config.model_dim,
        branches=base.num_branch,
        outputs=base.num_output,
        matrix_rank=2,
        adapt_recurrent=True,
    )
    population = 4
    steps = 9
    morphology = torch.tensor([0, 1, 1, 0])
    references = []
    packed = []
    morphology_ids = tuple(base.config.morphology_ids or ())
    for neuron in range(population):
        config = replace(
            base.temporal_config,
            behavior_adapter_morphology_id=(
                morphology_ids[morphology[neuron]]
            ),
            behavior_adapter_rank=2,
        )
        reference = AxoTemporalModel(base.backbone, config).eval()
        incompatible = reference.load_state_dict(
            base.state_dict(),
            strict=False,
        )
        assert not incompatible.unexpected_keys
        assert all(
            name.startswith("behavior_adapter.")
            for name in incompatible.missing_keys
        )
        adapter = reference.behavior_adapter
        assert adapter is not None
        with torch.no_grad():
            for parameter in adapter.parameters():
                parameter.normal_(mean=0.0, std=0.04)
        packed.append(
            pack_neuron_behavior_adapter(
                adapter,
                adaptation=adaptation,
            )
        )
        references.append(reference)
    behavior_parameters = torch.stack(packed)
    assert behavior_parameters.shape == (
        population,
        adaptation.parameter_count,
    )
    runner = AdaptedGRUBranchPopulationRunner(
        base,
        population=population,
        chunk_size=2,
        behavior_parameters=behavior_parameters,
        adaptation=adaptation,
        compile_step=False,
    )
    assert runner.compile_mode == "max-autotune-no-cudagraphs"

    event_indices = torch.randint(
        0,
        base.num_input,
        (population, steps, 3),
    )
    event_values = torch.randn(population, steps, 3)
    dense = torch.zeros(population, steps, base.num_input)
    dense.scatter_add_(2, event_indices, event_values)
    expected = torch.cat(
        [
            references[neuron](
                dense[neuron : neuron + 1],
                morphology_indices=morphology[neuron : neuron + 1],
            )
            for neuron in range(population)
        ]
    )
    actual_steps = []
    for time_index in range(steps):
        raw_branches = _raw_sparse_branch_inputs(
            base,
            event_indices[:, time_index],
            event_values[:, time_index],
            morphology,
        )
        actual_steps.append(
            runner.step(raw_branches, morphology).clone()
        )
    actual = torch.cat(actual_steps, dim=1)

    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert runner.adapted_parameters_per_neuron == 156
    assert runner.adapted_components == (
        "branch_bias",
        "branch_gain",
        "branch_projection_low_rank",
        "gru_input_low_rank",
        "gru_recurrent_low_rank",
        "latent_gain",
        "latent_offset",
        "readout",
        "recurrent_output_gain",
    )


def test_adapted_gru_branch_runner_rejects_incomplete_parameter_bank():
    base = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=4),
    ).eval()
    adaptation = NeuronBehaviorAdaptation(
        width=base.config.model_dim,
        branches=base.num_branch,
        outputs=base.num_output,
        matrix_rank=2,
        adapt_recurrent=True,
    )

    with pytest.raises(ValueError, match="behavior parameters"):
        AdaptedGRUBranchPopulationRunner(
            base,
            population=4,
            chunk_size=2,
            behavior_parameters=torch.zeros(
                4,
                adaptation.parameter_count - 1,
            ),
            adaptation=adaptation,
            compile_step=False,
        )


@pytest.mark.parametrize("patch_size", [1, 2])
def test_grouped_gru_population_runner_matches_shared_weights(patch_size):
    torch.manual_seed(59)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=patch_size),
    ).eval()
    population = 8
    morphology = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    shared = GRUPopulationRunner(
        model,
        population=population,
        chunk_size=4,
        compile_step=False,
    )
    grouped = GroupedGRUPopulationRunner(
        model,
        population=population,
        groups=2,
        chunk_size=4,
        compile_step=False,
    )

    for _ in range(7):
        event_indices = torch.randint(0, model.num_input, (population, 3))
        event_values = torch.randn(population, 3)
        expected = shared.step(event_indices, event_values, morphology)
        actual = grouped.step(event_indices, event_values, morphology)
        torch.testing.assert_close(actual, expected)

    torch.testing.assert_close(
        grouped.recurrent_state.flatten(0, 1),
        shared.recurrent_state,
    )
    assert grouped.group_specific_parameter_count > 0
    assert grouped.resident_parameter_count == (
        grouped.shared_parameter_count
        + 2 * grouped.group_specific_parameter_count
    )


def test_grouped_gru_population_runner_shares_patched_parameter_storage():
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru", patch_size=2),
    ).eval()
    runner = GroupedGRUPopulationRunner(
        model,
        population=8,
        groups=2,
        chunk_size=4,
        compile_step=False,
    )

    assert runner._open_step is not None
    assert runner._close_step is not None
    for name in (
        "group_weight_ih",
        "group_weight_hh",
        "group_bias_ih",
        "group_bias_hh",
        "group_residual_scale",
    ):
        assert getattr(runner._open_step, name) is getattr(
            runner._close_step,
            name,
        )


def test_grouped_gru_population_runner_loads_distinct_group_weights():
    torch.manual_seed(61)
    model = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru"),
    ).eval()
    source = create_temporal_model(
        _source_model(),
        AxoTemporalCoreConfig(kind="gru"),
    ).eval()
    runner = GroupedGRUPopulationRunner(
        model,
        population=8,
        groups=2,
        chunk_size=4,
        compile_step=False,
    )
    runner.load_group_recurrent_parameters(1, source)

    event_indices = torch.randint(0, model.num_input, (8, 3))
    event_values = torch.randn(8, 3)
    morphology = torch.zeros(8, dtype=torch.long)
    output = runner.step(event_indices, event_values, morphology)

    assert not torch.allclose(output[:4], output[4:])


def _streaming_state_tensors(state):
    core_state = (
        [state.core_state]
        if isinstance(state.core_state, torch.Tensor)
        else list(state.core_state)
    )
    return [
        *core_state,
        *state.local_tcn_histories,
        state.patch_sum,
        state.patch_correction,
    ]


def _raw_sparse_branch_inputs(
    model,
    event_indices,
    event_values,
    morphology,
):
    backbone = model.backbone
    slots = backbone.sparse_input_slot_indices.index_select(
        0,
        event_indices.reshape(-1),
    ).view_as(event_indices)
    weighted = event_values
    assert backbone.morphology_synapse_log_gain is not None
    assert backbone.config.morphology_synapse_gain_scale is not None
    gain = torch.exp(
        backbone.config.morphology_synapse_gain_scale
        * backbone.morphology_synapse_log_gain
    )
    if backbone.config.share_morphology_synapse_gain:
        weighted = weighted * gain.flatten().index_select(
            0,
            slots.flatten(),
        ).view_as(slots)
    else:
        slot_count = (
            backbone.num_branch * backbone.num_synapse_per_branch
        )
        morphology_slots = morphology.unsqueeze(1) * slot_count + slots
        weighted = weighted * gain.flatten().index_select(
            0,
            morphology_slots.flatten(),
        ).view_as(slots)
    feature_gains = backbone._all_morphology_feature_gains(
        dtype=weighted.dtype,
        device=weighted.device,
    )
    if feature_gains is not None:
        slot_count = (
            backbone.num_branch * backbone.num_synapse_per_branch
        )
        morphology_slots = morphology.unsqueeze(1) * slot_count + slots
        weighted = weighted * feature_gains.flatten().index_select(
            0,
            morphology_slots.flatten(),
        ).view_as(slots)
    branch_indices = torch.div(
        slots,
        backbone.num_synapse_per_branch,
        rounding_mode="floor",
    )
    branches = torch.zeros(
        event_indices.shape[0],
        backbone.num_branch,
        dtype=event_values.dtype,
    )
    return branches.scatter_add_(1, branch_indices, weighted)
