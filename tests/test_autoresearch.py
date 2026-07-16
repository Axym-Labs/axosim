from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from axosim.autoresearch import (
    AXOMAMBA_DEFAULT_BURN_IN,
    LONG_PRESENTATIONS,
    STANDARD_PRESENTATIONS,
    FrozenWindowDataset,
    PanelCandidate,
    practical_candidate_gate,
    benchmark_model_inference,
    benchmark_model_training_step,
    TrainingObjective,
    axomamba_training_objective,
    create_frozen_window_schedule,
    default_axomamba_training_objective,
    freeze_named_parameters,
    initialize_named_parameters,
    load_named_buffer_artifact,
    offset_named_parameters,
    output_distillation_loss,
    resolve_presentation_budget,
    train_candidate_panel,
    training_objective_loss,
    differentiable_near_spike_voltage_loss,
)
from axosim.data import ShardedNeuronIODataset
from axosim.axomamba import AxoMambaConfig, AxoPyTorchMamba
from axosim.model import BranchELM, BranchELMConfig


def _write_morphology_dataset(root) -> None:
    root.mkdir(parents=True)
    for shard_index, morphology in enumerate(("specimen_a", "specimen_b", "specimen_a", "specimen_b")):
        rng = np.random.default_rng(shard_index)
        inputs = rng.choice([-1.0, 0.0, 1.0], size=(4, 24, 4), p=[0.05, 0.9, 0.05]).astype(np.float32)
        targets = np.zeros((4, 24, 2), dtype=np.float32)
        targets[..., 1] = inputs.sum(axis=-1) * 0.01
        soma_mv = targets[..., 1] / 0.1 - 67.7
        sample_ids = np.asarray([f"{morphology}:trace_{shard_index}_{index}" for index in range(4)])
        np.savez_compressed(
            root / f"shard_{shard_index}.npz",
            inputs=inputs,
            targets=targets,
            soma_mv=soma_mv,
            sample_ids=sample_ids,
        )


def test_task_objective_recipes_keep_replicated_regression_objective_separate() -> None:
    dual = axomamba_training_objective("dual")
    regression = axomamba_training_objective("regression")
    spike = axomamba_training_objective("spike")

    assert dual == default_axomamba_training_objective()
    assert regression.direct_sera_loss_weight == 0.01
    assert regression.spike_loss_weight == 0.0
    assert regression.soma_loss_weight == 0.0
    assert regression.f1_curve_loss_weight == 0.0
    assert regression.near_spike_voltage_loss_weight == 0.0
    assert spike.soma_loss_weight == 0.1
    assert spike.direct_sera_loss_weight == 0.001
    assert spike.f1_curve_loss_weight == 0.05
    with np.testing.assert_raises_regex(ValueError, "unsupported AxoMamba task"):
        axomamba_training_objective("other")


def test_practical_gate_rejects_tiny_metric_only_changes() -> None:
    gate = practical_candidate_gate(
        {
            "voltage_sera_mv": -0.099,
            "dynamics_sera_mv_per_ms": -0.02,
            "response_sera_mv": 0.01,
            "spike_f1_area_0_5ms": 0.0199,
            "spike_f1_exact": 0.0,
            "spike_f1_5ms": 0.0,
        },
        assessments={
            "voltage_sera_mv": "improved",
            "dynamics_sera_mv_per_ms": "improved",
            "response_sera_mv": "uncertain",
            "spike_f1_area_0_5ms": "improved",
            "spike_f1_exact": "uncertain",
            "spike_f1_5ms": "uncertain",
        },
        baseline_training={
            "parameter_count": 100_000,
            "candidate_compute_presentations_per_second": 1_000,
        },
        candidate_training={
            "parameter_count": 100_000,
            "candidate_compute_presentations_per_second": 1_000,
        },
    )

    assert gate["automatically_rejected"] is True
    assert gate["decision"] == "reject_tiny_metric_only"


def test_practical_gate_retains_material_or_efficiency_changes() -> None:
    metric_gate = practical_candidate_gate(
        {"voltage_sera_mv": -0.1, "spike_f1_area_0_5ms": 0.0},
        assessments={
            "voltage_sera_mv": "improved",
            "spike_f1_area_0_5ms": "uncertain",
        },
        baseline_training={"parameter_count": 100_000},
        candidate_training={"parameter_count": 100_000},
    )
    efficiency_gate = practical_candidate_gate(
        {"voltage_sera_mv": -0.01, "spike_f1_area_0_5ms": 0.001},
        assessments={
            "voltage_sera_mv": "uncertain",
            "spike_f1_area_0_5ms": "uncertain",
        },
        baseline_training={"parameter_count": 100_000},
        candidate_training={"parameter_count": 89_000},
    )

    assert metric_gate["automatically_rejected"] is False
    assert metric_gate["decision"] == "review_material_gain"
    assert efficiency_gate["automatically_rejected"] is False
    assert efficiency_gate["decision"] == "review_efficiency_gain"


def test_practical_gate_requires_robust_training_speed_separation() -> None:
    common = {
        "voltage_sera_mv": 0.0,
    }
    assessments = {"voltage_sera_mv": "uncertain"}
    noisy = practical_candidate_gate(
        common,
        assessments=assessments,
        baseline_training={
            "parameter_count": 100_000,
            "training_step_presentations_per_second": 1_000,
            "training_step_latency_p10_ms": 10.0,
        },
        candidate_training={
            "parameter_count": 100_000,
            "training_step_presentations_per_second": 1_200,
            "training_step_latency_p90_ms": 10.5,
        },
    )
    separated = practical_candidate_gate(
        common,
        assessments=assessments,
        baseline_training={
            "parameter_count": 100_000,
            "training_step_presentations_per_second": 1_000,
            "training_step_latency_p10_ms": 10.0,
        },
        candidate_training={
            "parameter_count": 100_000,
            "training_step_presentations_per_second": 1_200,
            "training_step_latency_p90_ms": 8.8,
        },
    )

    assert noisy["automatically_rejected"] is True
    assert separated["automatically_rejected"] is False


def test_practical_gate_rejects_large_but_uncertain_metric_delta() -> None:
    gate = practical_candidate_gate(
        {"spike_f1_5ms": 0.04},
        assessments={"spike_f1_5ms": "uncertain"},
        baseline_training={"parameter_count": 100_000},
        candidate_training={"parameter_count": 100_000},
    )

    assert gate["automatically_rejected"] is True


def test_practical_gate_labels_efficiency_with_material_regression_as_tradeoff() -> None:
    gate = practical_candidate_gate(
        {"voltage_sera_mv": 0.11},
        assessments={"voltage_sera_mv": "regressed"},
        baseline_training={
            "parameter_count": 100_000,
            "inference_presentations_per_second": 1_000,
            "inference_latency_p10_ms": 10.0,
        },
        candidate_training={
            "parameter_count": 80_000,
            "inference_presentations_per_second": 1_200,
            "inference_latency_p90_ms": 8.8,
        },
    )

    assert gate["automatically_rejected"] is False
    assert gate["decision"] == "review_efficiency_tradeoff"
    assert gate["material_regressions"] == ["voltage_sera_mv"]


def test_practical_gate_ignores_metrics_outside_the_declared_task_family() -> None:
    gate = practical_candidate_gate(
        {
            "voltage_sera_mv": 0.01,
            "spike_f1_area_0_5ms": -0.08,
            "spike_f1_5ms": -0.09,
        },
        assessments={
            "voltage_sera_mv": "uncertain",
            "spike_f1_area_0_5ms": "regressed",
            "spike_f1_5ms": "regressed",
        },
        baseline_training={"parameter_count": 100_000},
        candidate_training={"parameter_count": 80_000},
        metric_families={"sera"},
    )

    assert gate["decision"] == "review_efficiency_gain"
    assert gate["material_regressions"] == []
    assert gate["metric_families"] == ["sera"]


def test_practical_gate_requires_robust_inference_speed_separation() -> None:
    common_deltas = {"voltage_sera_mv": 0.0}
    common_assessments = {"voltage_sera_mv": "uncertain"}
    noisy = practical_candidate_gate(
        common_deltas,
        assessments=common_assessments,
        baseline_training={
            "parameter_count": 100_000,
            "inference_presentations_per_second": 1_000,
            "inference_latency_p10_ms": 10.0,
        },
        candidate_training={
            "parameter_count": 100_000,
            "inference_presentations_per_second": 1_200,
            "inference_latency_p90_ms": 10.5,
        },
    )
    separated = practical_candidate_gate(
        common_deltas,
        assessments=common_assessments,
        baseline_training={
            "parameter_count": 100_000,
            "inference_presentations_per_second": 1_000,
            "inference_latency_p10_ms": 10.0,
        },
        candidate_training={
            "parameter_count": 100_000,
            "inference_presentations_per_second": 1_200,
            "inference_latency_p90_ms": 8.8,
        },
    )

    assert noisy["automatically_rejected"] is True
    assert separated["automatically_rejected"] is False


def test_output_distillation_matches_teacher_probabilities_and_voltage() -> None:
    teacher = torch.tensor(
        [[[0.0, 0.5], [2.0, -0.25], [-2.0, 0.75]]],
        dtype=torch.float32,
    )
    student = teacher.clone().requires_grad_(True)

    loss = output_distillation_loss(student, teacher, burn_in=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert student.grad is not None
    torch.testing.assert_close(student.grad[:, 1:, :], torch.zeros_like(student.grad[:, 1:, :]))


def test_inference_benchmark_records_repeated_latency():
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=8,
            num_output=2,
            num_branch=2,
            num_synapse_per_branch=4,
            model_dim=4,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
            expansion=2,
        )
    )

    result = benchmark_model_inference(
        model,
        device="cpu",
        batch_size=2,
        window_size=8,
        warmup_repeats=1,
        repeats=3,
    )

    assert result["inference_benchmark_repeats"] == 3
    assert result["inference_latency_median_ms"] > 0
    assert result["inference_presentations_per_second"] > 0


def test_training_benchmark_records_repeated_latency():
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=8,
            num_output=2,
            num_branch=2,
            num_synapse_per_branch=4,
            model_dim=4,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
            expansion=2,
        )
    )

    result = benchmark_model_training_step(
        model,
        device="cpu",
        batch_size=2,
        window_size=8,
        warmup_repeats=1,
        repeats=3,
    )

    assert result["training_step_benchmark_repeats"] == 3
    assert result["training_step_latency_median_ms"] > 0
    assert result["training_step_presentations_per_second"] > 0


def test_frozen_window_schedule_is_deterministic_and_morphology_balanced(tmp_path) -> None:
    data_root = tmp_path / "train"
    _write_morphology_dataset(data_root)
    base = ShardedNeuronIODataset(data_root, cache_shards=1)
    schedule_path = tmp_path / "schedule.npz"

    metadata = create_frozen_window_schedule(
        base,
        schedule_path,
        presentations=32,
        schedule_batch_size=4,
        shard_reuse_batches=2,
        window_size=8,
        start_offset=4,
        seed=17,
    )
    first = FrozenWindowDataset(base, schedule_path)
    second = FrozenWindowDataset(base, schedule_path)

    assert len(first) == len(second) == 32
    assert metadata["morphology_counts"] == {"specimen_a": 16, "specimen_b": 16}
    assert first.get_sample_ids(range(32)) == second.get_sample_ids(range(32))
    first_x, first_y = first.get_batch(range(8))
    second_x, second_y = second.get_batch(range(8))
    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)


def test_frozen_schedule_does_not_decompress_shards_to_read_sequence_lengths(tmp_path) -> None:
    data_root = tmp_path / "train"
    _write_morphology_dataset(data_root)
    base = ShardedNeuronIODataset(data_root, cache_shards=1)

    def fail_payload_load(_):
        raise AssertionError("schedule generation loaded a full shard")

    base._load_cached_shard = fail_payload_load
    metadata = create_frozen_window_schedule(
        base,
        tmp_path / "schedule.npz",
        presentations=32,
        schedule_batch_size=4,
        shard_reuse_batches=2,
        window_size=8,
        start_offset=4,
        seed=17,
    )

    assert metadata["morphology_counts"] == {"specimen_a": 16, "specimen_b": 16}


def test_frozen_schedule_shared_batch_includes_aligned_raw_soma(tmp_path) -> None:
    data_root = tmp_path / "train"
    _write_morphology_dataset(data_root)
    base = ShardedNeuronIODataset(data_root, cache_shards=1)
    schedule_path = tmp_path / "schedule.npz"
    create_frozen_window_schedule(
        base,
        schedule_path,
        presentations=8,
        schedule_batch_size=4,
        shard_reuse_batches=1,
        window_size=8,
        start_offset=4,
        seed=19,
    )
    frozen = FrozenWindowDataset(base, schedule_path)

    inputs, targets, raw_soma = frozen.get_shared_batch(range(4))

    assert inputs.shape == (4, 8, 4)
    assert targets.shape == (4, 8, 2)
    assert raw_soma.shape == (4, 8)
    np.testing.assert_allclose(raw_soma, targets[..., 1] / 0.1 - 67.7)


class _CountingDataset:
    def __init__(self, samples: int = 8) -> None:
        self.inputs = np.zeros((samples, 6, 4), dtype=np.float32)
        self.targets = np.zeros((samples, 6, 2), dtype=np.float32)
        self.calls = 0

    def __len__(self) -> int:
        return len(self.inputs)

    def get_batch(self, indices) -> tuple[np.ndarray, np.ndarray]:
        self.calls += 1
        selected = list(indices)
        return self.inputs[selected], self.targets[selected]

    def get_shared_batch(self, indices) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inputs, targets = self.get_batch(indices)
        raw_soma = targets[..., 1] / 0.1 - 67.7
        return inputs, targets, raw_soma

    def get_sample_ids(self, indices) -> list[str]:
        return [f"specimen_{'a' if int(index) % 2 == 0 else 'b'}:trace_{index}" for index in indices]


def test_candidate_panel_shares_loads_but_preserves_candidate_optimizer_steps(tmp_path) -> None:
    dataset = _CountingDataset()
    config = BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4)
    candidates = [
        PanelCandidate(
            candidate_id="batch2",
            model=BranchELM(config),
            batch_size=2,
            checkpoint_path=tmp_path / "batch2.pt",
            metrics_path=tmp_path / "batch2.json",
            burn_in=2,
        ),
        PanelCandidate(
            candidate_id="batch4",
            model=BranchELM(config),
            batch_size=4,
            checkpoint_path=tmp_path / "batch4.pt",
            metrics_path=tmp_path / "batch4.json",
        ),
    ]

    results = train_candidate_panel(
        candidates,
        dataset,
        objective=TrainingObjective(spike_loss_weight=0.25, soma_loss_weight=0.75),
        device="cpu",
        burn_in=1,
        seed=11,
    )

    assert dataset.calls == 2
    assert results["batch2"]["optimizer_steps"] == 4
    assert results["batch4"]["optimizer_steps"] == 2
    assert results["batch2"]["train_presentations"] == 8
    assert results["batch4"]["train_presentations"] == 8
    assert results["batch2"]["burn_in"] == 2
    assert results["batch4"]["burn_in"] == 1
    assert json.loads((tmp_path / "batch2.json").read_text())["candidate_id"] == "batch2"
    assert (tmp_path / "batch2.pt").exists()
    assert (tmp_path / "batch4.pt").exists()


def test_candidate_panel_passes_morphology_context_to_conditioned_model(tmp_path) -> None:
    dataset = _CountingDataset(samples=4)
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=4,
            num_branch=2,
            num_synapse_per_branch=2,
            model_dim=4,
            num_layers=1,
            state_dim=2,
            conv_kernel=2,
            expansion=2,
            morphology_ids=["specimen_a", "specimen_b"],
            morphology_embedding_scale=1.0,
        )
    )
    candidate = PanelCandidate(
        candidate_id="conditioned",
        model=model,
        batch_size=2,
        checkpoint_path=tmp_path / "conditioned.pt",
        metrics_path=tmp_path / "conditioned.json",
        burn_in=1,
    )

    result = train_candidate_panel(
        [candidate],
        dataset,
        objective=TrainingObjective(spike_loss_weight=0.25, soma_loss_weight=0.75),
        device="cpu",
        seed=13,
    )["conditioned"]

    assert result["morphology_conditioning"] == ["specimen_a", "specimen_b"]
    assert result["optimizer_steps"] == 2


def test_candidate_panel_defaults_to_promoted_axomamba_recipe(tmp_path) -> None:
    dataset = _CountingDataset(samples=2)
    dataset.inputs = np.zeros((2, 60, 4), dtype=np.float32)
    dataset.targets = np.zeros((2, 60, 2), dtype=np.float32)
    candidate = PanelCandidate(
        candidate_id="promoted-defaults",
        model=BranchELM(
            BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4)
        ),
        batch_size=2,
        checkpoint_path=tmp_path / "promoted.pt",
        metrics_path=tmp_path / "promoted.json",
    )

    result = train_candidate_panel([candidate], dataset, device="cpu", seed=11)[
        "promoted-defaults"
    ]

    assert result["burn_in"] == AXOMAMBA_DEFAULT_BURN_IN
    assert result["objective"] == {
        "base_soma_loss_weight": 0.0,
        "base_soma_sera_loss_weight": 0.0,
        "base_soma_target_mix": 1.0,
        "direct_sera_loss_weight": 0.01,
        "f1_curve_loss_weight": 0.05,
        "f1_tolerances_steps": (0, 1, 2, 3, 4, 5, 5),
        "near_spike_radius_steps": 5,
        "near_spike_voltage_loss_weight": 0.002,
        "raw_soma_target_mix": 1.0,
        "sera_threshold_count": 101,
        "soma_loss_weight": 0.75,
        "soma_scale": 0.1,
        "spike_loss_weight": 0.25,
    }


def test_candidate_panel_runs_one_shared_distillation_teacher(tmp_path) -> None:
    dataset = _CountingDataset(samples=4)
    config = BranchELMConfig(
        input_dim=4,
        memory_units=3,
        num_branches=2,
        hidden_units=4,
    )
    candidate = PanelCandidate(
        candidate_id="student",
        model=BranchELM(config),
        batch_size=2,
        checkpoint_path=tmp_path / "student.pt",
        metrics_path=tmp_path / "student.json",
        burn_in=1,
        output_distillation_weight=0.5,
    )
    objective = TrainingObjective(spike_loss_weight=0.25, soma_loss_weight=0.75)

    with np.testing.assert_raises_regex(ValueError, "requires a teacher"):
        train_candidate_panel(
            [candidate],
            dataset,
            objective=objective,
            device="cpu",
            seed=17,
        )
    result = train_candidate_panel(
        [candidate],
        dataset,
        objective=objective,
        device="cpu",
        seed=17,
        distillation_teacher=BranchELM(config),
    )["student"]

    assert result["output_distillation_weight"] == 0.5
    assert "output_distillation" in result["last_components"]
    assert result["shared_teacher_compute_seconds"] > 0


def test_named_parameter_initialization_changes_only_requested_tensors() -> None:
    model = BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    requested = {"readout.bias": 0.25}

    changed = initialize_named_parameters(model, requested)

    assert changed == ["readout.bias"]
    for name, parameter in model.named_parameters():
        if name == "readout.bias":
            torch.testing.assert_close(parameter, torch.full_like(parameter, 0.25))
        else:
            torch.testing.assert_close(parameter, before[name])

    with np.testing.assert_raises_regex(ValueError, "unknown model parameters"):
        initialize_named_parameters(model, {"missing.parameter": 1.0})


def test_named_parameter_offsets_apply_after_common_initialization() -> None:
    model = BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    changed = offset_named_parameters(model, {"readout.bias": 0.4})

    assert changed == ["readout.bias"]
    for name, parameter in model.named_parameters():
        if name == "readout.bias":
            torch.testing.assert_close(parameter, before[name] + 0.4)
        else:
            torch.testing.assert_close(parameter, before[name])

    with np.testing.assert_raises_regex(ValueError, "unknown model parameters"):
        offset_named_parameters(model, {"missing.parameter": 1.0})


def test_named_parameter_freeze_changes_only_requested_tensors() -> None:
    model = BranchELM(BranchELMConfig(input_dim=4, memory_units=3, num_branches=2, hidden_units=4))

    frozen = freeze_named_parameters(model, ["readout.bias"])

    assert frozen == ["readout.bias"]
    assert dict(model.named_parameters())["readout.bias"].requires_grad is False
    assert all(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name != "readout.bias"
    )
    with np.testing.assert_raises_regex(ValueError, "unknown model parameters"):
        freeze_named_parameters(model, ["missing.parameter"])


def test_named_buffer_artifact_loads_tensor_and_checks_morphology_order(tmp_path) -> None:
    morphology_ids = ("cell_a", "cell_b")
    model = AxoPyTorchMamba(
        AxoMambaConfig(
            num_input=4,
            num_branch=2,
            num_synapse_per_branch=2,
            model_dim=8,
            state_dim=4,
            num_layers=1,
            morphology_ids=morphology_ids,
            morphology_embedding_scale=0.0,
            morphology_synapse_feature_dim=3,
            morphology_synapse_feature_hidden=4,
            morphology_synapse_feature_scale=1.0,
        )
    )
    values = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    artifact_path = tmp_path / "routes.npz"
    np.savez_compressed(
        artifact_path,
        morphology_ids=np.asarray(morphology_ids),
        features=values,
    )

    metadata = load_named_buffer_artifact(
        model,
        buffer_name="morphology_synapse_features",
        artifact_path=artifact_path,
        key="features",
        expected_morphology_ids=morphology_ids,
    )

    torch.testing.assert_close(model.morphology_synapse_features, torch.from_numpy(values))
    assert metadata["shape"] == [2, 4, 3]
    assert len(metadata["sha256"]) == 64

    with np.testing.assert_raises_regex(ValueError, "morphology_ids do not match"):
        load_named_buffer_artifact(
            model,
            buffer_name="morphology_synapse_features",
            artifact_path=artifact_path,
            key="features",
            expected_morphology_ids=("cell_b", "cell_a"),
        )


def test_every_tenth_iteration_uses_long_budget() -> None:
    assert resolve_presentation_budget(iteration=1, force_long=False) == STANDARD_PRESENTATIONS
    assert resolve_presentation_budget(iteration=10, force_long=False) == LONG_PRESENTATIONS
    assert resolve_presentation_budget(iteration=3, force_long=True) == LONG_PRESENTATIONS


def test_metric_aligned_objective_is_zero_for_perfect_voltage_and_spikes() -> None:
    target = torch.zeros(2, 12, 2)
    target[:, (3, 8), 0] = 1.0
    target[..., 1] = torch.linspace(-0.2, 0.4, 12)
    prediction = target.clone()
    prediction[..., 0] = torch.where(target[..., 0] > 0.5, 20.0, -20.0)

    loss, components = training_objective_loss(
        prediction,
        target,
        objective=TrainingObjective(
            spike_loss_weight=0.0,
            soma_loss_weight=0.0,
            direct_sera_loss_weight=1.0,
            f1_curve_loss_weight=1.0,
        ),
        burn_in=0,
    )

    assert float(components["direct_sera"]) == 0.0
    assert float(components["f1_curve_loss"]) < 1e-6
    assert float(loss) < 1e-6


def test_near_spike_voltage_loss_ignores_errors_outside_event_neighborhood() -> None:
    target = torch.zeros(1, 12, 2)
    target[:, 6, 0] = 1.0
    prediction = target.clone()
    prediction[:, 1, 1] = 3.0

    outside = differentiable_near_spike_voltage_loss(
        prediction[..., 1], target, radius_steps=2, soma_scale=0.1
    )
    prediction[:, 5, 1] = 0.2
    inside = differentiable_near_spike_voltage_loss(
        prediction[..., 1], target, radius_steps=2, soma_scale=0.1
    )

    assert float(outside) == 0.0
    assert float(inside) > 0.0


def test_zero_weight_near_spike_objective_still_reports_matched_component() -> None:
    target = torch.zeros(1, 12, 2)
    target[:, 6, 0] = 1.0
    prediction = target.clone()
    prediction[:, 6, 1] = 0.2

    loss, components = training_objective_loss(
        prediction,
        target,
        objective=TrainingObjective(
            spike_loss_weight=0.0,
            soma_loss_weight=0.0,
            near_spike_voltage_loss_weight=0.0,
            near_spike_radius_steps=2,
        ),
        burn_in=0,
    )

    assert float(loss) == 0.0
    assert float(components["near_spike_voltage_loss"]) > 0.0


def test_raw_soma_target_mix_optimizes_the_scored_voltage_view() -> None:
    target = torch.zeros(1, 8, 2)
    target[..., 1] = 1.27
    raw_soma_mv = torch.full((1, 8), 32.3)
    prediction = target.clone()
    prediction[..., 1] = (raw_soma_mv + 67.7) * 0.1

    raw_loss, _ = training_objective_loss(
        prediction,
        target,
        objective=TrainingObjective(
            spike_loss_weight=0.0,
            soma_loss_weight=1.0,
            raw_soma_target_mix=1.0,
        ),
        burn_in=0,
        raw_soma_mv=raw_soma_mv,
    )
    clipped_loss, _ = training_objective_loss(
        prediction,
        target,
        objective=TrainingObjective(
            spike_loss_weight=0.0,
            soma_loss_weight=1.0,
            raw_soma_target_mix=0.0,
        ),
        burn_in=0,
        raw_soma_mv=raw_soma_mv,
    )

    assert float(raw_loss) == 0.0
    assert float(clipped_loss) > 0.0


def test_base_soma_auxiliary_can_decompose_clipped_and_raw_targets() -> None:
    target = torch.zeros(1, 8, 2)
    target[..., 1] = 1.27
    raw_soma_mv = torch.full((1, 8), 32.3)
    prediction = target.clone()
    prediction[..., 1] = (raw_soma_mv + 67.7) * 0.1
    clipped_base = target[..., 1:2].clone()

    decomposed, components = training_objective_loss(
        prediction,
        target,
        objective=TrainingObjective(
            spike_loss_weight=0.0,
            soma_loss_weight=1.0,
            raw_soma_target_mix=1.0,
            base_soma_loss_weight=1.0,
            base_soma_target_mix=0.0,
        ),
        burn_in=0,
        raw_soma_mv=raw_soma_mv,
        base_soma_prediction=clipped_base,
    )
    raw_base, _ = training_objective_loss(
        prediction,
        target,
        objective=TrainingObjective(
            spike_loss_weight=0.0,
            soma_loss_weight=1.0,
            raw_soma_target_mix=1.0,
            base_soma_loss_weight=1.0,
            base_soma_target_mix=1.0,
        ),
        burn_in=0,
        raw_soma_mv=raw_soma_mv,
        base_soma_prediction=clipped_base,
    )

    assert float(decomposed) == 0.0
    assert float(components["base_soma_mse"]) == 0.0
    assert float(raw_base) > 0.0


def test_promoted_axomamba_objective_duplicates_tolerant_f1_endpoint() -> None:
    objective = default_axomamba_training_objective()

    assert objective.direct_sera_loss_weight == 0.01
    assert objective.f1_curve_loss_weight == 0.05
    assert objective.near_spike_voltage_loss_weight == 0.002
    assert objective.near_spike_radius_steps == 5
    assert objective.sera_threshold_count == 101
    assert objective.f1_tolerances_steps == (0, 1, 2, 3, 4, 5, 5)
    assert AXOMAMBA_DEFAULT_BURN_IN == 50


def test_experiment_runner_dry_run_has_small_manifest_driven_interface(tmp_path) -> None:
    manifest = tmp_path / "iteration.json"
    manifest.write_text(
        json.dumps(
            {
                "iteration": 10,
                "name": "objective_probe",
                "candidates": [{"id": "sera", "config_overrides": {}}],
            }
        ),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            str(project_root / ".venv" / "bin" / "python"),
            str(project_root / "scripts" / "run_axomamba_autoresearch.py"),
            str(manifest),
            "--dry-run",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["presentations"] == LONG_PRESENTATIONS
    assert plan["candidate_ids"] == ["sera"]


def test_experiment_runner_allows_only_manifest_labeled_smoke_budget(tmp_path) -> None:
    manifest = tmp_path / "smoke.json"
    manifest.write_text(
        json.dumps(
            {
                "iteration": 0,
                "name": "plumbing",
                "smoke": True,
                "presentations": 64,
                "candidates": [{"id": "full", "config_overrides": {}}],
            }
        ),
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            str(project_root / ".venv" / "bin" / "python"),
            str(project_root / "scripts" / "run_axomamba_autoresearch.py"),
            str(manifest),
            "--dry-run",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["presentations"] == 64
