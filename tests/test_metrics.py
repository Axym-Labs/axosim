import numpy as np
import torch

from axosim.data import ShardedNeuronIODataset, write_demo_shards
from axosim.evaluate import evaluate_dataset
from axosim.metrics import binary_auc, soma_rmse, spike_auc


def test_soma_rmse_uses_second_output_channel():
    prediction = np.array([[[0.0, 1.0], [0.0, 3.0]]], dtype=np.float32)
    target = np.array([[[0.0, 2.0], [0.0, 5.0]]], dtype=np.float32)

    assert np.isclose(soma_rmse(prediction, target), np.sqrt(2.5))


def test_binary_auc_handles_perfect_ranking():
    labels = np.array([0, 0, 1, 1], dtype=np.float32)
    scores = np.array([-2.0, -1.0, 1.0, 2.0], dtype=np.float32)

    assert binary_auc(scores, labels) == 1.0


def test_binary_auc_averages_tied_ranks():
    labels = np.array([0, 1, 0, 1], dtype=np.float32)
    scores = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)

    assert binary_auc(scores, labels) == 0.5


def test_evaluate_dataset_matches_full_array_metrics(tmp_path):
    write_demo_shards(tmp_path, shard_count=1, samples_per_shard=3, time_steps=4, input_dim=3, seed=5)
    with np.load(tmp_path / "demo_000.npz") as data:
        inputs = data["inputs"]
        targets = data["targets"]
        sample_ids = data["sample_ids"]
    targets[..., 0] = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float32,
    )
    np.savez_compressed(tmp_path / "demo_000.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)
    dataset = ShardedNeuronIODataset(tmp_path)

    class TinyModel(torch.nn.Module):
        def forward(self, x):
            return torch.stack([x[..., 0], x[..., 1]], dim=-1)

    metrics = evaluate_dataset(TinyModel(), dataset, batch_size=2, soma_units="normalized")
    predictions = []
    targets = []
    for index in range(len(dataset)):
        sample = dataset[index]
        predictions.append(np.stack([sample.inputs[..., 0], sample.inputs[..., 1]], axis=-1))
        targets.append(sample.targets)
    prediction_array = np.stack(predictions)
    target_array = np.stack(targets)

    assert metrics["samples"] == 3
    assert metrics["soma_rmse_units"] == "normalized"
    assert metrics["pin_memory"] is False
    assert metrics["non_blocking"] is True
    assert metrics["total_seconds"] >= 0.0
    assert np.isclose(metrics["soma_rmse"], soma_rmse(prediction_array, target_array))
    assert np.isclose(metrics["spike_auc"], spike_auc(prediction_array, target_array))


def test_evaluate_dataset_can_report_soma_rmse_in_millivolts(tmp_path):
    write_demo_shards(tmp_path, shard_count=1, samples_per_shard=1, time_steps=2, input_dim=3, seed=5)
    dataset = ShardedNeuronIODataset(tmp_path)

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], 2)

    normalized = evaluate_dataset(ZeroModel(), dataset, batch_size=1, soma_units="normalized")
    millivolts = evaluate_dataset(ZeroModel(), dataset, batch_size=1, soma_units="millivolts")

    assert millivolts["soma_rmse_units"] == "millivolts"
    assert np.isclose(millivolts["soma_rmse"], normalized["soma_rmse"] * 10)


def test_evaluate_dataset_can_ignore_initial_timesteps(tmp_path):
    write_demo_shards(tmp_path, shard_count=1, samples_per_shard=1, time_steps=4, input_dim=3, seed=5)
    dataset = ShardedNeuronIODataset(tmp_path)

    class TimeRampModel(torch.nn.Module):
        def forward(self, x):
            soma = torch.arange(x.shape[1], dtype=x.dtype).expand(x.shape[0], -1)
            return torch.stack([torch.zeros_like(soma), soma], dim=-1)

    all_steps = evaluate_dataset(TimeRampModel(), dataset, batch_size=1, soma_units="normalized")
    tail_steps = evaluate_dataset(TimeRampModel(), dataset, batch_size=1, soma_units="normalized", ignore_start=2)

    assert tail_steps["metric_ignore_start"] == 2
    assert tail_steps["soma_rmse"] != all_steps["soma_rmse"]


def test_evaluate_dataset_can_affine_calibrate_soma_predictions(tmp_path):
    write_demo_shards(tmp_path, shard_count=1, samples_per_shard=2, time_steps=4, input_dim=3, seed=5)
    dataset = ShardedNeuronIODataset(tmp_path)

    class ScaledTargetModel(torch.nn.Module):
        def forward(self, x):
            soma = 2.0 * x[..., 1] + 3.0
            return torch.stack([x[..., 0], soma], dim=-1)

    uncalibrated = evaluate_dataset(ScaledTargetModel(), dataset, batch_size=1, soma_units="normalized")
    calibrated = evaluate_dataset(
        ScaledTargetModel(),
        dataset,
        batch_size=1,
        soma_units="normalized",
        soma_affine_calibration=True,
    )

    assert calibrated["soma_affine_calibration"] is True
    assert calibrated["soma_rmse"] < uncalibrated["soma_rmse"]


def test_evaluate_dataset_can_use_official_overlap_mask(tmp_path):
    inputs = np.zeros((3, 4, 2), dtype=np.float32)
    targets = np.zeros((3, 4, 2), dtype=np.float32)
    targets[..., 1] = np.array([[1, 1, 1, 1], [5, 5, 5, 5], [9, 9, 9, 9]], dtype=np.float32)
    sample_ids = np.array(["sim:0@500", "sim:0@502", "other:0@500"])
    np.savez_compressed(tmp_path / "demo_000.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)
    dataset = ShardedNeuronIODataset(tmp_path)

    class ZeroModel(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], x.shape[1], 2)

    masked = evaluate_dataset(
        ZeroModel(),
        dataset,
        batch_size=3,
        soma_units="normalized",
        mask_mode="official-overlap",
        stitch_burn_in=2,
    )

    expected = np.sqrt(np.mean(np.array([1, 1, 1, 1, 5, 5, 9, 9, 9, 9], dtype=np.float32) ** 2))
    assert masked["metric_mask_mode"] == "official-overlap"
    assert np.isclose(masked["soma_rmse"], expected)
