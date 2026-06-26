import json

import torch

from axosim.cli import main
from axosim.checkpoint import load_checkpoint
from axosim.train import _sera_soma_weighted_mse, _sparse_soma_weighted_mse


def test_sparse_soma_weighted_mse_uses_only_selected_timesteps():
    prediction = torch.tensor([[0.0, 2.0, 4.0, 8.0]])
    target = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    mask = torch.tensor([[False, True, False, True]])

    loss, fraction = _sparse_soma_weighted_mse(prediction, target, mask)

    assert torch.isclose(loss, torch.tensor((1.0**2 + 5.0**2) / 2.0))
    assert torch.isclose(fraction, torch.tensor(0.5))


def test_sparse_soma_weighted_mse_requires_matching_shapes():
    prediction = torch.zeros(1, 4)
    target = torch.zeros(1, 4)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    try:
        _sparse_soma_weighted_mse(prediction, target, mask)
    except ValueError as error:
        assert "matching shapes" in str(error)
    else:
        raise AssertionError("expected shape mismatch to fail")


def test_sera_soma_weighted_mse_weights_relevant_timesteps_more():
    targets = torch.zeros(1, 20, 2)
    targets[..., 0] = torch.tensor([[0.0] * 10 + [1.0] + [0.0] * 9])
    background_prediction = torch.zeros(1, 20)
    background_prediction[:, 0] = 10.0
    spike_prediction = torch.zeros(1, 20)
    spike_prediction[:, 10] = 10.0

    background_loss, _ = _sera_soma_weighted_mse(
        background_prediction,
        targets,
        min_weight=0.0,
        relevance_power=1.0,
    )
    spike_loss, relevance_mean = _sera_soma_weighted_mse(
        spike_prediction,
        targets,
        min_weight=0.0,
        relevance_power=1.0,
    )

    assert spike_loss > background_loss
    assert 0.0 < float(relevance_mean) <= 1.0


def test_sera_soma_weighted_mse_requires_matching_shapes():
    prediction = torch.zeros(1, 3)
    targets = torch.zeros(1, 4, 2)

    try:
        _sera_soma_weighted_mse(prediction, targets)
    except ValueError as error:
        assert "prediction_soma" in str(error)
    else:
        raise AssertionError("expected shape mismatch to fail")


def test_cli_train_writes_checkpoint_and_metrics(tmp_path):
    data_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics_path = tmp_path / "train_metrics.json"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "4", "--time-steps", "6", "--input-dim", "8"])

    result = main(
        [
            "train",
            "--preset",
            "legacy-baseline-smoke",
            "--data",
            str(data_dir),
            "--input-dim",
            "8",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--checkpoint",
            str(checkpoint_path),
            "--metrics",
            str(metrics_path),
        ]
    )

    assert result == 0
    assert checkpoint_path.exists()
    assert metrics_path.exists()
    model, metadata = load_checkpoint(checkpoint_path)
    assert model.config.input_dim == 8
    assert metadata["epoch"] == 1
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["epoch"] == 1


def test_cli_train_with_validation_writes_best_checkpoint(tmp_path):
    data_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "checkpoint.pt"
    best_checkpoint_path = tmp_path / "best.pt"
    metrics_path = tmp_path / "train_metrics.json"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "4", "--time-steps", "6", "--input-dim", "8"])

    result = main(
        [
            "train",
            "--preset",
            "legacy-baseline-smoke",
            "--data",
            str(data_dir),
            "--val-data",
            str(data_dir),
            "--input-dim",
            "8",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--window-size",
            "4",
            "--ignore-start",
            "1",
            "--window-sampling",
            "random-full-trace",
            "--epoch-samples",
            "4",
            "--shard-reuse-batches",
            "2",
            "--full-trace-length",
            "12",
            "--optimizer",
            "adamw",
            "--weight-decay",
            "0.0001",
            "--lr-schedule-steps",
            "30",
            "--val-window-size",
            "4",
            "--val-ignore-start",
            "1",
            "--val-metric-ignore-start",
            "1",
            "--val-metric-mask-mode",
            "ignore-start",
            "--val-metric-stitch-burn-in",
            "150",
            "--val-soma-affine-calibration",
            "--spike-loss-weight",
            "0.5",
            "--soma-loss-weight",
            "0.5",
            "--sparse-soma-loss-weight",
            "0.25",
            "--sparse-soma-high-voltage-quantile",
            "0.8",
            "--sparse-soma-high-dvdt-quantile",
            "0.8",
            "--sparse-soma-input-event-quantile",
            "0.8",
            "--sparse-soma-spike-window",
            "1",
            "--sparse-soma-post-event-window",
            "1",
            "--sera-soma-loss-weight",
            "0.125",
            "--sera-soma-min-weight",
            "0.1",
            "--sera-soma-relevance-power",
            "1.5",
            "--prefetch-batches",
            "1",
            "--pin-memory",
            "--best-checkpoint",
            str(best_checkpoint_path),
            "--checkpoint",
            str(checkpoint_path),
            "--metrics",
            str(metrics_path),
        ]
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert result == 0
    assert best_checkpoint_path.exists()
    assert metrics["best_epoch"] == 1
    assert metrics["samples"] == 4
    assert metrics["optimizer"] == "adamw"
    assert metrics["weight_decay"] == 0.0001
    assert metrics["lr_schedule_steps"] == 30
    assert metrics["spike_loss_weight"] == 0.5
    assert metrics["soma_loss_weight"] == 0.5
    assert metrics["sparse_soma_loss_weight"] == 0.25
    assert metrics["sparse_soma_high_voltage_quantile"] == 0.8
    assert metrics["sparse_soma_high_dvdt_quantile"] == 0.8
    assert metrics["sparse_soma_input_event_quantile"] == 0.8
    assert metrics["sparse_soma_spike_window"] == 1
    assert metrics["sparse_soma_post_event_window"] == 1
    assert metrics["sera_soma_loss_weight"] == 0.125
    assert metrics["sera_soma_min_weight"] == 0.1
    assert metrics["sera_soma_relevance_power"] == 1.5
    assert metrics["prefetch_batches"] == 1
    assert metrics["pin_memory"] is True
    assert metrics["non_blocking"] is True
    assert metrics["validation_soma_units"] == "millivolts"
    assert metrics["validation_metric_ignore_start"] == 1
    assert metrics["validation_metric_mask_mode"] == "ignore-start"
    assert metrics["validation_metric_stitch_burn_in"] == 150
    assert metrics["validation_soma_affine_calibration"] is True
    assert metrics["history"][0]["validation_metric_ignore_start"] == 1
    assert metrics["history"][0]["validation_metric_mask_mode"] == "ignore-start"
    assert metrics["history"][0]["validation_metric_stitch_burn_in"] is None
    assert metrics["history"][0]["validation_soma_affine_calibration"] is True
    assert metrics["history"][0]["validation_soma_rmse_units"] == "millivolts"
    assert "train_last_sparse_soma_loss" in metrics["history"][0]
    assert 0.0 <= metrics["history"][0]["train_last_sparse_soma_fraction"] <= 1.0
    assert "train_last_sera_soma_loss" in metrics["history"][0]
    assert 0.0 <= metrics["history"][0]["train_last_sera_soma_relevance_mean"] <= 1.0
    assert "validation_soma_rmse" in metrics["history"][0]
    assert (metrics_path.parent / "experiment-registry.jsonl").exists()


def test_cli_train_supports_official_full_trace_sampling(tmp_path):
    data_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics_path = tmp_path / "train_metrics.json"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "8", "--time-steps", "12", "--input-dim", "8"])

    result = main(
        [
            "train",
            "--data",
            str(data_dir),
            "--registry",
            "off",
            "--val-data",
            str(data_dir),
            "--input-dim",
            "8",
            "--memory-units",
            "4",
            "--branches",
            "2",
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--window-size",
            "4",
            "--ignore-start",
            "2",
            "--window-sampling",
            "official-full-trace",
            "--epoch-samples",
            "8",
            "--file-load-fraction",
            "0.5",
            "--full-trace-length",
            "12",
            "--checkpoint",
            str(checkpoint_path),
            "--metrics",
            str(metrics_path),
        ]
    )

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert result == 0
    assert checkpoint_path.exists()
    assert metrics["samples"] == 8
