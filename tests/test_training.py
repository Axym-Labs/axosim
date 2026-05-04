import json

from bnn_sim.cli import main
from bnn_sim.checkpoint import load_checkpoint


def test_cli_train_writes_checkpoint_and_metrics(tmp_path):
    data_dir = tmp_path / "data"
    checkpoint_path = tmp_path / "checkpoint.pt"
    metrics_path = tmp_path / "train_metrics.json"

    main(["make-demo-data", "--output", str(data_dir), "--samples", "4", "--time-steps", "6", "--input-dim", "8"])

    result = main(
        [
            "train",
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
    assert "validation_soma_rmse" in metrics["history"][0]
