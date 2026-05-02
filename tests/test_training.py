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
