from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
import types

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

REPO_ROOT = Path(__file__).resolve().parents[3] / "elmneuron"
BNN_SIM_SRC = Path(__file__).resolve().parents[2] / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BNN_SIM_SRC) not in sys.path:
    sys.path.insert(0, str(BNN_SIM_SRC))


def _install_viz_stubs() -> None:
    if "matplotlib" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        matplotlib.colors = types.SimpleNamespace(from_levels_and_colors=lambda *args, **kwargs: (None, None))
        sys.modules["matplotlib"] = matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
    if "seaborn" not in sys.modules:
        seaborn = types.ModuleType("seaborn")
        seaborn.color_palette = lambda *args, **kwargs: [(0.0, 0.0, 0.0)] * 5
        seaborn.set_palette = lambda *args, **kwargs: None
        seaborn.set_style = lambda *args, **kwargs: None
        sys.modules["seaborn"] = seaborn


from bnn_sim.data import OfficialStyleFullTraceWindowDataset, ShardedNeuronIODataset
from bnn_sim.evaluate import evaluate_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elmneuron-root", default=str(REPO_ROOT))
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epoch-samples", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=500)
    parser.add_argument("--ignore-start", type=int, default=500)
    parser.add_argument("--burn-in", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--lr-schedule-steps", type=int, default=342000)
    parser.add_argument("--file-load-fraction", type=float, default=0.3)
    parser.add_argument("--full-trace-length", type=int, default=6000)
    parser.add_argument("--cache-shards", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    elmneuron_root = Path(args.elmneuron_root)
    if str(elmneuron_root) not in sys.path:
        sys.path.insert(0, str(elmneuron_root))
    _install_viz_stubs()
    from src.expressive_leaky_memory_neuron import ELM
    from src.neuronio.neuronio_train_utils import NeuronioLoss

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = json.loads((elmneuron_root / "models/num_memory_30/model_config.json").read_text(encoding="utf-8"))
    model = ELM(**model_config).to(args.device)
    criterion = NeuronioLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.lr_schedule_steps)

    train_base = ShardedNeuronIODataset(args.train_data, cache_shards=args.cache_shards)
    train_dataset = OfficialStyleFullTraceWindowDataset(
        train_base,
        window_size=args.window_size,
        start_offset=args.ignore_start,
        samples_per_epoch=args.epoch_samples,
        batch_size=args.batch_size,
        file_load_fraction=args.file_load_fraction,
        sequence_length=args.full_trace_length,
        seed=args.seed,
    )
    val_dataset = ShardedNeuronIODataset(args.val_data, cache_shards=args.cache_shards)

    best_rmse = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    total_start = time.perf_counter()
    history_path = output_dir / "history.jsonl"

    for epoch in range(1, args.epochs + 1):
        train_dataset.reshuffle(args.seed + epoch - 1)
        model.train()
        epoch_start = time.perf_counter()
        last_loss = 0.0

        for start in range(0, len(train_dataset), args.batch_size):
            batch_indices = range(start, min(start + args.batch_size, len(train_dataset)))
            x_np, y_np = train_dataset.get_batch(batch_indices)
            x = torch.from_numpy(x_np).to(args.device)
            y = torch.from_numpy(y_np).to(args.device)

            spike_target = y[..., 0]
            soma_target = y[..., 1]
            optimizer.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = criterion(
                outputs[:, args.burn_in :, :],
                (
                    spike_target[:, args.burn_in :],
                    soma_target[:, args.burn_in :],
                ),
            )
            loss.backward()
            optimizer.step()
            scheduler.step()
            last_loss = float(loss.detach().cpu())

        epoch_seconds = time.perf_counter() - epoch_start
        train_samples = len(train_dataset)
        val_metrics = evaluate_dataset(
            model,
            val_dataset,
            batch_size=args.val_batch_size,
            device=args.device,
            soma_units="millivolts",
            ignore_start=500,
            soma_affine_calibration=True,
            pin_memory=True,
            non_blocking=True,
        )
        epoch_metrics = {
            "epoch": epoch,
            "epoch_seconds": epoch_seconds,
            "train_samples": train_samples,
            "train_samples_per_second": train_samples / epoch_seconds if epoch_seconds > 0 else float("inf"),
            "train_last_loss": last_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_samples": int(val_metrics["samples"]),
            "validation_soma_rmse": float(val_metrics["soma_rmse"]),
            "validation_spike_auc": float(val_metrics["spike_auc"]),
        }
        history.append(epoch_metrics)
        if epoch_metrics["validation_soma_rmse"] < best_rmse:
            best_rmse = float(epoch_metrics["validation_soma_rmse"])
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_state.pt")
        history_path.open("a", encoding="utf-8").write(json.dumps(epoch_metrics) + "\n")
        progress = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_validation_soma_rmse": best_rmse,
            "train_presentations": epoch * len(train_dataset),
            "elapsed_seconds": time.perf_counter() - total_start,
            "history": history,
        }
        (output_dir / "progress.json").write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")

    total_seconds = time.perf_counter() - total_start
    metrics = {
        "epoch": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_soma_rmse": best_rmse,
        "train_presentations": args.epochs * len(train_dataset),
        "total_seconds": total_seconds,
        "train_samples_per_second": (args.epochs * len(train_dataset)) / total_seconds if total_seconds > 0 else float("inf"),
        "last_loss": last_loss,
        "history": history,
    }
    torch.save(model.state_dict(), output_dir / "final_state.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
