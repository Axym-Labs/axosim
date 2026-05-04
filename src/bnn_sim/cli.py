from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.data import (
    DeterministicWindowDataset,
    OfficialStyleFullTraceWindowDataset,
    RandomFullTraceWindowDataset,
    ShardedNeuronIODataset,
    repack_shards_as_npy,
    write_demo_shards,
)
from bnn_sim.evaluate import evaluate_dataset, write_metrics
from bnn_sim.model import BranchELM, BranchELMConfig
from bnn_sim.neuronio_raw import convert_neuronio_pickles
from bnn_sim.official import create_official_elm, load_official_elm_config, load_official_state_dict
from bnn_sim.train import train_and_save

DEFAULT_OFFICIAL_MODEL_CONFIG = "configs/branch_elm_30_official.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bnn-sim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("make-demo-data", help="write deterministic demo NPZ shards")
    demo.add_argument("--output", required=True)
    demo.add_argument("--samples", type=int, default=8)
    demo.add_argument("--shards", type=int, default=1)
    demo.add_argument("--time-steps", type=int, default=32)
    demo.add_argument("--input-dim", type=int, default=64)
    demo.add_argument("--seed", type=int, default=0)

    convert = subparsers.add_parser("convert-neuronio", help="convert raw NeuronIO pickle files to deterministic shards")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--shard-size", type=int, default=128)
    convert.add_argument("--window-size", type=int)
    convert.add_argument("--window-stride", type=int)
    convert.add_argument("--ignore-start", type=int, default=0)

    repack = subparsers.add_parser("repack-shards", help="repack shards as sliceable .npy arrays")
    repack.add_argument("--input", required=True)
    repack.add_argument("--output", required=True)
    repack.add_argument("--input-dtype", choices=["int8", "float32"], default="int8")

    evaluate = subparsers.add_parser("evaluate", help="evaluate an untrained Branch-ELM smoke baseline")
    evaluate.add_argument("--data", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--input-dim", type=int, default=1278)
    evaluate.add_argument("--memory-units", type=int, default=30)
    evaluate.add_argument("--branches", type=int, default=32)
    evaluate.add_argument("--model-kind", choices=["baseline", "official"], default="baseline")
    evaluate.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=0)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--window-size", type=int)
    evaluate.add_argument("--window-stride", type=int)
    evaluate.add_argument("--ignore-start", type=int, default=0)
    evaluate.add_argument("--cache-shards", type=int, default=1)
    evaluate.add_argument("--soma-units", choices=["millivolts", "normalized"], default="millivolts")
    evaluate.add_argument("--metric-ignore-start", type=int, default=0)
    evaluate.add_argument("--metric-mask-mode", choices=["ignore-start", "official-overlap"], default="ignore-start")
    evaluate.add_argument("--metric-stitch-burn-in", type=int, default=150)
    evaluate.add_argument("--soma-affine-calibration", action="store_true")
    evaluate.add_argument("--pin-memory", action="store_true")
    evaluate.add_argument("--blocking-transfer", action="store_false", dest="non_blocking")
    evaluate.set_defaults(non_blocking=True)

    train = subparsers.add_parser("train", help="train a deterministic Branch-ELM baseline")
    train.add_argument("--data", required=True)
    train.add_argument("--checkpoint", required=True)
    train.add_argument("--metrics", required=True)
    train.add_argument("--input-dim", type=int, default=1278)
    train.add_argument("--memory-units", type=int, default=30)
    train.add_argument("--branches", type=int, default=32)
    train.add_argument("--model-kind", choices=["baseline", "official"], default="baseline")
    train.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--burn-in", type=int, default=0)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="cpu")
    train.add_argument("--window-size", type=int)
    train.add_argument("--window-stride", type=int)
    train.add_argument("--ignore-start", type=int, default=0)
    train.add_argument(
        "--window-sampling",
        choices=["deterministic", "random-full-trace", "official-full-trace"],
        default="deterministic",
    )
    train.add_argument("--epoch-samples", type=int)
    train.add_argument("--shard-reuse-batches", type=int, default=1)
    train.add_argument("--file-load-fraction", type=float, default=0.3)
    train.add_argument("--full-trace-length", type=int)
    train.add_argument("--shuffle-mode", choices=["sample", "shard", "none"], default="shard")
    train.add_argument("--cache-shards", type=int, default=1)
    train.add_argument("--val-data")
    train.add_argument("--val-batch-size", type=int)
    train.add_argument("--val-cache-shards", type=int, default=1)
    train.add_argument("--val-window-size", type=int)
    train.add_argument("--val-window-stride", type=int)
    train.add_argument("--val-ignore-start", type=int, default=0)
    train.add_argument("--val-soma-units", choices=["millivolts", "normalized"], default="millivolts")
    train.add_argument("--val-metric-ignore-start", type=int, default=500)
    train.add_argument("--val-metric-mask-mode", choices=["ignore-start", "official-overlap"], default="ignore-start")
    train.add_argument("--val-metric-stitch-burn-in", type=int, default=150)
    train.add_argument("--val-soma-affine-calibration", action="store_true", default=True)
    train.add_argument("--no-val-soma-affine-calibration", action="store_false", dest="val_soma_affine_calibration")
    train.add_argument("--best-checkpoint")
    train.add_argument("--max-train-batches", type=int)
    train.add_argument("--lr-schedule", choices=["constant", "cosine"], default="constant")
    train.add_argument("--lr-schedule-steps", type=int)
    train.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    train.add_argument("--weight-decay", type=float, default=0.0)
    train.add_argument("--l1-lambda", type=float, default=0.0)
    train.add_argument("--spike-loss-weight", type=float, default=0.5)
    train.add_argument("--soma-loss-weight", type=float, default=0.5)
    train.add_argument("--prefetch-batches", type=int, default=0)
    train.add_argument("--pin-memory", action="store_true")
    train.add_argument("--blocking-transfer", action="store_false", dest="non_blocking")
    train.set_defaults(non_blocking=True)

    benchmark = subparsers.add_parser("benchmark", help="time Branch-ELM forward passes on random input")
    benchmark.add_argument("--input-dim", type=int, default=1278)
    benchmark.add_argument("--memory-units", type=int, default=30)
    benchmark.add_argument("--branches", type=int, default=32)
    benchmark.add_argument("--model-kind", choices=["baseline", "official"], default="baseline")
    benchmark.add_argument("--model-config", default=DEFAULT_OFFICIAL_MODEL_CONFIG)
    benchmark.add_argument("--batch-size", type=int, default=1)
    benchmark.add_argument("--time-steps", type=int, default=500)
    benchmark.add_argument("--runs", type=int, default=10)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--compile", action="store_true", dest="compile_model")
    benchmark.add_argument("--output")

    args = parser.parse_args(argv)

    if args.command == "make-demo-data":
        write_demo_shards(
            args.output,
            shard_count=args.shards,
            samples_per_shard=args.samples,
            time_steps=args.time_steps,
            input_dim=args.input_dim,
            seed=args.seed,
        )
        return 0

    if args.command == "convert-neuronio":
        try:
            written = convert_neuronio_pickles(
                args.input,
                args.output,
                shard_size=args.shard_size,
                window_size=args.window_size,
                window_stride=args.window_stride,
                ignore_start=args.ignore_start,
            )
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(f"wrote_shards={len(written)}")
        return 0

    if args.command == "repack-shards":
        try:
            input_dtype = np.float32 if args.input_dtype == "float32" else np.int8
            written = repack_shards_as_npy(args.input, args.output, input_dtype=input_dtype)
        except (FileNotFoundError, ValueError) as error:
            print(str(error))
            return 2
        print(f"wrote_shards={len(written)}")
        return 0

    if args.command == "evaluate":
        torch.manual_seed(args.seed)
        dataset = _build_dataset(
            args.data,
            window_size=args.window_size,
            window_stride=args.window_stride,
            ignore_start=args.ignore_start,
            cache_shards=args.cache_shards,
        )
        if args.checkpoint:
            model = _load_checkpoint_for_evaluate(args)
        else:
            model = _build_model_from_args(args)
        metrics = evaluate_dataset(
            model,
            dataset,
            batch_size=args.batch_size,
            device=args.device,
            soma_units=args.soma_units,
            ignore_start=args.metric_ignore_start,
            mask_mode=args.metric_mask_mode,
            stitch_burn_in=args.metric_stitch_burn_in,
            soma_affine_calibration=args.soma_affine_calibration,
            pin_memory=args.pin_memory,
            non_blocking=args.non_blocking,
        )
        write_metrics(metrics, args.output)
        return 0

    if args.command == "train":
        torch.manual_seed(args.seed)
        dataset = _build_dataset(
            args.data,
            shuffle=args.shuffle_mode != "none",
            shuffle_mode="shard" if args.shuffle_mode == "none" else args.shuffle_mode,
            seed=args.seed,
            window_size=args.window_size,
            window_stride=args.window_stride,
            ignore_start=args.ignore_start,
            cache_shards=args.cache_shards,
            window_sampling=args.window_sampling,
            epoch_samples=args.epoch_samples,
            batch_size=args.batch_size,
            shard_reuse_batches=args.shard_reuse_batches,
            file_load_fraction=args.file_load_fraction,
            sequence_length=args.full_trace_length,
        )
        validation_dataset = None
        if args.val_data:
            validation_dataset = _build_dataset(
                args.val_data,
                window_size=args.val_window_size,
                window_stride=args.val_window_stride,
                ignore_start=args.val_ignore_start,
                cache_shards=args.val_cache_shards,
            )
        model = _build_model_from_args(args)
        best_checkpoint = args.best_checkpoint
        if validation_dataset is not None and best_checkpoint is None:
            checkpoint_path = Path(args.checkpoint)
            best_checkpoint = str(checkpoint_path.with_name(f"{checkpoint_path.stem}-best{checkpoint_path.suffix}"))
        train_and_save(
            model,
            dataset,
            checkpoint_path=args.checkpoint,
            metrics_path=args.metrics,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            burn_in=args.burn_in,
            device=args.device,
            seed=args.seed,
            validation_dataset=validation_dataset,
            validation_batch_size=args.val_batch_size,
            best_checkpoint_path=best_checkpoint,
            validation_soma_units=args.val_soma_units,
            validation_metric_ignore_start=args.val_metric_ignore_start,
            validation_metric_mask_mode=args.val_metric_mask_mode,
            validation_metric_stitch_burn_in=args.val_metric_stitch_burn_in,
            validation_soma_affine_calibration=args.val_soma_affine_calibration,
            max_train_batches=args.max_train_batches,
            lr_schedule=args.lr_schedule,
            lr_schedule_steps=args.lr_schedule_steps,
            optimizer_name=args.optimizer,
            weight_decay=args.weight_decay,
            l1_lambda=args.l1_lambda,
            spike_loss_weight=args.spike_loss_weight,
            soma_loss_weight=args.soma_loss_weight,
            prefetch_batches=args.prefetch_batches,
            pin_memory=args.pin_memory,
            non_blocking=args.non_blocking,
        )
        return 0

    if args.command == "benchmark":
        torch.manual_seed(0)
        model = _build_model_from_args(args).to(args.device).eval()
        if args.compile_model:
            model = torch.compile(model)
        input_dim = int(getattr(model, "num_input", args.input_dim))
        x = torch.randn(args.batch_size, args.time_steps, input_dim, device=args.device)
        with torch.inference_mode():
            model(x)
            start = time.perf_counter()
            for _ in range(args.runs):
                model(x)
            elapsed = time.perf_counter() - start
        report = {
            "batch_size": args.batch_size,
            "input_dim": input_dim,
            "model_kind": args.model_kind,
            "time_steps": args.time_steps,
            "runs": args.runs,
            "mean_forward_seconds": elapsed / args.runs,
            "compiled": bool(args.compile_model),
            "device": args.device,
        }
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            print(f"mean_forward_seconds={report['mean_forward_seconds']:.6f}")
        return 0

    raise AssertionError(f"unhandled command {args.command}")


def _build_dataset(
    data_path: str,
    *,
    shuffle: bool = False,
    shuffle_mode: str = "sample",
    seed: int = 0,
    window_size: int | None = None,
    window_stride: int | None = None,
    ignore_start: int = 0,
    cache_shards: int = 0,
    window_sampling: str = "deterministic",
    epoch_samples: int | None = None,
    batch_size: int = 8,
    shard_reuse_batches: int = 1,
    file_load_fraction: float = 0.3,
    sequence_length: int | None = None,
):
    dataset = ShardedNeuronIODataset(
        data_path,
        shuffle=shuffle,
        shuffle_mode=shuffle_mode,  # type: ignore[arg-type]
        seed=seed,
        cache_shards=cache_shards,
    )
    if window_size is None:
        return dataset
    if window_sampling == "random-full-trace":
        return RandomFullTraceWindowDataset(
            dataset,
            window_size=window_size,
            start_offset=ignore_start,
            samples_per_epoch=epoch_samples,
            batch_size=batch_size,
            shard_reuse_batches=shard_reuse_batches,
            sequence_length=sequence_length,
            seed=seed,
        )
    if window_sampling == "official-full-trace":
        return OfficialStyleFullTraceWindowDataset(
            dataset,
            window_size=window_size,
            start_offset=ignore_start,
            samples_per_epoch=epoch_samples,
            batch_size=batch_size,
            file_load_fraction=file_load_fraction,
            sequence_length=sequence_length,
            seed=seed,
        )
    if window_sampling != "deterministic":
        raise ValueError("window_sampling must be 'deterministic', 'random-full-trace', or 'official-full-trace'")
    return DeterministicWindowDataset(
        dataset,
        window_size=window_size,
        stride=window_stride,
        start_offset=ignore_start,
    )


def _build_model_from_args(args):
    if args.model_kind == "official":
        return create_official_elm(load_official_elm_config(args.model_config))
    config = BranchELMConfig(input_dim=args.input_dim, memory_units=args.memory_units, num_branches=args.branches)
    return BranchELM(config)


def _load_checkpoint_for_evaluate(args):
    try:
        model, _ = load_checkpoint(args.checkpoint, map_location=args.device)
        return model
    except (KeyError, ValueError):
        if args.model_kind != "official":
            raise
        model = create_official_elm(load_official_elm_config(args.model_config))
        state_dict = load_official_state_dict(args.checkpoint, map_location=args.device)
        model.load_state_dict(state_dict, strict=True)
        return model


if __name__ == "__main__":
    raise SystemExit(main())
