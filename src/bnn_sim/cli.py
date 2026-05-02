from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from bnn_sim.checkpoint import load_checkpoint
from bnn_sim.data import DeterministicWindowDataset, ShardedNeuronIODataset, write_demo_shards
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

    if args.command == "evaluate":
        torch.manual_seed(args.seed)
        dataset = _build_dataset(args.data, window_size=args.window_size, window_stride=args.window_stride, ignore_start=args.ignore_start)
        if args.checkpoint:
            model = _load_checkpoint_for_evaluate(args)
        else:
            model = _build_model_from_args(args)
        metrics = evaluate_dataset(model, dataset, batch_size=args.batch_size, device=args.device)
        write_metrics(metrics, args.output)
        return 0

    if args.command == "train":
        torch.manual_seed(args.seed)
        dataset = _build_dataset(
            args.data,
            shuffle=True,
            seed=args.seed,
            window_size=args.window_size,
            window_stride=args.window_stride,
            ignore_start=args.ignore_start,
        )
        model = _build_model_from_args(args)
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
    seed: int = 0,
    window_size: int | None = None,
    window_stride: int | None = None,
    ignore_start: int = 0,
):
    dataset = ShardedNeuronIODataset(data_path, shuffle=shuffle, seed=seed)
    if window_size is None:
        return dataset
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
