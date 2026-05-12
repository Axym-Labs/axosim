from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import shlex
import subprocess
import sys


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    slug: str


DEFAULT_DATASETS = (
    DatasetSpec("nmda-train", "selfishgene/single-neurons-as-deep-nets-nmda-train-data"),
    DatasetSpec("nmda-test", "selfishgene/single-neurons-as-deep-nets-nmda-test-data"),
)


@dataclass(frozen=True)
class SetupOptions:
    project_root: Path
    data_dir: Path
    install_kaggle: bool = False
    download_data: bool = False
    convert_raw: bool = False
    raw_dir: Path | None = None
    shard_dir: Path | None = None
    shard_size: int = 128
    skip_install: bool = False
    datasets: tuple[DatasetSpec, ...] = DEFAULT_DATASETS
    python_executable: str = sys.executable


@dataclass(frozen=True)
class SetupStep:
    description: str
    command: str | None = None
    argv: tuple[str, ...] | None = None
    mkdir: Path | None = None


@dataclass(frozen=True)
class SetupPlan:
    steps: tuple[SetupStep, ...]
    next_steps: tuple[str, ...] = field(default_factory=tuple)


def build_setup_plan(options: SetupOptions) -> SetupPlan:
    project_root = options.project_root
    data_dir = options.data_dir
    raw_root = data_dir / "raw"
    shard_root = data_dir / "shards"
    steps: list[SetupStep] = [
        SetupStep("Create data directory", mkdir=data_dir),
        SetupStep("Create raw data directory", mkdir=raw_root),
        SetupStep("Create shard directory", mkdir=shard_root),
        SetupStep("Create run output directory", mkdir=project_root / "runs"),
        SetupStep("Create checkpoint directory", mkdir=project_root / "checkpoints"),
    ]

    if not options.skip_install:
        editable_target = f"{project_root}[dev]"
        steps.append(
            _command_step(
                "Install the local package in editable mode",
                (options.python_executable, "-m", "pip", "install", "-e", editable_target),
            )
        )

    if options.install_kaggle or options.download_data:
        steps.append(
            _command_step(
                "Install Kaggle downloader dependencies",
                (options.python_executable, "-m", "pip", "install", "requests"),
            )
        )

    if options.download_data:
        for dataset in options.datasets:
            target = raw_root / dataset.name
            steps.append(SetupStep(f"Create raw directory for {dataset.name}", mkdir=target))
            steps.append(
                _command_step(
                    f"Download {dataset.name} from Kaggle",
                    (
                        options.python_executable,
                        "-m",
                        "bnn_sim.kaggle_download",
                        dataset.slug,
                        str(target / f"{dataset.name}.zip"),
                        "--unzip-to",
                        str(target),
                    ),
                )
            )

    if options.convert_raw:
        raw_dir = options.raw_dir or raw_root
        shard_dir = options.shard_dir or shard_root
        steps.append(SetupStep("Create conversion output directory", mkdir=shard_dir))
        steps.append(
            _command_step(
                "Convert raw NeuronIO pickles to deterministic shards",
                (
                    options.python_executable,
                    "-m",
                    "bnn_sim.cli",
                    "convert-neuronio",
                    "--input",
                    str(raw_dir),
                    "--output",
                    str(shard_dir),
                    "--shard-size",
                    str(options.shard_size),
                ),
            )
        )

    next_steps = (
        "Place kaggle.json under ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY before using --download-data.",
        "Convert downloaded raw files with --convert-raw if you did not convert during setup.",
        "Run the recommended preset with: python -m bnn_sim.cli train --preset paper-random-fulltrace --data <train-shards> --val-data <val-shards> --epochs 10 --epoch-samples 80000 --shard-reuse-batches 58 --checkpoint runs/paper-like.pt --metrics runs/paper-like-metrics.json",
    )
    return SetupPlan(steps=tuple(steps), next_steps=next_steps)


def run_setup_plan(plan: SetupPlan, *, dry_run: bool = False) -> None:
    for step in plan.steps:
        print(f"[setup] {step.description}")
        if step.mkdir is not None:
            print(f"  mkdir {step.mkdir}")
            if not dry_run:
                step.mkdir.mkdir(parents=True, exist_ok=True)
        if step.argv is not None:
            print(f"  {step.command}")
            if not dry_run:
                subprocess.run(step.argv, check=True)

    print("[setup] Next steps")
    for next_step in plan.next_steps:
        print(f"  - {next_step}")


def _command_step(description: str, argv: tuple[str, ...]) -> SetupStep:
    return SetupStep(description=description, command=_format_command(argv), argv=argv)


def _format_command(argv: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(part) for part in argv)
