from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import pickle

import numpy as np


DEFAULT_Y_TRAIN_SOMA_BIAS = -67.7
DEFAULT_Y_SOMA_THRESHOLD = -55.0
DEFAULT_Y_TRAIN_SOMA_SCALE = 1 / 10


@dataclass(frozen=True)
class RawNeuronIO:
    inputs: np.ndarray
    spikes: np.ndarray
    soma: np.ndarray
    metadata: dict[str, object]


def parse_neuronio_pickle(path: str | Path) -> RawNeuronIO:
    """Parse a public NeuronIO simulation pickle into `(sim, time, channel)` arrays."""

    path = Path(path)
    with path.open("rb") as handle:
        try:
            experiment = pickle.load(handle, encoding="ASCII")
        except UnicodeDecodeError:
            handle.seek(0)
            experiment = pickle.load(handle, encoding="latin1")

    simulations = experiment["Results"]["listOfSingleSimulationDicts"]
    params = experiment["Params"]
    num_simulations = len(simulations)
    num_segments = len(params["allSegmentsType"])
    sim_duration_ms = int(round(float(params["totalSimDurationInSec"]) * 1000))
    num_synapses = num_segments * 2

    inputs = np.zeros((num_simulations, sim_duration_ms, num_synapses), dtype=np.float32)
    spikes = np.zeros((num_simulations, sim_duration_ms), dtype=np.float32)
    soma = np.zeros((num_simulations, sim_duration_ms), dtype=np.float32)

    for sim_idx, sim in enumerate(simulations):
        ex = _dict_to_binary(sim["exInputSpikeTimes"], num_segments, sim_duration_ms)
        inh = _dict_to_binary(sim["inhInputSpikeTimes"], num_segments, sim_duration_ms)
        inputs[sim_idx] = np.vstack([ex, inh]).T

        spike_times = (np.asarray(sim["outputSpikeTimes"], dtype=float) - 0.5).astype(int)
        valid_spike_times = spike_times[(spike_times >= 0) & (spike_times < sim_duration_ms)]
        spikes[sim_idx, valid_spike_times] = 1.0
        soma[sim_idx] = np.asarray(sim["somaVoltageLowRes"], dtype=np.float32)[:sim_duration_ms]

    metadata = {
        "num_simulations": num_simulations,
        "num_segments": num_segments,
        "sim_duration_ms": sim_duration_ms,
        "num_synapses": num_synapses,
        "segment_type": list(params["allSegmentsType"]),
        "segment_dist_from_soma": list(params.get("allSegments_DistFromSoma", [])),
    }
    return RawNeuronIO(inputs=inputs, spikes=spikes, soma=soma, metadata=metadata)


def convert_neuronio_pickles(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    shard_size: int = 128,
    window_size: int | None = None,
    window_stride: int | None = None,
    ignore_start: int = 0,
    y_soma_threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    y_train_soma_bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    y_train_soma_scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
) -> list[Path]:
    """Convert raw NeuronIO pickle files to deterministic `.npz` shards."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if ignore_start < 0:
        raise ValueError("ignore_start must be non-negative")
    if window_size is not None and window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_stride is not None and window_stride <= 0:
        raise ValueError("window_stride must be positive")
    window_stride = window_size if window_stride is None else window_stride

    files = _discover_raw_files(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest_entries = _load_existing_manifest(output_dir)
    pending_inputs: list[np.ndarray] = []
    pending_targets: list[np.ndarray] = []
    pending_ids: list[str] = []
    shard_index = _next_shard_index(output_dir)

    for raw_file in files:
        raw = parse_neuronio_pickle(raw_file)
        signed_inputs = raw.inputs * create_neuronio_input_type(raw.inputs.shape[-1])
        normalized_soma = normalize_soma(raw.soma, y_soma_threshold, y_train_soma_bias, y_train_soma_scale)
        targets = np.stack([raw.spikes, normalized_soma], axis=-1).astype(np.float32)

        for sample_inputs, sample_targets, sample_id in _iter_samples(
            raw_file,
            signed_inputs,
            targets,
            window_size=window_size,
            window_stride=window_stride,
            ignore_start=ignore_start,
        ):
            pending_inputs.append(sample_inputs.astype(np.int8, copy=False))
            pending_targets.append(sample_targets.astype(np.float32, copy=False))
            pending_ids.append(sample_id)
            if len(pending_inputs) == shard_size:
                shard = _write_shard(output_dir, shard_index, pending_inputs, pending_targets, pending_ids)
                written.append(shard)
                manifest_entries.append(_manifest_entry(shard, len(pending_inputs), pending_ids))
                _write_manifest(output_dir, manifest_entries)
                shard_index += 1
                pending_inputs, pending_targets, pending_ids = [], [], []

    if pending_inputs:
        shard = _write_shard(output_dir, shard_index, pending_inputs, pending_targets, pending_ids)
        written.append(shard)
        manifest_entries.append(_manifest_entry(shard, len(pending_inputs), pending_ids))
        _write_manifest(output_dir, manifest_entries)

    return written


def create_neuronio_input_type(num_input: int) -> np.ndarray:
    half = num_input // 2
    if half * 2 != num_input:
        raise ValueError(f"NeuronIO input dimension must be even, got {num_input}")
    return np.concatenate([np.ones(half), -np.ones(half)]).astype(np.float32)


def normalize_soma(
    soma: np.ndarray,
    threshold: float = DEFAULT_Y_SOMA_THRESHOLD,
    bias: float = DEFAULT_Y_TRAIN_SOMA_BIAS,
    scale: float = DEFAULT_Y_TRAIN_SOMA_SCALE,
) -> np.ndarray:
    clipped = np.asarray(soma, dtype=np.float32).copy()
    clipped[clipped > threshold] = threshold
    return ((clipped - bias) * scale).astype(np.float32)


def _dict_to_binary(spike_map: dict[int, list[int]], num_segments: int, sim_duration_ms: int) -> np.ndarray:
    result = np.zeros((num_segments, sim_duration_ms), dtype=np.float32)
    for segment, times in spike_map.items():
        segment_idx = int(segment)
        if segment_idx < 0 or segment_idx >= num_segments:
            continue
        for time in times:
            time_idx = int(time)
            if 0 <= time_idx < sim_duration_ms:
                result[segment_idx, time_idx] = 1.0
    return result


def _discover_raw_files(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"raw NeuronIO path does not exist: {input_path}")

    files = sorted(path for path in input_path.rglob("*sim__*") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no NeuronIO '*sim__*' files found in {input_path}")
    return files


def _iter_samples(
    raw_file: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    *,
    window_size: int | None,
    window_stride: int | None,
    ignore_start: int,
):
    if window_size is None:
        for sample_idx in range(inputs.shape[0]):
            yield inputs[sample_idx], targets[sample_idx], f"{raw_file.stem}:{sample_idx}"
        return

    assert window_stride is not None
    for sample_idx in range(inputs.shape[0]):
        last_start = inputs.shape[1] - window_size
        for start in range(ignore_start, last_start + 1, window_stride):
            end = start + window_size
            yield inputs[sample_idx, start:end], targets[sample_idx, start:end], f"{raw_file.stem}:{sample_idx}@{start}"


def _write_shard(output_dir: Path, shard_index: int, inputs: list[np.ndarray], targets: list[np.ndarray], sample_ids: list[str]) -> Path:
    shard_path = output_dir / f"neuronio_{shard_index:05d}.npz"
    stacked_inputs = np.stack(inputs)
    np.savez_compressed(
        shard_path,
        inputs=stacked_inputs,
        targets=np.stack(targets).astype(np.float32),
        sample_ids=np.asarray(sample_ids),
    )
    return shard_path


def _next_shard_index(output_dir: Path) -> int:
    max_index = -1
    for path in output_dir.glob("neuronio_*.npz"):
        try:
            max_index = max(max_index, int(path.stem.rsplit("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max_index + 1


def _load_existing_manifest(output_dir: Path) -> list[dict[str, object]]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("shards", []))


def _manifest_entry(shard_path: Path, count: int, sample_ids: list[str]) -> dict[str, object]:
    return {"file": shard_path.name, "count": count, "sample_ids": list(sample_ids)}


def _write_manifest(output_dir: Path, entries: list[dict[str, object]]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps({"version": 1, "shards": entries}, separators=(",", ":")),
        encoding="utf-8",
    )
