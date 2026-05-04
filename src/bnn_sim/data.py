from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class NeuronIOSample:
    sample_id: str
    inputs: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class _ShardFiles:
    name: str
    path: Path | None = None
    inputs_path: Path | None = None
    targets_path: Path | None = None
    sample_ids_path: Path | None = None


class ShardedNeuronIODataset:
    """Deterministic reader for pre-sharded NeuronIO-style NPZ files."""

    def __init__(
        self,
        root: str | Path,
        *,
        shuffle: bool = False,
        seed: int = 0,
        shuffle_mode: Literal["sample", "shard"] = "sample",
        cache_shards: int = 0,
    ) -> None:
        self.root = Path(root)
        if shuffle_mode not in {"sample", "shard"}:
            raise ValueError("shuffle_mode must be 'sample' or 'shard'")
        if cache_shards < 0:
            raise ValueError("cache_shards must be non-negative")
        self.cache_shards = cache_shards
        self.shuffle_mode = shuffle_mode
        self.shuffle_enabled = shuffle
        self._shard_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.shards, self._index = self._load_manifest_shards_and_index()

        if not self.shards:
            self.shards = [_ShardFiles(name=path.name, path=path) for path in sorted(self.root.glob("*.npz"))]
        if not self.shards:
            raise FileNotFoundError(f"no supported shards found in {self.root}")
        if not self._index:
            for shard_idx, shard in enumerate(self.shards):
                with _open_shard(shard) as data:
                    inputs = data["inputs"]
                    sample_ids = _sample_ids(data, Path(shard.name).stem, inputs.shape[0])
                    for sample_idx, sample_id in enumerate(sample_ids):
                        self._index.append((shard_idx, sample_idx, str(sample_id)))
        self._base_index = list(self._index)

        if shuffle:
            self.reshuffle(seed)

    def __len__(self) -> int:
        return len(self._index)

    def reshuffle(self, seed: int) -> None:
        if not self.shuffle_enabled:
            return
        rng = np.random.default_rng(seed)
        if self.shuffle_mode == "sample":
            order = rng.permutation(len(self._base_index))
            self._index = [self._base_index[int(i)] for i in order]
        else:
            self._index = _shuffle_by_shard(self._base_index, rng)

    def __getitem__(self, index: int) -> NeuronIOSample:
        shard_idx, sample_idx, sample_id = self._index[index]
        inputs, targets = self._load_sample(shard_idx, sample_idx)

        if targets.ndim != 2 or targets.shape[-1] != 2:
            raise ValueError(f"targets must have shape (time, 2), got {targets.shape}")
        if inputs.ndim != 2:
            raise ValueError(f"inputs must have shape (time, channels), got {inputs.shape}")

        return NeuronIOSample(sample_id=sample_id, inputs=inputs, targets=targets)

    def get_batch(self, indices: range | list[int]) -> tuple[np.ndarray, np.ndarray]:
        entries = [self._index[int(index)] for index in indices]
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for shard_idx, sample_indices in _group_sample_indices_by_shard(entries):
            shard_inputs, shard_targets = self._load_cached_shard(shard_idx) if self.cache_shards > 0 else self._load_shard(shard_idx)
            inputs.append(np.asarray(shard_inputs[sample_indices], dtype=np.float32))
            targets.append(np.asarray(shard_targets[sample_indices], dtype=np.float32))
        return np.concatenate(inputs, axis=0), np.concatenate(targets, axis=0)

    def get_sample_ids(self, indices: range | list[int]) -> list[str]:
        return [self._index[int(index)][2] for index in indices]

    def _load_sample(self, shard_idx: int, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if self.cache_shards <= 0:
            with _open_shard(self.shards[shard_idx]) as data:
                inputs = np.asarray(data["inputs"][sample_idx], dtype=np.float32)
                targets = np.asarray(data["targets"][sample_idx], dtype=np.float32)
            return inputs, targets

        inputs, targets = self._load_cached_shard(shard_idx)
        return np.asarray(inputs[sample_idx], dtype=np.float32), np.asarray(targets[sample_idx], dtype=np.float32)

    def _load_cached_shard(self, shard_idx: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._shard_cache.get(shard_idx)
        if cached is not None:
            self._shard_cache.move_to_end(shard_idx)
            return cached

        with _open_shard(self.shards[shard_idx]) as data:
            cached = (
                np.asarray(data["inputs"]),
                np.asarray(data["targets"], dtype=np.float32),
            )
        self._shard_cache[shard_idx] = cached
        while len(self._shard_cache) > self.cache_shards:
            self._shard_cache.popitem(last=False)
        return cached

    def _load_shard(self, shard_idx: int) -> tuple[np.ndarray, np.ndarray]:
        with _open_shard(self.shards[shard_idx]) as data:
            return (
                np.asarray(data["inputs"]),
                np.asarray(data["targets"], dtype=np.float32),
            )

    def _load_manifest_shards_and_index(self) -> tuple[list[_ShardFiles], list[tuple[int, int, str]]]:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            return [], []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shards: list[_ShardFiles] = []
        for shard_entry in manifest.get("shards", []):
            if "inputs_file" in shard_entry and "targets_file" in shard_entry:
                sample_ids_file = shard_entry.get("sample_ids_file")
                name = str(shard_entry.get("file", shard_entry["inputs_file"]))
                shards.append(
                    _ShardFiles(
                        name=name,
                        inputs_path=self.root / shard_entry["inputs_file"],
                        targets_path=self.root / shard_entry["targets_file"],
                        sample_ids_path=self.root / sample_ids_file if sample_ids_file else None,
                    )
                )
        if not shards:
            shards = [_ShardFiles(name=path.name, path=path) for path in sorted(self.root.glob("*.npz"))]
        shard_lookup = {shard.name: idx for idx, shard in enumerate(shards)}
        index: list[tuple[int, int, str]] = []
        for shard_entry in manifest.get("shards", []):
            shard_name = str(shard_entry.get("file", shard_entry.get("inputs_file", "")))
            if shard_name not in shard_lookup:
                continue
            shard_idx = shard_lookup[shard_name]
            count = int(shard_entry["count"])
            sample_ids = shard_entry.get("sample_ids")
            if sample_ids is None:
                sample_ids = [f"{Path(shard_name).stem}:{sample_idx}" for sample_idx in range(count)]
            for sample_idx, sample_id in enumerate(sample_ids[:count]):
                index.append((shard_idx, sample_idx, str(sample_id)))
        return shards, index


class DeterministicWindowDataset:
    """Enumerate fixed-size windows from another NeuronIO-style dataset."""

    def __init__(
        self,
        base: ShardedNeuronIODataset,
        *,
        window_size: int,
        stride: int | None = None,
        start_offset: int = 0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        stride = window_size if stride is None else stride
        if stride <= 0:
            raise ValueError("stride must be positive")

        self.base = base
        self.window_size = window_size
        self.stride = stride
        self.start_offset = start_offset
        self._index: list[tuple[int, int]] = []

        for sample_idx in range(len(base)):
            sample_len = base[sample_idx].inputs.shape[0]
            last_start = sample_len - window_size
            for start in range(start_offset, last_start + 1, stride):
                self._index.append((sample_idx, start))

        if not self._index:
            raise ValueError("window configuration produced no samples")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> NeuronIOSample:
        sample_idx, start = self._index[index]
        base_sample = self.base[sample_idx]
        end = start + self.window_size
        return NeuronIOSample(
            sample_id=f"{base_sample.sample_id}@{start}",
            inputs=base_sample.inputs[start:end],
            targets=base_sample.targets[start:end],
        )

    def get_batch(self, indices: range | list[int]) -> tuple[np.ndarray, np.ndarray]:
        samples = [self[int(index)] for index in indices]
        return (
            np.stack([sample.inputs for sample in samples]).astype(np.float32, copy=False),
            np.stack([sample.targets for sample in samples]).astype(np.float32, copy=False),
        )

    def get_sample_ids(self, indices: range | list[int]) -> list[str]:
        return [self[int(index)].sample_id for index in indices]


class RandomFullTraceWindowDataset:
    """Sample deterministic random windows from cached full-trace shards."""

    def __init__(
        self,
        base: ShardedNeuronIODataset,
        *,
        window_size: int = 500,
        start_offset: int = 500,
        samples_per_epoch: int | None = None,
        batch_size: int = 8,
        shard_reuse_batches: int = 1,
        sequence_length: int | None = None,
        seed: int = 0,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if samples_per_epoch is not None and samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if shard_reuse_batches <= 0:
            raise ValueError("shard_reuse_batches must be positive")
        if sequence_length is not None and sequence_length <= 0:
            raise ValueError("sequence_length must be positive")

        self.base = base
        self.window_size = window_size
        self.start_offset = start_offset
        self.batch_size = batch_size
        self.shard_reuse_batches = shard_reuse_batches
        self.sequence_length = sequence_length
        self.samples_per_epoch = samples_per_epoch or len(base)
        self._by_shard: dict[int, list[tuple[int, str]]] = {}
        self._sequence_lengths: dict[int, int] = {}
        for shard_idx, sample_idx, sample_id in base._base_index:
            self._by_shard.setdefault(shard_idx, []).append((sample_idx, sample_id))
        if not self._by_shard:
            raise ValueError("base dataset has no samples")
        self.reshuffle(seed)

    def __len__(self) -> int:
        return len(self._index)

    def reshuffle(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        shard_indices = np.array(sorted(self._by_shard), dtype=np.int64)
        index: list[tuple[int, int, int, str]] = []
        num_batches = (self.samples_per_epoch + self.batch_size - 1) // self.batch_size
        batch_count = 0
        while batch_count < num_batches:
            shard_idx = int(rng.choice(shard_indices))
            for _ in range(min(self.shard_reuse_batches, num_batches - batch_count)):
                shard_samples = self._by_shard[shard_idx]
                replace = len(shard_samples) < self.batch_size
                chosen = rng.choice(len(shard_samples), size=self.batch_size, replace=replace)
                max_start = self._sequence_length(shard_idx) - self.window_size
                if self.start_offset > max_start:
                    raise ValueError(
                        f"start_offset={self.start_offset} leaves no {self.window_size}-step windows "
                        f"for shard {self.base.shards[shard_idx].name}"
                    )
                starts = rng.integers(self.start_offset, max_start + 1, size=self.batch_size)
                for chosen_idx, start in zip(chosen, starts):
                    sample_idx, sample_id = shard_samples[int(chosen_idx)]
                    index.append((shard_idx, sample_idx, int(start), f"{sample_id}@{int(start)}"))
                batch_count += 1
        self._index = index[: self.samples_per_epoch]

    def __getitem__(self, index: int) -> NeuronIOSample:
        shard_idx, sample_idx, start, sample_id = self._index[index]
        shard_inputs, shard_targets = self.base._load_cached_shard(shard_idx)
        end = start + self.window_size
        return NeuronIOSample(
            sample_id=sample_id,
            inputs=np.asarray(shard_inputs[sample_idx, start:end], dtype=np.float32),
            targets=np.asarray(shard_targets[sample_idx, start:end], dtype=np.float32),
        )

    def get_batch(self, indices: range | list[int]) -> tuple[np.ndarray, np.ndarray]:
        entries = [self._index[int(index)] for index in indices]
        inputs: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for shard_idx, shard_entries in _group_full_trace_window_entries_by_shard(entries):
            shard_inputs, shard_targets = self.base._load_cached_shard(shard_idx)
            for sample_idx, start, _ in shard_entries:
                end = start + self.window_size
                inputs.append(np.asarray(shard_inputs[sample_idx, start:end], dtype=np.float32))
                targets.append(np.asarray(shard_targets[sample_idx, start:end], dtype=np.float32))
        return np.stack(inputs).astype(np.float32, copy=False), np.stack(targets).astype(np.float32, copy=False)

    def get_sample_ids(self, indices: range | list[int]) -> list[str]:
        return [self._index[int(index)][3] for index in indices]

    def _sequence_length(self, shard_idx: int) -> int:
        if self.sequence_length is not None:
            return self.sequence_length
        if shard_idx not in self._sequence_lengths:
            with _open_shard(self.base.shards[shard_idx]) as data:
                self._sequence_lengths[shard_idx] = int(data["inputs"].shape[1])
        return self._sequence_lengths[shard_idx]


def write_demo_shards(
    root: str | Path,
    *,
    shard_count: int = 2,
    samples_per_shard: int = 8,
    time_steps: int = 32,
    input_dim: int = 64,
    seed: int = 0,
) -> None:
    """Create small deterministic NeuronIO-style shards for smoke tests."""

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for shard_idx in range(shard_count):
        inputs = rng.binomial(1, 0.03, size=(samples_per_shard, time_steps, input_dim))
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=inputs.shape)
        inputs = (inputs * signs).astype(np.float32)

        soma = inputs.sum(axis=-1, keepdims=True).astype(np.float32) / max(input_dim, 1)
        spikes = (soma[..., 0] > 0.02).astype(np.float32)[..., None]
        targets = np.concatenate([spikes, soma], axis=-1).astype(np.float32)
        sample_ids = np.array([f"shard{shard_idx:03d}_sample{i:04d}" for i in range(samples_per_shard)])

        np.savez_compressed(root / f"demo_{shard_idx:03d}.npz", inputs=inputs, targets=targets, sample_ids=sample_ids)


def repack_shards_as_npy(
    input_root: str | Path,
    output_root: str | Path,
    *,
    input_dtype: np.dtype | type = np.int8,
) -> list[Path]:
    """Repack existing shards as sliceable `.npy` arrays with identical sample order."""

    source = ShardedNeuronIODataset(input_root, cache_shards=0)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    manifest_entries: list[dict[str, object]] = []

    for shard_idx, shard in enumerate(source.shards):
        with _open_shard(shard) as data:
            inputs = np.asarray(data["inputs"], dtype=np.float32)
            targets = np.asarray(data["targets"], dtype=np.float32)
            sample_ids = _sample_ids(data, Path(shard.name).stem, inputs.shape[0]).astype(str)
        stored_inputs = inputs.astype(input_dtype, copy=False)
        if not np.array_equal(stored_inputs.astype(np.float32), inputs):
            raise ValueError(f"input_dtype={np.dtype(input_dtype)} would change values in shard {shard.name}")

        stem = f"neuronio_{shard_idx:05d}"
        inputs_file = f"{stem}_inputs.npy"
        targets_file = f"{stem}_targets.npy"
        sample_ids_file = f"{stem}_sample_ids.npy"
        np.save(output_root / inputs_file, stored_inputs)
        np.save(output_root / targets_file, targets)
        np.save(output_root / sample_ids_file, sample_ids)
        written.append(output_root / inputs_file)
        manifest_entries.append(
            {
                "file": stem,
                "inputs_file": inputs_file,
                "targets_file": targets_file,
                "sample_ids_file": sample_ids_file,
                "count": int(inputs.shape[0]),
                "sample_ids": sample_ids.tolist(),
            }
        )

    _write_manifest(output_root, manifest_entries)
    return written


def _sample_ids(data: np.lib.npyio.NpzFile, shard_name: str, size: int) -> np.ndarray:
    if "sample_ids" in data.files:
        return data["sample_ids"]
    return np.array([f"{shard_name}:{i}" for i in range(size)])


class _NpyShard:
    def __init__(self, shard: _ShardFiles) -> None:
        if shard.inputs_path is None or shard.targets_path is None:
            raise ValueError("npy shard requires inputs and targets paths")
        self.inputs = np.load(shard.inputs_path, mmap_mode="r")
        self.targets = np.load(shard.targets_path, mmap_mode="r")
        self.sample_ids = (
            np.load(shard.sample_ids_path, mmap_mode="r", allow_pickle=False)
            if shard.sample_ids_path is not None and shard.sample_ids_path.exists()
            else None
        )
        self.files = ["inputs", "targets"] + (["sample_ids"] if self.sample_ids is not None else [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def __getitem__(self, key: str) -> np.ndarray:
        if key == "inputs":
            return self.inputs
        if key == "targets":
            return self.targets
        if key == "sample_ids" and self.sample_ids is not None:
            return self.sample_ids
        raise KeyError(key)


def _open_shard(shard: _ShardFiles):
    if shard.path is not None:
        return np.load(shard.path, mmap_mode="r")
    return _NpyShard(shard)


def _write_manifest(output_root: Path, entries: list[dict[str, object]]) -> None:
    (output_root / "manifest.json").write_text(
        json.dumps({"version": 1, "shards": entries}, separators=(",", ":")),
        encoding="utf-8",
    )


def _shuffle_by_shard(
    index: list[tuple[int, int, str]], rng: np.random.Generator
) -> list[tuple[int, int, str]]:
    by_shard: dict[int, list[tuple[int, int, str]]] = {}
    for entry in index:
        by_shard.setdefault(entry[0], []).append(entry)

    shuffled: list[tuple[int, int, str]] = []
    shard_order = list(by_shard)
    rng.shuffle(shard_order)
    for shard_idx in shard_order:
        shard_entries = by_shard[shard_idx]
        order = rng.permutation(len(shard_entries))
        shuffled.extend(shard_entries[int(i)] for i in order)
    return shuffled


def _group_sample_indices_by_shard(
    entries: list[tuple[int, int, str]]
) -> list[tuple[int, list[int]]]:
    grouped: list[tuple[int, list[int]]] = []
    current_shard = -1
    current_indices: list[int] = []
    for shard_idx, sample_idx, _ in entries:
        if shard_idx != current_shard:
            if current_indices:
                grouped.append((current_shard, current_indices))
            current_shard = shard_idx
            current_indices = []
        current_indices.append(sample_idx)
    if current_indices:
        grouped.append((current_shard, current_indices))
    return grouped


def _group_full_trace_window_entries_by_shard(
    entries: list[tuple[int, int, int, str]]
) -> list[tuple[int, list[tuple[int, int, str]]]]:
    grouped: list[tuple[int, list[tuple[int, int, str]]]] = []
    current_shard = -1
    current_entries: list[tuple[int, int, str]] = []
    for shard_idx, sample_idx, start, sample_id in entries:
        if shard_idx != current_shard:
            if current_entries:
                grouped.append((current_shard, current_entries))
            current_shard = shard_idx
            current_entries = []
        current_entries.append((sample_idx, start, sample_id))
    if current_entries:
        grouped.append((current_shard, current_entries))
    return grouped
