from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NeuronIOSample:
    sample_id: str
    inputs: np.ndarray
    targets: np.ndarray


class ShardedNeuronIODataset:
    """Deterministic reader for pre-sharded NeuronIO-style NPZ files."""

    def __init__(self, root: str | Path, *, shuffle: bool = False, seed: int = 0) -> None:
        self.root = Path(root)
        self.shards = sorted(self.root.glob("*.npz"))
        if not self.shards:
            raise FileNotFoundError(f"no .npz shards found in {self.root}")

        self._index: list[tuple[int, int, str]] = self._load_manifest_index()
        if not self._index:
            for shard_idx, shard in enumerate(self.shards):
                with np.load(shard, mmap_mode="r") as data:
                    inputs = data["inputs"]
                    sample_ids = _sample_ids(data, shard.stem, inputs.shape[0])
                    for sample_idx, sample_id in enumerate(sample_ids):
                        self._index.append((shard_idx, sample_idx, str(sample_id)))

        if shuffle:
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(self._index))
            self._index = [self._index[int(i)] for i in order]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> NeuronIOSample:
        shard_idx, sample_idx, sample_id = self._index[index]
        with np.load(self.shards[shard_idx], mmap_mode="r") as data:
            inputs = np.asarray(data["inputs"][sample_idx], dtype=np.float32)
            targets = np.asarray(data["targets"][sample_idx], dtype=np.float32)

        if targets.ndim != 2 or targets.shape[-1] != 2:
            raise ValueError(f"targets must have shape (time, 2), got {targets.shape}")
        if inputs.ndim != 2:
            raise ValueError(f"inputs must have shape (time, channels), got {inputs.shape}")

        return NeuronIOSample(sample_id=sample_id, inputs=inputs, targets=targets)

    def _load_manifest_index(self) -> list[tuple[int, int, str]]:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            return []

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_lookup = {shard.name: idx for idx, shard in enumerate(self.shards)}
        index: list[tuple[int, int, str]] = []
        for shard_entry in manifest.get("shards", []):
            shard_name = shard_entry["file"]
            if shard_name not in shard_lookup:
                continue
            shard_idx = shard_lookup[shard_name]
            count = int(shard_entry["count"])
            sample_ids = shard_entry.get("sample_ids")
            if sample_ids is None:
                sample_ids = [f"{Path(shard_name).stem}:{sample_idx}" for sample_idx in range(count)]
            for sample_idx, sample_id in enumerate(sample_ids[:count]):
                index.append((shard_idx, sample_idx, str(sample_id)))
        return index


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


def _sample_ids(data: np.lib.npyio.NpzFile, shard_name: str, size: int) -> np.ndarray:
    if "sample_ids" in data.files:
        return data["sample_ids"]
    return np.array([f"{shard_name}:{i}" for i in range(size)])
