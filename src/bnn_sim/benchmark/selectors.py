"""Dataset selectors for the v1 benchmark layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from bnn_sim.benchmark.dataset_schema import INTERVENTION_CONDITIONS, ORDINARY_SPLITS
from bnn_sim.data import ShardedNeuronIODataset


DatasetCategory = Literal["train", "val", "test", "private-test"]


@dataclass(frozen=True)
class DatasetSelector:
    """Resolved dataset selector for one ordinary split or intervention."""

    root: Path
    split: str | None
    condition: str
    intervention: str | None = None

    def load(self, *, cache_shards: int = 1) -> ShardedNeuronIODataset:
        return ShardedNeuronIODataset(self.root, cache_shards=cache_shards)


class V1DatasetLayout:
    """Resolver for the v1 dataset structure."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def split(self, name: DatasetCategory = "val") -> DatasetSelector:
        if name not in ORDINARY_SPLITS:
            raise ValueError(f"unknown ordinary split {name!r}; expected one of {ORDINARY_SPLITS}")
        return DatasetSelector(root=self.root / name, split=name, condition=name)

    def intervention(self, name: str) -> DatasetSelector:
        path = self.root / "interventions" / name
        if name not in INTERVENTION_CONDITIONS and not path.exists():
            available = self.available_interventions()
            raise ValueError(
                f"unknown intervention {name!r}; expected one of {INTERVENTION_CONDITIONS} "
                f"or an existing intervention folder; available={available}"
            )
        return DatasetSelector(
            root=path,
            split=None,
            condition=f"interventions/{name}",
            intervention=name,
        )

    def default_validation(self) -> DatasetSelector:
        return self.split("val")

    def available_interventions(self) -> tuple[str, ...]:
        interventions_root = self.root / "interventions"
        if not interventions_root.exists():
            return ()
        names = {path.name for path in interventions_root.iterdir() if path.is_dir()}
        ordered = [name for name in INTERVENTION_CONDITIONS if name in names]
        ordered.extend(sorted(names - set(ordered)))
        return tuple(ordered)

    def validate_minimal(self, *, require_interventions: bool = False) -> None:
        missing = []
        for split in ORDINARY_SPLITS:
            path = self.root / split
            if not _has_manifest_or_shards(path):
                missing.append(str(path))
        if require_interventions:
            for intervention in INTERVENTION_CONDITIONS:
                path = self.root / "interventions" / intervention
                if not _has_manifest_or_shards(path):
                    missing.append(str(path))
        if missing:
            raise FileNotFoundError("missing v1 dataset categories: " + ", ".join(missing))


def _has_manifest_or_shards(path: Path) -> bool:
    return path.exists() and ((path / "manifest.json").exists() or any(path.glob("*.npz")))
