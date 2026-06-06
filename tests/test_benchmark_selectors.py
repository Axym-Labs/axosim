import pytest

from bnn_sim.benchmark.selectors import V1DatasetLayout
from bnn_sim.data import write_demo_shards


def test_v1_dataset_layout_resolves_default_validation_selector(tmp_path):
    write_demo_shards(tmp_path / "val", shard_count=1, samples_per_shard=2)

    selector = V1DatasetLayout(tmp_path).default_validation()
    dataset = selector.load()

    assert selector.root == tmp_path / "val"
    assert selector.split == "val"
    assert selector.condition == "val"
    assert len(dataset) == 2


def test_v1_dataset_layout_resolves_intervention_selector(tmp_path):
    write_demo_shards(tmp_path / "interventions" / "event_dropout", shard_count=1, samples_per_shard=1)

    layout = V1DatasetLayout(tmp_path)
    selector = layout.intervention("event_dropout")

    assert selector.root == tmp_path / "interventions" / "event_dropout"
    assert selector.split is None
    assert selector.condition == "interventions/event_dropout"
    assert selector.intervention == "event_dropout"
    assert layout.available_interventions() == ("event_dropout",)


def test_v1_dataset_layout_rejects_unknown_intervention(tmp_path):
    with pytest.raises(ValueError, match="unknown intervention"):
        V1DatasetLayout(tmp_path).intervention("nonsense")


def test_v1_dataset_layout_validates_minimal_categories(tmp_path):
    for split in ("train", "val", "test", "private-test"):
        write_demo_shards(tmp_path / split, shard_count=1, samples_per_shard=1)

    V1DatasetLayout(tmp_path).validate_minimal()

    with pytest.raises(FileNotFoundError, match="interventions/event_dropout"):
        V1DatasetLayout(tmp_path).validate_minimal(require_interventions=True)
