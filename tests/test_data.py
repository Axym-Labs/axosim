import numpy as np
import json

from bnn_sim.data import DeterministicWindowDataset, ShardedNeuronIODataset, write_demo_shards


def test_sharded_dataset_is_deterministic_when_shuffled_with_same_seed(tmp_path):
    write_demo_shards(tmp_path, shard_count=3, samples_per_shard=4, time_steps=5, input_dim=6, seed=7)

    dataset_a = ShardedNeuronIODataset(tmp_path, shuffle=True, seed=11)
    dataset_b = ShardedNeuronIODataset(tmp_path, shuffle=True, seed=11)

    order_a = [dataset_a[i].sample_id for i in range(len(dataset_a))]
    order_b = [dataset_b[i].sample_id for i in range(len(dataset_b))]

    assert order_a == order_b
    assert len(dataset_a) == 12


def test_sharded_dataset_returns_float32_inputs_and_two_label_outputs(tmp_path):
    write_demo_shards(tmp_path, shard_count=1, samples_per_shard=2, time_steps=4, input_dim=3, seed=5)

    sample = ShardedNeuronIODataset(tmp_path)[1]

    assert sample.inputs.shape == (4, 3)
    assert sample.targets.shape == (4, 2)
    assert sample.inputs.dtype == np.float32
    assert sample.targets.dtype == np.float32


def test_window_dataset_enumerates_fixed_windows_deterministically(tmp_path):
    write_demo_shards(tmp_path, shard_count=1, samples_per_shard=2, time_steps=10, input_dim=3, seed=5)

    base = ShardedNeuronIODataset(tmp_path)
    windows = DeterministicWindowDataset(base, window_size=4, stride=3, start_offset=1)

    assert len(windows) == 4
    assert windows[0].sample_id.endswith("@1")
    assert windows[1].sample_id.endswith("@4")
    assert windows[2].sample_id.endswith("@1")
    assert windows[0].inputs.shape == (4, 3)


def test_sharded_dataset_uses_manifest_without_opening_all_shards(tmp_path):
    write_demo_shards(tmp_path, shard_count=1, samples_per_shard=2, time_steps=4, input_dim=3, seed=5)
    manifest = {
        "version": 1,
        "shards": [
            {
                "file": "demo_000.npz",
                "count": 2,
                "sample_ids": ["manifest:0", "manifest:1"],
            }
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    dataset = ShardedNeuronIODataset(tmp_path)

    assert len(dataset) == 2
    assert dataset[1].sample_id == "manifest:1"
