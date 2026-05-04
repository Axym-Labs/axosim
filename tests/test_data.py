import numpy as np
import json

from bnn_sim.data import (
    DeterministicWindowDataset,
    RandomFullTraceWindowDataset,
    ShardedNeuronIODataset,
    repack_shards_as_npy,
    write_demo_shards,
)


def test_sharded_dataset_is_deterministic_when_shuffled_with_same_seed(tmp_path):
    write_demo_shards(tmp_path, shard_count=3, samples_per_shard=4, time_steps=5, input_dim=6, seed=7)

    dataset_a = ShardedNeuronIODataset(tmp_path, shuffle=True, seed=11)
    dataset_b = ShardedNeuronIODataset(tmp_path, shuffle=True, seed=11)

    order_a = [dataset_a[i].sample_id for i in range(len(dataset_a))]
    order_b = [dataset_b[i].sample_id for i in range(len(dataset_b))]

    assert order_a == order_b
    assert len(dataset_a) == 12


def test_sharded_dataset_can_shuffle_in_shard_local_blocks(tmp_path):
    write_demo_shards(tmp_path, shard_count=3, samples_per_shard=4, time_steps=5, input_dim=6, seed=7)

    dataset = ShardedNeuronIODataset(tmp_path, shuffle=True, shuffle_mode="shard", seed=11)
    order = [dataset[i].sample_id for i in range(len(dataset))]

    assert len(order) == 12
    assert len({sample_id.split("_sample")[0] for sample_id in order[:4]}) == 1
    assert len({sample_id.split("_sample")[0] for sample_id in order[4:8]}) == 1
    assert len({sample_id.split("_sample")[0] for sample_id in order[8:12]}) == 1


def test_sharded_dataset_can_reshuffle_deterministically(tmp_path):
    write_demo_shards(tmp_path, shard_count=3, samples_per_shard=4, time_steps=5, input_dim=6, seed=7)

    dataset_a = ShardedNeuronIODataset(tmp_path, shuffle=True, shuffle_mode="shard", seed=11)
    dataset_b = ShardedNeuronIODataset(tmp_path, shuffle=True, shuffle_mode="shard", seed=11)
    dataset_a.reshuffle(12)
    dataset_b.reshuffle(12)

    assert [dataset_a[i].sample_id for i in range(len(dataset_a))] == [
        dataset_b[i].sample_id for i in range(len(dataset_b))
    ]


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


def test_repack_shards_as_npy_preserves_samples(tmp_path):
    source_dir = tmp_path / "source"
    repacked_dir = tmp_path / "repacked"
    write_demo_shards(source_dir, shard_count=2, samples_per_shard=3, time_steps=4, input_dim=3, seed=5)

    written = repack_shards_as_npy(source_dir, repacked_dir)
    source = ShardedNeuronIODataset(source_dir)
    repacked = ShardedNeuronIODataset(repacked_dir)

    assert len(written) == 2
    assert len(source) == len(repacked)
    assert repacked[0].sample_id == source[0].sample_id
    assert np.array_equal(repacked.get_batch(range(1, 4))[0], source.get_batch(range(1, 4))[0])
    assert np.array_equal(repacked.get_batch(range(1, 4))[1], source.get_batch(range(1, 4))[1])


def test_random_full_trace_window_dataset_samples_cached_windows(tmp_path):
    write_demo_shards(tmp_path, shard_count=2, samples_per_shard=4, time_steps=12, input_dim=3, seed=5)
    base = ShardedNeuronIODataset(tmp_path, cache_shards=1)

    windows_a = RandomFullTraceWindowDataset(
        base,
        window_size=4,
        start_offset=2,
        samples_per_epoch=8,
        batch_size=2,
        shard_reuse_batches=2,
        sequence_length=12,
        seed=7,
    )
    windows_b = RandomFullTraceWindowDataset(
        base,
        window_size=4,
        start_offset=2,
        samples_per_epoch=8,
        batch_size=2,
        shard_reuse_batches=2,
        sequence_length=12,
        seed=7,
    )
    x, y = windows_a.get_batch(range(2))

    assert len(windows_a) == 8
    assert x.shape == (2, 4, 3)
    assert y.shape == (2, 4, 2)
    assert windows_a.get_sample_ids(range(8)) == windows_b.get_sample_ids(range(8))
    assert all(int(sample_id.rsplit("@", 1)[1]) >= 2 for sample_id in windows_a.get_sample_ids(range(8)))
