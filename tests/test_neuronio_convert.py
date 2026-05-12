import pickle

import numpy as np

from bnn_sim.data import ShardedNeuronIODataset
from bnn_sim.neuronio_raw import convert_neuronio_pickles, parse_neuronio_pickle


def test_parse_neuronio_pickle_matches_upstream_shapes(tmp_path):
    raw_file = tmp_path / "demo_sim__0"
    _write_raw_neuronio_pickle(raw_file)

    parsed = parse_neuronio_pickle(raw_file)

    assert parsed.inputs.shape == (4, 6, 4)
    assert parsed.spikes.shape == (4, 6)
    assert parsed.soma.shape == (4, 6)
    assert parsed.metadata["num_segments"] == 2


def test_convert_neuronio_pickles_writes_signed_normalized_shards(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw_neuronio_pickle(raw_dir / "batch_sim__0")
    output_dir = tmp_path / "shards"

    written = convert_neuronio_pickles(raw_dir, output_dir, shard_size=3)

    assert len(written) == 2
    dataset = ShardedNeuronIODataset(output_dir)
    first = dataset[0]
    assert len(dataset) == 4
    assert first.inputs.shape == (6, 4)
    assert first.targets.shape == (6, 2)
    assert first.inputs[1, 2] == -1.0
    assert np.isclose(first.targets[0, 1], 0.77)
    with np.load(written[0], allow_pickle=False) as data:
        assert data["inputs"].dtype == np.int8


def test_convert_neuronio_pickles_discovers_nested_files_and_writes_windows(tmp_path):
    raw_dir = tmp_path / "raw" / "nested"
    raw_dir.mkdir(parents=True)
    _write_raw_neuronio_pickle(raw_dir / "batch_sim__0")
    output_dir = tmp_path / "window_shards"

    written = convert_neuronio_pickles(
        tmp_path / "raw",
        output_dir,
        shard_size=3,
        window_size=3,
        window_stride=2,
        ignore_start=1,
    )

    assert len(written) == 3
    dataset = ShardedNeuronIODataset(output_dir)
    assert len(dataset) == 4 * 2
    assert dataset[0].inputs.shape == (3, 4)
    assert dataset[0].sample_id.endswith("@1")
    assert dataset[1].sample_id.endswith("@3")
    assert dataset[2].sample_id.endswith("@1")


def test_convert_neuronio_pickles_can_store_full_simulations_without_windowing(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw_neuronio_pickle(raw_dir / "batch_sim__0")
    output_dir = tmp_path / "full_shards"

    written = convert_neuronio_pickles(raw_dir, output_dir, shard_size=4)

    assert len(written) == 1
    dataset = ShardedNeuronIODataset(output_dir)
    assert len(dataset) == 4
    assert dataset[0].inputs.shape == (6, 4)
    assert dataset[0].sample_id.endswith(":0")
    with np.load(written[0], allow_pickle=False) as data:
        assert data["inputs"].dtype == np.int8
        assert data["sample_ids"][0].endswith(":0")


def test_convert_neuronio_pickles_appends_after_existing_shards(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_raw_neuronio_pickle(raw_dir / "batch_sim__0")
    output_dir = tmp_path / "shards"

    first = convert_neuronio_pickles(raw_dir, output_dir, shard_size=4)
    second = convert_neuronio_pickles(raw_dir, output_dir, shard_size=4)

    assert first[0].name == "neuronio_00000.npz"
    assert second[0].name == "neuronio_00001.npz"
    assert len(list(output_dir.glob("*.npz"))) == 2


def _write_raw_neuronio_pickle(path):
    sims = []
    for sim_idx in range(4):
        sims.append(
            {
                "exInputSpikeTimes": {0: [0, 3], 1: [2]},
                "inhInputSpikeTimes": {0: [1], 1: [4]},
                "outputSpikeTimes": np.array([1.5, 4.5]) if sim_idx == 0 else np.array([]),
                "somaVoltageLowRes": np.array([-60.0, -54.0, -70.0, -67.7, -65.0, -80.0]),
            }
        )
    payload = {
        "Params": {
            "allSegmentsType": ["basal", "apical"],
            "allSegments_DistFromSoma": [10.0, 20.0],
            "totalSimDurationInSec": 0.006,
        },
        "Results": {"listOfSingleSimulationDicts": sims},
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
