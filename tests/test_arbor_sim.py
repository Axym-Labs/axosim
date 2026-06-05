import numpy as np

from bnn_sim.arbor_sim import (
    ArborHeteroConfig,
    _interlocking_branch_routing,
    _label_spikes,
    _sample_sparse_inputs,
    _template_branch_info,
    _template_synapse_site_info,
)


def _event_count_autocorr(inputs: np.ndarray, lag: int) -> float:
    counts = np.abs(inputs).sum(axis=1).astype(np.float64)
    centered = counts - counts.mean()
    denom = float(np.dot(centered, centered))
    assert denom > 0
    return float(np.dot(centered[:-lag], centered[lag:]) / denom)


def _window_count_std(inputs: np.ndarray, window: int = 100) -> float:
    counts = np.abs(inputs).sum(axis=1).astype(np.float64)
    usable = (counts.shape[0] // window) * window
    return float(counts[:usable].reshape(-1, window).sum(axis=1).std())


def test_neuronio_smoothed_arbor_inputs_have_slow_drive_structure():
    input_dim = 256
    num_branch = 8
    routing = _interlocking_branch_routing(input_dim, num_branch)
    branch_regions = ["dend"] * 4 + ["apic"] * 4
    branch_lengths_um = np.asarray([80.0, 120.0, 160.0, 100.0, 220.0, 180.0, 260.0, 140.0])
    params = {
        "apic_synapse_fraction": 0.55,
        "ei_balance": 0.0,
        "input_rate_scale": 1.0,
        "burstiness_scale": 1.0,
        "synchrony_scale": 1.0,
    }

    poisson = _sample_sparse_inputs(
        ArborHeteroConfig(
            input_dim=input_dim,
            num_branch=num_branch,
            time_steps=3000,
            base_event_rate_hz=10.0,
            burst_probability=0.0,
            input_protocol="arbor_poisson",
        ),
        routing,
        branch_regions,
        params,
        np.random.default_rng(123),
        branch_lengths_um,
    )
    smoothed = _sample_sparse_inputs(
        ArborHeteroConfig(
            input_dim=input_dim,
            num_branch=num_branch,
            time_steps=3000,
            input_protocol="neuronio_smoothed",
        ),
        routing,
        branch_regions,
        params,
        np.random.default_rng(123),
        branch_lengths_um,
    )

    assert set(np.unique(smoothed)).issubset({-1, 0, 1})
    assert _event_count_autocorr(smoothed, 100) > 0.05
    assert abs(_event_count_autocorr(poisson, 100)) < 0.05
    assert _window_count_std(smoothed) > 2.0 * _window_count_std(poisson)


def test_distributed_synapse_routing_restores_many_physical_input_sites():
    config = ArborHeteroConfig(
        input_dim=1278,
        num_branch=45,
        synapse_routing="distributed",
        distributed_sites_per_polarity=639,
    )
    branch_info = _template_branch_info(config.num_branch)
    site_info = _template_synapse_site_info(config, branch_info)

    assert site_info["site_count"] == 639
    assert len(set(site_info["site_branch_indices"].tolist())) == len(branch_info["branch_ids"])
    assert site_info["channel_event_routing"][0] == site_info["channel_event_routing"][639]
    assert site_info["channel_event_routing"][638] == site_info["channel_event_routing"][1277]
    assert set(np.unique(site_info["site_regions"])).issubset({"dend", "apic"})
    assert np.all(site_info["site_exc_positions"] >= 0.15)
    assert np.all(site_info["site_exc_positions"] <= 0.90)


def test_local_peak_spike_labels_match_neuronio_style_peak_semantics():
    soma = np.asarray([-70.0, -20.0, -15.0, -30.0, -70.0, -24.0, -26.0], dtype=np.float32)
    threshold = _label_spikes(
        ArborHeteroConfig(spike_label_mode="threshold_crossing"),
        soma,
        {"spike_threshold_mv": -32.0},
    )
    peaks = _label_spikes(
        ArborHeteroConfig(spike_label_mode="local_peak", spike_peak_threshold_mv=-25.0),
        soma,
        {"spike_threshold_mv": -32.0},
    )

    assert np.flatnonzero(threshold).tolist() == [1, 5]
    assert np.flatnonzero(peaks).tolist() == [2, 5]
