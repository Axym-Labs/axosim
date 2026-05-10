from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import importlib.resources as resources
import json
from pathlib import Path
import time
from typing import Any

import numpy as np


HETEROGENEITY_NAMES = (
    "soma_cm_scale",
    "dend_cm_scale",
    "apic_cm_scale",
    "axon_cm_scale",
    "axial_rL_scale",
    "v_init_mv",
    "temperature_celsius",
    "soma_pas_g_scale",
    "dend_pas_g_scale",
    "apic_pas_g_scale",
    "axon_pas_g_scale",
    "soma_e_pas_shift_mv",
    "dend_e_pas_shift_mv",
    "apic_e_pas_shift_mv",
    "axon_e_pas_shift_mv",
    "soma_nav_scale",
    "axon_nav_scale",
    "dend_nav_scale",
    "apic_nav_scale",
    "soma_kv_scale",
    "axon_kv_scale",
    "dend_kv_scale",
    "apic_kv_scale",
    "dend_ih_scale",
    "apic_ih_scale",
    "dend_im_scale",
    "apic_im_scale",
    "soma_ca_scale",
    "dend_ca_scale",
    "apic_ca_scale",
    "soma_sk_scale",
    "axon_sk_scale",
    "ca_decay_scale",
    "ca_gamma_scale",
    "exc_tau1_ms",
    "exc_tau2_ms",
    "inh_tau1_ms",
    "inh_tau2_ms",
    "exc_weight_scale",
    "inh_weight_scale",
    "ei_balance",
    "input_rate_scale",
    "burstiness_scale",
    "synchrony_scale",
    "apic_synapse_fraction",
    "spike_threshold_mv",
)


@dataclass(frozen=True)
class ArborHeteroConfig:
    input_dim: int = 1278
    num_branch: int = 45
    time_steps: int = 6000
    dt_ms: float = 1.0
    integration_dt_ms: float = 0.025
    base_event_rate_hz: float = 2.0
    burst_probability: float = 0.015
    burst_event_rate_hz: float = 35.0
    soma_target_offset_mv: float = -67.7
    soma_target_scale: float = 0.1
    cv_max_extent_um: float = 25.0
    arbor_threads: int = 1
    arbor_gpu_id: int | None = 0
    sim_batch_size: int = 32


def write_arbor_hetero_splits(
    output_root: str | Path,
    *,
    split_samples: dict[str, int],
    shards_per_split: int = 1,
    seed: int = 0,
    config: ArborHeteroConfig | None = None,
) -> dict[str, Any]:
    if not split_samples:
        raise ValueError("split_samples must not be empty")
    if shards_per_split <= 0:
        raise ValueError("shards_per_split must be positive")
    config = config or ArborHeteroConfig()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    reports: dict[str, Any] = {}
    total_disk_bytes = 0
    split_stats = []
    for split_idx, (split_name, samples) in enumerate(split_samples.items()):
        report = write_arbor_hetero_dataset(
            output_root / split_name,
            samples=samples,
            shards=shards_per_split,
            seed=seed + 1009 * split_idx,
            config=config,
        )
        reports[split_name] = report
        total_disk_bytes += int(report["disk_bytes"])
        split_stats.append((split_name, report["mean_spike_rate_hz"], report["soma_mean_mv"]))
    summary = {
        "backend": "arbor",
        "output": str(output_root),
        "splits": reports,
        "split_balance": {
            "spike_rate_hz_by_split": {name: rate for name, rate, _ in split_stats},
            "soma_mean_mv_by_split": {name: mean for name, _, mean in split_stats},
        },
        "total_disk_bytes": total_disk_bytes,
        "elapsed_seconds": time.perf_counter() - start,
    }
    (output_root / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_arbor_hetero_dataset(
    output_root: str | Path,
    *,
    samples: int = 128,
    shards: int = 1,
    seed: int = 0,
    config: ArborHeteroConfig | None = None,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if shards <= 0:
        raise ValueError("shards must be positive")
    config = config or ArborHeteroConfig()
    _validate_config(config)
    arbor_version, arbor_config = _require_arbor_version(require_cuda=config.arbor_gpu_id is not None)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    branch_info = _template_branch_info(config.num_branch)
    routing = _interlocking_branch_routing(config.input_dim, config.num_branch)
    split_counts = _split_counts(samples, shards)
    manifest_entries: list[dict[str, Any]] = []
    disk_paths: list[Path] = []
    total_events = 0
    total_spikes = 0
    soma_sum = 0.0
    soma_square_sum = 0.0
    soma_count = 0
    sample_base = 0
    start_time = time.perf_counter()

    for shard_idx, shard_samples in enumerate(split_counts):
        shard_inputs = np.zeros((shard_samples, config.time_steps, config.input_dim), dtype=np.int8)
        shard_targets = np.zeros((shard_samples, config.time_steps, 2), dtype=np.float32)
        shard_soma_mv = np.zeros((shard_samples, config.time_steps), dtype=np.float32)
        shard_heterogeneity = np.zeros((shard_samples, len(HETEROGENEITY_NAMES)), dtype=np.float32)
        sample_ids = np.empty(shard_samples, dtype=f"<U{max(48, len(str(samples)) + 32)}")
        sample_params: list[dict[str, float]] = []

        for local_idx in range(shard_samples):
            sample_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            sample_rng = np.random.default_rng(sample_seed)
            params = _sample_heterogeneity(sample_rng)
            inputs = _sample_sparse_inputs(config, routing, branch_info["regions"], params, sample_rng)
            global_idx = sample_base + local_idx
            shard_inputs[local_idx] = inputs
            shard_heterogeneity[local_idx] = _heterogeneity_vector(params)
            sample_ids[local_idx] = f"arbor_allen_hetero:{global_idx:06d}:seed{sample_seed}"
            sample_params.append(params)
            total_events += int(np.count_nonzero(inputs))

        for start in range(0, shard_samples, config.sim_batch_size):
            stop = min(start + config.sim_batch_size, shard_samples)
            batch_soma = _simulate_arbor_batch(
                config,
                inputs=shard_inputs[start:stop],
                params_list=sample_params[start:stop],
                routing=routing,
                branch_ids=branch_info["branch_ids"],
            )
            shard_soma_mv[start:stop] = batch_soma

        for local_idx, params in enumerate(sample_params):
            soma_mv = shard_soma_mv[local_idx]
            spikes = _threshold_spikes(soma_mv, params["spike_threshold_mv"])
            shard_targets[local_idx] = np.stack(
                [
                    spikes,
                    (soma_mv - config.soma_target_offset_mv) * config.soma_target_scale,
                ],
                axis=1,
            ).astype(np.float32)
            total_spikes += int(spikes.sum())
            soma_sum += float(soma_mv.sum())
            soma_square_sum += float(np.square(soma_mv).sum())
            soma_count += int(soma_mv.size)

        sample_base += shard_samples
        shard_path = output_root / f"arbor_allen_hetero_{shard_idx:05d}.npz"
        np.savez_compressed(
            shard_path,
            inputs=shard_inputs,
            targets=shard_targets,
            sample_ids=sample_ids,
            heterogeneity=shard_heterogeneity,
            heterogeneity_names=np.asarray(HETEROGENEITY_NAMES),
            soma_mv=shard_soma_mv,
        )
        disk_paths.append(shard_path)
        manifest_entries.append(
            {
                "file": shard_path.name,
                "count": int(shard_samples),
                "sample_ids": sample_ids.tolist(),
                "heterogeneity_names": list(HETEROGENEITY_NAMES),
            }
        )

    simulator = _simulator_metadata(config, arbor_version, arbor_config, branch_info)
    manifest = {
        "version": 2,
        "dataset_kind": "arbor-allen-heterogeneous-cable",
        "simulator": simulator,
        "shards": manifest_entries,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    disk_bytes = sum(path.stat().st_size for path in disk_paths) + manifest_path.stat().st_size
    dense_bytes = samples * config.time_steps * (
        config.input_dim * np.dtype(np.int8).itemsize + 2 * np.dtype(np.float32).itemsize
    )
    soma_mean = soma_sum / max(soma_count, 1)
    soma_var = soma_square_sum / max(soma_count, 1) - soma_mean * soma_mean
    report = {
        "backend": "arbor",
        "output": str(output_root),
        "samples": samples,
        "shards": shards,
        "time_steps": config.time_steps,
        "input_dim": config.input_dim,
        "num_branch": config.num_branch,
        "elapsed_seconds": time.perf_counter() - start_time,
        "disk_bytes": int(disk_bytes),
        "estimated_dense_bytes": int(dense_bytes),
        "mean_input_event_fraction": total_events / max(samples * config.time_steps * config.input_dim, 1),
        "mean_spike_rate_hz": 1000.0 * total_spikes / max(samples * config.time_steps * config.dt_ms, 1e-12),
        "soma_mean_mv": soma_mean,
        "soma_std_mv": float(np.sqrt(max(soma_var, 0.0))),
        "simulator": simulator,
    }
    (output_root / "generation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def simulate_arbor_sample(
    config: ArborHeteroConfig,
    *,
    routing: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    branch_info = _template_branch_info(config.num_branch)
    params = _sample_heterogeneity(rng)
    inputs = _sample_sparse_inputs(config, routing, branch_info["regions"], params, rng)
    soma_mv = _simulate_arbor_batch(
        config,
        inputs=inputs[None, ...],
        params_list=[params],
        routing=routing,
        branch_ids=branch_info["branch_ids"],
    )[0]
    spikes = _threshold_spikes(soma_mv, params["spike_threshold_mv"])
    targets = np.stack(
        [spikes, (soma_mv - config.soma_target_offset_mv) * config.soma_target_scale],
        axis=1,
    ).astype(np.float32)
    return {
        "inputs": inputs,
        "targets": targets,
        "soma_mv": soma_mv,
        "heterogeneity": _heterogeneity_vector(params),
    }


def parse_split_samples(value: str) -> dict[str, int]:
    splits: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise ValueError("splits must be comma-separated NAME:SAMPLES pairs")
        name, count = part.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError("split name must not be empty")
        samples = int(count)
        if samples <= 0:
            raise ValueError("split sample counts must be positive")
        splits[name] = samples
    if not splits:
        raise ValueError("splits must not be empty")
    return splits


def _simulate_arbor_batch(
    config: ArborHeteroConfig,
    *,
    inputs: np.ndarray,
    params_list: list[dict[str, float]],
    routing: np.ndarray,
    branch_ids: list[int],
) -> np.ndarray:
    import arbor as A

    recipe = _make_batch_recipe(A, config, inputs, params_list, routing, branch_ids)
    ctx = A.context(threads=config.arbor_threads, gpu_id=config.arbor_gpu_id)
    decomp = A.partition_load_balance(recipe, ctx)
    sim = A.simulation(recipe, ctx, decomp)
    schedule = A.regular_schedule(config.dt_ms * A.units.ms)
    handles = [sim.sample((gid, "Um-soma"), schedule) for gid in range(len(params_list))]
    sim.run(config.time_steps * config.dt_ms * A.units.ms, config.integration_dt_ms * A.units.ms)
    soma = np.zeros((len(params_list), config.time_steps), dtype=np.float32)
    for gid, handle in enumerate(handles):
        samples = sim.samples(handle)[0][0]
        trace = np.asarray(samples[:, 1], dtype=np.float32)
        if trace.shape[0] < config.time_steps:
            trace = np.pad(trace, (0, config.time_steps - trace.shape[0]), mode="edge")
        soma[gid] = trace[: config.time_steps]
    return soma


def _make_batch_recipe(A, config: ArborHeteroConfig, inputs, params_list, routing, branch_ids):
    cells = [_build_cell(config, params, branch_ids) for params in params_list]
    event_times = [_branch_event_times(x, routing, config.dt_ms) for x in inputs]

    class BatchRecipe(A.recipe):
        def __init__(self) -> None:
            A.recipe.__init__(self)
            self.props = A.neuron_cable_properties()
            self.props.catalogue.extend(A.allen_catalogue())

        def num_cells(self):
            return len(cells)

        def cell_kind(self, gid):
            return A.cell_kind.cable

        def cell_description(self, gid):
            return cells[int(gid)]

        def probes(self, gid):
            return [A.cable_probe_membrane_voltage('"soma_mid"', "Um-soma")]

        def event_generators(self, gid):
            U = A.units
            params = params_list[int(gid)]
            exc_times, inh_times = event_times[int(gid)]
            generators = []
            for branch_idx, times in enumerate(exc_times):
                if times:
                    schedule = A.explicit_schedule([float(t) * U.ms for t in times])
                    generators.append(A.event_generator(f"exc_{branch_idx}", params["exc_weight_scale"], schedule))
            for branch_idx, times in enumerate(inh_times):
                if times:
                    schedule = A.explicit_schedule([float(t) * U.ms for t in times])
                    generators.append(A.event_generator(f"inh_{branch_idx}", params["inh_weight_scale"], schedule))
            return generators

        def global_properties(self, kind):
            return self.props

    return BatchRecipe()


def _build_cell(config: ArborHeteroConfig, params: dict[str, float], branch_ids: list[int]):
    import arbor as A

    U = A.units
    morphology, fit = _load_allen_template()
    labels = A.label_dict().add_swc_tags()
    labels["soma_mid"] = "(location 0 0.5)"
    for branch_idx, branch_id in enumerate(branch_ids):
        labels[f"branch_{branch_idx}"] = f"(branch {branch_id})"
        labels[f"branch_exc_{branch_idx}"] = f"(location {branch_id} 0.65)"
        labels[f"branch_inh_{branch_idx}"] = f"(location {branch_id} 0.35)"

    decor = A.decor()
    default = fit["default_properties"]
    decor.set_property(
        tempK=params["temperature_celsius"] * U.Celsius,
        Vm=params["v_init_mv"] * U.mV,
        cm=default["cm"] * U.uF / U.cm2,
        rL=default["rL"] * params["axial_rL_scale"] * U.Ohm * U.cm,
    )

    for region, region_params in fit["region_properties"].items():
        scale_key = f"{region}_cm_scale" if region in {"soma", "axon"} else f"{region}_cm_scale"
        cm_scale = params.get(scale_key, 1.0)
        decor.paint(
            f'"{region}"',
            cm=region_params["cm"] * cm_scale * U.uF / U.cm2,
            rL=region_params["rL"] * params["axial_rL_scale"] * U.Ohm * U.cm,
            Vm=params["v_init_mv"] * U.mV,
        )

    for region, ion, erev in fit["ion_reversal_mv"]:
        decor.paint(f'"{region}"', ion=ion, rev_pot=erev * U.mV)
    decor.set_ion("ca", int_con=5e-5 * U.mM, ext_con=2.0 * U.mM, method="nernst/x=ca")

    for (region, mech), values in fit["mechanisms"].items():
        painted_name, painted_values = _perturb_mechanism(region, mech, values, params)
        decor.paint(f'"{region}"', A.density(A.mechanism(painted_name, painted_values)))

    for branch_idx in range(len(branch_ids)):
        decor.place(
            f'"branch_exc_{branch_idx}"',
            A.synapse(
                "Exp2Syn",
                {"erev": 0.0, "tau1": params["exc_tau1_ms"], "tau2": params["exc_tau2_ms"]},
            ),
            f"exc_{branch_idx}",
        )
        decor.place(
            f'"branch_inh_{branch_idx}"',
            A.synapse(
                "Exp2Syn",
                {"erev": -82.0, "tau1": params["inh_tau1_ms"], "tau2": params["inh_tau2_ms"]},
            ),
            f"inh_{branch_idx}",
        )
    return A.cable_cell(morphology, decor, labels, A.cv_policy_max_extent(config.cv_max_extent_um))


def _perturb_mechanism(region: str, mech: str, values: dict[str, float], params: dict[str, float]) -> tuple[str, dict[str, float]]:
    name = mech
    out = dict(values)
    if mech == "pas":
        e_shift = params[f"{region}_e_pas_shift_mv"]
        name = f"pas/e={values['e'] + e_shift}"
        out = {"g": values["g"] * params[f"{region}_pas_g_scale"]}
        return name, out
    if mech in {"NaV", "NaTa", "NaTs", "Nap"}:
        _scale_gbar(out, params.get(f"{region}_nav_scale", 1.0))
    elif mech in {"K_T", "Kd", "Kv2like", "Kv3_1", "K_P"}:
        _scale_gbar(out, params.get(f"{region}_kv_scale", 1.0))
    elif mech == "Ih":
        _scale_gbar(out, params.get(f"{region}_ih_scale", 1.0))
    elif mech in {"Im", "Im_v2"}:
        _scale_gbar(out, params.get(f"{region}_im_scale", 1.0))
    elif mech in {"Ca_HVA", "Ca_LVA"}:
        _scale_gbar(out, params.get(f"{region}_ca_scale", 1.0))
    elif mech == "SK":
        _scale_gbar(out, params.get(f"{region}_sk_scale", 1.0))
    elif mech == "CaDynamics":
        out["decay"] = out.get("decay", 80.0) * params["ca_decay_scale"]
        out["gamma"] = out.get("gamma", 0.05) * params["ca_gamma_scale"]
    return name, out


def _scale_gbar(values: dict[str, float], scale: float) -> None:
    for key in list(values):
        if key.startswith("gbar") or key == "gbar":
            values[key] *= scale


def _sample_sparse_inputs(
    config: ArborHeteroConfig,
    routing: np.ndarray,
    branch_regions: list[str],
    params: dict[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    inputs = np.zeros((config.time_steps, config.input_dim), dtype=np.int8)
    exc_channels = config.input_dim // 2
    branch_is_apic = np.asarray([region == "apic" for region in branch_regions], dtype=np.float64)
    channel_branch = routing
    channel_is_exc = np.arange(config.input_dim) < exc_channels
    apic_weight = params["apic_synapse_fraction"]
    branch_weights = np.where(branch_is_apic, apic_weight, 1.0 - apic_weight)
    channel_weights = branch_weights[channel_branch]
    channel_weights *= np.where(channel_is_exc, 1.0 + params["ei_balance"], 1.0 - params["ei_balance"])
    channel_weights = np.clip(channel_weights, 1e-6, None)
    channel_prob = channel_weights / channel_weights.sum()
    base_p = config.base_event_rate_hz * config.dt_ms / 1000.0 * params["input_rate_scale"]
    burst_p = config.burst_event_rate_hz * config.dt_ms / 1000.0 * params["input_rate_scale"] * params["burstiness_scale"]
    burst_probability = np.clip(config.burst_probability * params["burstiness_scale"], 0.0, 0.35)
    for t in range(config.time_steps):
        p_event = burst_p if rng.random() < burst_probability else base_p
        event_count = min(int(rng.poisson(config.input_dim * p_event)), config.input_dim)
        if event_count:
            if rng.random() < 0.12 * params["synchrony_scale"]:
                center_branch = int(rng.integers(0, len(branch_regions)))
                nearby = np.flatnonzero(channel_branch == center_branch)
                if nearby.size:
                    local_count = min(event_count, int(nearby.size))
                    channels = rng.choice(nearby, size=local_count, replace=False)
                else:
                    channels = rng.choice(config.input_dim, size=event_count, replace=False, p=channel_prob)
            else:
                channels = rng.choice(config.input_dim, size=event_count, replace=False, p=channel_prob)
            inputs[t, channels] = np.where(channels < exc_channels, 1, -1).astype(np.int8)
    return inputs


def _branch_event_times(inputs: np.ndarray, routing: np.ndarray, dt_ms: float) -> tuple[list[list[float]], list[list[float]]]:
    num_branch = int(routing.max()) + 1
    exc_times: list[list[float]] = [[] for _ in range(num_branch)]
    inh_times: list[list[float]] = [[] for _ in range(num_branch)]
    event_t, event_ch = np.nonzero(inputs)
    for t, ch in zip(event_t.tolist(), event_ch.tolist()):
        branch = int(routing[ch])
        if inputs[t, ch] > 0:
            exc_times[branch].append(float(t * dt_ms))
        else:
            inh_times[branch].append(float(t * dt_ms))
    return exc_times, inh_times


def _threshold_spikes(soma_mv: np.ndarray, threshold_mv: float) -> np.ndarray:
    above = soma_mv >= threshold_mv
    spikes = np.zeros(soma_mv.shape, dtype=np.float32)
    spikes[1:] = np.logical_and(above[1:], np.logical_not(above[:-1])).astype(np.float32)
    return spikes


def _sample_heterogeneity(rng: np.random.Generator) -> dict[str, float]:
    logn = lambda sigma=0.25: float(np.exp(rng.normal(0.0, sigma)))
    return {
        "soma_cm_scale": logn(0.18),
        "dend_cm_scale": logn(0.22),
        "apic_cm_scale": logn(0.22),
        "axon_cm_scale": logn(0.18),
        "axial_rL_scale": float(rng.uniform(0.65, 1.55)),
        "v_init_mv": float(rng.uniform(-82.0, -66.0)),
        "temperature_celsius": float(rng.uniform(32.0, 36.5)),
        "soma_pas_g_scale": logn(0.35),
        "dend_pas_g_scale": logn(0.45),
        "apic_pas_g_scale": logn(0.45),
        "axon_pas_g_scale": logn(0.35),
        "soma_e_pas_shift_mv": float(rng.normal(0.0, 2.0)),
        "dend_e_pas_shift_mv": float(rng.normal(0.0, 3.0)),
        "apic_e_pas_shift_mv": float(rng.normal(0.0, 3.0)),
        "axon_e_pas_shift_mv": float(rng.normal(0.0, 2.0)),
        "soma_nav_scale": logn(0.28),
        "axon_nav_scale": logn(0.35),
        "dend_nav_scale": logn(0.45),
        "apic_nav_scale": logn(0.45),
        "soma_kv_scale": logn(0.30),
        "axon_kv_scale": logn(0.35),
        "dend_kv_scale": logn(0.45),
        "apic_kv_scale": logn(0.45),
        "dend_ih_scale": logn(0.55),
        "apic_ih_scale": logn(0.55),
        "dend_im_scale": logn(0.45),
        "apic_im_scale": logn(0.45),
        "soma_ca_scale": logn(0.40),
        "dend_ca_scale": logn(0.50),
        "apic_ca_scale": logn(0.50),
        "soma_sk_scale": logn(0.40),
        "axon_sk_scale": logn(0.35),
        "ca_decay_scale": float(rng.uniform(0.55, 1.8)),
        "ca_gamma_scale": float(rng.uniform(0.55, 1.8)),
        "exc_tau1_ms": float(rng.uniform(0.15, 0.9)),
        "exc_tau2_ms": float(rng.uniform(2.0, 8.5)),
        "inh_tau1_ms": float(rng.uniform(0.3, 1.8)),
        "inh_tau2_ms": float(rng.uniform(6.0, 20.0)),
        "exc_weight_scale": float(rng.uniform(0.0015, 0.018)),
        "inh_weight_scale": float(rng.uniform(0.001, 0.012)),
        "ei_balance": float(rng.uniform(-0.25, 0.25)),
        "input_rate_scale": float(rng.uniform(0.55, 1.75)),
        "burstiness_scale": float(rng.uniform(0.35, 2.4)),
        "synchrony_scale": float(rng.uniform(0.4, 2.2)),
        "apic_synapse_fraction": float(rng.uniform(0.35, 0.75)),
        "spike_threshold_mv": float(rng.uniform(-42.0, -32.0)),
    }


def _heterogeneity_vector(params: dict[str, float]) -> np.ndarray:
    return np.asarray([params[name] for name in HETEROGENEITY_NAMES], dtype=np.float32)


def _interlocking_branch_routing(input_dim: int, num_branch: int) -> np.ndarray:
    even = np.arange(0, input_dim, 2, dtype=np.int64)
    odd = np.arange(1, input_dim, 2, dtype=np.int64)
    interlocked = np.concatenate([even, odd])
    routing = np.empty(input_dim, dtype=np.int64)
    routing[interlocked] = np.arange(input_dim, dtype=np.int64) % num_branch
    return routing


def _simulator_metadata(
    config: ArborHeteroConfig,
    arbor_version: str,
    arbor_config: dict[str, Any],
    branch_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": "arbor",
        "arbor_version": arbor_version,
        "arbor_config": arbor_config,
        "description": (
            "GPU-capable Arbor batched independent-cell teacher based on an Allen L5 morphology/fit, "
            "with region-specific passive properties, Allen active channels, calcium dynamics, "
            "heterogeneous E/I Exp2Syn inputs, and per-sample physiology/stimulus heterogeneity."
        ),
        "template": {
            "morphology": "src/bnn_sim/assets/allen_l5_template.swc",
            "fit": "src/bnn_sim/assets/allen_l5_template_fit.json",
            "source": "Arbor v0.11 python example single_cell_allen",
        },
        "config": asdict(config),
        "heterogeneity_names": list(HETEROGENEITY_NAMES),
        "branch_ids": branch_info["branch_ids"],
        "branch_regions": branch_info["regions"],
        "routing": "interlocking-inputs-to-selected-allen-morphology-branches",
        "target_columns": ["spike", "soma"],
        "soma_target_units": "normalized_neuronio",
        "known_limitations": [
            "Excitatory synapses use conductance-based Exp2Syn AMPA-like kinetics; voltage-dependent NMDA requires a custom Arbor mechanism catalogue.",
        ],
    }


def _require_arbor_version(*, require_cuda: bool = True) -> tuple[str, dict[str, Any]]:
    try:
        import arbor as A
    except Exception as error:
        raise ValueError("Arbor is required for this command. Build/install Arbor with CUDA support for GPU generation.") from error
    config = A.config()
    version = str(config.get("version", getattr(A, "__version__", "unknown")))
    if require_cuda and config.get("gpu") != "cuda":
        raise ValueError(f"expected a CUDA-enabled Arbor build; found Arbor config {config}")
    return version, dict(config)


@lru_cache(maxsize=1)
def _load_allen_template():
    import arbor as A

    swc_path = resources.files("bnn_sim.assets").joinpath("allen_l5_template.swc")
    fit_path = resources.files("bnn_sim.assets").joinpath("allen_l5_template_fit.json")
    morphology = A.load_swc_neuron(str(swc_path)).morphology
    fit = json.loads(fit_path.read_text(encoding="utf-8"))
    return morphology, _parse_allen_fit(fit)


def _parse_allen_fit(fit: dict[str, Any]) -> dict[str, Any]:
    region_properties: dict[str, dict[str, float]] = {}
    mechanisms: dict[tuple[str, str], dict[str, float]] = {}
    for block in fit["genome"]:
        mech = block["mechanism"] or "pas"
        region = block["section"]
        name = block["name"]
        value = float(block["value"])
        if mech == "pas":
            if name == "cm":
                region_properties.setdefault(region, {})["cm"] = value
            elif name == "Ra":
                region_properties.setdefault(region, {})["rL"] = value
            elif name == "g_pas":
                mechanisms.setdefault((region, "pas"), {})["g"] = value
            elif name == "e_pas":
                mechanisms.setdefault((region, "pas"), {})["e"] = value
            elif name in {"Vm", "celsius"}:
                continue
            else:
                raise ValueError(f"unsupported Allen passive parameter: {name}")
            continue
        key = name[: -(len(mech) + 1)] if name.endswith("_" + mech) else name
        mechanisms.setdefault((region, mech), {})[key] = value

    default_properties = {
        "temp": float(fit["conditions"][0]["celsius"]),
        "Vm": float(fit["conditions"][0]["v_init"]),
        "rL": float(fit["passive"][0]["ra"]),
        "cm": float(np.mean([props["cm"] for props in region_properties.values() if "cm" in props])),
    }
    ion_reversal_mv = []
    for row in fit["conditions"][0]["erev"]:
        region = row["section"]
        for key, value in row.items():
            if key != "section":
                ion_reversal_mv.append((region, key[1:], float(value)))
    return {
        "default_properties": default_properties,
        "region_properties": region_properties,
        "ion_reversal_mv": ion_reversal_mv,
        "mechanisms": mechanisms,
        "junction_potential": float(fit["fitting"][0].get("junction_potential", 0.0)),
    }


@lru_cache(maxsize=8)
def _template_branch_info(num_branch: int) -> dict[str, Any]:
    morphology, _ = _load_allen_template()
    basal = []
    apical = []
    for branch_id in range(morphology.num_branches):
        tags = {segment.tag for segment in morphology.branch_segments(branch_id)}
        if 3 in tags:
            basal.append(branch_id)
        elif 4 in tags:
            apical.append(branch_id)
    if not basal or not apical:
        raise ValueError("Allen template morphology is missing basal/apical SWC tags")
    basal_count = min(len(basal), max(1, int(round(num_branch * 0.45))))
    apical_count = min(len(apical), num_branch - basal_count)
    if basal_count + apical_count < num_branch:
        basal_count = min(len(basal), num_branch - apical_count)
    basal_idx = np.linspace(0, len(basal) - 1, basal_count, dtype=int)
    apical_idx = np.linspace(0, len(apical) - 1, apical_count, dtype=int)
    branch_ids = [basal[idx] for idx in basal_idx] + [apical[idx] for idx in apical_idx]
    regions = ["dend"] * len(basal_idx) + ["apic"] * len(apical_idx)
    return {"branch_ids": branch_ids, "regions": regions}


def _split_counts(samples: int, shards: int) -> list[int]:
    base = samples // shards
    remainder = samples % shards
    return [base + (1 if idx < remainder else 0) for idx in range(shards)]


def _validate_config(config: ArborHeteroConfig) -> None:
    if config.input_dim <= 1:
        raise ValueError("input_dim must be greater than one")
    if config.num_branch <= 0:
        raise ValueError("num_branch must be positive")
    if config.time_steps <= 0:
        raise ValueError("time_steps must be positive")
    if config.dt_ms <= 0 or config.integration_dt_ms <= 0:
        raise ValueError("time steps must be positive")
    if config.sim_batch_size <= 0:
        raise ValueError("sim_batch_size must be positive")
    if config.arbor_threads <= 0:
        raise ValueError("arbor_threads must be positive")
