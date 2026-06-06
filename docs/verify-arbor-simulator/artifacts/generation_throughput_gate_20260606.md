# Generation Throughput Gate - 2026-06-06

## Environment

- Source root exists: `True`
- CPU mechanism exists: `True`
- GPU mechanism exists: `True`
- `libcoreneuron-cuda.so`: `None`
- GPU generation status: `blocked_missing_runtime`

## Sequential Timings

| Label | Backend | Samples | Duration ms | Elapsed s | s/trace | Simulated trace s / wall s |
|---|---:|---:|---:|---:|---:|---:|
| cpu_coreneuron_8x1000 | coreneuron | 8 | 1000 | 32.88 | 4.11 | 0.243 |
| cpu_neuron_8x1000 | neuron | 8 | 1000 | 34.05 | 4.26 | 0.235 |
| cpu_coreneuron_2x6000 | coreneuron | 2 | 6000 | 49.63 | 24.82 | 0.242 |
| cpu_neuron_2x6000 | neuron | 2 | 6000 | 51.17 | 25.59 | 0.234 |

## Parallel CPU CoreNEURON Timings

| Label | Parallelism | Samples | Duration ms | Elapsed s | s/trace | Simulated trace s / wall s |
|---|---:|---:|---:|---:|---:|---:|
| cpu_coreneuron_parallel_4x1x6000 | 4 | 4 | 6000 | 26.92 | 6.73 | 0.892 |
| cpu_coreneuron_parallel_8x1x6000 | 8 | 8 | 6000 | 28.08 | 3.51 | 1.709 |
| cpu_coreneuron_parallel_16x1x6000 | 16 | 16 | 6000 | 51.39 | 3.21 | 1.868 |

## Decision

Current promoted generation route: `cpu_coreneuron_parallel_shards` with parallelism `8`.
Peak measured parallelism in this run: `16`.

Use independent CPU CoreNEURON shards until a durable NEURON GPU runtime is rebuilt and beats the measured shard-parallel path without changing teacher outputs. The default parallelism is the smallest setting within 10% of peak measured throughput to preserve interactive and memory headroom.

This is not a CPU fallback for correctness. The GPU path is blocked in the current environment because the compiled GPU mechanism requires `libcoreneuron-cuda.so`, but no durable NEURON GPU runtime library is present. GPU remains a follow-up gate: rebuild the NEURON 9 GPU runtime in a durable location, rerun this script, and adopt GPU only if it preserves teacher equivalence and wins on throughput at dataset scale.

## GPU Mechanism `ldd` Excerpt

```text
linux-vdso.so.1 (0x00007bc2a1e6d000)
	libcoreneuron-cuda.so => not found
	libcudanvhpc.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libcudanvhpc.so (0x00007bc2a1200000)
	libacchost.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libacchost.so (0x00007bc2a0e00000)
	libaccdevaux.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libaccdevaux.so (0x00007bc2a0a00000)
	libaccdevice.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libaccdevice.so (0x00007bc2a0600000)
	libcudart.so.13 => /home/davwis/.local/nvidia/Linux_x86_64/26.3/cuda/13.1/lib64/libcudart.so.13 (0x00007bc2a0200000)
	libnvhpcatm.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libnvhpcatm.so (0x00007bc29fe00000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007bc29fa00000)
	libnvomp.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libnvomp.so (0x00007bc29e800000)
	libnvcpumath.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libnvcpumath.so (0x00007bc29e200000)
	libnvc.so => /home/davwis/.local/nvidia/Linux_x86_64/26.3/compilers/lib/libnvc.so (0x00007bc29de00000)
	libc.so.6 => /usr/lib/x86_64-linux-gnu/libc.so.6 (0x00007bc29da00000)
	libgcc_s.so.1 => /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007bc2a1e1c000)
	libm.so.6 => /usr/lib/x86_64-linux-gnu/libm.so.6 (0x00007bc2a1cf4000)
	/lib64/ld-linux-x86-64.so.2 (0x00007bc2a1e6f000)
	libdl.so.2 => /usr/lib/x86_64-linux-gnu/libdl.so.2 (0x00007bc2a1cef000)
	libpthread.so.0 => /usr/lib/x86_64-linux-gnu/libpthread.so.0 (0x00007bc2a1cea000)
	librt.so.1 => /usr/lib/x86_64-linux-gnu/librt.so.1 (0x00007bc2a1ce5000)
```
