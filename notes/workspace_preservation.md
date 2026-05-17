# Workspace Preservation Notes

Saved from local-only workspace state before teardown:

- `external_patches/arbor-gpu-v011-cxx-compat.patch`: local edits from
  `/workspace/deps/arbor-gpu-v011-src` against Arbor v0.11. These remove
  invalid `constexpr` defaults around `std::string` fields and spell out
  `cell_address_type` comparison operators. Apply from an Arbor v0.11 checkout
  with `git apply /path/to/arbor-gpu-v011-cxx-compat.patch`.
- `scripts/compat/run_fulltrace_compat.py`: compatibility trainer originally
  found as an untracked file in `/workspace/elmneuron/scripts/`. It trains the
  official `elmneuron` Branch-ELM implementation with `bnn_sim` full-trace
  datasets/evaluation.
