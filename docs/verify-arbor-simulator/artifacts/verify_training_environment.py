#!/usr/bin/env python3
"""Verify that the training environment is suitable for benchmark runs."""

from __future__ import annotations

import argparse
import json
import sys
import time

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-mamba-ssm", action="store_true")
    args = parser.parse_args()

    report: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_kernel_smoke": None,
        "mamba_ssm_available": False,
        "causal_conv1d_available": False,
    }
    ok = True

    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        arch = f"sm_{capability[0]}{capability[1]}"
        arch_list = torch.cuda.get_arch_list()
        report.update(
            {
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": capability,
                "device_arch": arch,
                "torch_arch_list": arch_list,
                "device_arch_supported": arch in arch_list,
            }
        )
        try:
            start = time.perf_counter()
            x = torch.randn(2048, 2048, device="cuda")
            y = x @ x.T
            torch.cuda.synchronize()
            report["cuda_kernel_smoke"] = {
                "status": "passed",
                "seconds": time.perf_counter() - start,
                "sample": float(y[0, 0].detach().cpu()),
            }
        except Exception as error:  # pragma: no cover - hardware dependent
            report["cuda_kernel_smoke"] = {"status": "failed", "error": repr(error)}
            ok = False
        if arch not in arch_list:
            ok = False
    elif args.require_gpu:
        ok = False

    try:
        import mamba_ssm  # type: ignore

        report["mamba_ssm_available"] = True
        report["mamba_ssm_version"] = getattr(mamba_ssm, "__version__", "unknown")
    except Exception as error:
        report["mamba_ssm_error"] = repr(error)
        if args.require_mamba_ssm:
            ok = False

    try:
        import causal_conv1d  # type: ignore

        report["causal_conv1d_available"] = True
        report["causal_conv1d_version"] = getattr(causal_conv1d, "__version__", "unknown")
    except Exception as error:
        report["causal_conv1d_error"] = repr(error)
        if args.require_mamba_ssm:
            ok = False

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
