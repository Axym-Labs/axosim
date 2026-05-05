# Official Mamba Setup

Environment used for the official Mamba probe on 2026-05-05.

## Local Stack

- Python `3.11`
- PyTorch `2.4.1+cu124`
- CUDA runtime `12.4`
- GPU `NVIDIA GeForce RTX 3090`
- Torch C++11 ABI: `False`

## Venv

```bash
python -m venv --system-site-packages /workspace/.venvs/mamba-official
/workspace/.venvs/mamba-official/bin/python -m pip install -U pip setuptools wheel
```

## Official Wheels

Use prebuilt wheels matching `cp311`, `cu12`, `torch2.4`, and `cxx11abiFALSE`.

```bash
/workspace/.venvs/mamba-official/bin/python -m pip install einops ninja
/workspace/.venvs/mamba-official/bin/python -m pip install --no-deps \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.3.post1/causal_conv1d-1.5.3.post1+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl' \
  'https://github.com/state-spaces/mamba/releases/download/v2.2.6.post3/mamba_ssm-2.2.6.post3+cu12torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
/workspace/.venvs/mamba-official/bin/python -m pip install \
  'transformers==4.44.2' 'huggingface_hub<1.0' 'numpy==1.26.3'
```

## Smoke Check

```bash
PYTHONPATH=src /workspace/.venvs/mamba-official/bin/python - <<'PY'
import numpy as np
import torch
import causal_conv1d_cuda
import selective_scan_cuda
from mamba_ssm import Mamba

print("numpy", np.__version__)
print("torch", torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)
model = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(2, 128, 64, device="cuda", requires_grad=True)
y = model(x)
y.square().mean().backward()
torch.cuda.synchronize()
print(tuple(y.shape), torch.isfinite(y).all().item())
PY
```

The smoke check passed for the canonical official Mamba probe.
