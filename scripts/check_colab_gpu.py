#!/usr/bin/env python3
"""Print GPU identity inside the active Colab runtime."""

from __future__ import annotations

import json
import subprocess

import torch


result = {
    "cuda_available": torch.cuda.is_available(),
    "torch_gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "nvidia_smi": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
    ).strip(),
}
print(json.dumps(result, ensure_ascii=False, indent=2))
