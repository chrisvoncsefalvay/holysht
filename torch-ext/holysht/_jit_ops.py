"""Local CUDA extension loader for HOLYSHT.

This keeps development off the heavyweight `kernel-builder` path by compiling a
small torch extension for the current machine only. The compiled artefacts are
cached under `build/torch_extensions`.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parents[2]
_BUILD_DIR = _ROOT / "build" / "torch_extensions"
_EXTENSION_NAME = "holysht_ops"


def _default_arch_list() -> str:
    if not torch.cuda.is_available():
        return ""

    major, minor = torch.cuda.get_device_capability(0)
    if (major, minor) >= (12, 1):
        return "12.0+PTX"
    return f"{major}.{minor}"


def _cuda_flags() -> list[str]:
    flags = [
        "-O3",
        "-lineinfo",
        "-Xptxas=-warn-spills",
        "--expt-relaxed-constexpr",
    ]
    if os.environ.get("HOLYSHT_USE_FAST_MATH", "1") != "0":
        flags.append("--use_fast_math")
    return flags


def _load_ops():
    os.environ.setdefault("MAX_JOBS", "1")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _default_arch_list())
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    load(
        name=_EXTENSION_NAME,
        sources=[
            str(_ROOT / "torch-ext" / "torch_binding.cpp"),
            str(_ROOT / "cuda" / "fused_legendre.cu"),
            str(_ROOT / "cuda" / "fused_sht.cu"),
        ],
        extra_include_paths=[str(_ROOT / "torch-ext")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=_cuda_flags(),
        build_directory=str(_BUILD_DIR),
        verbose=os.environ.get("HOLYSHT_VERBOSE_BUILD", "0") == "1",
        with_cuda=True,
        is_python_module=False,
    )
    return getattr(torch.ops, _EXTENSION_NAME)


ops = _load_ops()
