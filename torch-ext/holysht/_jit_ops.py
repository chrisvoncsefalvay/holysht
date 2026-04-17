"""Local GPU extension loader for HOLYSHT.

This keeps development off the heavyweight `kernel-builder` path by compiling a
small torch extension for the current machine only. CUDA builds reuse the
existing `.cu` sources; Apple Silicon builds compile the Objective-C++ MPS
bridge and load the Metal shader source from the repository at runtime. The
compiled artefacts are cached under `build/torch_extensions`.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_ROOT = Path(__file__).resolve().parents[2]
_BUILD_DIR = _ROOT / "build" / "torch_extensions"
_CUDA_EXTENSION_NAME = "holysht_ops_cuda"
_METAL_EXTENSION_NAME = "holysht_ops_metal"
_METAL_SHADER_PATH = _ROOT / "metal" / "fused_legendre.metal"


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


def _has_mps() -> bool:
    return (
        sys.platform == "darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _ensure_objcpp_extension():
    for module_name in ("setuptools._distutils.unixccompiler", "distutils.unixccompiler"):
        try:
            module = __import__(module_name, fromlist=["UnixCCompiler"])
            compiler = module.UnixCCompiler
            if ".mm" not in compiler.src_extensions:
                compiler.src_extensions.append(".mm")
                compiler.language_map[".mm"] = "objc"
            return
        except Exception:
            continue


def _metal_flags() -> list[str]:
    shader_path = str(_METAL_SHADER_PATH).replace("\\", "\\\\").replace('"', '\\"')
    return [
        "-O3",
        "-std=c++17",
        "-DMETAL_KERNEL",
        f'-DHOLYSHT_METAL_SHADER_PATH=\\"{shader_path}\\"',
    ]


def _load_ops():
    os.environ.setdefault("MAX_JOBS", "1")
    _BUILD_DIR.mkdir(parents=True, exist_ok=True)

    verbose = os.environ.get("HOLYSHT_VERBOSE_BUILD", "0") == "1"

    if torch.cuda.is_available():
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", _default_arch_list())
        load(
            name=_CUDA_EXTENSION_NAME,
            sources=[
                str(_ROOT / "torch-ext" / "torch_binding.cpp"),
                str(_ROOT / "cuda" / "fused_legendre.cu"),
                str(_ROOT / "cuda" / "fused_sht.cu"),
            ],
            extra_include_paths=[str(_ROOT / "torch-ext")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=_cuda_flags(),
            extra_ldflags=["-lcuda"],  # CUDA driver API for TMA (cuTensorMapEncodeTiled)
            build_directory=str(_BUILD_DIR),
            verbose=verbose,
            with_cuda=True,
            is_python_module=False,
        )
        return getattr(torch.ops, _CUDA_EXTENSION_NAME)

    if _has_mps():
        _ensure_objcpp_extension()
        load(
            name=_METAL_EXTENSION_NAME,
            sources=[
                str(_ROOT / "torch-ext" / "torch_binding.cpp"),
                str(_ROOT / "metal" / "fused_legendre.mm"),
            ],
            extra_include_paths=[str(_ROOT / "torch-ext")],
            extra_cflags=_metal_flags(),
            extra_ldflags=["-framework", "Foundation", "-framework", "Metal"],
            build_directory=str(_BUILD_DIR),
            verbose=verbose,
            with_cuda=False,
            is_python_module=False,
        )
        return getattr(torch.ops, _METAL_EXTENSION_NAME)

    raise ImportError("HOLYSHT could not find a supported GPU backend (CUDA or MPS).")


ops = _load_ops()
