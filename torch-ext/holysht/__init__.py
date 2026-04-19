"""HOLYSHT: Highly Optimised Legendre/Ylm/SHT.

CUDA-accelerated spherical harmonic transforms designed as a practical,
production-oriented companion to torch-harmonics.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

import contextlib
import json
import os
import time
from dataclasses import dataclass, asdict
from enum import IntEnum
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn

__all__ = [
    "RealSHT",
    "InverseRealSHT",
    "RealVectorSHT",
    "InverseRealVectorSHT",
    "GraphedModule",
    "legendre_forward",
    "legendre_inverse",
    "sht_forward",
    "sht_inverse",
]

# Prefer kernel-builder's generated alias module on packaged builds, then fall
# back to the local single-machine JIT loader for development.
try:
    from ._ops import ops as _ops
    _HAS_NATIVE_EXT = True
except ModuleNotFoundError:
    try:
        from ._jit_ops import ops as _ops
        _HAS_NATIVE_EXT = True
    except ImportError:
        _HAS_NATIVE_EXT = False
except ImportError:
    _HAS_NATIVE_EXT = False


def _can_use_cuda_legendre(input: torch.Tensor, weight_t: Optional[torch.Tensor]) -> bool:
    return (
        _HAS_NATIVE_EXT
        and weight_t is not None
        and input.is_cuda
        and weight_t.is_cuda
        and input.dtype == torch.complex64
        and weight_t.dtype == torch.float32
        and weight_t.is_contiguous()
    )


def _can_use_metal_complex_legendre(input: torch.Tensor, weight_t: Optional[torch.Tensor]) -> bool:
    return (
        _HAS_NATIVE_EXT
        and weight_t is not None
        and input.device.type == "mps"
        and weight_t.device.type == "mps"
        and input.dtype == torch.complex64
        and weight_t.dtype == torch.float32
        and input.is_contiguous()
        and weight_t.is_contiguous()
    )


def _can_use_native_complex_legendre(input: torch.Tensor, weight_t: Optional[torch.Tensor]) -> bool:
    return _can_use_cuda_legendre(input, weight_t) or _can_use_metal_complex_legendre(input, weight_t)


def _mps_scalar_native_max_nlat() -> int:
    raw = os.environ.get("HOLYSHT_MPS_SCALAR_NATIVE_MAX_NLAT")
    if raw is None:
        return 512
    try:
        return max(0, int(raw))
    except ValueError:
        return 512


def _prefer_metal_scalar_kernel(weight_t: Optional[torch.Tensor]) -> bool:
    return (
        weight_t is not None
        and weight_t.device.type == "mps"
        and weight_t.size(1) <= _mps_scalar_native_max_nlat()
    )


def _mps_vector_inverse_native_max_nlat() -> int:
    raw = os.environ.get("HOLYSHT_MPS_VECTOR_INVERSE_NATIVE_MAX_NLAT")
    if raw is None:
        return 512
    try:
        return max(0, int(raw))
    except ValueError:
        return 512


def _prefer_metal_vector_inverse_kernel(input: torch.Tensor, weight_t: Optional[torch.Tensor] = None) -> bool:
    raw = os.environ.get("HOLYSHT_MPS_VECTOR_INVERSE_NATIVE", "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return input.device.type != "mps"
    if input.device.type != "mps":
        return True
    return weight_t is not None and weight_t.size(1) <= _mps_vector_inverse_native_max_nlat()


def _can_use_cuda_real_legendre(input: torch.Tensor, weight_t: Optional[torch.Tensor]) -> bool:
    return (
        _HAS_NATIVE_EXT
        and weight_t is not None
        and input.is_cuda
        and weight_t.is_cuda
        and input.dtype in (torch.float32, torch.bfloat16)
        and weight_t.dtype == torch.float32
        and input.is_contiguous()
        and weight_t.is_contiguous()
    )


def _can_use_metal_real_legendre(input: torch.Tensor, weight_t: Optional[torch.Tensor]) -> bool:
    return (
        _HAS_NATIVE_EXT
        and weight_t is not None
        and input.device.type == "mps"
        and weight_t.device.type == "mps"
        and input.dtype in (torch.float32, torch.float16)
        and weight_t.dtype == torch.float32
        and input.is_contiguous()
        and weight_t.is_contiguous()
    )


def _can_use_cuda_vector(input: torch.Tensor, weight0_t: Optional[torch.Tensor], weight1_t: Optional[torch.Tensor]) -> bool:
    return (
        _HAS_NATIVE_EXT
        and weight0_t is not None
        and weight1_t is not None
        and input.is_cuda
        and weight0_t.is_cuda
        and weight1_t.is_cuda
        and input.dtype == torch.complex64
        and weight0_t.dtype == torch.float32
        and weight1_t.dtype == torch.float32
        and input.is_contiguous()
        and weight0_t.is_contiguous()
        and weight1_t.is_contiguous()
    )


def _can_use_metal_vector(input: torch.Tensor, weight0_t: Optional[torch.Tensor], weight1_t: Optional[torch.Tensor]) -> bool:
    return (
        _HAS_NATIVE_EXT
        and weight0_t is not None
        and weight1_t is not None
        and input.device.type == "mps"
        and weight0_t.device.type == "mps"
        and weight1_t.device.type == "mps"
        and input.dtype == torch.complex64
        and weight0_t.dtype == torch.float32
        and weight1_t.dtype == torch.float32
        and input.is_contiguous()
        and weight0_t.is_contiguous()
        and weight1_t.is_contiguous()
    )


def _can_use_native_real_legendre(input: torch.Tensor, weight_t: Optional[torch.Tensor]) -> bool:
    return _can_use_cuda_real_legendre(input, weight_t) or _can_use_metal_real_legendre(input, weight_t)


def _parse_nonnegative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


class _ForwardBackend(IntEnum):
    AUTO = 0
    FMA = 1
    TMA = 2
    TC_TF32 = 3
    TC_BF16 = 4


@dataclass(frozen=True)
class _AutotuneKey:
    device_name: str
    capability: str
    op_kind: str
    dtype_mode: str
    nlat: int
    lmax: int
    mmax: int
    batch_bucket: str


def _autotune_batch_bucket(batch_size: int) -> str:
    if batch_size <= 1:
        return "1"
    if batch_size == 2:
        return "2"
    if batch_size <= 4:
        return "3-4"
    return "5+"


def _default_autotune_cache_path() -> Path:
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base_dir = Path(cache_home) if cache_home else (Path.home() / ".cache")
    return base_dir / "holysht" / "holysht_autotune_cache.json"


class _AutotuneCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text())
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        return payload

    def load(self, key: _AutotuneKey) -> Optional[str]:
        return self._read().get(json.dumps(asdict(key), sort_keys=True))

    def store(self, key: _AutotuneKey, backend_name: str) -> None:
        payload = self._read()
        payload[json.dumps(asdict(key), sort_keys=True)] = backend_name
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _forced_cuda_forward_backend() -> _ForwardBackend:
    raw = os.environ.get("HOLYSHT_FORCE_BACKEND", "").strip().lower()
    mapping = {
        "": _ForwardBackend.AUTO,
        "auto": _ForwardBackend.AUTO,
        "fma": _ForwardBackend.FMA,
        "tma": _ForwardBackend.TMA,
        "tc_tf32": _ForwardBackend.TC_TF32,
        "tc_bf16": _ForwardBackend.TC_BF16,
    }
    return mapping.get(raw, _ForwardBackend.AUTO)


def _forced_vector_forward_strategy() -> str:
    raw = os.environ.get("HOLYSHT_FORCE_VECTOR_STRATEGY", "").strip().lower()
    mapping = {
        "": "auto",
        "auto": "auto",
        "native": "native_vector",
        "native_vector": "native_vector",
        "native-vector": "native_vector",
        "stacked": "stacked_real",
        "stacked_real": "stacked_real",
        "stacked-real": "stacked_real",
        "composed": "stacked_real",
    }
    return mapping.get(raw, "auto")


def _autotune_enabled() -> bool:
    raw = os.environ.get("HOLYSHT_AUTOTUNE", "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return False
    return True


def _deterministic_forward_backend(candidates: list[str]) -> str:
    for preferred in ("tma", "fma"):
        if preferred in candidates:
            return preferred
    if not candidates:
        raise ValueError("candidates must not be empty")
    return candidates[0]


def _deterministic_vector_forward_strategy(candidates: list[str]) -> str:
    for preferred in ("stacked_real", "native_vector"):
        if preferred in candidates:
            return preferred
    if not candidates:
        raise ValueError("candidates must not be empty")
    return candidates[0]


def _select_forward_backend_for_key(
    key: _AutotuneKey,
    candidates: list[str],
    benchmark,
    cache: Optional[_AutotuneCache] = None,
) -> str:
    forced = _forced_cuda_forward_backend()
    forced_name = {
        _ForwardBackend.FMA: "fma",
        _ForwardBackend.TMA: "tma",
        _ForwardBackend.TC_TF32: "tc_tf32",
        _ForwardBackend.TC_BF16: "tc_bf16",
    }.get(forced)
    if forced_name in candidates:
        return forced_name

    if not _autotune_enabled():
        return _deterministic_forward_backend(candidates)

    cache = cache or _AutotuneCache(
        Path(os.environ.get("HOLYSHT_AUTOTUNE_CACHE_PATH", _default_autotune_cache_path()))
    )
    if os.environ.get("HOLYSHT_AUTOTUNE_REBENCH", "0") != "1":
        cached = cache.load(key)
        if cached in candidates:
            return cached

    winner = min(candidates, key=benchmark)
    cache.store(key, winner)
    return winner


def _select_vector_forward_strategy_for_key(
    key: _AutotuneKey,
    candidates: list[str],
    benchmark,
    cache: Optional[_AutotuneCache] = None,
) -> str:
    forced = _forced_vector_forward_strategy()
    if forced in candidates:
        return forced

    if not _autotune_enabled():
        return _deterministic_vector_forward_strategy(candidates)

    cache = cache or _AutotuneCache(
        Path(os.environ.get("HOLYSHT_AUTOTUNE_CACHE_PATH", _default_autotune_cache_path()))
    )
    if os.environ.get("HOLYSHT_AUTOTUNE_REBENCH", "0") != "1":
        cached = cache.load(key)
        if cached in candidates:
            return cached

    winner = min(candidates, key=benchmark)
    cache.store(key, winner)
    return winner


def _backend_name_to_forward_hint(backend_name: str) -> _ForwardBackend:
    return {
        "fma": _ForwardBackend.FMA,
        "tma": _ForwardBackend.TMA,
        "tc_tf32": _ForwardBackend.TC_TF32,
        "tc_bf16": _ForwardBackend.TC_BF16,
    }[backend_name]


def _cuda_device_metadata(
    device: torch.device,
    device_name: Optional[str] = None,
    capability: Optional[str] = None,
) -> tuple[str, str]:
    if device_name is not None and capability is not None:
        return device_name, capability
    if device.type == "cuda":
        cap = torch.cuda.get_device_capability(device)
        return torch.cuda.get_device_name(device), f"{cap[0]}.{cap[1]}"
    return device.type, device.type


def _benchmark_real_forward_backend(
    input: torch.Tensor,
    weight_t: torch.Tensor,
    backend_hint: _ForwardBackend,
) -> float:
    if input.is_cuda:
        torch.cuda.synchronize(input.device)
    start = time.perf_counter()
    with torch.no_grad():
        _direct_legendre_forward_real(input, weight_t, backend_hint=backend_hint)
    if input.is_cuda:
        torch.cuda.synchronize(input.device)
    return time.perf_counter() - start


def _benchmark_device_callable(
    device: torch.device,
    run,
) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    with torch.no_grad():
        run()

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        run()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    return time.perf_counter() - start


def _stack_vector_forward_real_inputs(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, nlat, mmax = input.shape
    input_c = input.contiguous()
    if input_c.is_cuda and hasattr(_ops, "fused_vector_forward_pack_real"):
        stacked_w0 = torch.empty((4 * batch_size, nlat, mmax), device=input.device, dtype=torch.float32)
        stacked_w1 = torch.empty_like(stacked_w0)
        _ops.fused_vector_forward_pack_real(stacked_w0, stacked_w1, input_c)
        return stacked_w0, stacked_w1

    xr = torch.view_as_real(input_c)
    x00_flat = xr[:, 0, :, :, 0].reshape(-1, nlat, mmax)
    x01_flat = xr[:, 0, :, :, 1].reshape(-1, nlat, mmax)
    x10_flat = xr[:, 1, :, :, 0].reshape(-1, nlat, mmax)
    x11_flat = xr[:, 1, :, :, 1].reshape(-1, nlat, mmax)
    stacked_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0).contiguous()
    stacked_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0).contiguous()
    return stacked_w0, stacked_w1


def _recompose_vector_forward_real_outputs(
    out_w0: torch.Tensor,
    out_w1: torch.Tensor,
    *,
    batch_size: int,
    lmax: int,
    mmax: int,
) -> torch.Tensor:
    out_w0_c = out_w0.contiguous()
    out_w1_c = out_w1.contiguous()
    if out_w0_c.is_cuda and hasattr(_ops, "fused_vector_forward_recompose_real"):
        output = torch.empty((batch_size, 2, lmax, mmax), device=out_w0.device, dtype=torch.complex64)
        _ops.fused_vector_forward_recompose_real(output, out_w0_c, out_w1_c)
        return output

    r00 = out_w0_c[:batch_size]
    r01 = out_w0_c[batch_size:2 * batch_size]
    r10 = out_w0_c[2 * batch_size:3 * batch_size]
    r11 = out_w0_c[3 * batch_size:]
    s11 = out_w1_c[:batch_size]
    s10 = out_w1_c[batch_size:2 * batch_size]
    s01 = out_w1_c[2 * batch_size:3 * batch_size]
    s00 = out_w1_c[3 * batch_size:]

    xout = torch.empty((batch_size, 2, lmax, mmax, 2), dtype=torch.float32, device=out_w0.device)
    xout[:, 0, :, :, 0] = r00 - s11
    xout[:, 0, :, :, 1] = r01 + s10
    xout[:, 1, :, :, 0] = -s01 - r10
    xout[:, 1, :, :, 1] = s00 - r11
    return torch.view_as_complex(xout)


def _autotuned_direct_legendre_forward_real(
    input: torch.Tensor,
    weight_t: torch.Tensor,
    *,
    op_kind: str,
    dtype_mode: str,
    backend_candidates: list[str],
    padded_input: Optional[torch.Tensor] = None,
    padded_weight_t: Optional[torch.Tensor] = None,
    cache: Optional[_AutotuneCache] = None,
    device_name: Optional[str] = None,
    capability: Optional[str] = None,
) -> torch.Tensor:
    resolved_device_name, resolved_capability = _cuda_device_metadata(input.device, device_name, capability)
    key = _AutotuneKey(
        device_name=resolved_device_name,
        capability=resolved_capability,
        op_kind=op_kind,
        dtype_mode=dtype_mode,
        nlat=input.size(1),
        lmax=weight_t.size(0),
        mmax=weight_t.size(2),
        batch_bucket=_autotune_batch_bucket(int(input.size(0))),
    )

    def benchmark(name: str) -> float:
        backend_hint = _backend_name_to_forward_hint(name)
        bench_input = input
        bench_weight_t = weight_t
        if name == "tma" and padded_input is not None and padded_weight_t is not None:
            bench_input = padded_input
            bench_weight_t = padded_weight_t
        return _benchmark_real_forward_backend(bench_input, bench_weight_t, backend_hint)

    backend_name = _select_forward_backend_for_key(key, backend_candidates, benchmark, cache=cache)
    backend_hint = _backend_name_to_forward_hint(backend_name)
    if backend_name == "tma" and padded_input is not None and padded_weight_t is not None:
        return _direct_legendre_forward_real(padded_input, padded_weight_t, backend_hint=backend_hint)
    return _direct_legendre_forward_real(input, weight_t, backend_hint=backend_hint)


def _vector_forward_strategy_candidates(
    *,
    can_use_native: bool,
    forced_backend: _ForwardBackend,
) -> list[str]:
    candidates = ["stacked_real"]
    if can_use_native:
        candidates.append("native_vector")
    return candidates


def _native_vector_forward_backend_candidates(
    *,
    dtype_mode: str,
    can_use_tensor_core: bool,
) -> list[str]:
    if dtype_mode == "bf16":
        return ["tma", "tc_bf16"] if can_use_tensor_core else ["tma"]
    return ["fma", "tma", "tc_tf32"] if can_use_tensor_core else ["fma", "tma"]


def _autotuned_direct_vector_forward(
    input: torch.Tensor,
    weight0_t: torch.Tensor,
    weight1_t: torch.Tensor,
    *,
    dtype_mode: str,
    op_kind: str,
    backend_candidates: list[str],
    padded_input: Optional[torch.Tensor] = None,
    padded_weight0_t: Optional[torch.Tensor] = None,
    padded_weight1_t: Optional[torch.Tensor] = None,
    cache: Optional[_AutotuneCache] = None,
    device_name: Optional[str] = None,
    capability: Optional[str] = None,
) -> torch.Tensor:
    resolved_device_name, resolved_capability = _cuda_device_metadata(input.device, device_name, capability)
    key = _AutotuneKey(
        device_name=resolved_device_name,
        capability=resolved_capability,
        op_kind=op_kind,
        dtype_mode=dtype_mode,
        nlat=input.size(2),
        lmax=weight0_t.size(0),
        mmax=input.size(3),
        batch_bucket=_autotune_batch_bucket(int(input.size(0))),
    )

    forced_backend = _forced_cuda_forward_backend()
    if forced_backend in (_ForwardBackend.TC_TF32, _ForwardBackend.TC_BF16):
        forced_name = {
            _ForwardBackend.TC_TF32: "tc_tf32",
            _ForwardBackend.TC_BF16: "tc_bf16",
        }[forced_backend]
        if forced_name not in backend_candidates:
            raise RuntimeError(
                f"forced vector backend {forced_name} is unsupported for dtype_mode={dtype_mode}"
            )

    def benchmark(name: str) -> float:
        backend_hint = _backend_name_to_forward_hint(name)
        bench_input = padded_input if name == "tma" and padded_input is not None else input
        bench_weight0_t = padded_weight0_t if name == "tma" and padded_weight0_t is not None else weight0_t
        bench_weight1_t = padded_weight1_t if name == "tma" and padded_weight1_t is not None else weight1_t
        return _benchmark_device_callable(
            input.device,
            lambda: _direct_vector_legendre_forward(
                bench_input,
                bench_weight0_t,
                bench_weight1_t,
                backend_hint=backend_hint,
            ),
        )

    backend_name = _select_forward_backend_for_key(key, backend_candidates, benchmark, cache=cache)
    out_input = padded_input if backend_name == "tma" and padded_input is not None else input
    out_weight0_t = padded_weight0_t if backend_name == "tma" and padded_weight0_t is not None else weight0_t
    out_weight1_t = padded_weight1_t if backend_name == "tma" and padded_weight1_t is not None else weight1_t
    return _direct_vector_legendre_forward(
        out_input,
        out_weight0_t,
        out_weight1_t,
        backend_hint=_backend_name_to_forward_hint(backend_name),
    )[..., : input.size(3)]


def _run_vector_forward_native_cuda(
    input: torch.Tensor,
    weight0_t: torch.Tensor,
    weight1_t: torch.Tensor,
    *,
    dtype_mode: str,
    op_kind: str,
    mmax: int,
    tma_mmax_complex: int,
    weight0_t_tma_complex: torch.Tensor,
    weight1_t_tma_complex: torch.Tensor,
    cache: Optional[_AutotuneCache] = None,
    device_name: Optional[str] = None,
    capability: Optional[str] = None,
) -> torch.Tensor:
    can_use_tensor_core = (
        input.is_cuda
        and hasattr(torch.cuda, "get_device_capability")
        and torch.cuda.get_device_capability(input.device)[0] >= 9
        and hasattr(_ops, "fused_vector_legendre_forward_ex")
    )
    backend_candidates = _native_vector_forward_backend_candidates(
        dtype_mode=dtype_mode,
        can_use_tensor_core=can_use_tensor_core,
    )
    return _autotuned_direct_vector_forward(
        input,
        weight0_t,
        weight1_t,
        dtype_mode=dtype_mode,
        op_kind=op_kind,
        backend_candidates=backend_candidates,
        padded_input=_pad_last_dim(input, tma_mmax_complex) if _cuda_tma_available(input) and tma_mmax_complex != mmax else None,
        padded_weight0_t=weight0_t_tma_complex if _cuda_tma_available(input) and tma_mmax_complex != mmax else None,
        padded_weight1_t=weight1_t_tma_complex if _cuda_tma_available(input) and tma_mmax_complex != mmax else None,
        cache=cache,
        device_name=device_name,
        capability=capability,
    )


def _run_vector_forward_stacked_real_cuda(
    input: torch.Tensor,
    weight0_t: torch.Tensor,
    weight1_t: torch.Tensor,
    *,
    dtype_mode: str,
    lmax: int,
    mmax: int,
    tma_mmax_real: int,
    weight0_t_tma_real: torch.Tensor,
    weight1_t_tma_real: torch.Tensor,
    cache: Optional[_AutotuneCache] = None,
    device_name: Optional[str] = None,
    capability: Optional[str] = None,
) -> torch.Tensor:
    batch_size = input.size(0)
    stacked_w0, stacked_w1 = _stack_vector_forward_real_inputs(input)
    backend_candidates = ["fma", "tma", "tc_bf16"] if dtype_mode == "bf16" else ["fma", "tma", "tc_tf32"]
    stacked_w0_work = stacked_w0.bfloat16().contiguous() if dtype_mode == "bf16" else stacked_w0
    stacked_w1_work = stacked_w1.bfloat16().contiguous() if dtype_mode == "bf16" else stacked_w1
    tma_pad_w0 = None
    if _cuda_tma_available(stacked_w0_work) and tma_mmax_real != mmax:
        tma_pad_w0 = _pad_last_dim(stacked_w0_work, tma_mmax_real)
    out_w0 = _autotuned_direct_legendre_forward_real(
        stacked_w0_work,
        weight0_t,
        op_kind="vector-real-forward",
        dtype_mode=dtype_mode,
        backend_candidates=backend_candidates,
        padded_input=tma_pad_w0,
        padded_weight_t=weight0_t_tma_real if tma_pad_w0 is not None else None,
        cache=cache,
        device_name=device_name,
        capability=capability,
    )[..., :mmax]
    tma_pad_w1 = None
    if _cuda_tma_available(stacked_w1_work) and tma_mmax_real != mmax:
        tma_pad_w1 = _pad_last_dim(stacked_w1_work, tma_mmax_real)
    out_w1 = _autotuned_direct_legendre_forward_real(
        stacked_w1_work,
        weight1_t,
        op_kind="vector-real-forward",
        dtype_mode=dtype_mode,
        backend_candidates=backend_candidates,
        padded_input=tma_pad_w1,
        padded_weight_t=weight1_t_tma_real if tma_pad_w1 is not None else None,
        cache=cache,
        device_name=device_name,
        capability=capability,
    )[..., :mmax]
    return _recompose_vector_forward_real_outputs(
        out_w0,
        out_w1,
        batch_size=batch_size,
        lmax=lmax,
        mmax=mmax,
    )


def _run_vector_forward_stacked_real_fp32(
    input: torch.Tensor,
    weight0_t: torch.Tensor,
    weight1_t: torch.Tensor,
    *,
    lmax: int,
    mmax: int,
    tma_mmax_real: int,
    weight0_t_tma_real: torch.Tensor,
    weight1_t_tma_real: torch.Tensor,
    cache: Optional[_AutotuneCache] = None,
    device_name: Optional[str] = None,
    capability: Optional[str] = None,
) -> torch.Tensor:
    return _run_vector_forward_stacked_real_cuda(
        input,
        weight0_t,
        weight1_t,
        dtype_mode="fp32",
        lmax=lmax,
        mmax=mmax,
        tma_mmax_real=tma_mmax_real,
        weight0_t_tma_real=weight0_t_tma_real,
        weight1_t_tma_real=weight1_t_tma_real,
        cache=cache,
        device_name=device_name,
        capability=capability,
    )


def _mps_scalar_fallback_chunk_m(weight: Optional[torch.Tensor], inverse: bool) -> int:
    if weight is None or weight.device.type != "mps":
        return 0

    specific_env = (
        "HOLYSHT_MPS_SCALAR_INVERSE_EINSUM_M_CHUNK"
        if inverse
        else "HOLYSHT_MPS_SCALAR_FORWARD_EINSUM_M_CHUNK"
    )
    raw = os.environ.get(specific_env)
    if raw is None:
        raw = os.environ.get("HOLYSHT_MPS_SCALAR_EINSUM_M_CHUNK")
    if raw is not None:
        try:
            return max(0, int(raw))
        except ValueError:
            return 0

    # On unified-memory Apple GPUs, chunk the inverse fallback only when the
    # spectral weight slab is large enough that the full einsum risks paging.
    if not inverse:
        return 0

    threshold_mb = _parse_nonnegative_int_env(
        "HOLYSHT_MPS_SCALAR_INVERSE_EINSUM_CHUNK_THRESHOLD_MB",
        1024,
    )
    if threshold_mb == 0:
        return 0

    weight_bytes = weight.numel() * weight.element_size()
    if weight_bytes < threshold_mb * 1024 * 1024:
        return 0

    return 64 if weight.size(0) >= 512 else 128


def _weight_chunk_for_dtype(weight: torch.Tensor, m0: int, m1: int, dtype: torch.dtype) -> torch.Tensor:
    chunk = weight[m0:m1]
    return chunk if chunk.dtype == dtype else chunk.to(dtype)


def _chunked_legendre_forward_fallback(
    input: torch.Tensor,
    weights: torch.Tensor,
    chunk_m: int,
) -> torch.Tensor:
    x = torch.view_as_real(input)
    pieces = []
    for m0 in range(0, input.size(-1), chunk_m):
        m1 = min(input.size(-1), m0 + chunk_m)
        pieces.append(
            torch.einsum(
                "...kmr,mlk->...lmr",
                x[..., :, m0:m1, :],
                _weight_chunk_for_dtype(weights, m0, m1, x.dtype),
            ).contiguous()
        )

    out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2).contiguous()
    return torch.view_as_complex(out)


def _chunked_legendre_inverse_fallback(
    input: torch.Tensor,
    pct: torch.Tensor,
    chunk_m: int,
) -> torch.Tensor:
    x = torch.view_as_real(input)
    pieces = []
    for m0 in range(0, input.size(-1), chunk_m):
        m1 = min(input.size(-1), m0 + chunk_m)
        pieces.append(
            torch.einsum(
                "...lmr,mlk->...kmr",
                x[..., :, m0:m1, :],
                _weight_chunk_for_dtype(pct, m0, m1, x.dtype),
            ).contiguous()
        )

    out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2).contiguous()
    return torch.view_as_complex(out)


def _mul_i(x: torch.Tensor) -> torch.Tensor:
    """Multiply a complex tensor by +i without promoting dtype."""
    return torch.complex(-x.imag, x.real)


@contextlib.contextmanager
def _nvtx_range(name: str):
    enabled = os.environ.get("HOLYSHT_ENABLE_NVTX", "0") == "1"
    if enabled and torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
    else:
        yield


def _cuda_tma_requested() -> bool:
    raw = os.environ.get("HOLYSHT_USE_TMA")
    if raw is None or raw == "":
        return True
    if raw == "0":
        return False
    if raw == "1":
        return True
    return True


def _cuda_tma_available(tensor: torch.Tensor) -> bool:
    return (
        _HAS_NATIVE_EXT
        and tensor.is_cuda
        and _cuda_tma_requested()
        and torch.cuda.get_device_capability(tensor.device)[0] >= 9
    )


def _cuda_tma_batch_tile() -> int:
    raw = os.environ.get("HOLYSHT_TMA_BATCH_TILE")
    if raw is None or raw == "":
        return 2
    try:
        parsed = int(raw)
    except ValueError:
        return 2
    return 1 if parsed <= 1 else 2


def _aligned_mmax_for_tma(mmax: int, dtype: torch.dtype) -> int:
    if dtype == torch.complex64:
        elem_size = 8
    elif dtype == torch.float32:
        elem_size = 4
    elif dtype in (torch.bfloat16, torch.float16):
        elem_size = 2
    else:
        return mmax

    quantum = max(1, 16 // elem_size)
    return ((mmax + quantum - 1) // quantum) * quantum


def _aligned_mmax_for_tma_tile(mmax: int) -> int:
    # The public forward microkernels still advance `m` in 8-lane tiles, so
    # padding only to the minimal 16-byte alignment leaves a slower partial
    # tail. Keep both complex and real wrapper pads on the tile-aligned quantum.
    return _aligned_mmax_for_tma(mmax, torch.bfloat16)


def _pad_last_dim(x: torch.Tensor, padded_size: int) -> torch.Tensor:
    if x.size(-1) == padded_size:
        return x.contiguous()
    out_shape = list(x.shape)
    out_shape[-1] = padded_size
    out = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
    out[..., :x.size(-1)] = x
    return out


def _direct_legendre_forward_complex(input: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
    output = torch.empty(
        input.size(0), weight_t.size(0), input.size(2),
        device=input.device, dtype=torch.complex64
    )
    _ops.fused_legendre_forward(output, input.contiguous(), weight_t)
    return output


def _direct_legendre_forward_real(
    input: torch.Tensor,
    weight_t: torch.Tensor,
    backend_hint: _ForwardBackend = _ForwardBackend.AUTO,
) -> torch.Tensor:
    output = torch.empty(
        input.size(0), weight_t.size(0), input.size(2),
        device=input.device, dtype=torch.float32
    )
    if input.is_cuda and hasattr(_ops, "fused_legendre_forward_real_ex"):
        _ops.fused_legendre_forward_real_ex(output, input.contiguous(), weight_t, int(backend_hint))
    else:
        _ops.fused_legendre_forward_real(output, input.contiguous(), weight_t)
    return output


def _direct_vector_legendre_forward(
    input: torch.Tensor,
    weight0_t: torch.Tensor,
    weight1_t: torch.Tensor,
    backend_hint: _ForwardBackend = _ForwardBackend.AUTO,
) -> torch.Tensor:
    output = torch.empty(
        input.size(0), 2, weight0_t.size(0), input.size(3),
        device=input.device, dtype=torch.complex64
    )
    if input.is_cuda and hasattr(_ops, "fused_vector_legendre_forward_ex"):
        _ops.fused_vector_legendre_forward_ex(
            output,
            input.contiguous(),
            weight0_t,
            weight1_t,
            int(backend_hint),
        )
    else:
        _ops.fused_vector_legendre_forward(output, input.contiguous(), weight0_t, weight1_t)
    return output


def _prepare_irfft_input(x: torch.Tensor, nlon: int, active_mmax: Optional[int] = None) -> torch.Tensor:
    """Pad/clean an rFFT-format complex tensor before irfft."""
    active_mmax = x.size(-1) if active_mmax is None else active_mmax
    full_mmax = nlon // 2 + 1

    if x.size(-1) == full_mmax:
        out = x.contiguous()
    else:
        out_shape = list(x.shape)
        out_shape[-1] = full_mmax
        out = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        out[..., :x.size(-1)] = x

    if _HAS_NATIVE_EXT and (out.is_cuda or out.device.type == "mps") and out.dtype == torch.complex64:
        orig_shape = out.shape
        flat = out.reshape(-1, orig_shape[-2], orig_shape[-1]).contiguous()
        _ops.sht_prepare_irfft(flat, active_mmax, nlon)
        return flat.reshape(orig_shape)

    out[..., 0] = out[..., 0].real.to(torch.complex64)
    if nlon % 2 == 0:
        nyquist_idx = nlon // 2
        if nyquist_idx < out.size(-1):
            out[..., nyquist_idx] = out[..., nyquist_idx].real.to(torch.complex64)
    return out


class _FusedLegendreForwardFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        input_c = input.contiguous()
        output = torch.empty(
            input_c.size(0), weight_t.size(0), input_c.size(2),
            device=input_c.device, dtype=torch.complex64
        )
        _ops.fused_legendre_forward(output, input_c, weight_t)
        ctx.save_for_backward(weight_t)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight_t,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(
            grad_output.size(0), weight_t.size(1), grad_output.size(2),
            device=grad_output.device, dtype=torch.complex64
        )
        _ops.fused_legendre_inverse(grad_input, grad_output, weight_t)
        return grad_input, None


class _FusedLegendreInverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        input_c = input.contiguous()
        output = torch.empty(
            input_c.size(0), weight_t.size(1), input_c.size(2),
            device=input_c.device, dtype=torch.complex64
        )
        _ops.fused_legendre_inverse(output, input_c, weight_t)
        ctx.save_for_backward(weight_t)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight_t,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(
            grad_output.size(0), weight_t.size(0), grad_output.size(2),
            device=grad_output.device, dtype=torch.complex64
        )
        _ops.fused_legendre_forward(grad_input, grad_output, weight_t)
        return grad_input, None


class _FusedLegendreForwardRealFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight_t: torch.Tensor) -> torch.Tensor:
        input_c = input.contiguous()
        output = torch.empty(
            input_c.size(0), weight_t.size(0), input_c.size(2),
            device=input_c.device, dtype=torch.float32
        )
        _ops.fused_legendre_forward_real(output, input_c, weight_t)
        ctx.save_for_backward(weight_t)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (weight_t,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(
            grad_output.size(0), weight_t.size(1), grad_output.size(2),
            device=grad_output.device, dtype=torch.float32
        )
        _ops.fused_legendre_inverse_real(grad_input, grad_output, weight_t)
        return grad_input, None


class _FusedVectorLegendreForwardFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight0_t: torch.Tensor, weight1_t: torch.Tensor) -> torch.Tensor:
        input_c = input.contiguous()
        output = torch.empty(
            input_c.size(0), 2, weight0_t.size(0), input_c.size(3),
            device=input_c.device, dtype=torch.complex64
        )
        _ops.fused_vector_legendre_forward(output, input_c, weight0_t, weight1_t)
        ctx.save_for_backward(weight0_t, weight1_t)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight0_t, weight1_t = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(
            grad_output.size(0), 2, weight0_t.size(1), grad_output.size(3),
            device=grad_output.device, dtype=torch.complex64
        )
        _ops.fused_vector_legendre_inverse(grad_input, grad_output, weight0_t, weight1_t)
        return grad_input, None, None


class _FusedVectorLegendreInverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight0_t: torch.Tensor, weight1_t: torch.Tensor) -> torch.Tensor:
        input_c = input.contiguous()
        output = torch.empty(
            input_c.size(0), 2, weight0_t.size(1), input_c.size(3),
            device=input_c.device, dtype=torch.complex64
        )
        _ops.fused_vector_legendre_inverse(output, input_c, weight0_t, weight1_t)
        ctx.save_for_backward(weight0_t, weight1_t)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        weight0_t, weight1_t = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty(
            grad_output.size(0), 2, weight0_t.size(0), grad_output.size(3),
            device=grad_output.device, dtype=torch.complex64
        )
        _ops.fused_vector_legendre_forward(grad_input, grad_output, weight0_t, weight1_t)
        return grad_input, None, None


# ============================================================================
# Fused Legendre Transform
# ============================================================================

def fused_legendre_forward(
    input: torch.Tensor,       # [B, nlat, mmax] complex64
    weights: torch.Tensor,     # [mmax, lmax, nlat] float32 (original torch-harmonics layout)
    weight_t: Optional[torch.Tensor] = None,  # [lmax, nlat, mmax] pre-transposed
) -> torch.Tensor:
    """Fused forward Legendre transform operating on complex tensors.

    Computes out[b,l,m] = Σ_k weights[m,l,k] · input[b,k,m] for complex input,
    fusing the real and imaginary multiplications into a single pass.

    Uses the custom CUDA kernels when the extension is available, otherwise
    falls back to a stacked einsum.
    """
    B = input.size(0)
    nlat = input.size(1)
    mmax = input.size(2)
    lmax = weights.size(1)

    if weight_t is None:
        weight_t = weights.float().permute(1, 2, 0).contiguous()

    if _can_use_native_complex_legendre(input, weight_t) and (
        input.device.type != "mps" or _prefer_metal_scalar_kernel(weight_t)
    ):
        return _FusedLegendreForwardFn.apply(input, weight_t)
    else:
        chunk_m = _mps_scalar_fallback_chunk_m(weights, inverse=False)
        if 0 < chunk_m < input.size(-1):
            return _chunked_legendre_forward_fallback(input, weights, chunk_m)

        x = torch.view_as_real(input)  # [B, nlat, mmax, 2]
        out = torch.einsum("...kmr,mlk->...lmr", x, weights.to(x.dtype)).contiguous()
        return torch.view_as_complex(out)


def fused_legendre_inverse(
    input: torch.Tensor,       # [B, lmax, mmax] complex64
    pct: torch.Tensor,         # [mmax, lmax, nlat] float32
    pct_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused inverse Legendre transform."""
    B = input.size(0)
    lmax = input.size(1)
    mmax = input.size(2)
    nlat = pct.size(2)

    if pct_t is None:
        pct_t = pct.float().permute(1, 2, 0).contiguous()

    if _can_use_native_complex_legendre(input, pct_t) and (
        input.device.type != "mps" or _prefer_metal_scalar_kernel(pct_t)
    ):
        return _FusedLegendreInverseFn.apply(input, pct_t)
    else:
        chunk_m = _mps_scalar_fallback_chunk_m(pct, inverse=True)
        if 0 < chunk_m < input.size(-1):
            return _chunked_legendre_inverse_fallback(input, pct, chunk_m)

        x = torch.view_as_real(input)
        out = torch.einsum("...lmr,mlk->...kmr", x, pct.to(x.dtype)).contiguous()
        return torch.view_as_complex(out)


def fused_legendre_forward_real(
    input: torch.Tensor,       # [B, nlat, mmax] float32 or bfloat16
    weight_t: torch.Tensor,    # [lmax, nlat, mmax] float32
) -> torch.Tensor:
    """Real-valued forward Legendre transform with float accumulation."""
    if _can_use_native_real_legendre(input, weight_t):
        return _FusedLegendreForwardRealFn.apply(input, weight_t)

    return torch.einsum("bkm,lkm->blm", input.float(), weight_t)


def fused_legendre_inverse_real(
    input: torch.Tensor,       # [B, lmax, mmax] float32
    weight_t: torch.Tensor,    # [lmax, nlat, mmax] float32
) -> torch.Tensor:
    """Real-valued inverse Legendre transform with float accumulation."""
    if _can_use_native_real_legendre(input, weight_t):
        grad_output = input.contiguous()
        output = torch.empty(
            grad_output.size(0), weight_t.size(1), grad_output.size(2),
            device=grad_output.device, dtype=torch.float32
        )
        _ops.fused_legendre_inverse_real(output, grad_output, weight_t)
        return output

    return torch.einsum("blm,lkm->bkm", input.float(), weight_t)


# ============================================================================
# Fused SHT (complete pipeline)
# ============================================================================

def fused_sht_forward(
    x: torch.Tensor,           # [B, nlat, nlon] float32
    weights: torch.Tensor,     # [mmax, lmax, nlat] float32
    mmax: int,
    weight_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Complete fused forward SHT: rfft → fused Legendre → complex coefficients.

    Replaces RealSHT.forward() with fewer intermediate allocations.
    """
    with _nvtx_range("holysht.scalar_forward"):
        x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
        x_fft = x_fft[..., :mmax]
        return fused_legendre_forward(x_fft, weights, weight_t)


def fused_sht_inverse(
    coeffs: torch.Tensor,      # [B, lmax, mmax] complex64
    pct: torch.Tensor,         # [mmax, lmax, nlat] float32
    nlon: int,
    pct_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Complete fused inverse SHT: fused Legendre → zero-pad → irfft."""
    with _nvtx_range("holysht.scalar_inverse"):
        x = fused_legendre_inverse(coeffs, pct, pct_t)
        x = _prepare_irfft_input(x, nlon, coeffs.size(-1))
        return torch.fft.irfft(x, n=nlon, dim=-1, norm="forward")


# ============================================================================
# nn.Module wrappers (drop-in replacements for torch-harmonics)
# ============================================================================

class RealSHT(nn.Module):
    """Optimised drop-in replacement for ``torch_harmonics.RealSHT``.

    Args:
        dtype: Weight precision. ``"fp32"`` (default), ``"bf16"`` (CUDA), or ``"fp16"`` (MPS).
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        grid: str = "equiangular",
        norm: str = "ortho",
        csphase: bool = True,
        dtype: str = "fp32",
    ):
        super().__init__()
        from torch_harmonics import RealSHT
        ref = RealSHT(
            nlat,
            nlon,
            lmax=lmax,
            mmax=mmax,
            grid=grid,
            norm=norm,
            csphase=csphase,
        )
        self.nlat = ref.nlat
        self.nlon = ref.nlon
        self.lmax = ref.lmax
        self.mmax = ref.mmax
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self._use_bf16 = (dtype == "bf16")
        self._use_fp16 = (dtype == "fp16")
        if self._use_bf16:
            w_dtype = torch.bfloat16
        elif self._use_fp16:
            w_dtype = torch.float16
        else:
            w_dtype = torch.float32
        self.register_buffer("weights", ref.weights.to(w_dtype))
        self.register_buffer("weight_t", ref.weights.float().permute(1, 2, 0).contiguous())
        self._tma_mmax_complex = _aligned_mmax_for_tma_tile(self.mmax)
        self._tma_mmax_real = _aligned_mmax_for_tma_tile(self.mmax)
        self._tma_pad_complex_min_nlat = 512
        self.register_buffer(
            "weight_t_tma_complex",
            _pad_last_dim(self.weight_t, self._tma_mmax_complex),
            persistent=False,
        )
        self.register_buffer(
            "weight_t_tma_real",
            _pad_last_dim(self.weight_t, self._tma_mmax_real),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_bf16:
            with _nvtx_range("holysht.scalar_forward_bf16"):
                x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
                x_fft = x_fft[..., :self.mmax]
                xr = torch.view_as_real(x_fft)
                if _HAS_NATIVE_EXT and x.is_cuda and not x.requires_grad:
                    xr_bf16 = xr.bfloat16().contiguous()
                    tma_pad_input_re = None
                    tma_pad_input_im = None
                    tma_pad_weight = None
                    if _cuda_tma_available(xr_bf16) and self._tma_mmax_real != self.mmax:
                        tma_pad_input_re = _pad_last_dim(xr_bf16[..., 0].contiguous(), self._tma_mmax_real)
                        tma_pad_input_im = _pad_last_dim(xr_bf16[..., 1].contiguous(), self._tma_mmax_real)
                        tma_pad_weight = self.weight_t_tma_real
                    out_re = _autotuned_direct_legendre_forward_real(
                        xr_bf16[..., 0].contiguous(),
                        self.weight_t,
                        op_kind="scalar-real-forward",
                        dtype_mode="bf16",
                        backend_candidates=["fma", "tma", "tc_bf16"],
                        padded_input=tma_pad_input_re,
                        padded_weight_t=tma_pad_weight,
                    )[..., :self.mmax]
                    out_im = _autotuned_direct_legendre_forward_real(
                        xr_bf16[..., 1].contiguous(),
                        self.weight_t,
                        op_kind="scalar-real-forward",
                        dtype_mode="bf16",
                        backend_candidates=["fma", "tma", "tc_bf16"],
                        padded_input=tma_pad_input_im,
                        padded_weight_t=tma_pad_weight,
                    )[..., :self.mmax]
                    return torch.complex(out_re, out_im)

                B = x.size(0)
                xs = torch.cat([xr[..., 0], xr[..., 1]], dim=0).bfloat16()
                out = torch.einsum("bkm,mlk->blm", xs, self.weights).float()
                return torch.complex(out[:B], out[B:])
        if self._use_fp16:
            with _nvtx_range("holysht.scalar_forward_fp16"):
                x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
                x_fft = x_fft[..., :self.mmax]
                xr = torch.view_as_real(x_fft)
                if _HAS_NATIVE_EXT and x.device.type == "mps" and not x.requires_grad:
                    xr_fp16 = xr.half().contiguous()
                    out_re = fused_legendre_forward_real(xr_fp16[..., 0].contiguous(), self.weight_t)
                    out_im = fused_legendre_forward_real(xr_fp16[..., 1].contiguous(), self.weight_t)
                    return torch.complex(out_re, out_im)

                B = x.size(0)
                xs = torch.cat([xr[..., 0], xr[..., 1]], dim=0).half()
                out = torch.einsum("bkm,mlk->blm", xs, self.weights).float()
                return torch.complex(out[:B], out[B:])
        if _HAS_NATIVE_EXT and x.is_cuda and not x.requires_grad:
            with _nvtx_range("holysht.scalar_forward_fp32_autotune"):
                x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
                x_fft = x_fft[..., :self.mmax]
                xr = torch.view_as_real(x_fft)
                tma_pad_input_re = None
                tma_pad_input_im = None
                tma_pad_weight = None
                if _cuda_tma_available(x) and self._tma_mmax_real != self.mmax:
                    tma_pad_input_re = _pad_last_dim(xr[..., 0].contiguous(), self._tma_mmax_real)
                    tma_pad_input_im = _pad_last_dim(xr[..., 1].contiguous(), self._tma_mmax_real)
                    tma_pad_weight = self.weight_t_tma_real
                out_re = _autotuned_direct_legendre_forward_real(
                    xr[..., 0].contiguous(),
                    self.weight_t,
                    op_kind="scalar-real-forward",
                    dtype_mode="fp32",
                    backend_candidates=["fma", "tma", "tc_tf32"],
                    padded_input=tma_pad_input_re,
                    padded_weight_t=tma_pad_weight,
                )[..., :self.mmax]
                out_im = _autotuned_direct_legendre_forward_real(
                    xr[..., 1].contiguous(),
                    self.weight_t,
                    op_kind="scalar-real-forward",
                    dtype_mode="fp32",
                    backend_candidates=["fma", "tma", "tc_tf32"],
                    padded_input=tma_pad_input_im,
                    padded_weight_t=tma_pad_weight,
                )[..., :self.mmax]
                return torch.complex(out_re, out_im)
        if (
            _HAS_NATIVE_EXT
            and x.is_cuda
            and not x.requires_grad
            and _cuda_tma_available(x)
            and self._tma_mmax_complex != self.mmax
            and self.nlat >= self._tma_pad_complex_min_nlat
        ):
            with _nvtx_range("holysht.scalar_forward_tma_pad"):
                x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
                x_fft = x_fft[..., :self.mmax].contiguous()
                out = _direct_legendre_forward_complex(
                    _pad_last_dim(x_fft, self._tma_mmax_complex),
                    self.weight_t_tma_complex,
                )
                return out[..., :self.mmax]
        return fused_sht_forward(x, self.weights, self.mmax, self.weight_t)


class InverseRealSHT(nn.Module):
    """Optimised drop-in replacement for ``torch_harmonics.InverseRealSHT``."""

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        grid: str = "equiangular",
        norm: str = "ortho",
        csphase: bool = True,
    ):
        super().__init__()
        from torch_harmonics import InverseRealSHT
        ref = InverseRealSHT(
            nlat,
            nlon,
            lmax=lmax,
            mmax=mmax,
            grid=grid,
            norm=norm,
            csphase=csphase,
        )
        self.nlat = ref.nlat
        self.nlon = ref.nlon
        self.lmax = ref.lmax
        self.mmax = ref.mmax
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.register_buffer("pct", ref.pct.float())
        self.register_buffer("pct_t", ref.pct.float().permute(1, 2, 0).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fused_sht_inverse(x, self.pct, self.nlon, self.pct_t)


class RealVectorSHT(nn.Module):
    """Optimised drop-in replacement for ``torch_harmonics.RealVectorSHT``.

    Reduces eight reference einsums to two composed Legendre passes on the
    default FP32 CUDA/MPS path.

    Args:
        dtype: Weight precision. ``"fp32"`` (default), ``"bf16"`` (CUDA), or ``"fp16"`` (MPS).
    """

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        grid: str = "equiangular",
        norm: str = "ortho",
        csphase: bool = True,
        dtype: str = "fp32",
    ):
        super().__init__()
        from torch_harmonics import RealVectorSHT
        ref = RealVectorSHT(
            nlat,
            nlon,
            lmax=lmax,
            mmax=mmax,
            grid=grid,
            norm=norm,
            csphase=csphase,
        )
        self.nlat = ref.nlat
        self.nlon = ref.nlon
        self.lmax = ref.lmax
        self.mmax = ref.mmax
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self._use_bf16 = (dtype == "bf16")
        self._use_fp16 = (dtype == "fp16")
        if self._use_bf16:
            w_dtype = torch.bfloat16
        elif self._use_fp16:
            w_dtype = torch.float16
        else:
            w_dtype = torch.float32
        self.register_buffer("w0", ref.weights[0].to(w_dtype))  # [mmax, lmax, nlat]
        self.register_buffer("w1", ref.weights[1].to(w_dtype))  # [mmax, lmax, nlat]
        self.register_buffer("w0_t", ref.weights[0].float().permute(1, 2, 0).contiguous())
        self.register_buffer("w1_t", ref.weights[1].float().permute(1, 2, 0).contiguous())
        self._tma_mmax_complex = _aligned_mmax_for_tma_tile(self.mmax)
        self._tma_mmax_real = _aligned_mmax_for_tma_tile(self.mmax)
        self.register_buffer(
            "w0_t_tma_complex",
            _pad_last_dim(self.w0_t, self._tma_mmax_complex),
            persistent=False,
        )
        self.register_buffer(
            "w1_t_tma_complex",
            _pad_last_dim(self.w1_t, self._tma_mmax_complex),
            persistent=False,
        )
        self.register_buffer(
            "w0_t_tma_real",
            _pad_last_dim(self.w0_t, self._tma_mmax_real),
            persistent=False,
        )
        self.register_buffer(
            "w1_t_tma_real",
            _pad_last_dim(self.w1_t, self._tma_mmax_real),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.nlat and x.shape[-1] == self.nlon

        with _nvtx_range("holysht.vector_forward"):
            x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
            mmax = self.mmax
            x = x[..., :mmax].contiguous()

            if _HAS_NATIVE_EXT and x.is_cuda and not x.requires_grad and not self._use_fp16:
                leading_shape = x.shape[:-3]
                x_flat = x.reshape(-1, 2, self.nlat, mmax).contiguous()
                can_use_native = _can_use_cuda_vector(x_flat, self.w0_t, self.w1_t)
                forced_backend = _forced_cuda_forward_backend()
                dtype_mode = "bf16" if self._use_bf16 else "fp32"
                cache = _AutotuneCache(
                    Path(os.environ.get("HOLYSHT_AUTOTUNE_CACHE_PATH", _default_autotune_cache_path()))
                )
                device_name, capability = _cuda_device_metadata(x_flat.device)
                strategy_candidates = _vector_forward_strategy_candidates(
                    can_use_native=can_use_native,
                    forced_backend=forced_backend,
                )
                if can_use_native and len(strategy_candidates) > 1:
                    key = _AutotuneKey(
                        device_name=device_name,
                        capability=capability,
                        op_kind="vector-forward-strategy",
                        dtype_mode=dtype_mode,
                        nlat=self.nlat,
                        lmax=self.lmax,
                        mmax=mmax,
                        batch_bucket=_autotune_batch_bucket(int(x_flat.size(0))),
                    )

                    def benchmark(strategy_name: str) -> float:
                        if strategy_name == "native_vector":
                            return _benchmark_device_callable(
                                x_flat.device,
                                lambda: _run_vector_forward_native_cuda(
                                    x_flat,
                                    self.w0_t,
                                    self.w1_t,
                                    dtype_mode=dtype_mode,
                                    op_kind="vector-native-forward",
                                    mmax=mmax,
                                    tma_mmax_complex=self._tma_mmax_complex,
                                    weight0_t_tma_complex=self.w0_t_tma_complex,
                                    weight1_t_tma_complex=self.w1_t_tma_complex,
                                    cache=cache,
                                    device_name=device_name,
                                    capability=capability,
                                ),
                            )
                        return _benchmark_device_callable(
                            x_flat.device,
                            lambda: _run_vector_forward_stacked_real_cuda(
                                x_flat,
                                self.w0_t,
                                self.w1_t,
                                dtype_mode=dtype_mode,
                                lmax=self.lmax,
                                mmax=mmax,
                                tma_mmax_real=self._tma_mmax_real,
                                weight0_t_tma_real=self.w0_t_tma_real,
                                weight1_t_tma_real=self.w1_t_tma_real,
                                cache=cache,
                                device_name=device_name,
                                capability=capability,
                            ),
                        )

                    strategy = _select_vector_forward_strategy_for_key(
                        key,
                        strategy_candidates,
                        benchmark,
                        cache=cache,
                    )
                else:
                    strategy = strategy_candidates[0]

                if strategy == "native_vector":
                    out = _run_vector_forward_native_cuda(
                        x_flat,
                        self.w0_t,
                        self.w1_t,
                        dtype_mode=dtype_mode,
                        op_kind="vector-native-forward",
                        mmax=mmax,
                        tma_mmax_complex=self._tma_mmax_complex,
                        weight0_t_tma_complex=self.w0_t_tma_complex,
                        weight1_t_tma_complex=self.w1_t_tma_complex,
                        cache=cache,
                        device_name=device_name,
                        capability=capability,
                    )
                else:
                    out = _run_vector_forward_stacked_real_cuda(
                        x_flat,
                        self.w0_t,
                        self.w1_t,
                        dtype_mode=dtype_mode,
                        lmax=self.lmax,
                        mmax=mmax,
                        tma_mmax_real=self._tma_mmax_real,
                        weight0_t_tma_real=self.w0_t_tma_real,
                        weight1_t_tma_real=self.w1_t_tma_real,
                        cache=cache,
                        device_name=device_name,
                        capability=capability,
                    )
                return out.reshape(leading_shape + (2, self.lmax, mmax))
            elif (not self._use_bf16 and not self._use_fp16) and (_can_use_cuda_vector(x, self.w0_t, self.w1_t) or _can_use_metal_vector(x, self.w0_t, self.w1_t)):
                leading_shape = x.shape[:-3]
                x_flat = x.reshape(-1, 2, self.nlat, mmax).contiguous()
                if (
                    x_flat.is_cuda
                    and not x.requires_grad
                    and _cuda_tma_available(x_flat)
                    and self._tma_mmax_complex != mmax
                ):
                    out = _direct_vector_legendre_forward(
                        _pad_last_dim(x_flat, self._tma_mmax_complex),
                        self.w0_t_tma_complex,
                        self.w1_t_tma_complex,
                    )[..., :mmax]
                    return out.reshape(leading_shape + (2, self.lmax, mmax))
                out = _FusedVectorLegendreForwardFn.apply(x_flat, self.w0_t, self.w1_t)
                return out.reshape(leading_shape + (2, self.lmax, mmax))

            leading_shape = x.shape[:-3]
            x_flat = x.reshape(-1, 2, self.nlat, mmax).contiguous()
            B = x_flat.shape[0]
            stacked_w0, stacked_w1 = _stack_vector_forward_real_inputs(x_flat)

            if self._use_fp16 and _HAS_NATIVE_EXT and x_flat.device.type == "mps" and not x.requires_grad:
                out_w0 = fused_legendre_forward_real(stacked_w0.half().contiguous(), self.w0_t)
                out_w1 = fused_legendre_forward_real(stacked_w1.half().contiguous(), self.w1_t)
            elif self._use_bf16 and _HAS_NATIVE_EXT and x_flat.is_cuda and not x.requires_grad:
                stacked_w0_bf16 = stacked_w0.bfloat16().contiguous()
                tma_pad_w0 = None
                if _cuda_tma_available(stacked_w0_bf16) and self._tma_mmax_real != mmax:
                    tma_pad_w0 = _pad_last_dim(stacked_w0_bf16, self._tma_mmax_real)
                out_w0 = _autotuned_direct_legendre_forward_real(
                    stacked_w0_bf16,
                    self.w0_t,
                    op_kind="vector-real-forward",
                    dtype_mode="bf16",
                    backend_candidates=["fma", "tma", "tc_bf16"],
                    padded_input=tma_pad_w0,
                    padded_weight_t=self.w0_t_tma_real if tma_pad_w0 is not None else None,
                )[..., :mmax]

                stacked_w1_bf16 = stacked_w1.bfloat16().contiguous()
                tma_pad_w1 = None
                if _cuda_tma_available(stacked_w1_bf16) and self._tma_mmax_real != mmax:
                    tma_pad_w1 = _pad_last_dim(stacked_w1_bf16, self._tma_mmax_real)
                out_w1 = _autotuned_direct_legendre_forward_real(
                    stacked_w1_bf16,
                    self.w1_t,
                    op_kind="vector-real-forward",
                    dtype_mode="bf16",
                    backend_candidates=["fma", "tma", "tc_bf16"],
                    padded_input=tma_pad_w1,
                    padded_weight_t=self.w1_t_tma_real if tma_pad_w1 is not None else None,
                )[..., :mmax]
            elif (not self._use_bf16 and not self._use_fp16) and _can_use_native_real_legendre(stacked_w0, self.w0_t):
                out_w0 = fused_legendre_forward_real(stacked_w0, self.w0_t)
                out_w1 = fused_legendre_forward_real(stacked_w1, self.w1_t)
            else:
                _low_prec = self._use_bf16 or self._use_fp16
                if self._use_bf16:
                    stacked_w0 = stacked_w0.bfloat16()
                    stacked_w1 = stacked_w1.bfloat16()
                elif self._use_fp16:
                    stacked_w0 = stacked_w0.half()
                    stacked_w1 = stacked_w1.half()
                out_w0 = torch.einsum("bkm,mlk->blm", stacked_w0, self.w0)
                out_w1 = torch.einsum("bkm,mlk->blm", stacked_w1, self.w1)
                if _low_prec:
                    out_w0 = out_w0.float()
                    out_w1 = out_w1.float()

            out = _recompose_vector_forward_real_outputs(
                out_w0[..., :mmax],
                out_w1[..., :mmax],
                batch_size=B,
                lmax=self.lmax,
                mmax=mmax,
            )
            return out.reshape(leading_shape + (2, self.lmax, mmax))


class InverseRealVectorSHT(nn.Module):
    """Optimised drop-in replacement for ``torch_harmonics.InverseRealVectorSHT``."""

    def __init__(
        self,
        nlat: int,
        nlon: int,
        lmax: Optional[int] = None,
        mmax: Optional[int] = None,
        grid: str = "equiangular",
        norm: str = "ortho",
        csphase: bool = True,
    ):
        super().__init__()
        from torch_harmonics import InverseRealVectorSHT
        ref = InverseRealVectorSHT(
            nlat,
            nlon,
            lmax=lmax,
            mmax=mmax,
            grid=grid,
            norm=norm,
            csphase=csphase,
        )
        self.nlat = ref.nlat
        self.nlon = ref.nlon
        self.lmax = ref.lmax
        self.mmax = ref.mmax
        self.grid = grid
        self.norm = norm
        self.csphase = csphase
        self.register_buffer("d0", ref.dpct[0].float())  # [mmax, lmax, nlat]
        self.register_buffer("d1", ref.dpct[1].float())  # [mmax, lmax, nlat]
        self.register_buffer("d0_t", ref.dpct[0].float().permute(1, 2, 0).contiguous())
        self.register_buffer("d1_t", ref.dpct[1].float().permute(1, 2, 0).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.lmax and x.shape[-1] == self.mmax

        with _nvtx_range("holysht.vector_inverse"):
            x = x.contiguous()
            if _can_use_cuda_vector(x, self.d0_t, self.d1_t) or _can_use_metal_vector(x, self.d0_t, self.d1_t):
                B_shape = x.shape[:-3]
                x_flat = x.reshape(-1, 2, self.lmax, self.mmax).contiguous()
                x_out = _FusedVectorLegendreInverseFn.apply(x_flat, self.d0_t, self.d1_t)
                x_out = x_out.reshape(B_shape + (2, self.nlat, self.mmax))
                x_out = _prepare_irfft_input(x_out, self.nlon, self.mmax)
                return torch.fft.irfft(x_out, n=self.nlon, dim=-1, norm="forward")

            x = torch.view_as_real(x)  # [..., 2, lmax, mmax, 2]
            mmax = self.mmax

            x00 = x[..., 0, :, :, 0]
            x01 = x[..., 0, :, :, 1]
            x10 = x[..., 1, :, :, 0]
            x11 = x[..., 1, :, :, 1]

            B_shape = x00.shape[:-2]
            x00_flat = x00.reshape(-1, self.lmax, mmax)
            x01_flat = x01.reshape(-1, self.lmax, mmax)
            x10_flat = x10.reshape(-1, self.lmax, mmax)
            x11_flat = x11.reshape(-1, self.lmax, mmax)
            B = x00_flat.shape[0]

            if _can_use_native_real_legendre(x00_flat, self.d0_t) and _prefer_metal_vector_inverse_kernel(x00_flat, self.d0_t):
                stacked_d0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0).contiguous()
                out_d0 = fused_legendre_inverse_real(stacked_d0, self.d0_t)
                r00, r01, r10, r11 = out_d0[:B], out_d0[B:2 * B], out_d0[2 * B:3 * B], out_d0[3 * B:]

                stacked_d1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0).contiguous()
                out_d1 = fused_legendre_inverse_real(stacked_d1, self.d1_t)
                s11, s10, s01, s00 = out_d1[:B], out_d1[B:2 * B], out_d1[2 * B:3 * B], out_d1[3 * B:]
            else:
                stacked_d0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0)
                out_d0 = torch.einsum("blm,mlk->bkm", stacked_d0, self.d0)
                r00, r01, r10, r11 = out_d0[:B], out_d0[B:2 * B], out_d0[2 * B:3 * B], out_d0[3 * B:]

                stacked_d1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0)
                out_d1 = torch.einsum("blm,mlk->bkm", stacked_d1, self.d1)
                s11, s10, s01, s00 = out_d1[:B], out_d1[B:2 * B], out_d1[2 * B:3 * B], out_d1[3 * B:]

            srl = r00 - s11
            sim = r01 + s10
            trl = -s01 - r10
            tim = s00 - r11

            out_k = self.nlat
            out_shape = B_shape + (2, out_k, mmax, 2)
            xs = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            xs[..., 0, :, :, 0] = srl.reshape(B_shape + (out_k, mmax))
            xs[..., 0, :, :, 1] = sim.reshape(B_shape + (out_k, mmax))
            xs[..., 1, :, :, 0] = trl.reshape(B_shape + (out_k, mmax))
            xs[..., 1, :, :, 1] = tim.reshape(B_shape + (out_k, mmax))
            x_out = torch.view_as_complex(xs)
            x_out = _prepare_irfft_input(x_out, self.nlon, self.mmax)
            return torch.fft.irfft(x_out, n=self.nlon, dim=-1, norm="forward")


class GraphedModule(nn.Module):
    """Wraps an nn.Module with CUDA-graph capture+replay for the forward pass.

    The first call for a given (input shape, dtype) captures a graph; subsequent
    calls with a matching key copy the input into the captured static buffer
    and replay the graph. This eliminates per-launch host overhead and is most
    valuable on small grids where a non-trivial fraction of wall-clock time is
    spent in cudaLaunchKernel rather than on the device.

    Constraints (enforced by silently passing through):
      - input must be on CUDA
      - input must not require gradients (graphs replay forward-only)

    Multiple shapes are supported via a small LRU cache; on overflow, the
    oldest captured graph is dropped (along with its static buffers).
    """

    def __init__(self, module: nn.Module, num_warmup: int = 3, max_cached_shapes: int = 4):
        super().__init__()
        self.module = module
        self._num_warmup = num_warmup
        self._max_cached_shapes = max_cached_shapes
        # Stored as plain attrs so nn.Module doesn't try to wrap them as
        # parameters/buffers/submodules.
        self._graph_cache = {}
        self._graph_lru = []

    def reset_graph_cache(self) -> None:
        self._graph_cache.clear()
        self._graph_lru.clear()

    def _capture(self, x: torch.Tensor):
        static_in = torch.empty_like(x)
        static_in.copy_(x)

        # Warmup on a side stream so allocator state matches the steady-state
        # captured graph; this is the recommended PyTorch pattern.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(self._num_warmup):
                _ = self.module(static_in)
        torch.cuda.current_stream().wait_stream(side)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_out = self.module(static_in)
        return static_in, static_out, graph

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda or x.requires_grad:
            return self.module(x)

        key = (tuple(x.shape), x.dtype)
        entry = self._graph_cache.get(key)
        if entry is None:
            entry = self._capture(x)
            self._graph_cache[key] = entry
            self._graph_lru.append(key)
            while len(self._graph_lru) > self._max_cached_shapes:
                evicted = self._graph_lru.pop(0)
                self._graph_cache.pop(evicted, None)
        else:
            # Refresh LRU position
            self._graph_lru.remove(key)
            self._graph_lru.append(key)

        static_in, static_out, graph = entry
        static_in.copy_(x)
        graph.replay()
        # Clone so the caller's tensor is decoupled from the captured static
        # output buffer (the next replay will overwrite it).
        return static_out.clone()


legendre_forward = fused_legendre_forward
legendre_inverse = fused_legendre_inverse
sht_forward = fused_sht_forward
sht_inverse = fused_sht_inverse

# Backwards-compatible aliases from the research prototype.
FusedRealSHT = RealSHT
FusedInverseRealSHT = InverseRealSHT
FusedRealVectorSHT = RealVectorSHT
FusedInverseRealVectorSHT = InverseRealVectorSHT
