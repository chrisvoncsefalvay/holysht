#!/usr/bin/env python3
"""
Benchmark HOLYSHT vs torch-harmonics: correctness + performance.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

Mirrors the workloads found in torch-harmonics notebooks:
  - getting_started.ipynb:      RealSHT / InverseRealSHT roundtrip
  - partial_derivatives.ipynb:  RealVectorSHT / InverseRealVectorSHT
  - plot_spherical_harmonics:   Synthesising individual Y_n^m modes
  - conditioning_sht.ipynb:     SHT conditioning across resolutions

Run:
    python benchmarks/bench_torch_harmonics.py [--quick]

Outputs a Markdown-friendly table to stdout and writes structured JSON to
data/bench_torch_harmonics.json.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import List, Optional

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

# ---------------------------------------------------------------------------
# torch-harmonics (reference)
# ---------------------------------------------------------------------------
from torch_harmonics import (
    RealSHT,
    InverseRealSHT,
    RealVectorSHT,
    InverseRealVectorSHT,
)

# ---------------------------------------------------------------------------
# HOLYSHT (our implementation)
# ---------------------------------------------------------------------------
from holysht import (
    RealSHT as HollyRealSHT,
    InverseRealSHT as HollyInverseRealSHT,
    RealVectorSHT as HollyRealVectorSHT,
    InverseRealVectorSHT as HollyInverseRealVectorSHT,
)

def has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if has_mps():
            return torch.device("mps")
    elif requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")
    elif requested == "mps":
        if has_mps():
            return torch.device("mps")
        raise RuntimeError("MPS was requested but is not available.")
    raise RuntimeError("No supported GPU backend is available. Expected CUDA or MPS.")


DEVICE = resolve_device(os.environ.get("HOLYSHT_DEVICE", "auto"))
GiB = 1024 ** 3
DEFAULT_MAX_ALLOC_GIB = float(os.environ.get("HOLYSHT_MAX_ALLOC_GIB", "6.0"))
MEMORY_SAFETY_FACTOR = 1.2
MAX_CASE_ALLOC_BYTES: Optional[int] = None
SKIPPED_CASES = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    test_name: str
    grid: str
    batch_size: int
    ref_ms: float
    holysht_ms: float
    speedup: float
    max_abs_err: float
    max_rel_err: float
    correct: bool


def to_device_module(module: torch.nn.Module) -> torch.nn.Module:
    if DEVICE.type == "mps":
        for name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            if buf.is_floating_point():
                module._buffers[name] = buf.float()
            elif buf.is_complex():
                module._buffers[name] = buf.to(torch.complex64)
    return module.to(DEVICE)


def synchronize_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


def empty_device_cache():
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def device_timer(fn, n_warmup=20, n_iters=100):
    """Median-of-n GPU time in milliseconds."""
    for _ in range(n_warmup):
        fn()
    synchronize_device()

    if DEVICE.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]

        for i in range(n_iters):
            starts[i].record()
            fn()
            ends[i].record()
        synchronize_device()

        times = sorted(starts[i].elapsed_time(ends[i]) for i in range(n_iters))
        return times[n_iters // 2]

    times = []
    for _ in range(n_iters):
        synchronize_device()
        start = time.perf_counter()
        fn()
        synchronize_device()
        times.append((time.perf_counter() - start) * 1000.0)
    times.sort()
    return times[n_iters // 2]  # median


def rel_err(a, b):
    """Max relative error, guarded against div-by-zero."""
    denom = b.abs().clamp(min=1e-8)
    return ((a - b).abs() / denom).max().item()


def complex_energy(x):
    """Smooth scalar loss for forward+backward benchmarking."""
    return x.real.square().mean() + x.imag.square().mean()


def format_gib(num_bytes: float) -> str:
    return f"{num_bytes / GiB:.2f} GiB"


def estimate_legendre_volume(nlat: int, nlon: int,
                             lmax: Optional[int] = None,
                             mmax: Optional[int] = None):
    lmax = nlat if lmax is None else lmax
    mmax = (nlon // 2 + 1) if mmax is None else mmax
    return nlat * lmax * mmax, lmax, mmax


def estimate_case_allocation_bytes(test_name: str, nlat: int, nlon: int, batch_size: int,
                                   lmax: Optional[int] = None,
                                   mmax: Optional[int] = None) -> int:
    vol, lmax, mmax = estimate_legendre_volume(nlat, nlon, lmax=lmax, mmax=mmax)

    weight_fp32 = vol * 4
    weight_bf16 = vol * 2
    spatial = batch_size * nlat * nlon * 4
    fft = batch_size * nlat * mmax * 8
    coeff = batch_size * lmax * mmax * 8
    spatial_vec = spatial * 2
    coeff_vec = coeff * 2

    if test_name == "RealSHT.forward":
        total = 3 * weight_fp32 + 2 * spatial + 2 * fft + 2 * coeff
    elif test_name == "InverseRealSHT.forward":
        total = 4 * weight_fp32 + spatial + 2 * coeff + 3 * fft + 2 * spatial
    elif test_name == "Roundtrip(SHT→ISHT)":
        total = 6 * weight_fp32 + 4 * spatial + 4 * coeff + 4 * fft
    elif test_name == "RealVectorSHT.forward":
        total = 6 * weight_fp32 + 2 * spatial_vec + 4 * fft + 4 * coeff_vec
    elif test_name == "InverseRealVectorSHT.forward":
        total = 6 * weight_fp32 + 4 * spatial_vec + 4 * coeff_vec + 2 * fft
    elif test_name == "Y_n^m synthesis (ISHT, sparse)":
        total = 3 * weight_fp32 + coeff + 2 * fft + 2 * spatial
    elif test_name == "RealSHT.forward (BF16)":
        total = (weight_fp32 + weight_bf16 + weight_fp32) + 2 * spatial + 2 * fft + 2 * coeff
    elif test_name == "RealVectorSHT.forward (BF16)":
        total = 2 * (weight_fp32 + weight_bf16 + weight_fp32) + 2 * spatial_vec + 4 * fft + 4 * coeff_vec
    elif test_name == "RealSHT.forward+backward":
        total = 3 * weight_fp32 + 5 * spatial + 6 * coeff + 4 * fft
    elif test_name == "RealVectorSHT.forward+backward":
        total = 6 * weight_fp32 + 10 * spatial_vec + 8 * coeff_vec + 4 * fft
    else:
        total = 4 * weight_fp32 + 4 * spatial + 4 * coeff + 4 * fft

    return int(total * MEMORY_SAFETY_FACTOR)


def should_skip_case(test_name: str, nlat: int, nlon: int, batch_size: int,
                     lmax: Optional[int] = None, mmax: Optional[int] = None) -> bool:
    if MAX_CASE_ALLOC_BYTES is None:
        return False

    est_bytes = estimate_case_allocation_bytes(
        test_name,
        nlat,
        nlon,
        batch_size,
        lmax=lmax,
        mmax=mmax,
    )
    if est_bytes <= MAX_CASE_ALLOC_BYTES:
        return False

    entry = {
        "test_name": test_name,
        "grid": f"{nlat}x{nlon}",
        "batch_size": batch_size,
        "estimated_alloc_gib": round(est_bytes / GiB, 2),
        "cap_gib": round(MAX_CASE_ALLOC_BYTES / GiB, 2),
    }
    SKIPPED_CASES.append(entry)
    print(
        f"Skipping {test_name} @ {nlat}x{nlon} B={batch_size}: "
        f"estimated peak {format_gib(est_bytes)} exceeds cap {format_gib(MAX_CASE_ALLOC_BYTES)}"
    )
    return True


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

GRIDS_FULL = [
    (64, 128),
    (128, 256),
    (256, 512),
    (512, 1024),
    (720, 1440),
]

GRIDS_QUICK = [
    (64, 128),
    (256, 512),
    (512, 1024),
]

BATCH_SIZES = [1, 4]

SCENARIO_NAMES = {
    "scalar-forward",
    "scalar-inverse",
    "roundtrip",
    "vector-forward",
    "vector-inverse",
    "ynm",
    "bf16-forward",
    "scalar-train",
    "vector-train",
}


def parse_grid_list(raw: Optional[str]) -> Optional[List[tuple[int, int]]]:
    if raw is None:
        return None
    grids = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        nlat_str, nlon_str = item.split("x", 1)
        grids.append((int(nlat_str), int(nlon_str)))
    return grids


def parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_scenarios(raw: str) -> set[str]:
    if raw.strip().lower() == "all":
        return set(SCENARIO_NAMES)
    scenarios = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = sorted(scenarios - SCENARIO_NAMES)
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return scenarios


def bench_scalar_sht(grids, batch_sizes, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """RealSHT forward — the core workload of getting_started.ipynb."""
    results = []
    for nlat, nlon in grids:
        for B in batch_sizes:
            if should_skip_case("RealSHT.forward", nlat, nlon, B):
                continue
            ref_sht = to_device_module(RealSHT(nlat, nlon))
            fused_sht = to_device_module(HollyRealSHT(nlat, nlon))
            x = torch.randn(B, nlat, nlon, device=DEVICE)

            # Correctness
            with torch.no_grad():
                ref_out = ref_sht(x)
                fused_out = fused_sht(x)
            mae = (fused_out - ref_out).abs().max().item()
            mre = rel_err(fused_out, ref_out)
            ok = mae < 1e-3

            # Performance
            ref_ms = device_timer(lambda: ref_sht(x), n_warmup, n_iters)
            holysht_ms = device_timer(lambda: fused_sht(x), n_warmup, n_iters)

            results.append(BenchResult(
                test_name="RealSHT.forward",
                grid=f"{nlat}x{nlon}",
                batch_size=B,
                ref_ms=round(ref_ms, 4),
                holysht_ms=round(holysht_ms, 4),
                speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
                max_abs_err=mae,
                max_rel_err=mre,
                correct=ok,
            ))

            del ref_sht, fused_sht, x, ref_out, fused_out
            empty_device_cache()
    return results


def bench_scalar_isht(grids, batch_sizes, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """InverseRealSHT — the reconstruction step."""
    results = []
    for nlat, nlon in grids:
        for B in batch_sizes:
            if should_skip_case("InverseRealSHT.forward", nlat, nlon, B):
                continue
            ref_sht = to_device_module(RealSHT(nlat, nlon))
            ref_isht = to_device_module(InverseRealSHT(nlat, nlon))
            fused_isht = to_device_module(HollyInverseRealSHT(nlat, nlon))

            x = torch.randn(B, nlat, nlon, device=DEVICE)
            with torch.no_grad():
                coeffs = ref_sht(x)
                ref_out = ref_isht(coeffs)
                fused_out = fused_isht(coeffs)
            mae = (fused_out - ref_out).abs().max().item()
            mre = rel_err(fused_out, ref_out)
            ok = mae < 1e-2

            ref_ms = device_timer(lambda: ref_isht(coeffs), n_warmup, n_iters)
            holysht_ms = device_timer(lambda: fused_isht(coeffs), n_warmup, n_iters)

            results.append(BenchResult(
                test_name="InverseRealSHT.forward",
                grid=f"{nlat}x{nlon}",
                batch_size=B,
                ref_ms=round(ref_ms, 4),
                holysht_ms=round(holysht_ms, 4),
                speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
                max_abs_err=mae,
                max_rel_err=mre,
                correct=ok,
            ))

            del ref_sht, ref_isht, fused_isht, x, coeffs, ref_out, fused_out
            empty_device_cache()
    return results


def bench_roundtrip(grids, batch_sizes, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """Full SHT roundtrip: forward → inverse. Tests numerical stability."""
    results = []
    for nlat, nlon in grids:
        for B in batch_sizes:
            if should_skip_case("Roundtrip(SHT→ISHT)", nlat, nlon, B):
                continue
            ref_sht = to_device_module(RealSHT(nlat, nlon))
            ref_isht = to_device_module(InverseRealSHT(nlat, nlon))
            fused_sht = to_device_module(HollyRealSHT(nlat, nlon))
            fused_isht = to_device_module(HollyInverseRealSHT(nlat, nlon))

            x = torch.randn(B, nlat, nlon, device=DEVICE)

            with torch.no_grad():
                ref_rt = ref_isht(ref_sht(x))
                fused_rt = fused_isht(fused_sht(x))

            # Compare roundtrip vs original (both should be close to x)
            ref_rterr = (ref_rt - x).abs().max().item()
            fused_rterr = (fused_rt - x).abs().max().item()
            # Compare the two roundtrips against each other
            mae = (fused_rt - ref_rt).abs().max().item()
            mre = rel_err(fused_rt, ref_rt)
            ok = mae < 1e-2

            def ref_fn():
                ref_isht(ref_sht(x))

            def fused_fn():
                fused_isht(fused_sht(x))

            ref_ms = device_timer(ref_fn, n_warmup, n_iters)
            holysht_ms = device_timer(fused_fn, n_warmup, n_iters)

            results.append(BenchResult(
                test_name="Roundtrip(SHT→ISHT)",
                grid=f"{nlat}x{nlon}",
                batch_size=B,
                ref_ms=round(ref_ms, 4),
                holysht_ms=round(holysht_ms, 4),
                speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
                max_abs_err=mae,
                max_rel_err=mre,
                correct=ok,
            ))

            del ref_sht, ref_isht, fused_sht, fused_isht, x
            empty_device_cache()
    return results


def bench_vector_sht(grids, batch_sizes, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """RealVectorSHT forward — the partial_derivatives.ipynb workload."""
    results = []
    for nlat, nlon in grids:
        for B in batch_sizes:
            if should_skip_case("RealVectorSHT.forward", nlat, nlon, B):
                continue
            ref_vsht = to_device_module(RealVectorSHT(nlat, nlon))
            fused_vsht = to_device_module(HollyRealVectorSHT(nlat, nlon))
            x = torch.randn(B, 2, nlat, nlon, device=DEVICE)

            with torch.no_grad():
                ref_out = ref_vsht(x)
                fused_out = fused_vsht(x)
            mae = (fused_out - ref_out).abs().max().item()
            mre = rel_err(fused_out, ref_out)
            ok = mae < 1e-3

            ref_ms = device_timer(lambda: ref_vsht(x), n_warmup, n_iters)
            holysht_ms = device_timer(lambda: fused_vsht(x), n_warmup, n_iters)

            results.append(BenchResult(
                test_name="RealVectorSHT.forward",
                grid=f"{nlat}x{nlon}",
                batch_size=B,
                ref_ms=round(ref_ms, 4),
                holysht_ms=round(holysht_ms, 4),
                speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
                max_abs_err=mae,
                max_rel_err=mre,
                correct=ok,
            ))

            del ref_vsht, fused_vsht, x, ref_out, fused_out
            empty_device_cache()
    return results


def bench_vector_isht(grids, batch_sizes, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """InverseRealVectorSHT — vector field reconstruction."""
    results = []
    for nlat, nlon in grids:
        for B in batch_sizes:
            if should_skip_case("InverseRealVectorSHT.forward", nlat, nlon, B):
                continue
            ref_vsht = to_device_module(RealVectorSHT(nlat, nlon))
            ref_ivsht = to_device_module(InverseRealVectorSHT(nlat, nlon))
            fused_ivsht = to_device_module(HollyInverseRealVectorSHT(nlat, nlon))

            x = torch.randn(B, 2, nlat, nlon, device=DEVICE)
            with torch.no_grad():
                coeffs = ref_vsht(x)
                ref_out = ref_ivsht(coeffs)
                fused_out = fused_ivsht(coeffs)
            mae = (fused_out - ref_out).abs().max().item()
            mre = rel_err(fused_out, ref_out)
            ok = mae < 1e-2

            ref_ms = device_timer(lambda: ref_ivsht(coeffs), n_warmup, n_iters)
            holysht_ms = device_timer(lambda: fused_ivsht(coeffs), n_warmup, n_iters)

            results.append(BenchResult(
                test_name="InverseRealVectorSHT.forward",
                grid=f"{nlat}x{nlon}",
                batch_size=B,
                ref_ms=round(ref_ms, 4),
                holysht_ms=round(holysht_ms, 4),
                speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
                max_abs_err=mae,
                max_rel_err=mre,
                correct=ok,
            ))

            del ref_vsht, ref_ivsht, fused_ivsht, x, coeffs, ref_out, fused_out
            empty_device_cache()
    return results


def bench_ynm_synthesis(lmax=64, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """Synthesise individual Y_n^m basis functions — plot_spherical_harmonics workload.

    Creates spectral coefficients with a single non-zero mode, then inverts
    to spatial domain. Tests that HOLYSHT reproduces the same patterns as
    torch-harmonics for sparse coefficient tensors.
    """
    results = []
    nlat, nlon = 2 * lmax, 4 * lmax
    if should_skip_case("Y_n^m synthesis (ISHT, sparse)", nlat, nlon, 1, lmax=lmax, mmax=lmax):
        return results

    ref_isht = to_device_module(InverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax))
    fused_isht = to_device_module(HollyInverseRealSHT(nlat, nlon, lmax=lmax, mmax=lmax))

    # Test a selection of (n, m) modes
    test_modes = [(0, 0), (1, 0), (1, 1), (4, 2), (10, 5), (lmax-1, 0), (lmax-1, lmax-1)]
    max_err_overall = 0.0

    for n, m in test_modes:
        coeffs = torch.zeros(1, lmax, lmax, device=DEVICE, dtype=torch.complex64)
        coeffs[0, n, m] = 1.0 + 0.0j

        with torch.no_grad():
            ref_out = ref_isht(coeffs)
            fused_out = fused_isht(coeffs)
        err = (fused_out - ref_out).abs().max().item()
        max_err_overall = max(max_err_overall, err)

    ok = max_err_overall < 1e-3

    ref_ms = device_timer(lambda: ref_isht(coeffs), n_warmup, n_iters)
    holysht_ms = device_timer(lambda: fused_isht(coeffs), n_warmup, n_iters)

    results.append(BenchResult(
        test_name="Y_n^m synthesis (ISHT, sparse)",
        grid=f"{nlat}x{nlon} (lmax={lmax})",
        batch_size=1,
        ref_ms=round(ref_ms, 4),
        holysht_ms=round(holysht_ms, 4),
        speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
        max_abs_err=max_err_overall,
        max_rel_err=0.0,  # not meaningful for sparse coefficients
        correct=ok,
    ))

    del ref_isht, fused_isht
    empty_device_cache()
    return results


def bench_bf16_sht(grids, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """BF16 tensor-core path — tests additional speedup from lower precision."""
    if DEVICE.type != "cuda":
        print("Skipping BF16 benchmark on non-CUDA backend.")
        return []
    results = []
    B = 4
    for nlat, nlon in grids:
        if should_skip_case("RealSHT.forward (BF16)", nlat, nlon, B):
            continue
        ref_sht = to_device_module(RealSHT(nlat, nlon))
        fused_bf16 = to_device_module(HollyRealSHT(nlat, nlon, dtype="bf16"))

        x = torch.randn(B, nlat, nlon, device=DEVICE)
        with torch.no_grad():
            ref_out = ref_sht(x)
            bf16_out = fused_bf16(x)
        mae = (bf16_out - ref_out).abs().max().item()
        mre = rel_err(bf16_out, ref_out)
        # BF16 has lower precision — use absolute error threshold
        # (relative error is misleading for near-zero spectral coefficients)
        ok = mae < 5e-3

        ref_ms = device_timer(lambda: ref_sht(x), n_warmup, n_iters)
        holysht_ms = device_timer(lambda: fused_bf16(x), n_warmup, n_iters)

        results.append(BenchResult(
            test_name="RealSHT.forward (BF16)",
            grid=f"{nlat}x{nlon}",
            batch_size=B,
            ref_ms=round(ref_ms, 4),
            holysht_ms=round(holysht_ms, 4),
            speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
            max_abs_err=mae,
            max_rel_err=mre,
            correct=ok,
        ))

        del ref_sht, fused_bf16, x, ref_out, bf16_out
        empty_device_cache()
    return results


def bench_bf16_vector_sht(grids, n_warmup=20, n_iters=100) -> List[BenchResult]:
    """BF16 vector path on top of the custom CUDA real reductions."""
    if DEVICE.type != "cuda":
        print("Skipping BF16 vector benchmark on non-CUDA backend.")
        return []
    results = []
    B = 4
    for nlat, nlon in grids:
        if should_skip_case("RealVectorSHT.forward (BF16)", nlat, nlon, B):
            continue
        ref_vsht = to_device_module(RealVectorSHT(nlat, nlon))
        fused_bf16 = to_device_module(HollyRealVectorSHT(nlat, nlon, dtype="bf16"))

        x = torch.randn(B, 2, nlat, nlon, device=DEVICE)
        with torch.no_grad():
            ref_out = ref_vsht(x)
            bf16_out = fused_bf16(x)
        mae = (bf16_out - ref_out).abs().max().item()
        mre = rel_err(bf16_out, ref_out)
        ok = mae < 5e-3

        ref_ms = device_timer(lambda: ref_vsht(x), n_warmup, n_iters)
        holysht_ms = device_timer(lambda: fused_bf16(x), n_warmup, n_iters)

        results.append(BenchResult(
            test_name="RealVectorSHT.forward (BF16)",
            grid=f"{nlat}x{nlon}",
            batch_size=B,
            ref_ms=round(ref_ms, 4),
            holysht_ms=round(holysht_ms, 4),
            speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
            max_abs_err=mae,
            max_rel_err=mre,
            correct=ok,
        ))

        del ref_vsht, fused_bf16, x, ref_out, bf16_out
        empty_device_cache()
    return results


def bench_scalar_train(grids, batch_sizes, n_warmup=10, n_iters=50) -> List[BenchResult]:
    """Benchmark forward+backward scalar SHT to validate training realism."""
    results = []
    for nlat, nlon in grids:
        for B in batch_sizes:
            if should_skip_case("RealSHT.forward+backward", nlat, nlon, B):
                continue
            ref_sht = to_device_module(RealSHT(nlat, nlon))
            fused_sht = to_device_module(HollyRealSHT(nlat, nlon))

            x_base = torch.randn(B, nlat, nlon, device=DEVICE)
            x_ref = x_base.clone().requires_grad_(True)
            x_fused = x_base.clone().requires_grad_(True)

            complex_energy(ref_sht(x_ref)).backward()
            ref_grad = x_ref.grad.detach().clone()
            complex_energy(fused_sht(x_fused)).backward()
            fused_grad = x_fused.grad.detach().clone()

            mae = (fused_grad - ref_grad).abs().max().item()
            mre = rel_err(fused_grad, ref_grad)
            ok = mae < 1e-3

            def ref_fn():
                x_ref.grad = None
                complex_energy(ref_sht(x_ref)).backward()

            def holysht_fn():
                x_fused.grad = None
                complex_energy(fused_sht(x_fused)).backward()

            ref_ms = device_timer(ref_fn, n_warmup, n_iters)
            holysht_ms = device_timer(holysht_fn, n_warmup, n_iters)

            results.append(BenchResult(
                test_name="RealSHT.forward+backward",
                grid=f"{nlat}x{nlon}",
                batch_size=B,
                ref_ms=round(ref_ms, 4),
                holysht_ms=round(holysht_ms, 4),
                speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
                max_abs_err=mae,
                max_rel_err=mre,
                correct=ok,
            ))

            del ref_sht, fused_sht, x_base, x_ref, x_fused, ref_grad, fused_grad
            empty_device_cache()
    return results


def bench_vector_train(grids, batch_sizes, n_warmup=10, n_iters=50) -> List[BenchResult]:
    """Benchmark forward+backward vector SHT to validate training realism."""
    results = []
    for nlat, nlon in grids:
        for B in batch_sizes:
            if should_skip_case("RealVectorSHT.forward+backward", nlat, nlon, B):
                continue
            ref_vsht = to_device_module(RealVectorSHT(nlat, nlon))
            fused_vsht = to_device_module(HollyRealVectorSHT(nlat, nlon))

            x_base = torch.randn(B, 2, nlat, nlon, device=DEVICE)
            x_ref = x_base.clone().requires_grad_(True)
            x_fused = x_base.clone().requires_grad_(True)

            complex_energy(ref_vsht(x_ref)).backward()
            ref_grad = x_ref.grad.detach().clone()
            complex_energy(fused_vsht(x_fused)).backward()
            fused_grad = x_fused.grad.detach().clone()

            mae = (fused_grad - ref_grad).abs().max().item()
            mre = rel_err(fused_grad, ref_grad)
            ok = mae < 1e-3

            def ref_fn():
                x_ref.grad = None
                complex_energy(ref_vsht(x_ref)).backward()

            def holysht_fn():
                x_fused.grad = None
                complex_energy(fused_vsht(x_fused)).backward()

            ref_ms = device_timer(ref_fn, n_warmup, n_iters)
            holysht_ms = device_timer(holysht_fn, n_warmup, n_iters)

            results.append(BenchResult(
                test_name="RealVectorSHT.forward+backward",
                grid=f"{nlat}x{nlon}",
                batch_size=B,
                ref_ms=round(ref_ms, 4),
                holysht_ms=round(holysht_ms, 4),
                speedup=round(ref_ms / holysht_ms, 2) if holysht_ms > 0 else 0,
                max_abs_err=mae,
                max_rel_err=mre,
                correct=ok,
            ))

            del ref_vsht, fused_vsht, x_base, x_ref, x_fused, ref_grad, fused_grad
            empty_device_cache()
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(results: List[BenchResult]):
    """Print a Markdown table of results."""
    header = "| Test | Grid | B | TH (ms) | HOLYSHT (ms) | Speedup | MaxAbsErr | Correct |"
    sep    = "|------|------|---|---------|------------|---------|-----------|---------|"
    print(header)
    print(sep)
    for r in results:
        status = "PASS" if r.correct else "**FAIL**"
        print(f"| {r.test_name} | {r.grid} | {r.batch_size} "
              f"| {r.ref_ms:.3f} | {r.holysht_ms:.3f} "
              f"| {r.speedup:.1f}x | {r.max_abs_err:.2e} | {status} |")


def print_summary(results: List[BenchResult]):
    """Print overall summary stats."""
    n_pass = sum(1 for r in results if r.correct)
    n_total = len(results)
    speedups = [r.speedup for r in results if r.speedup > 0]
    mean_speedup = sum(speedups) / len(speedups) if speedups else 0

    print(f"\n## Summary")
    print(f"- **{n_pass}/{n_total}** tests passed")
    print(f"- Mean speedup: **{mean_speedup:.1f}x**")
    print(f"- Min speedup: **{min(speedups):.1f}x**" if speedups else "")
    print(f"- Max speedup: **{max(speedups):.1f}x**" if speedups else "")

    if n_pass < n_total:
        print(f"\n### Failures:")
        for r in results:
            if not r.correct:
                print(f"  - {r.test_name} @ {r.grid} B={r.batch_size}: "
                      f"abs_err={r.max_abs_err:.2e}, rel_err={r.max_rel_err:.2e}")

    if SKIPPED_CASES:
        print(f"\n### Skipped For Memory Budget")
        for case in SKIPPED_CASES:
            print(
                f"  - {case['test_name']} @ {case['grid']} B={case['batch_size']}: "
                f"est={case['estimated_alloc_gib']:.2f} GiB, cap={case['cap_gib']:.2f} GiB"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="HOLYSHT vs torch-harmonics benchmark")
    parser.add_argument("--quick", action="store_true",
                        help="Run a smaller subset of grids/iterations")
    parser.add_argument(
        "--device",
        default=os.environ.get("HOLYSHT_DEVICE", "auto"),
        choices=["auto", "cuda", "mps"],
        help="Execution backend to benchmark.",
    )
    parser.add_argument("--output", default=None,
                        help="JSON output path (default: data/bench_torch_harmonics.json)")
    parser.add_argument(
        "--scenarios",
        default="all",
        help=(
            "Comma-separated scenarios to run. "
            "Choices: scalar-forward, scalar-inverse, roundtrip, vector-forward, "
            "vector-inverse, ynm, bf16-forward, scalar-train, vector-train, or all."
        ),
    )
    parser.add_argument(
        "--grids",
        default=None,
        help="Comma-separated grids such as 256x512,512x1024. Applies to grid-based scenarios.",
    )
    parser.add_argument(
        "--batch-sizes",
        default=None,
        help="Comma-separated batch sizes such as 1,4.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override the number of warmup iterations.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override the number of timed iterations.",
    )
    parser.add_argument(
        "--max-alloc-gib",
        type=float,
        default=DEFAULT_MAX_ALLOC_GIB,
        help=(
            "Soft per-case allocation cap in GiB. "
            "Cases whose estimated peak allocation exceeds this are skipped. "
            "Use 0 to disable."
        ),
    )
    args = parser.parse_args()

    global DEVICE
    DEVICE = resolve_device(args.device)

    global MAX_CASE_ALLOC_BYTES
    MAX_CASE_ALLOC_BYTES = None if args.max_alloc_gib <= 0 else int(args.max_alloc_gib * GiB)
    SKIPPED_CASES.clear()

    try:
        selected_scenarios = parse_scenarios(args.scenarios)
    except ValueError as exc:
        parser.error(str(exc))

    grids = parse_grid_list(args.grids) or (GRIDS_QUICK if args.quick else GRIDS_FULL)
    batch_sizes = parse_int_list(args.batch_sizes) or ([4] if args.quick else [1, 4])
    n_warmup = args.warmup if args.warmup is not None else (10 if args.quick else 20)
    n_iters = args.iters if args.iters is not None else (50 if args.quick else 100)

    # GPU info
    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_mem_gib = torch.cuda.get_device_properties(0).total_memory / GiB
        backend_version = torch.version.cuda
    else:
        gpu_name = "Apple Metal (MPS)"
        total_mem = getattr(torch.mps, "recommended_max_memory", lambda: 0)()
        total_mem_gib = total_mem / GiB if total_mem else 0.0
        backend_version = None
    print(f"# HOLYSHT vs torch-harmonics Benchmark")
    print(f"Device: {gpu_name}")
    if total_mem_gib > 0:
        print(f"Visible GPU memory: {total_mem_gib:.2f} GiB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Backend: {DEVICE.type}")
    if backend_version:
        print(f"CUDA: {backend_version}")
    print(f"Grids: {grids}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Scenarios: {sorted(selected_scenarios)}")
    print(f"Iterations: {n_iters} (warmup: {n_warmup})")
    if MAX_CASE_ALLOC_BYTES is None:
        print("Allocation cap: disabled")
    else:
        print(
            f"Allocation cap: {args.max_alloc_gib:.2f} GiB "
            f"(safety factor {MEMORY_SAFETY_FACTOR:.1f}x)"
        )
    print()

    all_results: List[BenchResult] = []

    if "scalar-forward" in selected_scenarios:
        print("## Scalar SHT (forward)")
        r = bench_scalar_sht(grids, batch_sizes, n_warmup, n_iters)
        print_table(r)
        all_results.extend(r)
        print()

    if "scalar-inverse" in selected_scenarios:
        print("## Scalar SHT (inverse)")
        r = bench_scalar_isht(grids, batch_sizes, n_warmup, n_iters)
        print_table(r)
        all_results.extend(r)
        print()

    if "roundtrip" in selected_scenarios:
        print("## Roundtrip (forward + inverse)")
        r = bench_roundtrip(grids, batch_sizes, n_warmup, n_iters)
        print_table(r)
        all_results.extend(r)
        print()

    if "vector-forward" in selected_scenarios:
        print("## Vector SHT (forward)")
        r = bench_vector_sht(grids, batch_sizes, n_warmup, n_iters)
        print_table(r)
        all_results.extend(r)
        print()

    if "vector-inverse" in selected_scenarios:
        print("## Vector SHT (inverse)")
        r = bench_vector_isht(grids, batch_sizes, n_warmup, n_iters)
        print_table(r)
        all_results.extend(r)
        print()

    if "ynm" in selected_scenarios:
        print("## Y_n^m basis synthesis")
        r = bench_ynm_synthesis(lmax=64, n_warmup=n_warmup, n_iters=n_iters)
        print_table(r)
        all_results.extend(r)
        print()

    if "bf16-forward" in selected_scenarios:
        print("## BF16 tensor core path")
        bf16_grids = grids if args.grids is not None else ([(512, 1024)] if args.quick else [(512, 1024), (720, 1440)])
        r = bench_bf16_sht(bf16_grids, n_warmup, n_iters)
        rv = bench_bf16_vector_sht(bf16_grids, n_warmup, n_iters)
        print_table(r + rv)
        all_results.extend(r)
        all_results.extend(rv)
        print()

    train_warmup = max(5, n_warmup // 2)
    train_iters = max(20, n_iters // 2)
    train_grids = grids if args.grids is not None else ([(256, 512)] if args.quick else [(256, 512), (512, 1024), (720, 1440)])
    train_batch_sizes = batch_sizes if args.batch_sizes is not None else [4]

    if "scalar-train" in selected_scenarios:
        print("## Scalar SHT training (forward + backward)")
        r = bench_scalar_train(train_grids, train_batch_sizes, train_warmup, train_iters)
        print_table(r)
        all_results.extend(r)
        print()

    if "vector-train" in selected_scenarios:
        print("## Vector SHT training (forward + backward)")
        r = bench_vector_train(train_grids, train_batch_sizes, train_warmup, train_iters)
        print_table(r)
        all_results.extend(r)
        print()

    # --- Summary ---
    print_summary(all_results)

    # --- Save JSON ---
    out_path = args.output or os.path.join(
        os.path.dirname(__file__), "..", "data", "bench_torch_harmonics.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "gpu": gpu_name,
            "backend": DEVICE.type,
            "visible_gpu_memory_gib": round(total_mem_gib, 2),
            "pytorch": torch.__version__,
            "cuda": backend_version,
            "max_alloc_gib": args.max_alloc_gib,
            "memory_safety_factor": MEMORY_SAFETY_FACTOR,
            "skipped_cases": SKIPPED_CASES,
            "results": [asdict(r) for r in all_results],
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Exit code
    if all(r.correct for r in all_results):
        print("\nAll correctness checks PASSED.")
    else:
        print("\nSOME CORRECTNESS CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
