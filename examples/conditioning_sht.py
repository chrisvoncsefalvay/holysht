#!/usr/bin/env python3
"""HOLYSHT reimplementation of torch-harmonics conditioning_sht.ipynb.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

Workload:
  1. Build Vandermonde matrix via InverseRealSHT on sparse unit basis vectors
  2. Compute Gramian matrix from Vandermonde + quadrature weights
  3. Compute condition number of the Gramian
  4. Test SHT roundtrip on a delta signal

This exercises batched InverseRealSHT with many sparse inputs plus
forward SHT → InverseRealSHT roundtrip accuracy.
"""

import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

from torch_harmonics import RealSHT as RefRealSHT, InverseRealSHT as RefInverseRealSHT
from torch_harmonics.quadrature import legendre_gauss_weights
from holysht import RealSHT as HolySHT, InverseRealSHT as HolyInverseSHT

DEVICE = torch.device("cuda")


@contextmanager
def cuda_timer(label):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    cuda_timer.last = elapsed
    print(f"  {label}: {elapsed:.2f} ms")


cuda_timer.last = 0.0


def run_conditioning(label, sht_cls, isht_cls, nlat, nlon, grid="legendre-gauss"):
    print(f"\n--- {label} (grid={nlat}x{nlon}, {grid}) ---")

    if grid == "legendre-gauss":
        lmax = mmax = nlat
    else:
        lmax = mmax = nlat // 2 - 1

    xq, wq = legendre_gauss_weights(nlat)
    if isinstance(wq, np.ndarray):
        wq = torch.from_numpy(wq)
    omega = torch.pi * wq.float() / nlat
    omega = omega.reshape(-1, 1).to(DEVICE)

    sht = sht_cls(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(DEVICE)
    isht = isht_cls(nlat, nlon, lmax=lmax, mmax=mmax, grid=grid).to(DEVICE)

    # Build Vandermonde matrix
    nmodes = int(lmax * (lmax + 1) / 2)
    e = torch.zeros(nmodes, lmax, mmax, dtype=torch.complex64, device=DEVICE)
    midx = lambda l, m: l * (l + 1) // 2 + m
    for l in range(lmax):
        for m in range(l + 1):
            e[midx(l, m), l, m] = 1.0

    timings = {}

    # Warmup
    with torch.no_grad():
        _ = isht(e)
    torch.cuda.synchronize()

    with cuda_timer(f"Vandermonde ({nmodes} modes)"):
        with torch.no_grad():
            vdm = isht(e)
    timings["vandermonde"] = cuda_timer.last

    # Gramian
    with cuda_timer("Gramian computation"):
        gramian = torch.einsum("iqr,jqr,qr->ij", vdm, vdm, omega)
    timings["gramian"] = cuda_timer.last

    cond = np.linalg.cond(gramian.cpu().numpy())
    print(f"  Condition number: {cond:.4e}")
    print(f"  Gramian diag range: [{gramian.diag().min().item():.4e}, {gramian.diag().max().item():.4e}]")

    # Roundtrip test: delta signal
    field = torch.zeros(1, nlat, nlon, device=DEVICE)
    field[0, nlat // 2, 3] = 1.0

    with cuda_timer("Delta roundtrip (SHT → ISHT)"):
        with torch.no_grad():
            rt = isht(sht(field))
    timings["roundtrip"] = cuda_timer.last

    return vdm, gramian, cond, timings


def main():
    print("=" * 60)
    print("conditioning_sht: Vandermonde, Gramian & conditioning")
    print("=" * 60)

    configs = [
        (32, 64),
        (64, 128),
        (128, 256),
    ]

    for nlat, nlon in configs:
        print(f"\n{'='*60}")
        print(f"Grid: {nlat}x{nlon}")
        print(f"{'='*60}")

        ref_vdm, ref_gram, ref_cond, ref_t = run_conditioning(
            "torch-harmonics", RefRealSHT, RefInverseRealSHT, nlat, nlon
        )
        opt_vdm, opt_gram, opt_cond, opt_t = run_conditioning(
            "HOLYSHT", HolySHT, HolyInverseSHT, nlat, nlon
        )

        # Numerical comparison
        vdm_diff = (opt_vdm - ref_vdm).abs().max().item()
        gram_diff = (opt_gram - ref_gram).abs().max().item()
        cond_rel = abs(opt_cond - ref_cond) / ref_cond if ref_cond > 0 else 0
        print(f"\n  Vandermonde max diff:      {vdm_diff:.2e}")
        print(f"  Gramian max diff:          {gram_diff:.2e}")
        print(f"  Condition number rel diff: {cond_rel:.2e}")

        print(f"\n  {'Workload':<25} {'TH (ms)':>10} {'HOLYSHT (ms)':>13} {'Speedup':>8}")
        print(f"  {'-'*56}")
        for key in ["vandermonde", "gramian", "roundtrip"]:
            sp = ref_t[key] / opt_t[key] if opt_t[key] > 0 else 0
            print(f"  {key:<25} {ref_t[key]:>10.2f} {opt_t[key]:>13.2f} {sp:>7.1f}x")


if __name__ == "__main__":
    main()
