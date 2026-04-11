#!/usr/bin/env python3
"""HOLYSHT reimplementation of torch-harmonics plot_spherical_harmonics.ipynb.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

Workload:
  Synthesise individual Y_l^m basis functions by constructing sparse spectral
  coefficient tensors and inverting via InverseRealSHT. This exercises the
  inverse Legendre path with highly sparse inputs.
"""

import os
import sys
import time
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

from torch_harmonics import InverseRealSHT as RefInverseRealSHT
from holysht import InverseRealSHT as HolyInverseSHT

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


def midx(l, m):
    return l * (l + 1) + m


def run_synthesis(label, isht_cls, nlat, nlon, lmax, plt_lmax):
    print(f"\n--- {label} (nlat={nlat}, nlon={nlon}, lmax={lmax}) ---")

    isht = isht_cls(nlat, nlon, lmax=lmax, mmax=lmax).to(DEVICE)

    # Build the full Vandermonde: one sparse coefficient tensor per (l, m) mode
    nmodes = int(lmax * lmax)
    e = torch.zeros(nmodes, lmax, lmax, dtype=torch.complex64, device=DEVICE)
    for l in range(lmax):
        for m in range(-l, l + 1):
            e[midx(l, m), l, abs(m)] = 1.0 if m >= 0 else 1.0j

    # Warmup
    with torch.no_grad():
        _ = isht(e)
    torch.cuda.synchronize()

    with cuda_timer(f"Vandermonde synthesis ({nmodes} modes)"):
        with torch.no_grad():
            vdm = isht(e)

    # Also time single-mode synthesis (the common use case)
    single_e = torch.zeros(1, lmax, lmax, dtype=torch.complex64, device=DEVICE)
    single_e[0, 3, 2] = 1.0
    with torch.no_grad():
        _ = isht(single_e)
    torch.cuda.synchronize()

    with cuda_timer("Single mode synthesis (l=3, m=2)"):
        with torch.no_grad():
            single_mode = isht(single_e)

    return vdm, cuda_timer.last


def main():
    print("=" * 60)
    print("plot_spherical_harmonics: Y_l^m basis synthesis")
    print("=" * 60)

    configs = [
        (60, 120, 60, 3),
        (128, 256, 128, 6),
    ]

    for nlat, nlon, lmax, plt_lmax in configs:
        print(f"\n{'='*60}")
        print(f"Grid: {nlat}x{nlon}, lmax={lmax}")
        print(f"{'='*60}")

        ref_vdm, _ = run_synthesis(
            "torch-harmonics", RefInverseRealSHT, nlat, nlon, lmax, plt_lmax
        )
        opt_vdm, _ = run_synthesis(
            "HOLYSHT", HolyInverseSHT, nlat, nlon, lmax, plt_lmax
        )

        # Numerical comparison
        diff = (opt_vdm - ref_vdm).abs().max().item()
        rel_diff = ((opt_vdm - ref_vdm).abs() / ref_vdm.abs().clamp(min=1e-8)).max().item()
        print(f"\n  Vandermonde max abs diff: {diff:.2e}")
        print(f"  Vandermonde max rel diff: {rel_diff:.2e}")


if __name__ == "__main__":
    main()
