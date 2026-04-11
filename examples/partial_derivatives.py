#!/usr/bin/env python3
"""HOLYSHT reimplementation of torch-harmonics partial_derivatives.ipynb.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

Workload:
  Compute the gradient of f(theta, phi) = sin(theta)*cos(phi) on the sphere
  via SHT → InverseVectorSHT, then compare against the analytic derivatives.
"""

import os
import sys
import time
from contextlib import contextmanager

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

from torch_harmonics import (
    RealSHT as RefRealSHT,
    InverseRealVectorSHT as RefInverseRealVectorSHT,
)
from holysht import (
    RealSHT as HolySHT,
    InverseRealVectorSHT as HolyInverseRealVectorSHT,
)

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


def f(theta, phi):
    return torch.sin(theta) * torch.cos(phi)


def df_dtheta(theta, phi):
    return torch.cos(theta) * torch.cos(phi)


def df_dphi(theta, phi):
    return -torch.sin(theta) * torch.sin(phi)


def run_gradient(label, sht_cls, ivsht_cls, n_theta, n_phi, lmax):
    print(f"\n--- {label} (grid={n_theta}x{n_phi}, lmax={lmax}) ---")

    theta = torch.linspace(0, torch.pi, n_theta, device=DEVICE)
    phi = torch.linspace(0, 2 * torch.pi, n_phi, device=DEVICE)
    T, P = torch.meshgrid(theta, phi, indexing="ij")

    f_grid = f(T, P).unsqueeze(0)  # [1, n_theta, n_phi]
    nft_grid = df_dtheta(T, P)
    nfp_grid = df_dphi(T, P)

    sht = sht_cls(n_theta, n_phi, lmax=lmax).to(DEVICE)
    ivsht = ivsht_cls(n_theta, n_phi, lmax=lmax).to(DEVICE)

    # Warmup
    with torch.no_grad():
        sh_coeffs = sht(f_grid)
        vector_coeffs = torch.zeros(1, 2, sh_coeffs.shape[-2], sh_coeffs.shape[-1],
                                    dtype=sh_coeffs.dtype, device=DEVICE)
        vector_coeffs[:, 0] = sh_coeffs
        _ = ivsht(vector_coeffs)
    torch.cuda.synchronize()

    timings = {}

    with cuda_timer("Forward SHT"):
        with torch.no_grad():
            sh_coeffs = sht(f_grid)
    timings["sht"] = cuda_timer.last

    vector_coeffs = torch.zeros(1, 2, sh_coeffs.shape[-2], sh_coeffs.shape[-1],
                                dtype=sh_coeffs.dtype, device=DEVICE)
    vector_coeffs[:, 0] = sh_coeffs

    with cuda_timer("Inverse Vector SHT"):
        with torch.no_grad():
            nabla_f = ivsht(vector_coeffs)
    timings["ivsht"] = cuda_timer.last

    # IVSHT gives 1/sin(theta) * d/dphi, so multiply by sin(theta)
    nabla_f = nabla_f.squeeze(0)  # [2, n_theta, n_phi]
    nabla_f[1] = nabla_f[1] * torch.sin(T)

    err_theta = (nabla_f[0] - nft_grid).abs().max().item()
    err_phi = (nabla_f[1] - nfp_grid).abs().max().item()
    print(f"  d/dtheta max error: {err_theta:.2e}")
    print(f"  d/dphi max error:   {err_phi:.2e}")

    return nabla_f, timings, err_theta, err_phi


def main():
    print("=" * 60)
    print("partial_derivatives: Gradient via SHT + InverseVectorSHT")
    print("=" * 60)

    configs = [
        (128, 256, 5),
        (256, 512, 10),
        (512, 1024, 20),
    ]

    for n_theta, n_phi, lmax in configs:
        print(f"\n{'='*60}")
        print(f"Grid: {n_theta}x{n_phi}, lmax={lmax}")
        print(f"{'='*60}")

        ref_nabla, ref_t, ref_et, ref_ep = run_gradient(
            "torch-harmonics", RefRealSHT, RefInverseRealVectorSHT, n_theta, n_phi, lmax
        )
        opt_nabla, opt_t, opt_et, opt_ep = run_gradient(
            "HOLYSHT", HolySHT, HolyInverseRealVectorSHT, n_theta, n_phi, lmax
        )

        # Numerical agreement
        diff = (opt_nabla - ref_nabla).abs().max().item()
        print(f"\n  Gradient max diff (HOLYSHT vs TH): {diff:.2e}")

        print(f"\n  {'Workload':<25} {'TH (ms)':>10} {'HOLYSHT (ms)':>13} {'Speedup':>8}")
        print(f"  {'-'*56}")
        for key in ["sht", "ivsht"]:
            sp = ref_t[key] / opt_t[key] if opt_t[key] > 0 else 0
            print(f"  {key.upper():<25} {ref_t[key]:>10.2f} {opt_t[key]:>13.2f} {sp:>7.1f}x")


if __name__ == "__main__":
    main()
