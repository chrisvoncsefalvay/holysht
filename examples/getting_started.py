#!/usr/bin/env python3
"""HOLYSHT reimplementation of torch-harmonics getting_started.ipynb.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

Workload:
  1. Forward SHT on a 512x1024 signal (Mars MOLA elevation analog)
  2. Inverse SHT (reconstruction)
  3. Roundtrip error measurement
  4. Spectral coefficient fitting via gradient descent (120 modes, 500 iters)
"""

import os
import sys
import time
from contextlib import contextmanager

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

from torch_harmonics import RealSHT as RefRealSHT, InverseRealSHT as RefInverseRealSHT
from holysht import RealSHT as HolySHT, InverseRealSHT as HolyInverseSHT

DEVICE = torch.device("cuda")
N_THETA, N_LAMBDA = 512, 1024


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


def make_signal():
    """Synthetic elevation signal (deterministic, no external data dependency)."""
    torch.manual_seed(42)
    theta = torch.linspace(0, torch.pi, N_THETA, device=DEVICE)
    lam = torch.linspace(0, 2 * torch.pi, N_LAMBDA, device=DEVICE)
    T, L = torch.meshgrid(theta, lam, indexing="ij")
    # Approximate a real-ish elevation field: sum of a few SH-like modes
    signal = (
        0.5 * torch.sin(T) * torch.cos(L)
        + 0.3 * torch.sin(2 * T) * torch.cos(3 * L)
        + 0.2 * torch.sin(5 * T) * torch.cos(7 * L)
        + 0.1 * torch.randn(N_THETA, N_LAMBDA, device=DEVICE)
    )
    return signal.unsqueeze(0)  # [1, nlat, nlon]


class SpectralModel(nn.Module):
    def __init__(self, n_modes, out_dims, isht_cls):
        super().__init__()
        mmax = n_modes + 1
        self.coeffs = nn.Parameter(
            torch.randn(1, n_modes, mmax, dtype=torch.complex64, device=DEVICE) * 0.01
        )
        self.isht = isht_cls(out_dims[0], out_dims[1], lmax=n_modes, mmax=mmax, grid="equiangular").to(DEVICE)

    def forward(self):
        return self.isht(self.coeffs)


def run_transform(label, sht_cls, isht_cls, signal):
    print(f"\n--- {label} ---")
    sht = sht_cls(N_THETA, N_LAMBDA, grid="equiangular").to(DEVICE)
    isht = isht_cls(N_THETA, N_LAMBDA, grid="equiangular").to(DEVICE)

    # Warmup
    with torch.no_grad():
        _ = sht(signal)
    torch.cuda.synchronize()

    timings = {}

    with cuda_timer("Forward SHT"):
        with torch.no_grad():
            coeffs = sht(signal)
    timings["forward"] = cuda_timer.last

    with cuda_timer("Inverse SHT"):
        with torch.no_grad():
            recon = isht(coeffs)
    timings["inverse"] = cuda_timer.last

    roundtrip_err = (recon - signal).abs().max().item()
    print(f"  Roundtrip max error: {roundtrip_err:.2e}")

    return coeffs, recon, timings


def run_fitting(label, isht_cls, signal, n_modes=120, n_iters=500):
    print(f"\n--- {label}: Spectral fitting ({n_modes} modes, {n_iters} iters) ---")
    model = SpectralModel(n_modes, (N_THETA, N_LAMBDA), isht_cls).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-2)

    # Warmup
    loss = (model() - signal).pow(2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    with cuda_timer(f"Fitting {n_iters} iterations"):
        for i in range(n_iters):
            loss = (model() - signal).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    final_loss = loss.item()
    print(f"  Final loss: {final_loss:.6e}")
    return final_loss, cuda_timer.last


def main():
    print("=" * 60)
    print("getting_started: SHT forward/inverse + spectral fitting")
    print("=" * 60)

    signal = make_signal()

    # Reference
    ref_coeffs, ref_recon, ref_timings = run_transform(
        "torch-harmonics (reference)", RefRealSHT, RefInverseRealSHT, signal
    )

    # HOLYSHT
    opt_coeffs, opt_recon, opt_timings = run_transform(
        "HOLYSHT", HolySHT, HolyInverseSHT, signal
    )

    # Numerical comparison
    coeff_err = (opt_coeffs - ref_coeffs).abs().max().item()
    recon_err = (opt_recon - ref_recon).abs().max().item()
    print(f"\n  Coefficient max diff: {coeff_err:.2e}")
    print(f"  Reconstruction max diff: {recon_err:.2e}")

    # Spectral fitting
    ref_loss, ref_fit_ms = run_fitting(
        "torch-harmonics (reference)", RefInverseRealSHT, signal
    )
    opt_loss, opt_fit_ms = run_fitting(
        "HOLYSHT", HolyInverseSHT, signal
    )

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Workload':<35} {'TH (ms)':>10} {'HOLYSHT (ms)':>13} {'Speedup':>8}")
    print("-" * 60)
    for key in ["forward", "inverse"]:
        sp = ref_timings[key] / opt_timings[key] if opt_timings[key] > 0 else 0
        print(f"{'SHT ' + key:<35} {ref_timings[key]:>10.2f} {opt_timings[key]:>13.2f} {sp:>7.1f}x")
    sp = ref_fit_ms / opt_fit_ms if opt_fit_ms > 0 else 0
    print(f"{'Spectral fitting (500 iters)':<35} {ref_fit_ms:>10.2f} {opt_fit_ms:>13.2f} {sp:>7.1f}x")
    print("-" * 60)
    print(f"{'Coefficient agreement':<35} {'max diff':>10} {coeff_err:>13.2e}")
    print(f"{'Reconstruction agreement':<35} {'max diff':>10} {recon_err:>13.2e}")
    print(f"{'Ref final fit loss':<35} {ref_loss:>24.6e}")
    print(f"{'HOLYSHT final fit loss':<35} {opt_loss:>24.6e}")


if __name__ == "__main__":
    main()
