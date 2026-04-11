#!/usr/bin/env python3
"""Public API parity tests for HOLYSHT.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

import holysht
from holysht import InverseRealSHT, InverseRealVectorSHT, RealSHT, RealVectorSHT
from holysht import _prepare_irfft_input
from torch_harmonics import (
    InverseRealSHT as RefInverseRealSHT,
    InverseRealVectorSHT as RefInverseRealVectorSHT,
    RealSHT as RefRealSHT,
    RealVectorSHT as RefRealVectorSHT,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")

DEVICE = torch.device("cuda")


def complex_energy(x: torch.Tensor) -> torch.Tensor:
    return x.real.square().mean() + x.imag.square().mean()


def test_aliases_are_available():
    assert holysht.FusedRealSHT is holysht.RealSHT
    assert holysht.FusedInverseRealSHT is holysht.InverseRealSHT
    assert holysht.FusedRealVectorSHT is holysht.RealVectorSHT
    assert holysht.FusedInverseRealVectorSHT is holysht.InverseRealVectorSHT


# ============================================================================
# Scalar SHT
# ============================================================================

@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_scalar_forward_and_inverse_match_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_sht = RefRealSHT(nlat, nlon).to(DEVICE)
    ref_isht = RefInverseRealSHT(nlat, nlon).to(DEVICE)
    opt_sht = RealSHT(nlat, nlon).to(DEVICE)
    opt_isht = InverseRealSHT(nlat, nlon).to(DEVICE)

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_coeffs = ref_sht(x)
        opt_coeffs = opt_sht(x)
        ref_back = ref_isht(ref_coeffs)
        opt_back = opt_isht(opt_coeffs)

    assert (opt_coeffs - ref_coeffs).abs().max().item() < 1e-3
    assert (opt_back - ref_back).abs().max().item() < 1e-2


@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_scalar_backward_matches_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_sht = RefRealSHT(nlat, nlon).to(DEVICE)
    opt_sht = RealSHT(nlat, nlon).to(DEVICE)

    x_ref = torch.randn(2, nlat, nlon, device=DEVICE, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    complex_energy(ref_sht(x_ref)).backward()
    complex_energy(opt_sht(x_opt)).backward()

    assert (x_opt.grad - x_ref.grad).abs().max().item() < 1e-3


@pytest.mark.parametrize("nlat,nlon", [(64, 128)])
def test_scalar_forward_accepts_noncontiguous_input(nlat, nlon):
    torch.manual_seed(0)
    ref_sht = RefRealSHT(nlat, nlon).to(DEVICE)
    opt_sht = RealSHT(nlat, nlon).to(DEVICE)

    x = torch.randn(2, nlon, nlat, device=DEVICE).transpose(-1, -2)
    assert not x.is_contiguous()

    with torch.no_grad():
        ref_out = ref_sht(x)
        opt_out = opt_sht(x)

    assert (opt_out - ref_out).abs().max().item() < 1e-3


# ============================================================================
# Vector SHT
# ============================================================================

@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_vector_forward_and_inverse_match_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_vsht = RefRealVectorSHT(nlat, nlon).to(DEVICE)
    ref_ivsht = RefInverseRealVectorSHT(nlat, nlon).to(DEVICE)
    opt_vsht = RealVectorSHT(nlat, nlon).to(DEVICE)
    opt_ivsht = InverseRealVectorSHT(nlat, nlon).to(DEVICE)

    x = torch.randn(2, 2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_coeffs = ref_vsht(x)
        opt_coeffs = opt_vsht(x)
        ref_back = ref_ivsht(ref_coeffs)
        opt_back = opt_ivsht(opt_coeffs)

    assert (opt_coeffs - ref_coeffs).abs().max().item() < 1e-3
    assert (opt_back - ref_back).abs().max().item() < 1e-2


@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_vector_backward_matches_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_vsht = RefRealVectorSHT(nlat, nlon).to(DEVICE)
    opt_vsht = RealVectorSHT(nlat, nlon).to(DEVICE)

    x_ref = torch.randn(2, 2, nlat, nlon, device=DEVICE, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    complex_energy(ref_vsht(x_ref)).backward()
    complex_energy(opt_vsht(x_opt)).backward()

    assert (x_opt.grad - x_ref.grad).abs().max().item() < 1e-3


@pytest.mark.parametrize("nlat,nlon", [(64, 128)])
def test_vector_forward_accepts_noncontiguous_input(nlat, nlon):
    torch.manual_seed(0)
    ref_vsht = RefRealVectorSHT(nlat, nlon).to(DEVICE)
    opt_vsht = RealVectorSHT(nlat, nlon).to(DEVICE)

    x = torch.randn(2, 2, nlon, nlat, device=DEVICE).transpose(-1, -2)
    assert not x.is_contiguous()

    with torch.no_grad():
        ref_out = ref_vsht(x)
        opt_out = opt_vsht(x)

    assert (opt_out - ref_out).abs().max().item() < 1e-3


# ============================================================================
# Large-grid tests (nlat > 128 triggers the tiled shared-memory kernels)
# ============================================================================

@pytest.mark.parametrize("nlat,nlon", [(256, 512), (512, 1024)])
def test_large_grid_scalar_forward(nlat, nlon):
    """Tests the large-grid tiled kernel path (nlat > SMALL_GRID_THRESHOLD=128)."""
    torch.manual_seed(42)
    ref_sht = RefRealSHT(nlat, nlon).to(DEVICE)
    opt_sht = RealSHT(nlat, nlon).to(DEVICE)

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_out = ref_sht(x)
        opt_out = opt_sht(x)

    assert (opt_out - ref_out).abs().max().item() < 1e-3


@pytest.mark.parametrize("nlat,nlon", [(256, 512), (512, 1024)])
def test_large_grid_scalar_inverse(nlat, nlon):
    """Tests the large-grid inverse kernel path."""
    torch.manual_seed(42)
    ref_sht = RefRealSHT(nlat, nlon).to(DEVICE)
    ref_isht = RefInverseRealSHT(nlat, nlon).to(DEVICE)
    opt_isht = InverseRealSHT(nlat, nlon).to(DEVICE)

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        coeffs = ref_sht(x)
        ref_out = ref_isht(coeffs)
        opt_out = opt_isht(coeffs)

    assert (opt_out - ref_out).abs().max().item() < 1e-2


@pytest.mark.parametrize("nlat,nlon", [(256, 512)])
def test_large_grid_scalar_backward(nlat, nlon):
    """Tests autograd through the large-grid tiled kernel."""
    torch.manual_seed(42)
    ref_sht = RefRealSHT(nlat, nlon).to(DEVICE)
    opt_sht = RealSHT(nlat, nlon).to(DEVICE)

    x_ref = torch.randn(2, nlat, nlon, device=DEVICE, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    complex_energy(ref_sht(x_ref)).backward()
    complex_energy(opt_sht(x_opt)).backward()

    assert (x_opt.grad - x_ref.grad).abs().max().item() < 1e-3


@pytest.mark.parametrize("nlat,nlon", [(256, 512)])
def test_large_grid_vector_forward(nlat, nlon):
    """Tests vector SHT through the large-grid path."""
    torch.manual_seed(42)
    ref_vsht = RefRealVectorSHT(nlat, nlon).to(DEVICE)
    opt_vsht = RealVectorSHT(nlat, nlon).to(DEVICE)

    x = torch.randn(2, 2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_out = ref_vsht(x)
        opt_out = opt_vsht(x)

    assert (opt_out - ref_out).abs().max().item() < 1e-3


# ============================================================================
# BF16 tests
# ============================================================================

@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_bf16_scalar_forward(nlat, nlon):
    """Tests the BF16 einsum path produces results within bf16 tolerance."""
    torch.manual_seed(0)
    ref_sht = RefRealSHT(nlat, nlon).to(DEVICE)
    bf16_sht = RealSHT(nlat, nlon, dtype="bf16").to(DEVICE)

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_out = ref_sht(x)
        bf16_out = bf16_sht(x)

    # BF16 has lower precision — 5e-3 absolute tolerance
    assert (bf16_out - ref_out).abs().max().item() < 5e-3


@pytest.mark.parametrize("nlat,nlon", [(64, 128)])
def test_bf16_vector_forward(nlat, nlon):
    """Tests BF16 vector SHT path."""
    torch.manual_seed(0)
    ref_vsht = RefRealVectorSHT(nlat, nlon).to(DEVICE)
    bf16_vsht = RealVectorSHT(nlat, nlon, dtype="bf16").to(DEVICE)

    x = torch.randn(2, 2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_out = ref_vsht(x)
        bf16_out = bf16_vsht(x)

    assert (bf16_out - ref_out).abs().max().item() < 5e-3


# ============================================================================
# _prepare_irfft_input tests
# ============================================================================

def test_prepare_irfft_zeros_dc_imaginary():
    """DC mode (m=0) must have zero imaginary part after preparation."""
    nlon = 128
    nlat = 32
    mmax = nlon // 2 + 1
    x = torch.randn(2, nlat, mmax, dtype=torch.complex64, device=DEVICE)
    out = _prepare_irfft_input(x, nlon)
    assert out[..., 0].imag.abs().max().item() == 0.0


def test_prepare_irfft_zeros_nyquist_imaginary():
    """Nyquist mode must have zero imaginary part for even nlon."""
    nlon = 128  # even
    nlat = 32
    mmax = nlon // 2 + 1
    x = torch.randn(2, nlat, mmax, dtype=torch.complex64, device=DEVICE)
    out = _prepare_irfft_input(x, nlon)
    nyquist_idx = nlon // 2
    assert out[..., nyquist_idx].imag.abs().max().item() == 0.0


def test_prepare_irfft_pads_correctly():
    """When input mmax < nlon//2+1, output must be zero-padded."""
    nlon = 128
    nlat = 32
    active_mmax = 33  # less than nlon//2+1 = 65
    x = torch.randn(2, nlat, active_mmax, dtype=torch.complex64, device=DEVICE)
    out = _prepare_irfft_input(x, nlon, active_mmax)
    full_mmax = nlon // 2 + 1
    assert out.size(-1) == full_mmax
    # Padded region should be zeros
    assert out[..., active_mmax:].abs().max().item() == 0.0


def test_prepare_irfft_odd_nlon():
    """For odd nlon there is no Nyquist mode — only DC gets cleaned."""
    nlon = 127  # odd
    nlat = 32
    mmax = nlon // 2 + 1  # 64
    x = torch.randn(2, nlat, mmax, dtype=torch.complex64, device=DEVICE)
    out = _prepare_irfft_input(x, nlon)
    assert out[..., 0].imag.abs().max().item() == 0.0
    # Last mode should NOT have imaginary zeroed (it's not Nyquist)
    # Just check output shape is correct
    assert out.size(-1) == mmax
