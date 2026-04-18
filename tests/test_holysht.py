#!/usr/bin/env python3
"""Public API parity tests for HOLYSHT.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

import os
import sys
from pathlib import Path

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


HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

pytestmark = pytest.mark.skipif(not (HAS_CUDA or HAS_MPS), reason="CUDA or MPS is required")

DEVICE = torch.device("cuda" if HAS_CUDA else "mps")
COEFF_ATOL = 1e-3 if DEVICE.type == "cuda" else 3e-3
INVERSE_ATOL = 1e-2 if DEVICE.type == "cuda" else 2e-2
GRAD_ATOL = 1e-3 if DEVICE.type == "cuda" else 3e-3


def _to_test_device(module: torch.nn.Module) -> torch.nn.Module:
    if DEVICE.type == "mps":
        for name, buf in list(module._buffers.items()):
            if buf is None:
                continue
            if buf.is_floating_point():
                module._buffers[name] = buf.float()
            elif buf.is_complex():
                module._buffers[name] = buf.to(torch.complex64)
    return module.to(DEVICE)


def complex_energy(x: torch.Tensor) -> torch.Tensor:
    return x.real.square().mean() + x.imag.square().mean()


def test_aliases_are_available():
    assert holysht.FusedRealSHT is holysht.RealSHT
    assert holysht.FusedInverseRealSHT is holysht.InverseRealSHT
    assert holysht.FusedRealVectorSHT is holysht.RealVectorSHT
    assert holysht.FusedInverseRealVectorSHT is holysht.InverseRealVectorSHT


def test_autotune_batch_bucket_is_stable():
    assert holysht._autotune_batch_bucket(1) == "1"
    assert holysht._autotune_batch_bucket(2) == "2"
    assert holysht._autotune_batch_bucket(3) == "3-4"
    assert holysht._autotune_batch_bucket(4) == "3-4"
    assert holysht._autotune_batch_bucket(5) == "5+"


def test_autotune_cache_roundtrip(tmp_path):
    cache_path = tmp_path / "autotune.json"
    cache = holysht._AutotuneCache(cache_path)
    key = holysht._AutotuneKey(
        device_name="test-gpu",
        capability="9.0",
        op_kind="scalar-real-forward",
        dtype_mode="fp32",
        nlat=256,
        lmax=256,
        mmax=257,
        batch_bucket="3-4",
    )
    cache.store(key, "tc_tf32")

    reloaded = holysht._AutotuneCache(cache_path)
    assert reloaded.load(key) == "tc_tf32"


def test_force_backend_env_parser_accepts_known_values(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_tf32")
    assert holysht._forced_cuda_forward_backend() == holysht._ForwardBackend.TC_TF32

    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_bf16")
    assert holysht._forced_cuda_forward_backend() == holysht._ForwardBackend.TC_BF16

    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "bogus")
    assert holysht._forced_cuda_forward_backend() == holysht._ForwardBackend.AUTO


# ============================================================================
# Scalar SHT
# ============================================================================

@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_scalar_forward_and_inverse_match_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_sht = _to_test_device(RefRealSHT(nlat, nlon))
    ref_isht = _to_test_device(RefInverseRealSHT(nlat, nlon))
    opt_sht = _to_test_device(RealSHT(nlat, nlon))
    opt_isht = _to_test_device(InverseRealSHT(nlat, nlon))

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_coeffs = ref_sht(x)
        opt_coeffs = opt_sht(x)
        ref_back = ref_isht(ref_coeffs)
        opt_back = opt_isht(opt_coeffs)

    assert (opt_coeffs - ref_coeffs).abs().max().item() < COEFF_ATOL
    assert (opt_back - ref_back).abs().max().item() < INVERSE_ATOL


@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_scalar_backward_matches_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_sht = _to_test_device(RefRealSHT(nlat, nlon))
    opt_sht = _to_test_device(RealSHT(nlat, nlon))

    x_ref = torch.randn(2, nlat, nlon, device=DEVICE, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    complex_energy(ref_sht(x_ref)).backward()
    complex_energy(opt_sht(x_opt)).backward()

    assert (x_opt.grad - x_ref.grad).abs().max().item() < GRAD_ATOL


@pytest.mark.parametrize("nlat,nlon", [(64, 128)])
def test_scalar_forward_accepts_noncontiguous_input(nlat, nlon):
    torch.manual_seed(0)
    ref_sht = _to_test_device(RefRealSHT(nlat, nlon))
    opt_sht = _to_test_device(RealSHT(nlat, nlon))

    x = torch.randn(2, nlon, nlat, device=DEVICE).transpose(-1, -2)
    assert not x.is_contiguous()

    with torch.no_grad():
        ref_out = ref_sht(x)
        opt_out = opt_sht(x)

    assert (opt_out - ref_out).abs().max().item() < COEFF_ATOL


@pytest.mark.skipif(DEVICE.type != "mps", reason="Chunked fallback is MPS-only")
def test_mps_scalar_chunked_fallback_matches_reference(monkeypatch):
    torch.manual_seed(0)
    monkeypatch.setenv("HOLYSHT_MPS_SCALAR_NATIVE_MAX_NLAT", "0")
    monkeypatch.setenv("HOLYSHT_MPS_SCALAR_FORWARD_EINSUM_M_CHUNK", "32")
    monkeypatch.setenv("HOLYSHT_MPS_SCALAR_INVERSE_EINSUM_M_CHUNK", "32")

    ref_sht = _to_test_device(RefRealSHT(64, 128))
    ref_isht = _to_test_device(RefInverseRealSHT(64, 128))
    opt_sht = _to_test_device(RealSHT(64, 128))
    opt_isht = _to_test_device(InverseRealSHT(64, 128))

    x = torch.randn(2, 64, 128, device=DEVICE)
    with torch.no_grad():
        ref_coeffs = ref_sht(x)
        opt_coeffs = opt_sht(x)
        ref_back = ref_isht(ref_coeffs)
        opt_back = opt_isht(opt_coeffs)

    assert (opt_coeffs - ref_coeffs).abs().max().item() < COEFF_ATOL
    assert (opt_back - ref_back).abs().max().item() < INVERSE_ATOL


# ============================================================================
# Vector SHT
# ============================================================================

@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_vector_forward_and_inverse_match_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_vsht = _to_test_device(RefRealVectorSHT(nlat, nlon))
    ref_ivsht = _to_test_device(RefInverseRealVectorSHT(nlat, nlon))
    opt_vsht = _to_test_device(RealVectorSHT(nlat, nlon))
    opt_ivsht = _to_test_device(InverseRealVectorSHT(nlat, nlon))

    x = torch.randn(2, 2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_coeffs = ref_vsht(x)
        opt_coeffs = opt_vsht(x)
        ref_back = ref_ivsht(ref_coeffs)
        opt_back = opt_ivsht(opt_coeffs)

    assert (opt_coeffs - ref_coeffs).abs().max().item() < COEFF_ATOL
    assert (opt_back - ref_back).abs().max().item() < INVERSE_ATOL


@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_vector_backward_matches_reference(nlat, nlon):
    torch.manual_seed(0)
    ref_vsht = _to_test_device(RefRealVectorSHT(nlat, nlon))
    opt_vsht = _to_test_device(RealVectorSHT(nlat, nlon))

    x_ref = torch.randn(2, 2, nlat, nlon, device=DEVICE, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    complex_energy(ref_vsht(x_ref)).backward()
    complex_energy(opt_vsht(x_opt)).backward()

    assert (x_opt.grad - x_ref.grad).abs().max().item() < GRAD_ATOL


@pytest.mark.parametrize("nlat,nlon", [(64, 128)])
def test_vector_forward_accepts_noncontiguous_input(nlat, nlon):
    torch.manual_seed(0)
    ref_vsht = _to_test_device(RefRealVectorSHT(nlat, nlon))
    opt_vsht = _to_test_device(RealVectorSHT(nlat, nlon))

    x = torch.randn(2, 2, nlon, nlat, device=DEVICE).transpose(-1, -2)
    assert not x.is_contiguous()

    with torch.no_grad():
        ref_out = ref_vsht(x)
        opt_out = opt_vsht(x)

    assert (opt_out - ref_out).abs().max().item() < COEFF_ATOL


# ============================================================================
# Large-grid tests (nlat > 128 triggers the tiled shared-memory kernels)
# ============================================================================

@pytest.mark.parametrize("nlat,nlon", [(256, 512), (512, 1024)])
def test_large_grid_scalar_forward(nlat, nlon):
    """Tests the large-grid tiled kernel path (nlat > SMALL_GRID_THRESHOLD=128)."""
    torch.manual_seed(42)
    ref_sht = _to_test_device(RefRealSHT(nlat, nlon))
    opt_sht = _to_test_device(RealSHT(nlat, nlon))

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_out = ref_sht(x)
        opt_out = opt_sht(x)

    assert (opt_out - ref_out).abs().max().item() < COEFF_ATOL


@pytest.mark.parametrize("nlat,nlon", [(256, 512), (512, 1024)])
def test_large_grid_scalar_inverse(nlat, nlon):
    """Tests the large-grid inverse kernel path."""
    torch.manual_seed(42)
    ref_sht = _to_test_device(RefRealSHT(nlat, nlon))
    ref_isht = _to_test_device(RefInverseRealSHT(nlat, nlon))
    opt_isht = _to_test_device(InverseRealSHT(nlat, nlon))

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        coeffs = ref_sht(x)
        ref_out = ref_isht(coeffs)
        opt_out = opt_isht(coeffs)

    assert (opt_out - ref_out).abs().max().item() < INVERSE_ATOL


@pytest.mark.parametrize("nlat,nlon", [(256, 512)])
def test_large_grid_scalar_backward(nlat, nlon):
    """Tests autograd through the large-grid tiled kernel."""
    torch.manual_seed(42)
    ref_sht = _to_test_device(RefRealSHT(nlat, nlon))
    opt_sht = _to_test_device(RealSHT(nlat, nlon))

    x_ref = torch.randn(2, nlat, nlon, device=DEVICE, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)

    complex_energy(ref_sht(x_ref)).backward()
    complex_energy(opt_sht(x_opt)).backward()

    assert (x_opt.grad - x_ref.grad).abs().max().item() < GRAD_ATOL


@pytest.mark.parametrize("nlat,nlon", [(256, 512)])
def test_large_grid_vector_forward(nlat, nlon):
    """Tests vector SHT through the large-grid path."""
    torch.manual_seed(42)
    ref_vsht = _to_test_device(RefRealVectorSHT(nlat, nlon))
    opt_vsht = _to_test_device(RealVectorSHT(nlat, nlon))

    x = torch.randn(2, 2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_out = ref_vsht(x)
        opt_out = opt_vsht(x)

    assert (opt_out - ref_out).abs().max().item() < COEFF_ATOL


# ============================================================================
# BF16 tests
# ============================================================================

@pytest.mark.skipif(DEVICE.type != "cuda", reason="BF16 path is CUDA-only")
@pytest.mark.parametrize("nlat,nlon", [(64, 128), (256, 512)])
def test_bf16_scalar_forward(nlat, nlon):
    """Tests the BF16 einsum path produces results within bf16 tolerance."""
    torch.manual_seed(0)
    ref_sht = _to_test_device(RefRealSHT(nlat, nlon))
    bf16_sht = _to_test_device(RealSHT(nlat, nlon, dtype="bf16"))

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        ref_out = ref_sht(x)
        bf16_out = bf16_sht(x)

    # BF16 has lower precision — 5e-3 absolute tolerance
    assert (bf16_out - ref_out).abs().max().item() < 5e-3


@pytest.mark.skipif(DEVICE.type != "cuda", reason="BF16 path is CUDA-only")
@pytest.mark.parametrize("nlat,nlon", [(64, 128)])
def test_bf16_vector_forward(nlat, nlon):
    """Tests BF16 vector SHT path."""
    torch.manual_seed(0)
    ref_vsht = _to_test_device(RefRealVectorSHT(nlat, nlon))
    bf16_vsht = _to_test_device(RealVectorSHT(nlat, nlon, dtype="bf16"))

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
