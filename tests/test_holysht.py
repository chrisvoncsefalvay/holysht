#!/usr/bin/env python3
"""Public API parity tests for HOLYSHT."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

import holysht
from holysht import InverseRealSHT, InverseRealVectorSHT, RealSHT, RealVectorSHT
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
