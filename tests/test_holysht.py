#!/usr/bin/env python3
"""Public API parity tests for HOLYSHT.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "torch-ext"))

import holysht
from holysht import InverseRealSHT, InverseRealVectorSHT, RealSHT, RealVectorSHT
from holysht import _aligned_mmax_for_tma_tile, _prepare_irfft_input
from torch_harmonics import (
    InverseRealSHT as RefInverseRealSHT,
    InverseRealVectorSHT as RefInverseRealVectorSHT,
    RealSHT as RefRealSHT,
    RealVectorSHT as RefRealVectorSHT,
)


HAS_CUDA = torch.cuda.is_available()
HAS_MPS = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
CUDA_MAJOR = torch.cuda.get_device_capability()[0] if HAS_CUDA else 0

DEVICE = torch.device("cuda" if HAS_CUDA else "mps")
COEFF_ATOL = 1e-3 if DEVICE.type == "cuda" else 3e-3
INVERSE_ATOL = 1e-2 if DEVICE.type == "cuda" else 2e-2
GRAD_ATOL = 1e-3 if DEVICE.type == "cuda" else 3e-3


@pytest.fixture(autouse=True)
def _skip_gpu_only_tests(request):
    if HAS_CUDA or HAS_MPS:
        return
    if request.node.get_closest_marker("cpu_ok") is not None:
        return
    pytest.skip("CUDA or MPS is required")


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


def upper_triangle_mask(lmax: int, mmax: int, device: torch.device) -> torch.Tensor:
    l = torch.arange(lmax, device=device).unsqueeze(1)
    m = torch.arange(mmax, device=device).unsqueeze(0)
    return m > l


def vector_forward_reference(
    x: torch.Tensor,
    weight0_t: torch.Tensor,
    weight1_t: torch.Tensor,
) -> torch.Tensor:
    xr = torch.view_as_real(x.contiguous())
    x0_re = xr[:, 0, :, :, 0]
    x0_im = xr[:, 0, :, :, 1]
    x1_re = xr[:, 1, :, :, 0]
    x1_im = xr[:, 1, :, :, 1]

    sph_re = torch.einsum("bnm,lnm->blm", x0_re, weight0_t) - torch.einsum("bnm,lnm->blm", x1_im, weight1_t)
    sph_im = torch.einsum("bnm,lnm->blm", x0_im, weight0_t) + torch.einsum("bnm,lnm->blm", x1_re, weight1_t)
    tor_re = -torch.einsum("bnm,lnm->blm", x0_im, weight1_t) - torch.einsum("bnm,lnm->blm", x1_re, weight0_t)
    tor_im = torch.einsum("bnm,lnm->blm", x0_re, weight1_t) - torch.einsum("bnm,lnm->blm", x1_im, weight0_t)

    out = torch.empty((x.size(0), 2, weight0_t.size(0), x.size(3), 2), device=x.device, dtype=torch.float32)
    out[:, 0, :, :, 0] = sph_re
    out[:, 0, :, :, 1] = sph_im
    out[:, 1, :, :, 0] = tor_re
    out[:, 1, :, :, 1] = tor_im
    return torch.view_as_complex(out)


def test_aliases_are_available():
    assert holysht.FusedRealSHT is holysht.RealSHT
    assert holysht.FusedInverseRealSHT is holysht.InverseRealSHT
    assert holysht.FusedRealVectorSHT is holysht.RealVectorSHT
    assert holysht.FusedInverseRealVectorSHT is holysht.InverseRealVectorSHT


@pytest.mark.cpu_ok
def test_autotune_batch_bucket_is_stable():
    assert holysht._autotune_batch_bucket(1) == "1"
    assert holysht._autotune_batch_bucket(2) == "2"
    assert holysht._autotune_batch_bucket(3) == "3-4"
    assert holysht._autotune_batch_bucket(4) == "3-4"
    assert holysht._autotune_batch_bucket(5) == "5+"


@pytest.mark.cpu_ok
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


@pytest.mark.cpu_ok
def test_force_backend_env_parser_accepts_known_values(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_tf32")
    assert holysht._forced_cuda_forward_backend() == holysht._ForwardBackend.TC_TF32

    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_bf16")
    assert holysht._forced_cuda_forward_backend() == holysht._ForwardBackend.TC_BF16

    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "bogus")
    assert holysht._forced_cuda_forward_backend() == holysht._ForwardBackend.AUTO


@pytest.mark.cpu_ok
def test_force_vector_strategy_env_parser_accepts_known_values(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_VECTOR_STRATEGY", "native")
    assert holysht._forced_vector_forward_strategy() == "native_vector"

    monkeypatch.setenv("HOLYSHT_FORCE_VECTOR_STRATEGY", "stacked_real")
    assert holysht._forced_vector_forward_strategy() == "stacked_real"

    monkeypatch.setenv("HOLYSHT_FORCE_VECTOR_STRATEGY", "bogus")
    assert holysht._forced_vector_forward_strategy() == "auto"


@pytest.mark.cpu_ok
def test_autotune_prefers_forced_backend(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tma")
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

    winner = holysht._select_forward_backend_for_key(
        key,
        candidates=["fma", "tma", "tc_tf32"],
        benchmark=lambda name: {"fma": 3.0, "tma": 2.0, "tc_tf32": 1.0}[name],
    )

    assert winner == "tma"


@pytest.mark.cpu_ok
def test_vector_strategy_autotune_prefers_forced_strategy(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_VECTOR_STRATEGY", "native")
    key = holysht._AutotuneKey(
        device_name="test-gpu",
        capability="9.0",
        op_kind="vector-forward-strategy",
        dtype_mode="fp32",
        nlat=256,
        lmax=256,
        mmax=257,
        batch_bucket="3-4",
    )

    winner = holysht._select_vector_forward_strategy_for_key(
        key,
        candidates=["stacked_real", "native_vector"],
        benchmark=lambda name: {"stacked_real": 3.0, "native_vector": 2.0}[name],
    )

    assert winner == "native_vector"


@pytest.mark.cpu_ok
def test_autotune_uses_cached_winner(tmp_path, monkeypatch):
    monkeypatch.delenv("HOLYSHT_FORCE_BACKEND", raising=False)
    cache = holysht._AutotuneCache(tmp_path / "cache.json")
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

    winner = holysht._select_forward_backend_for_key(
        key,
        candidates=["fma", "tma", "tc_tf32"],
        benchmark=lambda name: (_ for _ in ()).throw(AssertionError("benchmark should not run")),
        cache=cache,
    )

    assert winner == "tc_tf32"


@pytest.mark.cpu_ok
def test_vector_strategy_autotune_uses_cached_winner(tmp_path, monkeypatch):
    monkeypatch.delenv("HOLYSHT_FORCE_VECTOR_STRATEGY", raising=False)
    cache = holysht._AutotuneCache(tmp_path / "cache.json")
    key = holysht._AutotuneKey(
        device_name="test-gpu",
        capability="9.0",
        op_kind="vector-forward-strategy",
        dtype_mode="fp32",
        nlat=256,
        lmax=256,
        mmax=257,
        batch_bucket="3-4",
    )
    cache.store(key, "native_vector")

    winner = holysht._select_vector_forward_strategy_for_key(
        key,
        candidates=["stacked_real", "native_vector"],
        benchmark=lambda name: (_ for _ in ()).throw(AssertionError("benchmark should not run")),
        cache=cache,
    )

    assert winner == "native_vector"


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (["fma", "tma", "tc_tf32"], "tma"),
        (["fma", "tc_tf32"], "fma"),
    ],
)
def test_autotune_disabled_uses_deterministic_fallback(monkeypatch, candidates, expected):
    monkeypatch.delenv("HOLYSHT_FORCE_BACKEND", raising=False)
    monkeypatch.setenv("HOLYSHT_AUTOTUNE", "0")
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

    winner = holysht._select_forward_backend_for_key(
        key,
        candidates=candidates,
        benchmark=lambda name: (_ for _ in ()).throw(AssertionError("benchmark should not run")),
    )

    assert winner == expected


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("candidates", "expected"),
    [
        (["stacked_real", "native_vector"], "stacked_real"),
        (["native_vector"], "native_vector"),
    ],
)
def test_vector_strategy_autotune_disabled_uses_deterministic_fallback(monkeypatch, candidates, expected):
    monkeypatch.delenv("HOLYSHT_FORCE_VECTOR_STRATEGY", raising=False)
    monkeypatch.setenv("HOLYSHT_AUTOTUNE", "0")
    key = holysht._AutotuneKey(
        device_name="test-gpu",
        capability="9.0",
        op_kind="vector-forward-strategy",
        dtype_mode="fp32",
        nlat=256,
        lmax=256,
        mmax=257,
        batch_bucket="3-4",
    )

    winner = holysht._select_vector_forward_strategy_for_key(
        key,
        candidates=candidates,
        benchmark=lambda name: (_ for _ in ()).throw(AssertionError("benchmark should not run")),
    )

    assert winner == expected


@pytest.mark.cpu_ok
def test_vector_strategy_candidates_keep_native_for_tensor_core_forces():
    assert holysht._vector_forward_strategy_candidates(
        can_use_native=True,
        forced_backend=holysht._ForwardBackend.TC_TF32,
    ) == ["stacked_real", "native_vector"]
    assert holysht._vector_forward_strategy_candidates(
        can_use_native=True,
        forced_backend=holysht._ForwardBackend.TC_BF16,
    ) == ["stacked_real", "native_vector"]
    assert holysht._vector_forward_strategy_candidates(
        can_use_native=True,
        forced_backend=holysht._ForwardBackend.TMA,
    ) == ["stacked_real", "native_vector"]


@pytest.mark.cpu_ok
def test_native_vector_backend_candidates_cover_fp32_and_bf16():
    assert holysht._native_vector_forward_backend_candidates(
        dtype_mode="fp32",
        can_use_tensor_core=True,
    ) == ["fma", "tma", "tc_tf32"]
    assert holysht._native_vector_forward_backend_candidates(
        dtype_mode="fp32",
        can_use_tensor_core=False,
    ) == ["fma", "tma"]
    assert holysht._native_vector_forward_backend_candidates(
        dtype_mode="bf16",
        can_use_tensor_core=True,
    ) == ["tma", "tc_bf16"]
    assert holysht._native_vector_forward_backend_candidates(
        dtype_mode="bf16",
        can_use_tensor_core=False,
    ) == ["tma"]


@pytest.mark.cpu_ok
def test_autotuned_direct_vector_forward_uses_cached_winner(tmp_path, monkeypatch):
    cache = holysht._AutotuneCache(tmp_path / "cache.json")
    key = holysht._AutotuneKey(
        device_name="test-gpu",
        capability="9.0",
        op_kind="vector-native-forward",
        dtype_mode="fp32",
        nlat=8,
        lmax=16,
        mmax=9,
        batch_bucket="2",
    )
    cache.store(key, "tc_tf32")

    calls = []

    def fake_direct(input, weight0_t, weight1_t, backend_hint=holysht._ForwardBackend.AUTO):
        calls.append((tuple(input.shape), tuple(weight0_t.shape), tuple(weight1_t.shape), backend_hint))
        return torch.zeros(input.size(0), 2, weight0_t.size(0), input.size(3), dtype=torch.complex64)

    monkeypatch.setattr(holysht, "_direct_vector_legendre_forward", fake_direct)

    x = torch.randn(2, 2, 8, 9, dtype=torch.complex64)
    w0 = torch.randn(16, 8, 9)
    w1 = torch.randn(16, 8, 9)

    out = holysht._autotuned_direct_vector_forward(
        x,
        w0,
        w1,
        dtype_mode="fp32",
        op_kind="vector-native-forward",
        backend_candidates=["fma", "tma", "tc_tf32"],
        cache=cache,
        device_name="test-gpu",
        capability="9.0",
    )

    assert calls == [(tuple(x.shape), tuple(w0.shape), tuple(w1.shape), holysht._ForwardBackend.TC_TF32)]
    assert out.shape == (2, 2, 16, 9)


@pytest.mark.cpu_ok
def test_autotuned_direct_vector_forward_rejects_unsupported_forced_tensor_core(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_tf32")

    x = torch.randn(2, 2, 8, 9, dtype=torch.complex64)
    w0 = torch.randn(16, 8, 9)
    w1 = torch.randn(16, 8, 9)

    with pytest.raises(RuntimeError, match="forced vector backend tc_tf32 is unsupported"):
        holysht._autotuned_direct_vector_forward(
            x,
            w0,
            w1,
            dtype_mode="fp32",
            op_kind="vector-native-forward",
            backend_candidates=["fma", "tma"],
            device_name="test-gpu",
            capability="9.0",
        )


@pytest.mark.cpu_ok
def test_run_vector_forward_native_cuda_routes_through_autotuned_helper(monkeypatch):
    calls = {}

    def fake_autotuned(
        input,
        weight0_t,
        weight1_t,
        *,
        dtype_mode,
        op_kind,
        backend_candidates,
        padded_input=None,
        padded_weight0_t=None,
        padded_weight1_t=None,
        cache=None,
        device_name=None,
        capability=None,
    ):
        calls["input_shape"] = tuple(input.shape)
        calls["weight0_shape"] = tuple(weight0_t.shape)
        calls["weight1_shape"] = tuple(weight1_t.shape)
        calls["dtype_mode"] = dtype_mode
        calls["op_kind"] = op_kind
        calls["backend_candidates"] = tuple(backend_candidates)
        calls["padded_input_shape"] = None if padded_input is None else tuple(padded_input.shape)
        calls["padded_weight0_shape"] = None if padded_weight0_t is None else tuple(padded_weight0_t.shape)
        calls["padded_weight1_shape"] = None if padded_weight1_t is None else tuple(padded_weight1_t.shape)
        calls["cache"] = cache
        calls["device_name"] = device_name
        calls["capability"] = capability
        return torch.zeros(input.size(0), 2, weight0_t.size(0), input.size(3), dtype=torch.complex64)

    monkeypatch.setattr(holysht, "_autotuned_direct_vector_forward", fake_autotuned)
    monkeypatch.setattr(holysht, "_cuda_tma_available", lambda tensor: True)

    x = torch.randn(2, 2, 8, 5, dtype=torch.complex64)
    w0 = torch.randn(16, 8, 5)
    w1 = torch.randn(16, 8, 5)
    padded_w0 = torch.randn(16, 8, 16)
    padded_w1 = torch.randn(16, 8, 16)
    cache = object()

    out = holysht._run_vector_forward_native_cuda(
        x,
        w0,
        w1,
        dtype_mode="fp32",
        op_kind="vector-native-forward",
        mmax=5,
        tma_mmax_complex=16,
        weight0_t_tma_complex=padded_w0,
        weight1_t_tma_complex=padded_w1,
        cache=cache,
        device_name="test-gpu",
        capability="9.0",
    )

    assert calls["dtype_mode"] == "fp32"
    assert calls["op_kind"] == "vector-native-forward"
    assert calls["backend_candidates"] == ("fma", "tma")
    assert calls["padded_input_shape"] == (2, 2, 8, 16)
    assert calls["padded_weight0_shape"] == (16, 8, 16)
    assert calls["padded_weight1_shape"] == (16, 8, 16)
    assert calls["cache"] is cache
    assert calls["device_name"] == "test-gpu"
    assert calls["capability"] == "9.0"
    assert out.shape == (2, 2, 16, 5)


@pytest.mark.cpu_ok
def test_stack_vector_forward_real_inputs_matches_reference_pack():
    xr = torch.arange(2 * 2 * 3 * 4 * 2, dtype=torch.float32).reshape(2, 2, 3, 4, 2)
    x = torch.view_as_complex(xr.contiguous())

    stacked_w0, stacked_w1 = holysht._stack_vector_forward_real_inputs(x)

    x00_flat = xr[:, 0, :, :, 0].reshape(-1, x.size(2), x.size(3))
    x01_flat = xr[:, 0, :, :, 1].reshape(-1, x.size(2), x.size(3))
    x10_flat = xr[:, 1, :, :, 0].reshape(-1, x.size(2), x.size(3))
    x11_flat = xr[:, 1, :, :, 1].reshape(-1, x.size(2), x.size(3))

    expected_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0).contiguous()
    expected_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0).contiguous()

    torch.testing.assert_close(stacked_w0, expected_w0)
    torch.testing.assert_close(stacked_w1, expected_w1)


@pytest.mark.cpu_ok
def test_recompose_vector_forward_real_outputs_matches_reference_formula():
    batch_size, lmax, mmax = 2, 3, 4
    out_w0 = torch.arange(4 * batch_size * lmax * mmax, dtype=torch.float32).reshape(4 * batch_size, lmax, mmax)
    out_w1 = (1000 + torch.arange(4 * batch_size * lmax * mmax, dtype=torch.float32)).reshape(4 * batch_size, lmax, mmax)

    out = holysht._recompose_vector_forward_real_outputs(
        out_w0,
        out_w1,
        batch_size=batch_size,
        lmax=lmax,
        mmax=mmax,
    )

    r00, r01, r10, r11 = out_w0[:batch_size], out_w0[batch_size:2 * batch_size], out_w0[2 * batch_size:3 * batch_size], out_w0[3 * batch_size:]
    s11, s10, s01, s00 = out_w1[:batch_size], out_w1[batch_size:2 * batch_size], out_w1[2 * batch_size:3 * batch_size], out_w1[3 * batch_size:]

    expected = torch.empty((batch_size, 2, lmax, mmax, 2), dtype=torch.float32)
    expected[:, 0, :, :, 0] = r00 - s11
    expected[:, 0, :, :, 1] = r01 + s10
    expected[:, 1, :, :, 0] = -s01 - r10
    expected[:, 1, :, :, 1] = s00 - r11

    torch.testing.assert_close(out, torch.view_as_complex(expected))


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
def test_cuda_stack_vector_forward_real_inputs_matches_reference_pack():
    xr = torch.arange(2 * 2 * 3 * 4 * 2, dtype=torch.float32, device="cuda").reshape(2, 2, 3, 4, 2)
    x = torch.view_as_complex(xr.contiguous())

    stacked_w0, stacked_w1 = holysht._stack_vector_forward_real_inputs(x)

    x00_flat = xr[:, 0, :, :, 0].reshape(-1, x.size(2), x.size(3))
    x01_flat = xr[:, 0, :, :, 1].reshape(-1, x.size(2), x.size(3))
    x10_flat = xr[:, 1, :, :, 0].reshape(-1, x.size(2), x.size(3))
    x11_flat = xr[:, 1, :, :, 1].reshape(-1, x.size(2), x.size(3))

    expected_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0).contiguous()
    expected_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0).contiguous()

    torch.testing.assert_close(stacked_w0, expected_w0)
    torch.testing.assert_close(stacked_w1, expected_w1)


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
def test_cuda_recompose_vector_forward_real_outputs_matches_reference_formula():
    batch_size, lmax, mmax = 2, 3, 4
    out_w0 = torch.arange(4 * batch_size * lmax * mmax, dtype=torch.float32, device="cuda").reshape(4 * batch_size, lmax, mmax)
    out_w1 = (1000 + torch.arange(4 * batch_size * lmax * mmax, dtype=torch.float32, device="cuda")).reshape(4 * batch_size, lmax, mmax)

    out = holysht._recompose_vector_forward_real_outputs(
        out_w0,
        out_w1,
        batch_size=batch_size,
        lmax=lmax,
        mmax=mmax,
    )

    r00, r01, r10, r11 = out_w0[:batch_size], out_w0[batch_size:2 * batch_size], out_w0[2 * batch_size:3 * batch_size], out_w0[3 * batch_size:]
    s11, s10, s01, s00 = out_w1[:batch_size], out_w1[batch_size:2 * batch_size], out_w1[2 * batch_size:3 * batch_size], out_w1[3 * batch_size:]

    expected = torch.empty((batch_size, 2, lmax, mmax, 2), dtype=torch.float32, device="cuda")
    expected[:, 0, :, :, 0] = r00 - s11
    expected[:, 0, :, :, 1] = r01 + s10
    expected[:, 1, :, :, 0] = -s01 - r10
    expected[:, 1, :, :, 1] = s00 - r11

    torch.testing.assert_close(out, torch.view_as_complex(expected))


@pytest.mark.cpu_ok
def test_run_vector_forward_stacked_real_fp32_uses_pack_and_recompose_helpers(monkeypatch):
    calls = []

    def fake_pack(input):
        calls.append(("pack", tuple(input.shape)))
        return (
            torch.ones(8, 3, 4, dtype=torch.float32),
            torch.full((8, 3, 4), 2.0, dtype=torch.float32),
        )

    def fake_forward(
        input,
        weight_t,
        *,
        op_kind,
        dtype_mode,
        backend_candidates,
        padded_input=None,
        padded_weight_t=None,
        cache=None,
        device_name=None,
        capability=None,
    ):
        calls.append(("forward", tuple(input.shape), tuple(weight_t.shape), op_kind, dtype_mode, tuple(backend_candidates)))
        fill = 3.0 if len([item for item in calls if item[0] == "forward"]) == 1 else 5.0
        return torch.full((input.size(0), weight_t.size(0), input.size(2)), fill, dtype=torch.float32)

    def fake_recompose(out_w0, out_w1, *, batch_size, lmax, mmax):
        calls.append(("recompose", tuple(out_w0.shape), tuple(out_w1.shape), batch_size, lmax, mmax))
        return torch.zeros(batch_size, 2, lmax, mmax, dtype=torch.complex64)

    monkeypatch.setattr(holysht, "_stack_vector_forward_real_inputs", fake_pack)
    monkeypatch.setattr(holysht, "_autotuned_direct_legendre_forward_real", fake_forward)
    monkeypatch.setattr(holysht, "_recompose_vector_forward_real_outputs", fake_recompose)

    x = torch.randn(2, 2, 3, 4, dtype=torch.complex64)
    w0 = torch.randn(5, 3, 4)
    w1 = torch.randn(5, 3, 4)

    out = holysht._run_vector_forward_stacked_real_fp32(
        x,
        w0,
        w1,
        lmax=5,
        mmax=4,
        tma_mmax_real=4,
        weight0_t_tma_real=w0,
        weight1_t_tma_real=w1,
    )

    assert out.shape == (2, 2, 5, 4)
    assert [item[0] for item in calls] == ["pack", "forward", "forward", "recompose"]


@pytest.mark.cpu_ok
def test_readme_covers_forward_only_autotune_policy():
    readme = Path(os.path.join(os.path.dirname(__file__), "..", "README.md")).read_text()
    assert "forward-only" in readme.lower()
    assert "HOLYSHT_FORCE_BACKEND" in readme
    assert "HOLYSHT_FORCE_VECTOR_STRATEGY" in readme
    assert "HOLYSHT_AUTOTUNE" in readme
    assert "XDG_CACHE_HOME" in readme or "holysht_autotune_cache.json" in readme


def test_public_tma_padding_uses_tile_aligned_m_quantum():
    scalar_fp32 = RealSHT(256, 512)
    scalar_bf16 = RealSHT(256, 512, dtype="bf16")
    vector_fp32 = RealVectorSHT(256, 512)
    vector_bf16 = RealVectorSHT(256, 512, dtype="bf16")

    assert scalar_fp32._tma_mmax_complex == _aligned_mmax_for_tma_tile(scalar_fp32.mmax)
    assert scalar_fp32.weight_t_tma_complex.size(-1) == scalar_fp32._tma_mmax_complex
    assert scalar_bf16._tma_mmax_real == _aligned_mmax_for_tma_tile(scalar_bf16.mmax)
    assert scalar_bf16.weight_t_tma_real.size(-1) == scalar_bf16._tma_mmax_real

    assert vector_fp32._tma_mmax_complex == _aligned_mmax_for_tma_tile(vector_fp32.mmax)
    assert vector_fp32.w0_t_tma_complex.size(-1) == vector_fp32._tma_mmax_complex
    assert vector_fp32.w1_t_tma_complex.size(-1) == vector_fp32._tma_mmax_complex
    assert vector_bf16._tma_mmax_real == _aligned_mmax_for_tma_tile(vector_bf16.mmax)
    assert vector_bf16.w0_t_tma_real.size(-1) == vector_bf16._tma_mmax_real
    assert vector_bf16.w1_t_tma_real.size(-1) == vector_bf16._tma_mmax_real


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


@pytest.mark.parametrize("nlat,nlon", [(256, 512)])
def test_large_grid_scalar_forward_upper_triangle_is_zero(nlat, nlon):
    torch.manual_seed(7)
    opt_sht = _to_test_device(RealSHT(nlat, nlon))

    x = torch.randn(2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        coeffs = opt_sht(x)

    mask = upper_triangle_mask(opt_sht.lmax, opt_sht.mmax, coeffs.device)
    assert coeffs[:, mask].abs().max().item() == 0.0


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


@pytest.mark.parametrize("nlat,nlon", [(256, 512)])
def test_large_grid_vector_forward_upper_triangle_is_zero(nlat, nlon):
    torch.manual_seed(7)
    opt_vsht = _to_test_device(RealVectorSHT(nlat, nlon))

    x = torch.randn(2, 2, nlat, nlon, device=DEVICE)
    with torch.no_grad():
        coeffs = opt_vsht(x)

    mask = upper_triangle_mask(opt_vsht.lmax, opt_vsht.mmax, coeffs.device)
    assert coeffs[:, :, mask].abs().max().item() == 0.0


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


@pytest.mark.skipif(not HAS_CUDA or CUDA_MAJOR < 9, reason="TMA path requires CUDA SM90+")
def test_cuda_tma_large_forward_paths_match_reference():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env = os.environ.copy()
    env["HOLYSHT_USE_TMA"] = "1"
    env["HOLYSHT_TMA_BATCH_TILE"] = "2"
    env["PYTHONPATH"] = os.path.join(repo_root, "torch-ext") + os.pathsep + env.get("PYTHONPATH", "")

    script = r"""
import torch
from holysht import RealSHT, RealVectorSHT, _cuda_tma_batch_tile
from torch_harmonics import RealSHT as RefRealSHT
from torch_harmonics import RealVectorSHT as RefRealVectorSHT

device = "cuda"
if _cuda_tma_batch_tile() != 2:
    raise SystemExit("expected HOLYSHT_TMA_BATCH_TILE=2 to be visible to the runtime")
cases = [
    ("scalar_fp32", lambda: RefRealSHT(256, 512).to(device), lambda: RealSHT(256, 512).to(device), (4, 256, 512), 1e-3),
    ("scalar_bf16", lambda: RefRealSHT(256, 512).to(device), lambda: RealSHT(256, 512, dtype="bf16").to(device), (4, 256, 512), 5e-3),
    ("vector_fp32", lambda: RefRealVectorSHT(256, 512).to(device), lambda: RealVectorSHT(256, 512).to(device), (4, 2, 256, 512), 1e-3),
    ("scalar_fp32_tail", lambda: RefRealSHT(256, 512).to(device), lambda: RealSHT(256, 512).to(device), (3, 256, 512), 1e-3),
]

for name, ref_factory, opt_factory, shape, atol in cases:
    torch.manual_seed(0)
    x = torch.randn(*shape, device=device)
    ref = ref_factory()
    opt = opt_factory()
    if opt.mmax % 8 == 0:
        raise SystemExit(f"{name} unexpectedly has aligned mmax={opt.mmax}; test no longer exercises the padded TMA public path")
    with torch.no_grad():
        y_ref = ref(x)
        y_opt = opt(x)
    err = (y_opt - y_ref).abs().max().item()
    if err >= atol:
        raise SystemExit(f"{name} max error {err} >= {atol}")
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
def test_direct_real_forward_accepts_backend_hint():
    weight_t = torch.randn(32, 16, 17, device="cuda")
    x = torch.randn(2, 16, 17, device="cuda")

    out_auto = holysht._direct_legendre_forward_real(x, weight_t)
    out_fma = holysht._direct_legendre_forward_real(
        x,
        weight_t,
        backend_hint=holysht._ForwardBackend.FMA,
    )

    assert out_auto.shape == out_fma.shape


@pytest.mark.skipif(not HAS_CUDA, reason="CUDA required")
def test_direct_vector_forward_accepts_backend_hint():
    weight0_t = torch.randn(32, 16, 17, device="cuda")
    weight1_t = torch.randn(32, 16, 17, device="cuda")
    x = torch.randn(2, 2, 16, 17, device="cuda", dtype=torch.complex64)

    out = holysht._direct_vector_legendre_forward(
        x,
        weight0_t,
        weight1_t,
        backend_hint=holysht._ForwardBackend.FMA,
    )

    assert out.shape == (2, 2, 32, 17)


@pytest.mark.skipif(not HAS_CUDA or CUDA_MAJOR < 9, reason="SM90+ CUDA required")
def test_forced_tensor_core_direct_vector_forward_matches_reference():
    torch.manual_seed(0)
    weight0_t = torch.randn(256, 256, 257, device="cuda")
    weight1_t = torch.randn(256, 256, 257, device="cuda")
    mask = upper_triangle_mask(weight0_t.shape[0], weight0_t.shape[2], weight0_t.device).unsqueeze(1)
    weight0_t = weight0_t.masked_fill(mask, 0.0)
    weight1_t = weight1_t.masked_fill(mask, 0.0)
    x = torch.randn(16, 2, 256, 257, device="cuda", dtype=torch.complex64)

    out = holysht._direct_vector_legendre_forward(
        x,
        weight0_t,
        weight1_t,
        backend_hint=holysht._ForwardBackend.TC_TF32,
    )
    ref = vector_forward_reference(x, weight0_t, weight1_t)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-3)


@pytest.mark.skipif(not HAS_CUDA or CUDA_MAJOR < 9, reason="SM90+ CUDA required")
def test_forced_tensor_core_direct_vector_forward_bf16_matches_reference():
    torch.manual_seed(0)
    weight0_t = torch.randn(256, 256, 257, device="cuda")
    weight1_t = torch.randn(256, 256, 257, device="cuda")
    mask = upper_triangle_mask(weight0_t.shape[0], weight0_t.shape[2], weight0_t.device).unsqueeze(1)
    weight0_t = weight0_t.masked_fill(mask, 0.0)
    weight1_t = weight1_t.masked_fill(mask, 0.0)
    x = torch.randn(4, 2, 256, 257, device="cuda", dtype=torch.complex64)

    out = holysht._direct_vector_legendre_forward(
        x,
        weight0_t,
        weight1_t,
        backend_hint=holysht._ForwardBackend.TC_BF16,
    )
    ref = vector_forward_reference(x, weight0_t, weight1_t)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=2e-1, rtol=2e-2)


@pytest.mark.skipif(not HAS_CUDA or CUDA_MAJOR < 9, reason="SM90+ CUDA required")
def test_forced_tensor_core_scalar_forward_matches_reference():
    torch.manual_seed(0)
    weight_t = torch.randn(256, 256, 257, device="cuda")
    weight_t = weight_t.masked_fill(
        upper_triangle_mask(weight_t.shape[0], weight_t.shape[2], weight_t.device).unsqueeze(1),
        0.0,
    )
    x = torch.randn(16, 256, 257, device="cuda")

    out = holysht._direct_legendre_forward_real(
        x,
        weight_t,
        backend_hint=holysht._ForwardBackend.TC_TF32,
    )
    ref = torch.einsum("bnm,lnm->blm", x, weight_t)

    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=5e-3)


@pytest.mark.skipif(not HAS_CUDA or CUDA_MAJOR < 9, reason="SM90+ CUDA required")
def test_forced_tensor_core_scalar_module_forward_matches_reference(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_tf32")
    torch.manual_seed(0)
    ref = _to_test_device(RefRealSHT(256, 512))
    opt = _to_test_device(RealSHT(256, 512))
    x = torch.randn(16, 256, 512, device="cuda")

    with torch.no_grad():
        y_ref = ref(x)
        y_opt = opt(x)

    assert y_opt.shape == y_ref.shape
    assert (y_opt - y_ref).abs().max().item() < COEFF_ATOL


@pytest.mark.skipif(not HAS_CUDA or CUDA_MAJOR < 9, reason="SM90+ CUDA required")
def test_forced_tensor_core_vector_forward_matches_reference(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_tf32")
    monkeypatch.setenv("HOLYSHT_FORCE_VECTOR_STRATEGY", "native")
    torch.manual_seed(0)
    ref = _to_test_device(RefRealVectorSHT(256, 512))
    opt = _to_test_device(RealVectorSHT(256, 512))
    x = torch.randn(16, 2, 256, 512, device="cuda")

    with torch.no_grad():
        y_ref = ref(x)
        y_opt = opt(x)

    assert y_opt.shape == y_ref.shape
    assert (y_opt - y_ref).abs().max().item() < COEFF_ATOL


@pytest.mark.skipif(not HAS_CUDA or CUDA_MAJOR < 9, reason="SM90+ CUDA required")
def test_forced_tensor_core_vector_forward_bf16_matches_reference(monkeypatch):
    monkeypatch.setenv("HOLYSHT_FORCE_BACKEND", "tc_bf16")
    monkeypatch.setenv("HOLYSHT_FORCE_VECTOR_STRATEGY", "native")
    torch.manual_seed(0)
    ref = _to_test_device(RefRealVectorSHT(256, 512))
    opt = _to_test_device(RealVectorSHT(256, 512, dtype="bf16"))
    x = torch.randn(2, 2, 256, 512, device="cuda")

    with torch.no_grad():
        y_ref = ref(x)
        y_opt = opt(x)

    assert (y_opt - y_ref).abs().max().item() < COEFF_ATOL


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "op_kind,input_shape,weight_shape,padded_shape",
    [
        ("scalar-real-forward", (2, 4, 5), (8, 4, 5), (2, 4, 8)),
        ("vector-real-forward", (4, 4, 5), (8, 4, 5), (4, 4, 8)),
    ],
)
def test_autotuned_real_forward_routes_selected_tma_path(monkeypatch, op_kind, input_shape, weight_shape, padded_shape):
    calls = []
    selector_calls = []

    def fake_select(key, candidates, benchmark, cache=None):
        selector_calls.append((key, tuple(candidates)))
        return "tma"

    def fake_direct(input, weight_t, backend_hint=holysht._ForwardBackend.AUTO):
        calls.append((tuple(input.shape), tuple(weight_t.shape), backend_hint))
        return torch.zeros(input.size(0), weight_t.size(0), input.size(2), dtype=torch.float32)

    monkeypatch.setattr(holysht, "_select_forward_backend_for_key", fake_select)
    monkeypatch.setattr(holysht, "_direct_legendre_forward_real", fake_direct)

    x = torch.randn(*input_shape)
    weight_t = torch.randn(*weight_shape)
    padded_x = torch.randn(*padded_shape)
    padded_weight_t = torch.randn(weight_shape[0], weight_shape[1], padded_shape[-1])

    out = holysht._autotuned_direct_legendre_forward_real(
        x,
        weight_t,
        op_kind=op_kind,
        dtype_mode="bf16",
        backend_candidates=["fma", "tma", "tc_bf16"],
        padded_input=padded_x,
        padded_weight_t=padded_weight_t,
        device_name="test-gpu",
        capability="9.0",
    )

    assert selector_calls and selector_calls[0][1] == ("fma", "tma", "tc_bf16")
    assert calls == [(tuple(padded_x.shape), tuple(padded_weight_t.shape), holysht._ForwardBackend.TMA)]
    assert out.shape == (input_shape[0], weight_shape[0], padded_shape[-1])


@pytest.mark.cpu_ok
def test_autotuned_real_forward_uses_explicit_tc_bf16_fallback(monkeypatch):
    calls = []

    def fake_select(key, candidates, benchmark, cache=None):
        assert tuple(candidates) == ("fma", "tma", "tc_bf16")
        return "tc_bf16"

    def fake_direct(input, weight_t, backend_hint=holysht._ForwardBackend.AUTO):
        calls.append((tuple(input.shape), tuple(weight_t.shape), backend_hint))
        return torch.zeros(input.size(0), weight_t.size(0), input.size(2), dtype=torch.float32)

    monkeypatch.setattr(holysht, "_select_forward_backend_for_key", fake_select)
    monkeypatch.setattr(holysht, "_direct_legendre_forward_real", fake_direct)

    x = torch.randn(2, 4, 5)
    weight_t = torch.randn(8, 4, 5)

    out = holysht._autotuned_direct_legendre_forward_real(
        x,
        weight_t,
        op_kind="scalar-real-forward",
        dtype_mode="bf16",
        backend_candidates=["fma", "tma", "tc_bf16"],
        device_name="test-gpu",
        capability="9.0",
    )

    assert calls == [(tuple(x.shape), tuple(weight_t.shape), holysht._ForwardBackend.TC_BF16)]
    assert out.shape == (2, 8, 5)


def test_direct_real_forward_uses_legacy_op_off_cuda(monkeypatch):
    calls = []

    class FakeOps:
        def fused_legendre_forward_real(self, output, input, weight_t):
            calls.append("legacy")
            return output

        def fused_legendre_forward_real_ex(self, output, input, weight_t, backend_hint):
            calls.append("ex")
            return output

    monkeypatch.setattr(holysht, "_ops", FakeOps())

    weight_t = torch.randn(8, 4, 5)
    x = torch.randn(2, 4, 5)

    out = holysht._direct_legendre_forward_real(
        x,
        weight_t,
        backend_hint=holysht._ForwardBackend.FMA,
    )

    assert out.shape == (2, 8, 5)
    assert calls == ["legacy"]


def test_direct_vector_forward_uses_legacy_op_off_cuda(monkeypatch):
    calls = []

    class FakeOps:
        def fused_vector_legendre_forward(self, output, input, weight0_t, weight1_t):
            calls.append("legacy")
            return output

        def fused_vector_legendre_forward_ex(self, output, input, weight0_t, weight1_t, backend_hint):
            calls.append("ex")
            return output

    monkeypatch.setattr(holysht, "_ops", FakeOps())

    weight0_t = torch.randn(8, 4, 5)
    weight1_t = torch.randn(8, 4, 5)
    x = torch.randn(2, 2, 4, 5, dtype=torch.complex64)

    out = holysht._direct_vector_legendre_forward(
        x,
        weight0_t,
        weight1_t,
        backend_hint=holysht._ForwardBackend.FMA,
    )

    assert out.shape == (2, 2, 8, 5)
    assert calls == ["legacy"]


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
