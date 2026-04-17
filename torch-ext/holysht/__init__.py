"""HOLYSHT: Highly Optimised Legendre/Ylm/SHT.

CUDA-accelerated spherical harmonic transforms designed as a practical,
production-oriented companion to torch-harmonics.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

import contextlib
import os
from typing import Optional
import torch
import torch.nn as nn

__all__ = [
    "RealSHT",
    "InverseRealSHT",
    "RealVectorSHT",
    "InverseRealVectorSHT",
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_bf16:
            with _nvtx_range("holysht.scalar_forward_bf16"):
                x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
                x_fft = x_fft[..., :self.mmax]
                xr = torch.view_as_real(x_fft)
                if _HAS_NATIVE_EXT and x.is_cuda and not x.requires_grad:
                    xr_bf16 = xr.bfloat16().contiguous()
                    out_re = fused_legendre_forward_real(xr_bf16[..., 0].contiguous(), self.weight_t)
                    out_im = fused_legendre_forward_real(xr_bf16[..., 1].contiguous(), self.weight_t)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.nlat and x.shape[-1] == self.nlon

        with _nvtx_range("holysht.vector_forward"):
            x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
            mmax = self.mmax
            x = x[..., :mmax].contiguous()

            if (not self._use_bf16 and not self._use_fp16) and (_can_use_cuda_vector(x, self.w0_t, self.w1_t) or _can_use_metal_vector(x, self.w0_t, self.w1_t)):
                B_shape = x.shape[:-3]
                x_flat = x.reshape(-1, 2, self.nlat, mmax).contiguous()
                out = _FusedVectorLegendreForwardFn.apply(x_flat, self.w0_t, self.w1_t)
                return out.reshape(B_shape + (2, self.lmax, mmax))

            x = torch.view_as_real(x)  # [..., 2, nlat, mmax, 2]

            x00 = x[..., 0, :, :, 0]
            x01 = x[..., 0, :, :, 1]
            x10 = x[..., 1, :, :, 0]
            x11 = x[..., 1, :, :, 1]

            B_shape = x00.shape[:-2]
            x00_flat = x00.reshape(-1, self.nlat, mmax)
            x01_flat = x01.reshape(-1, self.nlat, mmax)
            x10_flat = x10.reshape(-1, self.nlat, mmax)
            x11_flat = x11.reshape(-1, self.nlat, mmax)
            B = x00_flat.shape[0]

            if self._use_fp16 and _HAS_NATIVE_EXT and x00_flat.device.type == "mps" and not x.requires_grad:
                stacked_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0).half().contiguous()
                out_w0 = fused_legendre_forward_real(stacked_w0, self.w0_t)
                r00, r01, r10, r11 = out_w0[:B], out_w0[B:2 * B], out_w0[2 * B:3 * B], out_w0[3 * B:]

                stacked_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0).half().contiguous()
                out_w1 = fused_legendre_forward_real(stacked_w1, self.w1_t)
                s11, s10, s01, s00 = out_w1[:B], out_w1[B:2 * B], out_w1[2 * B:3 * B], out_w1[3 * B:]
            elif self._use_bf16 and _HAS_NATIVE_EXT and x00_flat.is_cuda and not x.requires_grad:
                stacked_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0).bfloat16().contiguous()
                out_w0 = fused_legendre_forward_real(stacked_w0, self.w0_t)
                r00, r01, r10, r11 = out_w0[:B], out_w0[B:2 * B], out_w0[2 * B:3 * B], out_w0[3 * B:]

                stacked_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0).bfloat16().contiguous()
                out_w1 = fused_legendre_forward_real(stacked_w1, self.w1_t)
                s11, s10, s01, s00 = out_w1[:B], out_w1[B:2 * B], out_w1[2 * B:3 * B], out_w1[3 * B:]
            elif (not self._use_bf16 and not self._use_fp16) and _can_use_native_real_legendre(x00_flat, self.w0_t):
                stacked_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0).contiguous()
                out_w0 = fused_legendre_forward_real(stacked_w0, self.w0_t)
                r00, r01, r10, r11 = out_w0[:B], out_w0[B:2 * B], out_w0[2 * B:3 * B], out_w0[3 * B:]

                stacked_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0).contiguous()
                out_w1 = fused_legendre_forward_real(stacked_w1, self.w1_t)
                s11, s10, s01, s00 = out_w1[:B], out_w1[B:2 * B], out_w1[2 * B:3 * B], out_w1[3 * B:]
            else:
                _low_prec = self._use_bf16 or self._use_fp16
                stacked_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0)
                if self._use_bf16:
                    stacked_w0 = stacked_w0.bfloat16()
                elif self._use_fp16:
                    stacked_w0 = stacked_w0.half()
                out_w0 = torch.einsum("bkm,mlk->blm", stacked_w0, self.w0)
                if _low_prec:
                    out_w0 = out_w0.float()
                r00, r01, r10, r11 = out_w0[:B], out_w0[B:2 * B], out_w0[2 * B:3 * B], out_w0[3 * B:]

                stacked_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0)
                if self._use_bf16:
                    stacked_w1 = stacked_w1.bfloat16()
                elif self._use_fp16:
                    stacked_w1 = stacked_w1.half()
                out_w1 = torch.einsum("bkm,mlk->blm", stacked_w1, self.w1)
                if _low_prec:
                    out_w1 = out_w1.float()
                s11, s10, s01, s00 = out_w1[:B], out_w1[B:2 * B], out_w1[2 * B:3 * B], out_w1[3 * B:]

            sph_re = r00 - s11
            sph_im = r01 + s10
            tor_re = -s01 - r10
            tor_im = s00 - r11

            out_shape = list(B_shape) + [2, self.lmax, mmax, 2]
            xout = torch.empty(out_shape, dtype=x.dtype, device=x.device)
            xout[..., 0, :, :, 0] = sph_re.reshape(B_shape + (self.lmax, mmax))
            xout[..., 0, :, :, 1] = sph_im.reshape(B_shape + (self.lmax, mmax))
            xout[..., 1, :, :, 0] = tor_re.reshape(B_shape + (self.lmax, mmax))
            xout[..., 1, :, :, 1] = tor_im.reshape(B_shape + (self.lmax, mmax))
            return torch.view_as_complex(xout)


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


legendre_forward = fused_legendre_forward
legendre_inverse = fused_legendre_inverse
sht_forward = fused_sht_forward
sht_inverse = fused_sht_inverse

# Backwards-compatible aliases from the research prototype.
FusedRealSHT = RealSHT
FusedInverseRealSHT = InverseRealSHT
FusedRealVectorSHT = RealVectorSHT
FusedInverseRealVectorSHT = InverseRealVectorSHT
