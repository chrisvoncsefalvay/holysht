"""HOLYSHT: Highly Optimised Legendre/Ylm/SHT.

CUDA-accelerated spherical harmonic transforms designed as a practical,
production-oriented companion to torch-harmonics.
"""

from typing import Optional
import torch
import torch.nn as nn

__all__ = [
    "legendre_forward",
    "legendre_inverse",
    "sht_forward",
    "sht_inverse",
    "RealSHT",
    "InverseRealSHT",
    "RealVectorSHT",
    "InverseRealVectorSHT",
    "fused_legendre_forward",
    "fused_legendre_inverse",
    "fused_sht_forward",
    "fused_sht_inverse",
    "FusedRealSHT",
    "FusedInverseRealSHT",
    "FusedRealVectorSHT",
    "FusedInverseRealVectorSHT",
]

# Try to load compiled CUDA extension
try:
    from ._ops import ops as _ops
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False


def _can_use_cuda_legendre(input: torch.Tensor, weight_t: Optional[torch.Tensor]) -> bool:
    return (
        _HAS_CUDA_EXT
        and weight_t is not None
        and input.is_cuda
        and weight_t.is_cuda
        and input.dtype == torch.complex64
        and weight_t.dtype == torch.float32
        and weight_t.is_contiguous()
    )


def _mul_i(x: torch.Tensor) -> torch.Tensor:
    """Multiply a complex tensor by +i without promoting dtype."""
    return torch.complex(-x.imag, x.real)


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

    if _HAS_CUDA_EXT and out.is_cuda and out.dtype == torch.complex64:
        orig_shape = out.shape
        flat = out.reshape(-1, orig_shape[-2], orig_shape[-1]).contiguous()
        _ops.sht_prepare_irfft(flat, active_mmax, nlon)
        return flat.reshape(orig_shape)

    out[..., 0] = out[..., 0].real.to(torch.complex64)
    if nlon % 2 == 0 and nlon // 2 < out.size(-1):
        out[..., nlon // 2] = out[..., nlon // 2].real.to(torch.complex64)
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

    For small grids (nlat ≤ 128), uses a custom CUDA kernel.
    For larger grids, uses a single batched einsum (stacking re/im).
    """
    B = input.size(0)
    nlat = input.size(1)
    mmax = input.size(2)
    lmax = weights.size(1)

    if weight_t is None:
        weight_t = weights.float().permute(1, 2, 0).contiguous()

    if _can_use_cuda_legendre(input, weight_t):
        # Adaptive CUDA kernel: small-grid direct path + large-grid tiled path.
        return _FusedLegendreForwardFn.apply(input, weight_t)
    else:
        # Fallback: stacked einsum (1.9x speedup over reference 2x einsum)
        x = torch.view_as_real(input)  # [B, nlat, mmax, 2]
        x_stacked = torch.cat([x[..., 0], x[..., 1]], dim=0)  # [2B, nlat, mmax]
        w = weights.to(x_stacked.dtype)
        out_stacked = torch.einsum("bkm,mlk->blm", x_stacked, w)
        out_re = out_stacked[:B]
        out_im = out_stacked[B:]
        return torch.complex(out_re, out_im)


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

    if _can_use_cuda_legendre(input, pct_t):
        return _FusedLegendreInverseFn.apply(input, pct_t)
    else:
        x = torch.view_as_real(input)
        x_stacked = torch.cat([x[..., 0], x[..., 1]], dim=0)
        p = pct.to(x_stacked.dtype)
        out_stacked = torch.einsum("blm,mlk->bkm", x_stacked, p)
        return torch.complex(out_stacked[:B], out_stacked[B:])


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
    # Step 1: Real FFT along longitude
    x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")

    # Step 2: Slice to mmax wavenumbers
    x_fft = x_fft[..., :mmax]

    # Step 3: Fused Legendre transform (handles complex natively)
    return fused_legendre_forward(x_fft, weights, weight_t)


def fused_sht_inverse(
    coeffs: torch.Tensor,      # [B, lmax, mmax] complex64
    pct: torch.Tensor,         # [mmax, lmax, nlat] float32
    nlon: int,
    pct_t: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Complete fused inverse SHT: fused Legendre → zero-pad → irfft."""
    # Step 1: Fused inverse Legendre
    x = fused_legendre_inverse(coeffs, pct, pct_t)

    # Step 2: Zero-pad plus clean DC/Nyquist using the CUDA helper when available
    x = _prepare_irfft_input(x, nlon, coeffs.size(-1))

    # Step 3: Inverse real FFT
    return torch.fft.irfft(x, n=nlon, dim=-1, norm="forward")


# ============================================================================
# nn.Module wrappers (drop-in replacements for torch-harmonics)
# ============================================================================

class RealSHT(nn.Module):
    """Optimized drop-in replacement for ``torch_harmonics.RealSHT``.

    Args:
        dtype: Weight precision. ``"fp32"`` (default) or ``"bf16"``.
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
        w_dtype = torch.bfloat16 if self._use_bf16 else torch.float32
        self.register_buffer("weights", ref.weights.to(w_dtype))
        self.register_buffer("weight_t", ref.weights.float().permute(1, 2, 0).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_bf16:
            x_fft = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
            x_fft = x_fft[..., :self.mmax]
            xr = torch.view_as_real(x_fft)
            B = x.size(0)
            xs = torch.cat([xr[..., 0], xr[..., 1]], dim=0).bfloat16()
            out = torch.einsum("bkm,mlk->blm", xs, self.weights).float()
            return torch.complex(out[:B], out[B:])
        return fused_sht_forward(x, self.weights, self.mmax, self.weight_t)


class InverseRealSHT(nn.Module):
    """Optimized drop-in replacement for ``torch_harmonics.InverseRealSHT``."""

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
    """Optimized drop-in replacement for ``torch_harmonics.RealVectorSHT``.

    Reduces eight reference einsums to two composed Legendre passes on the
    default FP32 CUDA path.

    Args:
        dtype: Weight precision. ``"fp32"`` (default) or ``"bf16"``.
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
        w_dtype = torch.bfloat16 if self._use_bf16 else torch.float32
        self.register_buffer("w0", ref.weights[0].to(w_dtype))  # [mmax, lmax, nlat]
        self.register_buffer("w1", ref.weights[1].to(w_dtype))  # [mmax, lmax, nlat]
        self.register_buffer("w0_t", ref.weights[0].float().permute(1, 2, 0).contiguous())
        self.register_buffer("w1_t", ref.weights[1].float().permute(1, 2, 0).contiguous())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-2] == self.nlat and x.shape[-1] == self.nlon

        # rfft along longitude
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm="forward")
        if (not self._use_bf16) and _HAS_CUDA_EXT and x.is_cuda:
            mmax = self.mmax
            comp0 = x[..., 0, :, :mmax]
            comp1 = x[..., 1, :, :mmax]
            B_shape = comp0.shape[:-2]
            comp0_flat = comp0.reshape(-1, self.nlat, mmax).contiguous()
            comp1_flat = comp1.reshape(-1, self.nlat, mmax).contiguous()
            B = comp0_flat.shape[0]

            packed = torch.cat([comp0_flat, comp1_flat], dim=0)
            out_w0 = fused_legendre_forward(packed, self.w0.float(), self.w0_t)
            out_w1 = fused_legendre_forward(packed, self.w1.float(), self.w1_t)

            y0, y1 = out_w0[:B], out_w0[B:]
            z0, z1 = out_w1[:B], out_w1[B:]

            sph = y0 + _mul_i(z1)
            tor = _mul_i(z0) - y1

            sph = sph.reshape(B_shape + (self.lmax, mmax))
            tor = tor.reshape(B_shape + (self.lmax, mmax))
            return torch.stack((sph, tor), dim=-3)

        x = torch.view_as_real(x)  # [..., 2, nlat, nlon//2+1, 2]

        mmax = self.mmax

        # Extract the 4 input slices: x[..., comp, :, :mmax, re/im]
        x00 = x[..., 0, :, :mmax, 0]  # comp0, real
        x01 = x[..., 0, :, :mmax, 1]  # comp0, imag
        x10 = x[..., 1, :, :mmax, 0]  # comp1, real
        x11 = x[..., 1, :, :mmax, 1]  # comp1, imag

        # Stack all 4 inputs for W0: [4*B, nlat, mmax]
        B_shape = x00.shape[:-2]  # batch dimensions
        x00_flat = x00.reshape(-1, self.nlat, mmax)
        x01_flat = x01.reshape(-1, self.nlat, mmax)
        x10_flat = x10.reshape(-1, self.nlat, mmax)
        x11_flat = x11.reshape(-1, self.nlat, mmax)
        B = x00_flat.shape[0]

        # W0 einsums: x00, x01, x10, x11 (used in spheroidal_re, spheroidal_im, toroidal_re, toroidal_im)
        stacked_w0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0)
        if self._use_bf16:
            stacked_w0 = stacked_w0.bfloat16()
        out_w0 = torch.einsum("bkm,mlk->blm", stacked_w0, self.w0)
        if self._use_bf16:
            out_w0 = out_w0.float()
        r00, r01, r10, r11 = out_w0[:B], out_w0[B:2*B], out_w0[2*B:3*B], out_w0[3*B:]

        # W1 einsums: x11, x10, x01, x00 (used in spheroidal_re, spheroidal_im, toroidal_re, toroidal_im)
        stacked_w1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0)
        if self._use_bf16:
            stacked_w1 = stacked_w1.bfloat16()
        out_w1 = torch.einsum("bkm,mlk->blm", stacked_w1, self.w1)
        if self._use_bf16:
            out_w1 = out_w1.float()
        s11, s10, s01, s00 = out_w1[:B], out_w1[B:2*B], out_w1[2*B:3*B], out_w1[3*B:]

        # Combine with correct signs:
        # spheroidal_re = +einsum(x00, W0) - einsum(x11, W1)
        # spheroidal_im = +einsum(x01, W0) + einsum(x10, W1)
        # toroidal_re   = -einsum(x01, W1) - einsum(x10, W0)
        # toroidal_im   = +einsum(x00, W1) - einsum(x11, W0)
        sph_re = r00 - s11
        sph_im = r01 + s10
        tor_re = -s01 - r10
        tor_im = s00 - r11

        # Reconstruct output: [..., 2, lmax, mmax] complex
        out_shape = list(B_shape) + [2, self.lmax, mmax, 2]
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        xout[..., 0, :, :, 0] = sph_re.reshape(B_shape + (self.lmax, mmax))
        xout[..., 0, :, :, 1] = sph_im.reshape(B_shape + (self.lmax, mmax))
        xout[..., 1, :, :, 0] = tor_re.reshape(B_shape + (self.lmax, mmax))
        xout[..., 1, :, :, 1] = tor_im.reshape(B_shape + (self.lmax, mmax))

        return torch.view_as_complex(xout)


class InverseRealVectorSHT(nn.Module):
    """Optimized drop-in replacement for ``torch_harmonics.InverseRealVectorSHT``."""

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

        if _HAS_CUDA_EXT and x.is_cuda:
            sph = x[..., 0, :, :]
            tor = x[..., 1, :, :]
            B_shape = sph.shape[:-2]
            sph_flat = sph.reshape(-1, self.lmax, self.mmax).contiguous()
            tor_flat = tor.reshape(-1, self.lmax, self.mmax).contiguous()
            B = sph_flat.shape[0]

            packed = torch.cat([sph_flat, tor_flat], dim=0)
            out_d0 = fused_legendre_inverse(packed, self.d0, self.d0_t)
            out_d1 = fused_legendre_inverse(packed, self.d1, self.d1_t)

            y0, y1 = out_d0[:B], out_d0[B:]
            z0, z1 = out_d1[:B], out_d1[B:]

            comp0 = y0 + _mul_i(z1)
            comp1 = _mul_i(z0) - y1

            comp0 = comp0.reshape(B_shape + (self.nlat, self.mmax))
            comp1 = comp1.reshape(B_shape + (self.nlat, self.mmax))
            x_out = torch.stack((comp0, comp1), dim=-3)
            x_out = _prepare_irfft_input(x_out, self.nlon, self.mmax)
            return torch.fft.irfft(x_out, n=self.nlon, dim=-1, norm="forward")

        x = torch.view_as_real(x)  # [..., 2, lmax, mmax, 2]

        mmax = self.mmax

        # Extract 4 input slices
        x00 = x[..., 0, :, :, 0]  # comp0, real
        x01 = x[..., 0, :, :, 1]  # comp0, imag
        x10 = x[..., 1, :, :, 0]  # comp1, real
        x11 = x[..., 1, :, :, 1]  # comp1, imag

        B_shape = x00.shape[:-2]
        x00_flat = x00.reshape(-1, self.lmax, mmax)
        x01_flat = x01.reshape(-1, self.lmax, mmax)
        x10_flat = x10.reshape(-1, self.lmax, mmax)
        x11_flat = x11.reshape(-1, self.lmax, mmax)
        B = x00_flat.shape[0]

        # d0 einsums: x00, x01, x10, x11
        stacked_d0 = torch.cat([x00_flat, x01_flat, x10_flat, x11_flat], dim=0)
        out_d0 = torch.einsum("blm,mlk->bkm", stacked_d0, self.d0)
        r00, r01, r10, r11 = out_d0[:B], out_d0[B:2*B], out_d0[2*B:3*B], out_d0[3*B:]

        # d1 einsums: x11, x10, x01, x00
        stacked_d1 = torch.cat([x11_flat, x10_flat, x01_flat, x00_flat], dim=0)
        out_d1 = torch.einsum("blm,mlk->bkm", stacked_d1, self.d1)
        s11, s10, s01, s00 = out_d1[:B], out_d1[B:2*B], out_d1[2*B:3*B], out_d1[3*B:]

        # Combine with correct signs (same as forward):
        # srl = +einsum(x00, d0) - einsum(x11, d1)
        # sim = +einsum(x01, d0) + einsum(x10, d1)
        # trl = -einsum(x01, d1) - einsum(x10, d0)
        # tim = +einsum(x00, d1) - einsum(x11, d0)
        srl = r00 - s11
        sim = r01 + s10
        trl = -s01 - r10
        tim = s00 - r11

        # Reassemble
        out_k = self.nlat
        srl = srl.reshape(B_shape + (out_k, mmax))
        sim = sim.reshape(B_shape + (out_k, mmax))
        trl = trl.reshape(B_shape + (out_k, mmax))
        tim = tim.reshape(B_shape + (out_k, mmax))

        s = torch.stack((srl, sim), -1)
        t = torch.stack((trl, tim), -1)
        xs = torch.stack((s, t), -4)
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
