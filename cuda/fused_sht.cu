// HOLYSHT: SHT helper kernels
// Author: Chris von Csefalvay <chris@chrisvoncsefalvay.com>
//
// This kernel fuses the entire forward SHT pipeline into a single pass:
//   1. Real FFT along longitude (via cuFFT device API or register-level butterfly)
//   2. Legendre-Gauss quadrature (fused real+imaginary as complex)
//
// For the inverse SHT:
//   1. Legendre synthesis (fused complex)
//   2. Inverse real FFT along longitude
//   3. Zero imaginary parts for 0th and Nyquist modes
//
// This file intentionally stays small. The main Legendre work lives in
// fused_legendre.cu; here we keep only the small-grid helpers plus the
// preparation step needed before irfft.
//
// The key optimization is the FUSED COMPLEX LEGENDRE TRANSFORM, which operates
// directly on the complex FFT output without:
//   - view_as_real conversion
//   - .real/.imag extraction + contiguous copy
//   - separate einsum calls for real and imaginary parts
//
// Optimisation techniques:
//   - __launch_bounds__ for register pressure control and occupancy targeting
//   - __ldg() read-only cache path for all global memory reads
//   - Grid-stride loop in prepare_irfft for robustness and flexibility
//   - Partial loop unrolling in small-grid kernels for ILP
//
// For small grids (nlat <= 128), the dedicated kernels here still win on launch
// overhead. Large-grid CUDA kernels live in fused_legendre.cu.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Thread block dimensions for small-grid Legendre kernel
constexpr int SM_TILE_M = 32;    // Wavenumber tile (coalesced dim)
constexpr int SM_TILE_L = 8;     // Degree tile

// =============================================================================
// Small-grid complex Legendre (identical to fused_legendre.cu)
// Works best for nlat ≤ 128 where launch overhead matters
// =============================================================================

__launch_bounds__(256, 4)
__global__ void small_grid_legendre_forward(
    const float2* __restrict__ input,     // [batch, nlat, mmax] complex64
    const float*  __restrict__ weight_t,  // [lmax, nlat, mmax] transposed
    float2* __restrict__ output,          // [batch, lmax, mmax] complex64
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * SM_TILE_M + threadIdx.x;
    const int l = blockIdx.y * SM_TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || l >= lmax || b >= batch_size) return;

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    const long long in_base = (long long)b * nlat * mmax + m;
    const long long w_base = (long long)l * nlat * mmax + m;

    #pragma unroll 4
    for (int k = 0; k < nlat; k++) {
        const float w = __ldg(&weight_t[w_base + (long long)k * mmax]);
        const float2 v = __ldg(&input[in_base + (long long)k * mmax]);
        acc_re = fmaf(w, v.x, acc_re);
        acc_im = fmaf(w, v.y, acc_im);
    }

    output[(long long)b * lmax * mmax + (long long)l * mmax + m] = make_float2(acc_re, acc_im);
}

__launch_bounds__(256, 4)
__global__ void small_grid_legendre_inverse(
    const float2* __restrict__ input,
    const float*  __restrict__ weight_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * SM_TILE_M + threadIdx.x;
    const int k = blockIdx.y * SM_TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || k >= nlat || b >= batch_size) return;

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    const long long in_base = (long long)b * lmax * mmax + m;
    const long long w_off = (long long)k * mmax + m;

    #pragma unroll 4
    for (int l = 0; l < lmax; l++) {
        const float w = __ldg(&weight_t[(long long)l * nlat * mmax + w_off]);
        const float2 v = __ldg(&input[in_base + (long long)l * mmax]);
        acc_re = fmaf(w, v.x, acc_re);
        acc_im = fmaf(w, v.y, acc_im);
    }

    output[(long long)b * nlat * mmax + (long long)k * mmax + m] = make_float2(acc_re, acc_im);
}


// =============================================================================
// Fused complex-to-real handling for inverse SHT
// =============================================================================
// After inverse Legendre, we need to:
//   1. Zero out imaginary part of m=0 mode
//   2. Zero out imaginary part of Nyquist mode (if nlon is even)
//   3. Zero-pad from mmax to nlon//2+1 if mmax < nlon//2+1
// This kernel does all three in one pass, avoiding separate operations.
// Uses a grid-stride loop for robustness across all grid sizes.

__launch_bounds__(256, 8)
__global__ void prepare_irfft_inplace(
    float2* __restrict__ data,    // [batch, nlat, mmax_padded] complex64 (in-place)
    const int batch_size,
    const int nlat,
    const int mmax,               // actual mmax from Legendre output
    const int mmax_padded,        // nlon//2 + 1 (full rfft size)
    const int nlon
) {
    const int total = batch_size * nlat * mmax_padded;
    const int stride = gridDim.x * blockDim.x;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += stride) {
        const int m = idx % mmax_padded;
        const int tmp = idx / mmax_padded;
        const int k = tmp % nlat;
        const int b = tmp / nlat;

        float2 val;
        if (m < mmax) {
            val = data[(long long)b * nlat * mmax_padded + (long long)k * mmax_padded + m];
        } else {
            val = make_float2(0.0f, 0.0f);  // zero-pad
        }

        // Zero imaginary part of DC and Nyquist modes
        if (m == 0) val.y = 0.0f;
        if (nlon % 2 == 0 && m == nlon / 2) val.y = 0.0f;

        data[(long long)b * nlat * mmax_padded + (long long)k * mmax_padded + m] = val;
    }
}


// =============================================================================
// C++ dispatch wrappers
// =============================================================================

void sht_legendre_forward_small(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t
) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && weight_t.is_cuda());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight_t.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat);

    const int B = input.size(0);
    const int nlat = input.size(1);
    const int mmax = input.size(2);
    const int lmax = weight_t.size(0);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    dim3 grid((mmax + SM_TILE_M - 1) / SM_TILE_M, (lmax + SM_TILE_L - 1) / SM_TILE_L, B);
    dim3 block(SM_TILE_M, SM_TILE_L);

    small_grid_legendre_forward<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(input.data_ptr()),
        weight_t.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr()),
        B, nlat, lmax, mmax
    );
}

void sht_legendre_inverse_small(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t
) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && weight_t.is_cuda());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight_t.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat);

    const int B = input.size(0);
    const int lmax = input.size(1);
    const int mmax = input.size(2);
    const int nlat = weight_t.size(1);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    dim3 grid((mmax + SM_TILE_M - 1) / SM_TILE_M, (nlat + SM_TILE_L - 1) / SM_TILE_L, B);
    dim3 block(SM_TILE_M, SM_TILE_L);

    small_grid_legendre_inverse<<<grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(input.data_ptr()),
        weight_t.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr()),
        B, nlat, lmax, mmax
    );
}

void sht_prepare_irfft(
    torch::Tensor& data,
    const int mmax,
    const int nlon
) {
    TORCH_CHECK(data.is_cuda() && data.is_contiguous());
    TORCH_CHECK(data.scalar_type() == torch::kComplexFloat);

    const int B = data.size(0);
    const int nlat = data.size(1);
    const int mmax_padded = data.size(2);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const int total = B * nlat * mmax_padded;
    dim3 grid((total + 255) / 256);
    dim3 block(256);

    prepare_irfft_inplace<<<grid, block, 0, stream>>>(
        reinterpret_cast<float2*>(data.data_ptr()),
        B, nlat, mmax, mmax_padded, nlon
    );
}
