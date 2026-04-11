// HOLYSHT: SHT helper kernels
// SPDX-License-Identifier: MIT
// Author: Chris von Csefalvay
// Repository: https://github.com/chrisvoncsefalvay/holysht
// Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
//
// This file contains the prepare_irfft kernel used before the inverse real FFT
// in the inverse SHT pipeline. The main Legendre work lives in fused_legendre.cu.
//
// After inverse Legendre, before irfft we need to:
//   1. Zero out imaginary part of DC mode (m=0)
//   2. Zero out imaginary part of Nyquist mode (if nlon is even)
//   3. Zero-pad from mmax to nlon//2+1 if needed
//
// This kernel does all three in one pass, avoiding separate Python-side operations.
//
// Optimisation techniques:
//   - __launch_bounds__ for register pressure control and occupancy targeting
//   - Grid-stride loop for robustness across all grid sizes

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

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
// C++ dispatch wrapper
// =============================================================================

void sht_prepare_irfft(
    torch::Tensor& data,
    const int64_t mmax,
    const int64_t nlon
) {
    TORCH_CHECK(data.is_cuda() && data.is_contiguous());
    TORCH_CHECK(data.scalar_type() == torch::kComplexFloat);
    TORCH_CHECK(data.dim() == 3, "data must be rank 3");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(data));
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
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
