// HOLYSHT: highly optimised Legendre and vector SHT CUDA kernels
// SPDX-License-Identifier: MIT
// Author: Chris von Csefalvay
// Repository: https://github.com/chrisvoncsefalvay/holysht
// Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <cstdlib>

namespace {

constexpr int TILE_M = 32;
constexpr int TILE_K = 16;  // reduction tile size (decoupled from TILE_L)

struct LaunchConfig {
    int tile_l;
    int small_grid_threshold;
};

inline int env_int(const char* name, const int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    const long parsed = std::strtol(raw, nullptr, 10);
    return parsed > 0 ? static_cast<int>(parsed) : fallback;
}

inline LaunchConfig select_launch_config() {
    const int forced_tile_l = env_int("HOLYSHT_TILE_L", 0);
    const int forced_threshold = env_int("HOLYSHT_SMALL_GRID_THRESHOLD", 0);

    if (forced_tile_l == 4 || forced_tile_l == 8) {
        return {
            forced_tile_l,
            forced_threshold > 0 ? forced_threshold : 128,
        };
    }

    const auto* props = at::cuda::getCurrentDeviceProperties();
    if (props->major >= 12) {
        return {8, forced_threshold > 0 ? forced_threshold : 192};
    }
    if (props->major >= 9) {
        return {8, forced_threshold > 0 ? forced_threshold : 160};
    }
    return {4, forced_threshold > 0 ? forced_threshold : 128};
}

template <typename scalar_t>
__device__ inline float load_scalar(const scalar_t* ptr) {
    return static_cast<float>(*ptr);
}

template <>
__device__ inline float load_scalar<float>(const float* ptr) {
    return __ldg(ptr);
}

__device__ inline float2 load_complex(const float2* ptr) {
    return __ldg(ptr);
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int l = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || l >= lmax || b >= batch_size) {
        return;
    }

    const long long out_idx = (long long)b * lmax * mmax + (long long)l * mmax + m;
    if (m > l) {
        output[out_idx] = make_float2(0.0f, 0.0f);
        return;
    }

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const long long in_base = (long long)b * nlat * mmax + m;
    const long long w_base = (long long)l * nlat * mmax + m;

    #pragma unroll 4
    for (int k = 0; k < nlat; ++k) {
        const float w = __ldg(&weight_t[w_base + (long long)k * mmax]);
        const float2 v = load_complex(&input[in_base + (long long)k * mmax]);
        acc_re = fmaf(w, v.x, acc_re);
        acc_im = fmaf(w, v.y, acc_im);
    }

    output[out_idx] = make_float2(acc_re, acc_im);
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_large_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int l = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (m <= l);
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float2 sm_input[TILE_K][TILE_M + 1];

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const long long in_base = (long long)b * nlat * mmax + m;
    const long long w_base = (long long)l * nlat * mmax + m;

    // Load first tile cooperatively: all threads load TILE_K / TILE_L rows each
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        const int k_global = row;
        if (can_load && k_global < nlat) {
            sm_input[row][threadIdx.x] = load_complex(&input[in_base + (long long)k_global * mmax]);
        } else {
            sm_input[row][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        // Prefetch next tile into registers
        float2 prefetch[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_k = k0 + TILE_K + row;
            prefetch[pass] = (can_load && next_k < nlat)
                ? load_complex(&input[in_base + (long long)next_k * mmax])
                : make_float2(0.0f, 0.0f);
        }

        // Pre-load weights into registers, then compute from shmem + registers
        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (long long)k_idx * mmax]) : 0.0f;
            }
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const float2 v = sm_input[kk][threadIdx.x];
                acc_re = fmaf(w_reg[kk], v.x, acc_re);
                acc_im = fmaf(w_reg[kk], v.y, acc_im);
            }
        }

        // Store prefetched data to shared memory
        __syncthreads();
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            sm_input[row][threadIdx.x] = prefetch[pass];
        }
        __syncthreads();
    }

    if (valid_out) {
        const long long out_idx = (long long)b * lmax * mmax + (long long)l * mmax + m;
        output[out_idx] = active ? make_float2(acc_re, acc_im) : make_float2(0.0f, 0.0f);
    }
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_inverse_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int k = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || k >= nlat || b >= batch_size) {
        return;
    }

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const long long in_base = (long long)b * lmax * mmax + m;
    const long long w_offset = (long long)k * mmax + m;

    #pragma unroll 4
    for (int l = 0; l < lmax; ++l) {
        const float w = __ldg(&weight_t[(long long)l * nlat * mmax + w_offset]);
        const float2 v = load_complex(&input[in_base + (long long)l * mmax]);
        acc_re = fmaf(w, v.x, acc_re);
        acc_im = fmaf(w, v.y, acc_im);
    }

    const long long out_idx = (long long)b * nlat * mmax + (long long)k * mmax + m;
    output[out_idx] = make_float2(acc_re, acc_im);
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_inverse_large_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int k = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (k < nlat) && (m < mmax);
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float2 sm_input[TILE_K][TILE_M + 1];

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const long long in_base = (long long)b * lmax * mmax + m;
    const long long w_offset = (long long)k * mmax + m;

    // Load first tile cooperatively
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        if (can_load && row < lmax) {
            sm_input[row][threadIdx.x] = load_complex(&input[in_base + (long long)row * mmax]);
        } else {
            sm_input[row][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int l0 = 0; l0 < lmax; l0 += TILE_K) {
        // Prefetch next tile into registers
        float2 prefetch[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_l = l0 + TILE_K + row;
            prefetch[pass] = (can_load && next_l < lmax)
                ? load_complex(&input[in_base + (long long)next_l * mmax])
                : make_float2(0.0f, 0.0f);
        }

        // Pre-load weights into registers, then compute from shmem + registers
        if (valid_out) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                w_reg[ll] = (l_idx < lmax && l_idx >= m)
                    ? __ldg(&weight_t[(long long)l_idx * nlat * mmax + w_offset])
                    : 0.0f;
            }
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const float2 v = sm_input[ll][threadIdx.x];
                acc_re = fmaf(w_reg[ll], v.x, acc_re);
                acc_im = fmaf(w_reg[ll], v.y, acc_im);
            }
        }

        // Store prefetched data to shared memory
        __syncthreads();
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            sm_input[row][threadIdx.x] = prefetch[pass];
        }
        __syncthreads();
    }

    if (valid_out) {
        const long long out_idx = (long long)b * nlat * mmax + (long long)k * mmax + m;
        output[out_idx] = make_float2(acc_re, acc_im);
    }
}

template <typename scalar_t, int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_real_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight_t,
    float* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int l = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || l >= lmax || b >= batch_size) {
        return;
    }

    const long long out_idx = (long long)b * lmax * mmax + (long long)l * mmax + m;
    if (m > l) {
        output[out_idx] = 0.0f;
        return;
    }

    float acc = 0.0f;
    const long long in_base = (long long)b * nlat * mmax + m;
    const long long w_base = (long long)l * nlat * mmax + m;

    #pragma unroll 4
    for (int k = 0; k < nlat; ++k) {
        const float w = __ldg(&weight_t[w_base + (long long)k * mmax]);
        const float v = load_scalar(&input[in_base + (long long)k * mmax]);
        acc = fmaf(w, v, acc);
    }

    output[out_idx] = acc;
}

template <typename scalar_t, int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_real_large_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight_t,
    float* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int l = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (m <= l);
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float sm_input[TILE_K][TILE_M + 1];

    float acc = 0.0f;
    const long long in_base = (long long)b * nlat * mmax + m;
    const long long w_base = (long long)l * nlat * mmax + m;

    // Load first tile cooperatively
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        if (can_load && row < nlat) {
            sm_input[row][threadIdx.x] = load_scalar(&input[in_base + (long long)row * mmax]);
        } else {
            sm_input[row][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        // Prefetch next tile into registers
        float prefetch[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_k = k0 + TILE_K + row;
            prefetch[pass] = (can_load && next_k < nlat)
                ? load_scalar(&input[in_base + (long long)next_k * mmax])
                : 0.0f;
        }

        // Pre-load weights into registers, then compute from shmem + registers
        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (long long)k_idx * mmax]) : 0.0f;
            }
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                acc = fmaf(w_reg[kk], sm_input[kk][threadIdx.x], acc);
            }
        }

        // Store prefetched data to shared memory
        __syncthreads();
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            sm_input[row][threadIdx.x] = prefetch[pass];
        }
        __syncthreads();
    }

    if (valid_out) {
        const long long out_idx = (long long)b * lmax * mmax + (long long)l * mmax + m;
        output[out_idx] = active ? acc : 0.0f;
    }
}

template <typename scalar_t, int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_inverse_real_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight_t,
    float* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int k = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || k >= nlat || b >= batch_size) {
        return;
    }

    float acc = 0.0f;
    const long long in_base = (long long)b * lmax * mmax + m;
    const long long w_offset = (long long)k * mmax + m;

    #pragma unroll 4
    for (int l = 0; l < lmax; ++l) {
        const float w = __ldg(&weight_t[(long long)l * nlat * mmax + w_offset]);
        const float v = load_scalar(&input[in_base + (long long)l * mmax]);
        acc = fmaf(w, v, acc);
    }

    const long long out_idx = (long long)b * nlat * mmax + (long long)k * mmax + m;
    output[out_idx] = acc;
}

template <typename scalar_t, int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_inverse_real_large_kernel(
    const scalar_t* __restrict__ input,
    const float* __restrict__ weight_t,
    float* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int k = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (k < nlat) && (m < mmax);
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float sm_input[TILE_K][TILE_M + 1];

    float acc = 0.0f;
    const long long in_base = (long long)b * lmax * mmax + m;
    const long long w_offset = (long long)k * mmax + m;

    // Load first tile cooperatively
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        if (can_load && row < lmax) {
            sm_input[row][threadIdx.x] = load_scalar(&input[in_base + (long long)row * mmax]);
        } else {
            sm_input[row][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    for (int l0 = 0; l0 < lmax; l0 += TILE_K) {
        // Prefetch next tile into registers
        float prefetch[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_l = l0 + TILE_K + row;
            prefetch[pass] = (can_load && next_l < lmax)
                ? load_scalar(&input[in_base + (long long)next_l * mmax])
                : 0.0f;
        }

        // Pre-load weights into registers, then compute from shmem + registers
        if (valid_out) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                w_reg[ll] = (l_idx < lmax && l_idx >= m)
                    ? __ldg(&weight_t[(long long)l_idx * nlat * mmax + w_offset])
                    : 0.0f;
            }
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                acc = fmaf(w_reg[ll], sm_input[ll][threadIdx.x], acc);
            }
        }

        // Store prefetched data to shared memory
        __syncthreads();
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            sm_input[row][threadIdx.x] = prefetch[pass];
        }
        __syncthreads();
    }

    if (valid_out) {
        const long long out_idx = (long long)b * nlat * mmax + (long long)k * mmax + m;
        output[out_idx] = acc;
    }
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_vector_legendre_forward_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight0_t,
    const float* __restrict__ weight1_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int l = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || l >= lmax || b >= batch_size) {
        return;
    }

    const long long out_base = (long long)b * 2 * lmax * mmax;
    const long long sph_idx = out_base + (long long)l * mmax + m;
    const long long tor_idx = out_base + (long long)lmax * mmax + (long long)l * mmax + m;

    if (m > l) {
        output[sph_idx] = make_float2(0.0f, 0.0f);
        output[tor_idx] = make_float2(0.0f, 0.0f);
        return;
    }

    float sph_re = 0.0f;
    float sph_im = 0.0f;
    float tor_re = 0.0f;
    float tor_im = 0.0f;

    const long long in_base = (long long)b * 2 * nlat * mmax;
    const long long w_base = (long long)l * nlat * mmax + m;

    #pragma unroll 4
    for (int k = 0; k < nlat; ++k) {
        const float2 comp0 = load_complex(&input[in_base + (long long)k * mmax + m]);
        const float2 comp1 = load_complex(&input[in_base + (long long)nlat * mmax + (long long)k * mmax + m]);
        const float w0 = __ldg(&weight0_t[w_base + (long long)k * mmax]);
        const float w1 = __ldg(&weight1_t[w_base + (long long)k * mmax]);

        sph_re = fmaf(w0, comp0.x, sph_re);
        sph_re = fmaf(-w1, comp1.y, sph_re);
        sph_im = fmaf(w0, comp0.y, sph_im);
        sph_im = fmaf(w1, comp1.x, sph_im);

        tor_re = fmaf(-w1, comp0.y, tor_re);
        tor_re = fmaf(-w0, comp1.x, tor_re);
        tor_im = fmaf(w1, comp0.x, tor_im);
        tor_im = fmaf(-w0, comp1.y, tor_im);
    }

    output[sph_idx] = make_float2(sph_re, sph_im);
    output[tor_idx] = make_float2(tor_re, tor_im);
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_vector_legendre_forward_large_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight0_t,
    const float* __restrict__ weight1_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int l = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (m <= l);
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float2 sm_comp0[TILE_K][TILE_M + 1];
    __shared__ float2 sm_comp1[TILE_K][TILE_M + 1];

    float sph_re = 0.0f;
    float sph_im = 0.0f;
    float tor_re = 0.0f;
    float tor_im = 0.0f;

    const long long in_base = (long long)b * 2 * nlat * mmax;
    const long long w_base = (long long)l * nlat * mmax + m;

    // Load first tile cooperatively
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        if (can_load && row < nlat) {
            sm_comp0[row][threadIdx.x] = load_complex(&input[in_base + (long long)row * mmax + m]);
            sm_comp1[row][threadIdx.x] = load_complex(&input[in_base + (long long)nlat * mmax + (long long)row * mmax + m]);
        } else {
            sm_comp0[row][threadIdx.x] = make_float2(0.0f, 0.0f);
            sm_comp1[row][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        // Prefetch next tile into registers
        float2 prefetch0[TILE_K / TILE_L];
        float2 prefetch1[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_k = k0 + TILE_K + row;
            if (can_load && next_k < nlat) {
                prefetch0[pass] = load_complex(&input[in_base + (long long)next_k * mmax + m]);
                prefetch1[pass] = load_complex(&input[in_base + (long long)nlat * mmax + (long long)next_k * mmax + m]);
            } else {
                prefetch0[pass] = make_float2(0.0f, 0.0f);
                prefetch1[pass] = make_float2(0.0f, 0.0f);
            }
        }

        // Compute from shmem (no weight pre-loading for vector kernels — register pressure)
        if (active) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                if (k_idx < nlat) {
                    const float2 comp0 = sm_comp0[kk][threadIdx.x];
                    const float2 comp1 = sm_comp1[kk][threadIdx.x];
                    const float w0 = __ldg(&weight0_t[w_base + (long long)k_idx * mmax]);
                    const float w1 = __ldg(&weight1_t[w_base + (long long)k_idx * mmax]);

                    sph_re = fmaf(w0, comp0.x, sph_re);
                    sph_re = fmaf(-w1, comp1.y, sph_re);
                    sph_im = fmaf(w0, comp0.y, sph_im);
                    sph_im = fmaf(w1, comp1.x, sph_im);

                    tor_re = fmaf(-w1, comp0.y, tor_re);
                    tor_re = fmaf(-w0, comp1.x, tor_re);
                    tor_im = fmaf(w1, comp0.x, tor_im);
                    tor_im = fmaf(-w0, comp1.y, tor_im);
                }
            }
        }

        // Store prefetched data to shared memory
        __syncthreads();
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            sm_comp0[row][threadIdx.x] = prefetch0[pass];
            sm_comp1[row][threadIdx.x] = prefetch1[pass];
        }
        __syncthreads();
    }

    if (valid_out) {
        const long long out_base = (long long)b * 2 * lmax * mmax;
        const long long sph_idx = out_base + (long long)l * mmax + m;
        const long long tor_idx = out_base + (long long)lmax * mmax + (long long)l * mmax + m;
        output[sph_idx] = active ? make_float2(sph_re, sph_im) : make_float2(0.0f, 0.0f);
        output[tor_idx] = active ? make_float2(tor_re, tor_im) : make_float2(0.0f, 0.0f);
    }
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_vector_legendre_inverse_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight0_t,
    const float* __restrict__ weight1_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int k = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || k >= nlat || b >= batch_size) {
        return;
    }

    float comp0_re = 0.0f;
    float comp0_im = 0.0f;
    float comp1_re = 0.0f;
    float comp1_im = 0.0f;

    const long long in_base = (long long)b * 2 * lmax * mmax;

    #pragma unroll 4
    for (int l = 0; l < lmax; ++l) {
        if (l < m) {
            continue;
        }

        const float2 sph = load_complex(&input[in_base + (long long)l * mmax + m]);
        const float2 tor = load_complex(&input[in_base + (long long)lmax * mmax + (long long)l * mmax + m]);
        const float w0 = __ldg(&weight0_t[(long long)l * nlat * mmax + (long long)k * mmax + m]);
        const float w1 = __ldg(&weight1_t[(long long)l * nlat * mmax + (long long)k * mmax + m]);

        comp0_re = fmaf(w0, sph.x, comp0_re);
        comp0_re = fmaf(-w1, tor.y, comp0_re);
        comp0_im = fmaf(w0, sph.y, comp0_im);
        comp0_im = fmaf(w1, tor.x, comp0_im);

        comp1_re = fmaf(-w1, sph.y, comp1_re);
        comp1_re = fmaf(-w0, tor.x, comp1_re);
        comp1_im = fmaf(w1, sph.x, comp1_im);
        comp1_im = fmaf(-w0, tor.y, comp1_im);
    }

    const long long out_base = (long long)b * 2 * nlat * mmax;
    const long long comp0_idx = out_base + (long long)k * mmax + m;
    const long long comp1_idx = out_base + (long long)nlat * mmax + (long long)k * mmax + m;
    output[comp0_idx] = make_float2(comp0_re, comp0_im);
    output[comp1_idx] = make_float2(comp1_re, comp1_im);
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_vector_legendre_inverse_large_kernel(
    const float2* __restrict__ input,
    const float* __restrict__ weight0_t,
    const float* __restrict__ weight1_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int k = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (k < nlat) && (m < mmax);
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float2 sm_sph[TILE_K][TILE_M + 1];
    __shared__ float2 sm_tor[TILE_K][TILE_M + 1];

    float comp0_re = 0.0f;
    float comp0_im = 0.0f;
    float comp1_re = 0.0f;
    float comp1_im = 0.0f;

    const long long in_base = (long long)b * 2 * lmax * mmax;
    const long long w_offset = (long long)k * mmax + m;

    // Load first tile cooperatively
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        if (can_load && row < lmax) {
            sm_sph[row][threadIdx.x] = load_complex(&input[in_base + (long long)row * mmax + m]);
            sm_tor[row][threadIdx.x] = load_complex(&input[in_base + (long long)lmax * mmax + (long long)row * mmax + m]);
        } else {
            sm_sph[row][threadIdx.x] = make_float2(0.0f, 0.0f);
            sm_tor[row][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int l0 = 0; l0 < lmax; l0 += TILE_K) {
        // Prefetch next tile into registers
        float2 prefetch_sph[TILE_K / TILE_L];
        float2 prefetch_tor[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_l = l0 + TILE_K + row;
            if (can_load && next_l < lmax) {
                prefetch_sph[pass] = load_complex(&input[in_base + (long long)next_l * mmax + m]);
                prefetch_tor[pass] = load_complex(&input[in_base + (long long)lmax * mmax + (long long)next_l * mmax + m]);
            } else {
                prefetch_sph[pass] = make_float2(0.0f, 0.0f);
                prefetch_tor[pass] = make_float2(0.0f, 0.0f);
            }
        }

        // Compute from shmem (no weight pre-loading for vector kernels — register pressure)
        if (valid_out) {
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                if ((l_idx < lmax) && (l_idx >= m)) {
                    const float2 sph = sm_sph[ll][threadIdx.x];
                    const float2 tor = sm_tor[ll][threadIdx.x];
                    const float w0 = __ldg(&weight0_t[(long long)l_idx * nlat * mmax + w_offset]);
                    const float w1 = __ldg(&weight1_t[(long long)l_idx * nlat * mmax + w_offset]);

                    comp0_re = fmaf(w0, sph.x, comp0_re);
                    comp0_re = fmaf(-w1, tor.y, comp0_re);
                    comp0_im = fmaf(w0, sph.y, comp0_im);
                    comp0_im = fmaf(w1, tor.x, comp0_im);

                    comp1_re = fmaf(-w1, sph.y, comp1_re);
                    comp1_re = fmaf(-w0, tor.x, comp1_re);
                    comp1_im = fmaf(w1, sph.x, comp1_im);
                    comp1_im = fmaf(-w0, tor.y, comp1_im);
                }
            }
        }

        // Store prefetched data to shared memory
        __syncthreads();
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            sm_sph[row][threadIdx.x] = prefetch_sph[pass];
            sm_tor[row][threadIdx.x] = prefetch_tor[pass];
        }
        __syncthreads();
    }

    if (valid_out) {
        const long long out_base = (long long)b * 2 * nlat * mmax;
        const long long comp0_idx = out_base + (long long)k * mmax + m;
        const long long comp1_idx = out_base + (long long)nlat * mmax + (long long)k * mmax + m;
        output[comp0_idx] = make_float2(comp0_re, comp0_im);
        output[comp1_idx] = make_float2(comp1_re, comp1_im);
    }
}

template <int TILE_L>
void launch_forward_complex(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const int batch_size = input.size(0);
    const int nlat = input.size(1);
    const int mmax = input.size(2);
    const int lmax = weight_t.size(0);

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (lmax + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= config.small_grid_threshold && lmax <= config.small_grid_threshold) {
        fused_legendre_forward_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_legendre_forward_large_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    }
}

template <int TILE_L>
void launch_inverse_complex(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const int batch_size = input.size(0);
    const int lmax = input.size(1);
    const int mmax = input.size(2);
    const int nlat = weight_t.size(1);

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (nlat + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= config.small_grid_threshold && lmax <= config.small_grid_threshold) {
        fused_legendre_inverse_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_legendre_inverse_large_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    }
}

template <typename scalar_t, int TILE_L>
void launch_forward_real(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const int batch_size = input.size(0);
    const int nlat = input.size(1);
    const int mmax = input.size(2);
    const int lmax = weight_t.size(0);

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (lmax + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= config.small_grid_threshold && lmax <= config.small_grid_threshold) {
        fused_legendre_forward_real_kernel<scalar_t, TILE_L><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight_t.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_legendre_forward_real_large_kernel<scalar_t, TILE_L><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight_t.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, nlat, lmax, mmax
        );
    }
}

template <typename scalar_t, int TILE_L>
void launch_inverse_real(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const int batch_size = input.size(0);
    const int lmax = input.size(1);
    const int mmax = input.size(2);
    const int nlat = weight_t.size(1);

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (nlat + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= config.small_grid_threshold && lmax <= config.small_grid_threshold) {
        fused_legendre_inverse_real_kernel<scalar_t, TILE_L><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight_t.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_legendre_inverse_real_large_kernel<scalar_t, TILE_L><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight_t.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, nlat, lmax, mmax
        );
    }
}

template <int TILE_L>
void launch_vector_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const int batch_size = input.size(0);
    const int nlat = input.size(2);
    const int mmax = input.size(3);
    const int lmax = weight0_t.size(0);

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (lmax + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= config.small_grid_threshold && lmax <= config.small_grid_threshold) {
        fused_vector_legendre_forward_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight0_t.data_ptr<float>(),
            weight1_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_vector_legendre_forward_large_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight0_t.data_ptr<float>(),
            weight1_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    }
}

template <int TILE_L>
void launch_vector_inverse(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const int batch_size = input.size(0);
    const int lmax = input.size(2);
    const int mmax = input.size(3);
    const int nlat = weight0_t.size(1);

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (nlat + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= config.small_grid_threshold && lmax <= config.small_grid_threshold) {
        fused_vector_legendre_inverse_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight0_t.data_ptr<float>(),
            weight1_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_vector_legendre_inverse_large_kernel<TILE_L><<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight0_t.data_ptr<float>(),
            weight1_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    }
}

void check_complex_legendre_args(
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const bool inverse
) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && weight_t.is_cuda());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight_t.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat, "input must be complex64");
    TORCH_CHECK(output.scalar_type() == torch::kComplexFloat, "output must be complex64");
    TORCH_CHECK(weight_t.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be rank 3");
    TORCH_CHECK(output.dim() == 3, "output must be rank 3");
    TORCH_CHECK(weight_t.dim() == 3, "weight_t must be rank 3");

    if (!inverse) {
        const int batch_size = input.size(0);
        const int nlat = input.size(1);
        const int mmax = input.size(2);
        const int lmax = weight_t.size(0);
        TORCH_CHECK(weight_t.size(1) == nlat, "weight dim 1 must match nlat");
        TORCH_CHECK(weight_t.size(2) == mmax, "weight dim 2 must match mmax");
        TORCH_CHECK(output.size(0) == batch_size);
        TORCH_CHECK(output.size(1) == lmax);
        TORCH_CHECK(output.size(2) == mmax);
    } else {
        const int batch_size = input.size(0);
        const int lmax = input.size(1);
        const int mmax = input.size(2);
        const int nlat = weight_t.size(1);
        TORCH_CHECK(weight_t.size(0) == lmax, "weight dim 0 must match lmax");
        TORCH_CHECK(weight_t.size(2) == mmax, "weight dim 2 must match mmax");
        TORCH_CHECK(output.size(0) == batch_size);
        TORCH_CHECK(output.size(1) == nlat);
        TORCH_CHECK(output.size(2) == mmax);
    }
}

void check_real_legendre_args(
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const bool inverse
) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && weight_t.is_cuda());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight_t.is_contiguous());
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kBFloat16,
        "input must be float32 or bfloat16"
    );
    TORCH_CHECK(output.scalar_type() == torch::kFloat32, "output must be float32");
    TORCH_CHECK(weight_t.scalar_type() == torch::kFloat32, "weight must be float32");
    TORCH_CHECK(input.dim() == 3, "input must be rank 3");
    TORCH_CHECK(output.dim() == 3, "output must be rank 3");
    TORCH_CHECK(weight_t.dim() == 3, "weight_t must be rank 3");

    if (!inverse) {
        const int batch_size = input.size(0);
        const int nlat = input.size(1);
        const int mmax = input.size(2);
        const int lmax = weight_t.size(0);
        TORCH_CHECK(weight_t.size(1) == nlat, "weight dim 1 must match nlat");
        TORCH_CHECK(weight_t.size(2) == mmax, "weight dim 2 must match mmax");
        TORCH_CHECK(output.size(0) == batch_size);
        TORCH_CHECK(output.size(1) == lmax);
        TORCH_CHECK(output.size(2) == mmax);
    } else {
        const int batch_size = input.size(0);
        const int lmax = input.size(1);
        const int mmax = input.size(2);
        const int nlat = weight_t.size(1);
        TORCH_CHECK(weight_t.size(0) == lmax, "weight dim 0 must match lmax");
        TORCH_CHECK(weight_t.size(2) == mmax, "weight dim 2 must match mmax");
        TORCH_CHECK(output.size(0) == batch_size);
        TORCH_CHECK(output.size(1) == nlat);
        TORCH_CHECK(output.size(2) == mmax);
    }
}

void check_vector_legendre_args(
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const bool inverse
) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && weight0_t.is_cuda() && weight1_t.is_cuda());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight0_t.is_contiguous() && weight1_t.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat, "input must be complex64");
    TORCH_CHECK(output.scalar_type() == torch::kComplexFloat, "output must be complex64");
    TORCH_CHECK(weight0_t.scalar_type() == torch::kFloat32, "weight0_t must be float32");
    TORCH_CHECK(weight1_t.scalar_type() == torch::kFloat32, "weight1_t must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be rank 4");
    TORCH_CHECK(output.dim() == 4, "output must be rank 4");
    TORCH_CHECK(input.size(1) == 2, "input component dimension must be 2");
    TORCH_CHECK(output.size(1) == 2, "output component dimension must be 2");
    TORCH_CHECK(weight0_t.sizes() == weight1_t.sizes(), "weight tensors must match");

    if (!inverse) {
        const int batch_size = input.size(0);
        const int nlat = input.size(2);
        const int mmax = input.size(3);
        const int lmax = weight0_t.size(0);
        TORCH_CHECK(weight0_t.size(1) == nlat, "weight dim 1 must match nlat");
        TORCH_CHECK(weight0_t.size(2) == mmax, "weight dim 2 must match mmax");
        TORCH_CHECK(output.size(0) == batch_size);
        TORCH_CHECK(output.size(2) == lmax);
        TORCH_CHECK(output.size(3) == mmax);
    } else {
        const int batch_size = input.size(0);
        const int lmax = input.size(2);
        const int mmax = input.size(3);
        const int nlat = weight0_t.size(1);
        TORCH_CHECK(weight0_t.size(0) == lmax, "weight dim 0 must match lmax");
        TORCH_CHECK(weight0_t.size(2) == mmax, "weight dim 2 must match mmax");
        TORCH_CHECK(output.size(0) == batch_size);
        TORCH_CHECK(output.size(2) == nlat);
        TORCH_CHECK(output.size(3) == mmax);
    }
}

}  // namespace

void fused_legendre_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t
) {
    check_complex_legendre_args(output, input, weight_t, false);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();

    if (config.tile_l == 8) {
        launch_forward_complex<8>(output, input, weight_t, config, stream);
    } else {
        launch_forward_complex<4>(output, input, weight_t, config, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_legendre_inverse(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t
) {
    check_complex_legendre_args(output, input, weight_t, true);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();

    if (config.tile_l == 8) {
        launch_inverse_complex<8>(output, input, weight_t, config, stream);
    } else {
        launch_inverse_complex<4>(output, input, weight_t, config, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_legendre_forward_real(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t
) {
    check_real_legendre_args(output, input, weight_t, false);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();

    if (input.scalar_type() == torch::kFloat32) {
        if (config.tile_l == 8) {
            launch_forward_real<float, 8>(output, input, weight_t, config, stream);
        } else {
            launch_forward_real<float, 4>(output, input, weight_t, config, stream);
        }
    } else {
        if (config.tile_l == 8) {
            launch_forward_real<at::BFloat16, 8>(output, input, weight_t, config, stream);
        } else {
            launch_forward_real<at::BFloat16, 4>(output, input, weight_t, config, stream);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_legendre_inverse_real(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t
) {
    check_real_legendre_args(output, input, weight_t, true);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();

    if (input.scalar_type() == torch::kFloat32) {
        if (config.tile_l == 8) {
            launch_inverse_real<float, 8>(output, input, weight_t, config, stream);
        } else {
            launch_inverse_real<float, 4>(output, input, weight_t, config, stream);
        }
    } else {
        if (config.tile_l == 8) {
            launch_inverse_real<at::BFloat16, 8>(output, input, weight_t, config, stream);
        } else {
            launch_inverse_real<at::BFloat16, 4>(output, input, weight_t, config, stream);
        }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_vector_legendre_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t
) {
    check_vector_legendre_args(output, input, weight0_t, weight1_t, false);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();

    if (config.tile_l == 8) {
        launch_vector_forward<8>(output, input, weight0_t, weight1_t, config, stream);
    } else {
        launch_vector_forward<4>(output, input, weight0_t, weight1_t, config, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fused_vector_legendre_inverse(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t
) {
    check_vector_legendre_args(output, input, weight0_t, weight1_t, true);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();

    if (config.tile_l == 8) {
        launch_vector_inverse<8>(output, input, weight0_t, weight1_t, config, stream);
    } else {
        launch_vector_inverse<4>(output, input, weight0_t, weight1_t, config, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
