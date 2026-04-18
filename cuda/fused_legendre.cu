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
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <list>
#include <mutex>
#include <unordered_map>

// TMA (Tensor Memory Accelerator) support for SM 9.0+ (Hopper/Blackwell)
#if CUDA_VERSION >= 12000
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#define HOLYSHT_HAS_TMA 1
using barrier_t = cuda::barrier<cuda::thread_scope_block>;
#else
#define HOLYSHT_HAS_TMA 0
#endif

namespace {

constexpr int TILE_M = 32;
constexpr int TILE_K = 16;  // reduction tile size (decoupled from TILE_L)

// Design notes on considered-and-rejected optimizations:
//
// Stream overlap: The SHT pipeline (FFT → Legendre → iFFT) is sequential on the
// same tensor — overlap is not possible within a single transform. Cross-batch
// pipelining would happen at the Python level with torch.cuda.Stream. Chris plz
// remember this and stop reimplementing this in hopes that it'll magically
// work @notetoself.
//
// Compile-time mmax specialization: mmax appears only in address strides, not in
// the inner compute loop (which iterates over TILE_K, already a compile-time
// constant). Specializing would replace one integer multiply with a shift+add —
// negligible vs. the FMA-dominated compute. The code bloat (one kernel per
// standard resolution × TILE_L × dtype) is not justified.

struct LaunchConfig {
    int tile_l;
    int small_grid_threshold;
    bool use_tma;
    int tma_batch_tile;
};

enum class ForwardBackendHint : int64_t {
    Auto = 0,
    Fma = 1,
    Tma = 2,
    TcTf32 = 3,
    TcBf16 = 4,
};

inline ForwardBackendHint normalize_backend_hint(const int64_t raw) {
    switch (raw) {
        case 1:
            return ForwardBackendHint::Fma;
        case 2:
            return ForwardBackendHint::Tma;
        case 3:
            return ForwardBackendHint::TcTf32;
        case 4:
            return ForwardBackendHint::TcBf16;
        default:
            return ForwardBackendHint::Auto;
    }
}

inline LaunchConfig apply_backend_hint(LaunchConfig config, const ForwardBackendHint hint) {
    switch (hint) {
        case ForwardBackendHint::Fma:
            config.use_tma = false;
            break;
        case ForwardBackendHint::Tma:
            config.use_tma = false;
#if HOLYSHT_HAS_TMA
            config.use_tma = at::cuda::getCurrentDeviceProperties()->major >= 9;
#endif
            break;
        default:
            break;
    }
    return config;
}

inline int env_int(const char* name, const int fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return fallback;
    }
    const long parsed = std::strtol(raw, nullptr, 10);
    return parsed > 0 ? static_cast<int>(parsed) : fallback;
}

inline int env_tristate(const char* name) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || raw[0] == '\0') {
        return -1;
    }
    if (std::strcmp(raw, "0") == 0) {
        return 0;
    }
    if (std::strcmp(raw, "1") == 0) {
        return 1;
    }
    return -1;
}

inline const LaunchConfig& select_launch_config() {
    // Cached: computed once per process, avoids per-launch device property queries
    static const LaunchConfig cached = [] {
        const int forced_tile_l = env_int("HOLYSHT_TILE_L", 0);
        const int forced_threshold = env_int("HOLYSHT_SMALL_GRID_THRESHOLD", 0);
        const int forced_tma = env_tristate("HOLYSHT_USE_TMA");
        const int forced_tma_batch_tile = env_int("HOLYSHT_TMA_BATCH_TILE", 0);

        const auto* props = at::cuda::getCurrentDeviceProperties();
#if HOLYSHT_HAS_TMA
        const bool tma_capable = (props->major >= 9);
#else
        const bool tma_capable = false;
#endif
        const bool use_tma = tma_capable && forced_tma != 0;
        const int tma_batch_tile = use_tma
            ? std::max(1, std::min(forced_tma_batch_tile > 0 ? forced_tma_batch_tile : 2, 2))
            : 1;

        if (forced_tile_l == 4 || forced_tile_l == 8) {
            return LaunchConfig{
                forced_tile_l,
                forced_threshold > 0 ? forced_threshold : 128,
                use_tma,
                tma_batch_tile,
            };
        }
        if (props->major >= 12) {
            return LaunchConfig{8, forced_threshold > 0 ? forced_threshold : 192, use_tma, tma_batch_tile};
        }
        if (props->major >= 9) {
            return LaunchConfig{8, forced_threshold > 0 ? forced_threshold : 160, use_tma, tma_batch_tile};
        }
        return LaunchConfig{4, forced_threshold > 0 ? forced_threshold : 128, false, 1};
    }();
    return cached;
}

// Occupancy: query max active blocks per SM for a given kernel
// Used at init time to validate __launch_bounds__ hints and for diagnostics
template <typename KernelFunc>
int query_occupancy(KernelFunc kernel, int block_size, size_t dynamic_smem = 0) {
    int max_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel, block_size, dynamic_smem);
    return max_blocks;
}

// 32-bit indexing: verify tensor dimensions fit in int32 at launch time
inline void check_32bit_indexing(int64_t batch, int64_t dim1, int64_t dim2) {
    TORCH_CHECK(
        batch * dim1 * dim2 <= static_cast<int64_t>(INT32_MAX),
        "Tensor too large for 32-bit indexing: ", batch, " x ", dim1, " x ", dim2,
        " = ", batch * dim1 * dim2, " > INT32_MAX"
    );
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

template <typename scalar_t>
__device__ inline float cast_scalar(const scalar_t value) {
    return static_cast<float>(value);
}

template <>
__device__ inline float cast_scalar<float>(const float value) {
    return value;
}

template <int TILE_L>
struct PackedForwardTile {
    int m_tile;
    int l_tile;
    bool full_triangle;
};

template <int TILE_L>
inline int packed_forward_tile_count(const int lmax, const int mmax) {
    static_assert(TILE_M % TILE_L == 0, "TILE_M must be divisible by TILE_L");
    constexpr int kTileRatio = TILE_M / TILE_L;
    const int l_tiles = (lmax + TILE_L - 1) / TILE_L;
    const int m_tiles = (mmax + TILE_M - 1) / TILE_M;

    int64_t total = 0;
    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
        const int first_l_tile = m_tile * kTileRatio;
        if (first_l_tile >= l_tiles) {
            break;
        }
        total += static_cast<int64_t>(l_tiles - first_l_tile);
    }

    TORCH_CHECK(total <= static_cast<int64_t>(INT32_MAX), "Packed forward tile count exceeds int32");
    return static_cast<int>(total);
}

template <int TILE_L>
__device__ inline PackedForwardTile<TILE_L> decode_packed_forward_tile(
    const int packed_tile,
    const int lmax,
    const int mmax
) {
    static_assert(TILE_M % TILE_L == 0, "TILE_M must be divisible by TILE_L");
    constexpr int kTileRatio = TILE_M / TILE_L;
    const int l_tiles = (lmax + TILE_L - 1) / TILE_L;
    const int m_tiles = (mmax + TILE_M - 1) / TILE_M;

    int remaining = packed_tile;
    for (int m_tile = 0; m_tile < m_tiles; ++m_tile) {
        const int first_l_tile = m_tile * kTileRatio;
        const int tile_count = l_tiles - first_l_tile;
        if (tile_count <= 0) {
            break;
        }
        if (remaining < tile_count) {
            return PackedForwardTile<TILE_L>{
                m_tile,
                first_l_tile + remaining,
                remaining > 0,
            };
        }
        remaining -= tile_count;
    }

    return PackedForwardTile<TILE_L>{0, 0, false};
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

    const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
    if (m > l) {
        output[out_idx] = make_float2(0.0f, 0.0f);
        return;
    }

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const int in_base = (int)b * nlat * mmax + m;
    const int w_base = (int)l * nlat * mmax + m;

    #pragma unroll 4
    for (int k = 0; k < nlat; ++k) {
        const float w = __ldg(&weight_t[w_base + (int)k * mmax]);
        const float2 v = load_complex(&input[in_base + (int)k * mmax]);
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
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m = tile.m_tile * TILE_M + threadIdx.x;
    const int l = tile.l_tile * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (tile.full_triangle || (m <= l));
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float2 sm_input[TILE_K][TILE_M + 1];

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const int in_base = (int)b * nlat * mmax + m;
    const int w_base = (int)l * nlat * mmax + m;

    // Load first tile cooperatively: all threads load TILE_K / TILE_L rows each
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        const int k_global = row;
        if (can_load && k_global < nlat) {
            sm_input[row][threadIdx.x] = load_complex(&input[in_base + (int)k_global * mmax]);
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
                ? load_complex(&input[in_base + (int)next_k * mmax])
                : make_float2(0.0f, 0.0f);
        }

        // Pre-load weights into registers, then compute from shmem + registers
        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (int)k_idx * mmax]) : 0.0f;
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

    if (active) {
        const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
        output[out_idx] = make_float2(acc_re, acc_im);
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
    const int in_base = (int)b * lmax * mmax + m;
    const int w_offset = (int)k * mmax + m;

    #pragma unroll 4
    for (int l = m; l < lmax; ++l) {
        const float w = __ldg(&weight_t[(int)l * nlat * mmax + w_offset]);
        const float2 v = load_complex(&input[in_base + (int)l * mmax]);
        acc_re = fmaf(w, v.x, acc_re);
        acc_im = fmaf(w, v.y, acc_im);
    }

    const int out_idx = (int)b * nlat * mmax + (int)k * mmax + m;
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
    const int in_base = (int)b * lmax * mmax + m;
    const int w_offset = (int)k * mmax + m;

    // Skip tiles where all l < m_min (triangular constraint)
    const int m_min = blockIdx.x * TILE_M;
    const int l0_start = (m_min / TILE_K) * TILE_K;

    // Load first tile cooperatively — starting from l0_start
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        const int l_global = l0_start + row;
        if (can_load && l_global < lmax) {
            sm_input[row][threadIdx.x] = load_complex(&input[in_base + (int)l_global * mmax]);
        } else {
            sm_input[row][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int l0 = l0_start; l0 < lmax; l0 += TILE_K) {
        // Prefetch next tile into registers
        float2 prefetch[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_l = l0 + TILE_K + row;
            prefetch[pass] = (can_load && next_l < lmax)
                ? load_complex(&input[in_base + (int)next_l * mmax])
                : make_float2(0.0f, 0.0f);
        }

        // Pre-load weights with warp ballot for efficient l >= m skipping
        if (valid_out) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                const bool needed = (l_idx < lmax) && (l_idx >= m);
                const unsigned any_needed = __ballot_sync(0xFFFFFFFF, needed);
                w_reg[ll] = (any_needed != 0u && needed)
                    ? __ldg(&weight_t[(int)l_idx * nlat * mmax + w_offset])
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
        const int out_idx = (int)b * nlat * mmax + (int)k * mmax + m;
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

    const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
    if (m > l) {
        output[out_idx] = 0.0f;
        return;
    }

    float acc = 0.0f;
    const int in_base = (int)b * nlat * mmax + m;
    const int w_base = (int)l * nlat * mmax + m;

    #pragma unroll 4
    for (int k = 0; k < nlat; ++k) {
        const float w = __ldg(&weight_t[w_base + (int)k * mmax]);
        const float v = load_scalar(&input[in_base + (int)k * mmax]);
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
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m = tile.m_tile * TILE_M + threadIdx.x;
    const int l = tile.l_tile * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (tile.full_triangle || (m <= l));
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float sm_input[TILE_K][TILE_M + 1];

    float acc = 0.0f;
    const int in_base = (int)b * nlat * mmax + m;
    const int w_base = (int)l * nlat * mmax + m;

    // Load first tile cooperatively
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        if (can_load && row < nlat) {
            sm_input[row][threadIdx.x] = load_scalar(&input[in_base + (int)row * mmax]);
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
                ? load_scalar(&input[in_base + (int)next_k * mmax])
                : 0.0f;
        }

        // Pre-load weights into registers, then compute from shmem + registers
        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (int)k_idx * mmax]) : 0.0f;
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

    if (active) {
        const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
        output[out_idx] = acc;
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
    const int in_base = (int)b * lmax * mmax + m;
    const int w_offset = (int)k * mmax + m;

    #pragma unroll 4
    for (int l = m; l < lmax; ++l) {
        const float w = __ldg(&weight_t[(int)l * nlat * mmax + w_offset]);
        const float v = load_scalar(&input[in_base + (int)l * mmax]);
        acc = fmaf(w, v, acc);
    }

    const int out_idx = (int)b * nlat * mmax + (int)k * mmax + m;
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
    const int in_base = (int)b * lmax * mmax + m;
    const int w_offset = (int)k * mmax + m;

    // Skip tiles where all l < m_min (triangular constraint)
    const int m_min = blockIdx.x * TILE_M;
    const int l0_start = (m_min / TILE_K) * TILE_K;

    // Load first tile cooperatively — starting from l0_start
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        const int l_global = l0_start + row;
        if (can_load && l_global < lmax) {
            sm_input[row][threadIdx.x] = load_scalar(&input[in_base + (int)l_global * mmax]);
        } else {
            sm_input[row][threadIdx.x] = 0.0f;
        }
    }
    __syncthreads();

    for (int l0 = l0_start; l0 < lmax; l0 += TILE_K) {
        // Prefetch next tile into registers
        float prefetch[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_l = l0 + TILE_K + row;
            prefetch[pass] = (can_load && next_l < lmax)
                ? load_scalar(&input[in_base + (int)next_l * mmax])
                : 0.0f;
        }

        // Pre-load weights with warp ballot for efficient l >= m skipping
        if (valid_out) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                const bool needed = (l_idx < lmax) && (l_idx >= m);
                const unsigned any_needed = __ballot_sync(0xFFFFFFFF, needed);
                w_reg[ll] = (any_needed != 0u && needed)
                    ? __ldg(&weight_t[(int)l_idx * nlat * mmax + w_offset])
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
        const int out_idx = (int)b * nlat * mmax + (int)k * mmax + m;
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

    const int out_base = (int)b * 2 * lmax * mmax;
    const int sph_idx = out_base + (int)l * mmax + m;
    const int tor_idx = out_base + (int)lmax * mmax + (int)l * mmax + m;

    if (m > l) {
        output[sph_idx] = make_float2(0.0f, 0.0f);
        output[tor_idx] = make_float2(0.0f, 0.0f);
        return;
    }

    float sph_re = 0.0f;
    float sph_im = 0.0f;
    float tor_re = 0.0f;
    float tor_im = 0.0f;

    const int in_base = (int)b * 2 * nlat * mmax;
    const int w_base = (int)l * nlat * mmax + m;

    #pragma unroll 4
    for (int k = 0; k < nlat; ++k) {
        const float2 comp0 = load_complex(&input[in_base + (int)k * mmax + m]);
        const float2 comp1 = load_complex(&input[in_base + (int)nlat * mmax + (int)k * mmax + m]);
        const float w0 = __ldg(&weight0_t[w_base + (int)k * mmax]);
        const float w1 = __ldg(&weight1_t[w_base + (int)k * mmax]);

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
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m = tile.m_tile * TILE_M + threadIdx.x;
    const int l = tile.l_tile * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (tile.full_triangle || (m <= l));
    const bool can_load = (b < batch_size) && (m < mmax);

    __shared__ float2 sm_comp0[TILE_K][TILE_M + 1];
    __shared__ float2 sm_comp1[TILE_K][TILE_M + 1];

    float sph_re = 0.0f;
    float sph_im = 0.0f;
    float tor_re = 0.0f;
    float tor_im = 0.0f;

    const int in_base = (int)b * 2 * nlat * mmax;
    const int w_base = (int)l * nlat * mmax + m;

    // Load first tile cooperatively
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        if (can_load && row < nlat) {
            sm_comp0[row][threadIdx.x] = load_complex(&input[in_base + (int)row * mmax + m]);
            sm_comp1[row][threadIdx.x] = load_complex(&input[in_base + (int)nlat * mmax + (int)row * mmax + m]);
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
                prefetch0[pass] = load_complex(&input[in_base + (int)next_k * mmax + m]);
                prefetch1[pass] = load_complex(&input[in_base + (int)nlat * mmax + (int)next_k * mmax + m]);
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
                    const float w0 = __ldg(&weight0_t[w_base + (int)k_idx * mmax]);
                    const float w1 = __ldg(&weight1_t[w_base + (int)k_idx * mmax]);

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

    if (active) {
        const int out_base = (int)b * 2 * lmax * mmax;
        const int sph_idx = out_base + (int)l * mmax + m;
        const int tor_idx = out_base + (int)lmax * mmax + (int)l * mmax + m;
        output[sph_idx] = make_float2(sph_re, sph_im);
        output[tor_idx] = make_float2(tor_re, tor_im);
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

    const int in_base = (int)b * 2 * lmax * mmax;

    #pragma unroll 4
    for (int l = m; l < lmax; ++l) {
        const float2 sph = load_complex(&input[in_base + (int)l * mmax + m]);
        const float2 tor = load_complex(&input[in_base + (int)lmax * mmax + (int)l * mmax + m]);
        const float w0 = __ldg(&weight0_t[(int)l * nlat * mmax + (int)k * mmax + m]);
        const float w1 = __ldg(&weight1_t[(int)l * nlat * mmax + (int)k * mmax + m]);

        comp0_re = fmaf(w0, sph.x, comp0_re);
        comp0_re = fmaf(-w1, tor.y, comp0_re);
        comp0_im = fmaf(w0, sph.y, comp0_im);
        comp0_im = fmaf(w1, tor.x, comp0_im);

        comp1_re = fmaf(-w1, sph.y, comp1_re);
        comp1_re = fmaf(-w0, tor.x, comp1_re);
        comp1_im = fmaf(w1, sph.x, comp1_im);
        comp1_im = fmaf(-w0, tor.y, comp1_im);
    }

    const int out_base = (int)b * 2 * nlat * mmax;
    const int comp0_idx = out_base + (int)k * mmax + m;
    const int comp1_idx = out_base + (int)nlat * mmax + (int)k * mmax + m;
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

    const int in_base = (int)b * 2 * lmax * mmax;
    const int w_offset = (int)k * mmax + m;

    // Skip tiles where all l < m_min (triangular constraint)
    const int m_min = blockIdx.x * TILE_M;
    const int l0_start = (m_min / TILE_K) * TILE_K;

    // Load first tile cooperatively — starting from l0_start
    #pragma unroll
    for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
        const int row = pass * TILE_L + threadIdx.y;
        const int l_global = l0_start + row;
        if (can_load && l_global < lmax) {
            sm_sph[row][threadIdx.x] = load_complex(&input[in_base + (int)l_global * mmax + m]);
            sm_tor[row][threadIdx.x] = load_complex(&input[in_base + (int)lmax * mmax + (int)l_global * mmax + m]);
        } else {
            sm_sph[row][threadIdx.x] = make_float2(0.0f, 0.0f);
            sm_tor[row][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int l0 = l0_start; l0 < lmax; l0 += TILE_K) {
        // Prefetch next tile into registers
        float2 prefetch_sph[TILE_K / TILE_L];
        float2 prefetch_tor[TILE_K / TILE_L];
        #pragma unroll
        for (int pass = 0; pass < TILE_K / TILE_L; ++pass) {
            const int row = pass * TILE_L + threadIdx.y;
            const int next_l = l0 + TILE_K + row;
            if (can_load && next_l < lmax) {
                prefetch_sph[pass] = load_complex(&input[in_base + (int)next_l * mmax + m]);
                prefetch_tor[pass] = load_complex(&input[in_base + (int)lmax * mmax + (int)next_l * mmax + m]);
            } else {
                prefetch_sph[pass] = make_float2(0.0f, 0.0f);
                prefetch_tor[pass] = make_float2(0.0f, 0.0f);
            }
        }

        // Compute from shmem with warp ballot for efficient l >= m skipping
        if (valid_out) {
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                const bool needed = (l_idx < lmax) && (l_idx >= m);
                const unsigned any_needed = __ballot_sync(0xFFFFFFFF, needed);
                if (any_needed == 0u) continue;

                if (needed) {
                    const float2 sph = sm_sph[ll][threadIdx.x];
                    const float2 tor = sm_tor[ll][threadIdx.x];
                    const float w0 = __ldg(&weight0_t[(int)l_idx * nlat * mmax + w_offset]);
                    const float w1 = __ldg(&weight1_t[(int)l_idx * nlat * mmax + w_offset]);

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
        const int out_base = (int)b * 2 * nlat * mmax;
        const int comp0_idx = out_base + (int)k * mmax + m;
        const int comp1_idx = out_base + (int)nlat * mmax + (int)k * mmax + m;
        output[comp0_idx] = make_float2(comp0_re, comp0_im);
        output[comp1_idx] = make_float2(comp1_re, comp1_im);
    }
}

// ============================================================================
// TMA (Tensor Memory Accelerator) kernels — SM 9.0+ (Hopper/Blackwell)
// Double-buffered async bulk copy with mbarrier synchronization
// ============================================================================

#if HOLYSHT_HAS_TMA

inline CUtensorMapDataType tma_dtype_for_scalar_type(c10::ScalarType scalar_type) {
    switch (scalar_type) {
        case c10::ScalarType::Float:
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
        case c10::ScalarType::BFloat16:
            return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        case c10::ScalarType::ComplexFloat:
            // TMA has no complex64 descriptor type. Treat complex64 tiles as
            // 8-byte floating-point elements so zero-fill OOB handling stays
            // legal; the kernels still interpret the payload as raw float2 data.
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
        default:
            TORCH_CHECK(false, "Unsupported scalar type for TMA descriptor");
            return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    }
}

// Contiguous scalar/complex tensors can use the original descriptor builder
// directly. This path is still the most reliable option on current GB10
// drivers, while the more general TmaPlan/TmaDescCache path remains useful for
// genuinely strided views like input.select(1, k) in the vector kernels.
inline bool tma_strides_aligned(int dim0, size_t elem_size) {
    return (static_cast<size_t>(dim0) * elem_size) % 16 == 0;
}

inline CUtensorMap make_tma_desc_3d(
    const void* base_ptr,
    int dim0,
    int dim1,
    int dim2,
    CUtensorMapDataType dtype,
    size_t elem_size,
    int tile0,
    int tile1
) {
    CUtensorMap tensor_map{};
    const uint64_t gdim[3] = {
        static_cast<uint64_t>(dim0),
        static_cast<uint64_t>(dim1),
        static_cast<uint64_t>(dim2)
    };
    const uint64_t gstride[2] = {
        static_cast<uint64_t>(dim0) * elem_size,
        static_cast<uint64_t>(dim1) * static_cast<uint64_t>(dim0) * elem_size
    };
    const uint32_t box[3] = {
        static_cast<uint32_t>(tile0),
        static_cast<uint32_t>(tile1),
        1u
    };
    const uint32_t estride[3] = {1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map, dtype, 3,
        const_cast<void*>(base_ptr),
        gdim, gstride, box, estride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    );
    TORCH_CHECK(result == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", result);
    return tensor_map;
}

// Strided TMA plan describing a 3-D tile view onto a (possibly non-contiguous)
// region of memory. Computed from real tensor sizes and strides — not assumed
// from "tight" layout, which is wrong for sliced views like input.select(1, k).
struct TmaPlan {
    const void* ptr;
    int dim0;                       // innermost; element stride must be 1
    int dim1;
    int dim2;
    uint64_t row_stride_bytes;      // bytes between successive dim1 indices
    uint64_t batch_stride_bytes;    // bytes between successive dim2 indices
    uint32_t elem_size;
    uint32_t dtype;
    uint32_t tile0;
    uint32_t tile1;
    uint32_t tile2;
    bool valid;                     // alignment + innermost-stride OK
};

// Build a TmaPlan from a 3-D torch tensor laid out as [dim2, dim1, dim0].
// Reads strides() rather than assuming contiguity, so sliced views work.
inline TmaPlan tma_plan_from_3d_tensor(
    const torch::Tensor& t,
    size_t elem_size,
    int tile0,
    int tile1,
    int tile2 = 1
) {
    TmaPlan p;
    p.ptr = t.data_ptr();
    p.dim0 = static_cast<int>(t.size(2));
    p.dim1 = static_cast<int>(t.size(1));
    p.dim2 = static_cast<int>(t.size(0));
    p.row_stride_bytes = static_cast<uint64_t>(t.stride(1)) * elem_size;
    p.batch_stride_bytes = static_cast<uint64_t>(t.stride(0)) * elem_size;
    p.elem_size = static_cast<uint32_t>(elem_size);
    p.dtype = static_cast<uint32_t>(tma_dtype_for_scalar_type(t.scalar_type()));
    p.tile0 = static_cast<uint32_t>(tile0);
    p.tile1 = static_cast<uint32_t>(tile1);
    p.tile2 = static_cast<uint32_t>(tile2);
    p.valid =
        (t.stride(2) == 1) &&
        (p.row_stride_bytes % 16 == 0) &&
        (p.batch_stride_bytes % 16 == 0) &&
        ((reinterpret_cast<uintptr_t>(p.ptr) % 16) == 0);
    return p;
}

// Same, plus an explicit byte offset onto the base pointer (for sliced views
// where we'd rather not materialise a TensorImpl).
inline TmaPlan tma_plan_with_offset(TmaPlan base, uint64_t offset_bytes) {
    base.ptr = static_cast<const char*>(base.ptr) + offset_bytes;
    base.valid = base.valid && ((reinterpret_cast<uintptr_t>(base.ptr) % 16) == 0);
    return base;
}

struct TmaCacheKey {
    const void* ptr;
    int dim0;
    int dim1;
    int dim2;
    uint64_t row_stride_bytes;
    uint64_t batch_stride_bytes;
    uint32_t elem_size;
    uint32_t dtype;
    uint32_t tile0;
    uint32_t tile1;
    uint32_t tile2;

    bool operator==(const TmaCacheKey& o) const noexcept {
        return ptr == o.ptr
            && dim0 == o.dim0 && dim1 == o.dim1 && dim2 == o.dim2
            && row_stride_bytes == o.row_stride_bytes
            && batch_stride_bytes == o.batch_stride_bytes
            && elem_size == o.elem_size
            && dtype == o.dtype
            && tile0 == o.tile0 && tile1 == o.tile1 && tile2 == o.tile2;
    }
};

struct TmaCacheKeyHash {
    size_t operator()(const TmaCacheKey& k) const noexcept {
        size_t h = reinterpret_cast<size_t>(k.ptr);
        auto mix = [&h](size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(static_cast<size_t>(k.dim0));
        mix(static_cast<size_t>(k.dim1));
        mix(static_cast<size_t>(k.dim2));
        mix(static_cast<size_t>(k.row_stride_bytes));
        mix(static_cast<size_t>(k.batch_stride_bytes));
        mix(static_cast<size_t>(k.dtype));
        mix((static_cast<size_t>(k.tile0) << 32) | static_cast<size_t>(k.tile1));
        mix(static_cast<size_t>(k.tile2));
        mix(static_cast<size_t>(k.elem_size));
        return h;
    }
};

// Process-wide LRU cache for CUtensorMap descriptors. Each entry is ~128 B,
// so the bound on memory is trivial; the win is avoiding the host-side
// cuTensorMapEncodeTiled driver call on every kernel launch.
//
// Cache entries are safe to reuse even if the data at `ptr` has changed:
// the descriptor encodes a memory layout, not contents. PyTorch's caching
// allocator may recycle an address with a different shape/dtype, but in
// that case the new request will produce a different key and miss.
class TmaDescCache {
public:
    static constexpr size_t kCapacity = 64;

    const CUtensorMap& get(const TmaPlan& plan) {
        TORCH_CHECK(plan.valid, "TMA plan is not valid (alignment or stride violation)");

        TmaCacheKey key{
            plan.ptr,
            plan.dim0, plan.dim1, plan.dim2,
            plan.row_stride_bytes, plan.batch_stride_bytes,
            plan.elem_size, plan.dtype, plan.tile0, plan.tile1, plan.tile2,
        };

        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            order_.splice(order_.begin(), order_, it->second.lru_it);
            return it->second.desc;
        }
        if (map_.size() >= kCapacity) {
            const auto& victim = order_.back();
            map_.erase(victim);
            order_.pop_back();
        }
        order_.push_front(key);
        Entry entry{build_descriptor(plan), order_.begin()};
        auto inserted = map_.emplace(key, entry).first;
        return inserted->second.desc;
    }

private:
    struct Entry {
        CUtensorMap desc;
        std::list<TmaCacheKey>::iterator lru_it;
    };

    static CUtensorMap build_descriptor(const TmaPlan& plan) {
        CUtensorMap tensor_map{};
        const uint64_t gdim[3] = {
            static_cast<uint64_t>(plan.dim0),
            static_cast<uint64_t>(plan.dim1),
            static_cast<uint64_t>(plan.dim2),
        };
        const uint64_t gstride[2] = {plan.row_stride_bytes, plan.batch_stride_bytes};
        const uint32_t box[3] = {plan.tile0, plan.tile1, plan.tile2};
        const uint32_t estride[3] = {1, 1, 1};
        const auto dtype = static_cast<CUtensorMapDataType>(plan.dtype);

        CUresult r = cuTensorMapEncodeTiled(
            &tensor_map, dtype, 3,
            const_cast<void*>(plan.ptr),
            gdim, gstride, box, estride,
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_NONE,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
        );
        TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", r);
        return tensor_map;
    }

    std::mutex mu_;
    std::unordered_map<TmaCacheKey, Entry, TmaCacheKeyHash> map_;
    std::list<TmaCacheKey> order_;
};

inline TmaDescCache& tma_cache() {
    static TmaDescCache cache;
    return cache;
}

// TMA load helper: one thread issues the bulk copy via PTX, all threads arrive at barrier
__device__ inline void tma_load_3d(
    void* smem_dst,
    const CUtensorMap& tma_map,
    int coord0, int coord1, int coord2,
    barrier_t& bar,
    uint32_t expected_bytes,
    int tid
) {
    if (tid == 0) {
        // Expect bytes on the barrier for tracking TMA completion
        cuda::device::barrier_expect_tx(bar, expected_bytes);
        const int32_t coords[3] = {coord0, coord1, coord2};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global,
            smem_dst, &tma_map, coords,
            reinterpret_cast<uint64_t*>(cuda::device::barrier_native_handle(bar))
        );
    }
}

// --- Forward complex TMA kernel ---

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_large_tma_kernel(
    __grid_constant__ const CUtensorMap input_tma,
    const float* __restrict__ weight_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m_tile = tile.m_tile * TILE_M;
    const int l_tile = tile.l_tile * TILE_L;
    const int m = m_tile + threadIdx.x;
    const int l = l_tile + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (tile.full_triangle || (m <= l));

    __shared__ __align__(128) float2 sm_input[2][TILE_K][TILE_M];
    __shared__ barrier_t bar;

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) init(&bar, blockDim.x * blockDim.y);
    __syncthreads();

    const uint32_t tile_bytes = TILE_K * TILE_M * sizeof(float2);
    int buf = 0;

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const int w_base = (int)l * nlat * mmax + m;

    // Load first tile
    tma_load_3d(&sm_input[0][0][0], input_tma, m_tile, 0, b, bar, tile_bytes, tid);
    bar.wait(bar.arrive());

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_k0 = k0 + TILE_K;

        // Issue next TMA load (non-blocking)
        if (next_k0 < nlat) {
            tma_load_3d(&sm_input[next_buf][0][0], input_tma, m_tile, next_k0, b,
                        bar, tile_bytes, tid);
        }

        // Compute from current buffer
        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (int)k_idx * mmax]) : 0.0f;
            }
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const float2 v = sm_input[buf][kk][threadIdx.x];
                acc_re = fmaf(w_reg[kk], v.x, acc_re);
                acc_im = fmaf(w_reg[kk], v.y, acc_im);
            }
        }

        // Wait for next tile
        if (next_k0 < nlat) {
            bar.wait(bar.arrive());
        }
        buf = next_buf;
    }

    if (active) {
        const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
        output[out_idx] = make_float2(acc_re, acc_im);
    }
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_large_tma_batch2_kernel(
    __grid_constant__ const CUtensorMap input_tma,
    const float* __restrict__ weight_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    constexpr int B_TILE = 2;
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m_tile = tile.m_tile * TILE_M;
    const int l_tile = tile.l_tile * TILE_L;
    const int m = m_tile + threadIdx.x;
    const int l = l_tile + threadIdx.y;
    const int batch0 = blockIdx.z * B_TILE;

    const bool valid_lm = (l < lmax) && (m < mmax);
    const bool active = valid_lm && (tile.full_triangle || (m <= l));

    __shared__ __align__(128) float2 sm_input[2][B_TILE][TILE_K][TILE_M];
    __shared__ barrier_t bars[B_TILE];

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int bi = 0; bi < B_TILE; ++bi) {
            init(&bars[bi], blockDim.x * blockDim.y);
        }
    }
    __syncthreads();

    const uint32_t tile_bytes = TILE_K * TILE_M * sizeof(float2);
    int buf = 0;

    float acc_re[B_TILE] = {0.0f, 0.0f};
    float acc_im[B_TILE] = {0.0f, 0.0f};
    const int w_base = (int)l * nlat * mmax + m;

    #pragma unroll
    for (int bi = 0; bi < B_TILE; ++bi) {
        const int b = batch0 + bi;
        if (b < batch_size) {
            tma_load_3d(&sm_input[0][bi][0][0], input_tma, m_tile, 0, b, bars[bi], tile_bytes, tid);
        }
    }
    #pragma unroll
    for (int bi = 0; bi < B_TILE; ++bi) {
        bars[bi].wait(bars[bi].arrive());
    }

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_k0 = k0 + TILE_K;

        if (next_k0 < nlat) {
            #pragma unroll
            for (int bi = 0; bi < B_TILE; ++bi) {
                const int b = batch0 + bi;
                if (b < batch_size) {
                    tma_load_3d(&sm_input[next_buf][bi][0][0], input_tma, m_tile, next_k0, b,
                                bars[bi], tile_bytes, tid);
                }
            }
        }

        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (int)k_idx * mmax]) : 0.0f;
            }
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const float w = w_reg[kk];
                #pragma unroll
                for (int bi = 0; bi < B_TILE; ++bi) {
                    if (batch0 + bi < batch_size) {
                        const float2 v = sm_input[buf][bi][kk][threadIdx.x];
                        acc_re[bi] = fmaf(w, v.x, acc_re[bi]);
                        acc_im[bi] = fmaf(w, v.y, acc_im[bi]);
                    }
                }
            }
        }

        if (next_k0 < nlat) {
            #pragma unroll
            for (int bi = 0; bi < B_TILE; ++bi) {
                bars[bi].wait(bars[bi].arrive());
            }
        }
        buf = next_buf;
    }

    if (active) {
        #pragma unroll
        for (int bi = 0; bi < B_TILE; ++bi) {
            const int b = batch0 + bi;
            if (b < batch_size) {
                const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
                output[out_idx] = make_float2(acc_re[bi], acc_im[bi]);
            }
        }
    }
}

// --- Inverse complex TMA kernel ---

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_inverse_large_tma_kernel(
    __grid_constant__ const CUtensorMap input_tma,
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

    __shared__ __align__(128) float2 sm_input[2][TILE_K][TILE_M];
    __shared__ barrier_t bar;

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) init(&bar, blockDim.x * blockDim.y);
    __syncthreads();

    const uint32_t tile_bytes = TILE_K * TILE_M * sizeof(float2);
    const int m_tile = blockIdx.x * TILE_M;
    const int m_min = m_tile;
    const int l0_start = (m_min / TILE_K) * TILE_K;
    int buf = 0;

    float acc_re = 0.0f;
    float acc_im = 0.0f;
    const int w_offset = (int)k * mmax + m;

    // Load first tile (starting from l0_start)
    tma_load_3d(&sm_input[0][0][0], input_tma, m_tile, l0_start, b, bar, tile_bytes, tid);
    bar.wait(bar.arrive());

    for (int l0 = l0_start; l0 < lmax; l0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_l0 = l0 + TILE_K;

        if (next_l0 < lmax) {
            tma_load_3d(&sm_input[next_buf][0][0], input_tma, m_tile, next_l0, b,
                        bar, tile_bytes, tid);
        }

        if (valid_out) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                const bool needed = (l_idx < lmax) && (l_idx >= m);
                const unsigned any_needed = __ballot_sync(0xFFFFFFFF, needed);
                w_reg[ll] = (any_needed != 0u && needed)
                    ? __ldg(&weight_t[(int)l_idx * nlat * mmax + w_offset])
                    : 0.0f;
            }
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const float2 v = sm_input[buf][ll][threadIdx.x];
                acc_re = fmaf(w_reg[ll], v.x, acc_re);
                acc_im = fmaf(w_reg[ll], v.y, acc_im);
            }
        }

        if (next_l0 < lmax) {
            bar.wait(bar.arrive());
        }
        buf = next_buf;
    }

    if (valid_out) {
        const int out_idx = (int)b * nlat * mmax + (int)k * mmax + m;
        output[out_idx] = make_float2(acc_re, acc_im);
    }
}

// --- Forward real TMA kernel ---

template <typename scalar_t, int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_real_large_tma_kernel(
    __grid_constant__ const CUtensorMap input_tma,
    const float* __restrict__ weight_t,
    float* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m_tile = tile.m_tile * TILE_M;
    const int l_tile = tile.l_tile * TILE_L;
    const int m = m_tile + threadIdx.x;
    const int l = l_tile + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (tile.full_triangle || (m <= l));

    __shared__ __align__(128) scalar_t sm_input[2][TILE_K][TILE_M];
    __shared__ barrier_t bar;

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) init(&bar, blockDim.x * blockDim.y);
    __syncthreads();

    const uint32_t tile_bytes = TILE_K * TILE_M * sizeof(scalar_t);
    float acc = 0.0f;
    const int w_base = (int)l * nlat * mmax + m;
    int buf = 0;

    tma_load_3d(&sm_input[0][0][0], input_tma, m_tile, 0, b, bar, tile_bytes, tid);
    bar.wait(bar.arrive());

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_k0 = k0 + TILE_K;

        if (next_k0 < nlat) {
            tma_load_3d(&sm_input[next_buf][0][0], input_tma, m_tile, next_k0, b,
                        bar, tile_bytes, tid);
        }

        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (int)k_idx * mmax]) : 0.0f;
            }
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const float v = cast_scalar(sm_input[buf][kk][threadIdx.x]);
                acc = fmaf(w_reg[kk], v, acc);
            }
        }

        if (next_k0 < nlat) {
            bar.wait(bar.arrive());
        }
        buf = next_buf;
    }

    if (active) {
        const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
        output[out_idx] = acc;
    }
}

template <typename scalar_t, int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_forward_real_large_tma_batch2_kernel(
    __grid_constant__ const CUtensorMap input_tma,
    const float* __restrict__ weight_t,
    float* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    constexpr int B_TILE = 2;
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m_tile = tile.m_tile * TILE_M;
    const int l_tile = tile.l_tile * TILE_L;
    const int m = m_tile + threadIdx.x;
    const int l = l_tile + threadIdx.y;
    const int batch0 = blockIdx.z * B_TILE;

    const bool valid_lm = (l < lmax) && (m < mmax);
    const bool active = valid_lm && (tile.full_triangle || (m <= l));

    __shared__ __align__(128) scalar_t sm_input[2][B_TILE][TILE_K][TILE_M];
    __shared__ barrier_t bars[B_TILE];

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int bi = 0; bi < B_TILE; ++bi) {
            init(&bars[bi], blockDim.x * blockDim.y);
        }
    }
    __syncthreads();

    const uint32_t tile_bytes = TILE_K * TILE_M * sizeof(scalar_t);
    float acc[B_TILE] = {0.0f, 0.0f};
    const int w_base = (int)l * nlat * mmax + m;
    int buf = 0;

    #pragma unroll
    for (int bi = 0; bi < B_TILE; ++bi) {
        const int b = batch0 + bi;
        if (b < batch_size) {
            tma_load_3d(&sm_input[0][bi][0][0], input_tma, m_tile, 0, b, bars[bi], tile_bytes, tid);
        }
    }
    #pragma unroll
    for (int bi = 0; bi < B_TILE; ++bi) {
        bars[bi].wait(bars[bi].arrive());
    }

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_k0 = k0 + TILE_K;

        if (next_k0 < nlat) {
            #pragma unroll
            for (int bi = 0; bi < B_TILE; ++bi) {
                const int b = batch0 + bi;
                if (b < batch_size) {
                    tma_load_3d(&sm_input[next_buf][bi][0][0], input_tma, m_tile, next_k0, b,
                                bars[bi], tile_bytes, tid);
                }
            }
        }

        if (active) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                w_reg[kk] = (k_idx < nlat) ? __ldg(&weight_t[w_base + (int)k_idx * mmax]) : 0.0f;
            }
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const float w = w_reg[kk];
                #pragma unroll
                for (int bi = 0; bi < B_TILE; ++bi) {
                    if (batch0 + bi < batch_size) {
                        const float v = cast_scalar(sm_input[buf][bi][kk][threadIdx.x]);
                        acc[bi] = fmaf(w, v, acc[bi]);
                    }
                }
            }
        }

        if (next_k0 < nlat) {
            #pragma unroll
            for (int bi = 0; bi < B_TILE; ++bi) {
                bars[bi].wait(bars[bi].arrive());
            }
        }
        buf = next_buf;
    }

    if (active) {
        #pragma unroll
        for (int bi = 0; bi < B_TILE; ++bi) {
            const int b = batch0 + bi;
            if (b < batch_size) {
                const int out_idx = (int)b * lmax * mmax + (int)l * mmax + m;
                output[out_idx] = acc[bi];
            }
        }
    }
}

// --- Inverse real TMA kernel ---

template <typename scalar_t, int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_legendre_inverse_real_large_tma_kernel(
    __grid_constant__ const CUtensorMap input_tma,
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

    __shared__ __align__(128) float sm_input[2][TILE_K][TILE_M];
    __shared__ barrier_t bar;

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) init(&bar, blockDim.x * blockDim.y);
    __syncthreads();

    const uint32_t tile_bytes = TILE_K * TILE_M * sizeof(float);
    const int m_tile = blockIdx.x * TILE_M;
    const int m_min = m_tile;
    const int l0_start = (m_min / TILE_K) * TILE_K;
    int buf = 0;

    float acc = 0.0f;
    const int w_offset = (int)k * mmax + m;

    tma_load_3d(&sm_input[0][0][0], input_tma, m_tile, l0_start, b, bar, tile_bytes, tid);
    bar.wait(bar.arrive());

    for (int l0 = l0_start; l0 < lmax; l0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_l0 = l0 + TILE_K;

        if (next_l0 < lmax) {
            tma_load_3d(&sm_input[next_buf][0][0], input_tma, m_tile, next_l0, b,
                        bar, tile_bytes, tid);
        }

        if (valid_out) {
            float w_reg[TILE_K];
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                const bool needed = (l_idx < lmax) && (l_idx >= m);
                const unsigned any_needed = __ballot_sync(0xFFFFFFFF, needed);
                w_reg[ll] = (any_needed != 0u && needed)
                    ? __ldg(&weight_t[(int)l_idx * nlat * mmax + w_offset])
                    : 0.0f;
            }
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                acc = fmaf(w_reg[ll], sm_input[buf][ll][threadIdx.x], acc);
            }
        }

        if (next_l0 < lmax) {
            bar.wait(bar.arrive());
        }
        buf = next_buf;
    }

    if (valid_out) {
        const int out_idx = (int)b * nlat * mmax + (int)k * mmax + m;
        output[out_idx] = acc;
    }
}

// --- Vector forward TMA kernel ---

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_vector_legendre_forward_large_tma_kernel(
    __grid_constant__ const CUtensorMap comp0_tma,
    __grid_constant__ const CUtensorMap comp1_tma,
    const float* __restrict__ weight0_t,
    const float* __restrict__ weight1_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m_tile = tile.m_tile * TILE_M;
    const int l_tile = tile.l_tile * TILE_L;
    const int m = m_tile + threadIdx.x;
    const int l = l_tile + threadIdx.y;
    const int b = blockIdx.z;

    const bool valid_out = (b < batch_size) && (l < lmax) && (m < mmax);
    const bool active = valid_out && (tile.full_triangle || (m <= l));

    __shared__ __align__(128) float2 sm_comp0[2][TILE_K][TILE_M];
    __shared__ __align__(128) float2 sm_comp1[2][TILE_K][TILE_M];
    __shared__ barrier_t bar0;
    __shared__ barrier_t bar1;

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) {
        init(&bar0, blockDim.x * blockDim.y);
        init(&bar1, blockDim.x * blockDim.y);
    }
    __syncthreads();

    const uint32_t input_tile_bytes = TILE_K * TILE_M * sizeof(float2);
    int buf = 0;

    float sph_re = 0.0f, sph_im = 0.0f;
    float tor_re = 0.0f, tor_im = 0.0f;
    const int w_base = (int)l * nlat * mmax + m;

    // Load first tiles for both components
    tma_load_3d(&sm_comp0[0][0][0], comp0_tma, m_tile, 0, b, bar0, input_tile_bytes, tid);
    tma_load_3d(&sm_comp1[0][0][0], comp1_tma, m_tile, 0, b, bar1, input_tile_bytes, tid);
    bar0.wait(bar0.arrive());
    bar1.wait(bar1.arrive());

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_k0 = k0 + TILE_K;

        if (next_k0 < nlat) {
            tma_load_3d(&sm_comp0[next_buf][0][0], comp0_tma, m_tile, next_k0, b, bar0, input_tile_bytes, tid);
            tma_load_3d(&sm_comp1[next_buf][0][0], comp1_tma, m_tile, next_k0, b, bar1, input_tile_bytes, tid);
        }

        if (active) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                if (k_idx < nlat) {
                    const float2 c0 = sm_comp0[buf][kk][threadIdx.x];
                    const float2 c1 = sm_comp1[buf][kk][threadIdx.x];
                    const float w0 = __ldg(&weight0_t[w_base + (int)k_idx * mmax]);
                    const float w1 = __ldg(&weight1_t[w_base + (int)k_idx * mmax]);

                    sph_re = fmaf(w0, c0.x, sph_re); sph_re = fmaf(-w1, c1.y, sph_re);
                    sph_im = fmaf(w0, c0.y, sph_im); sph_im = fmaf(w1, c1.x, sph_im);
                    tor_re = fmaf(-w1, c0.y, tor_re); tor_re = fmaf(-w0, c1.x, tor_re);
                    tor_im = fmaf(w1, c0.x, tor_im); tor_im = fmaf(-w0, c1.y, tor_im);
                }
            }
        }

        if (next_k0 < nlat) {
            bar0.wait(bar0.arrive());
            bar1.wait(bar1.arrive());
        }
        buf = next_buf;
    }

    if (active) {
        const int out_base = (int)b * 2 * lmax * mmax;
        const int sph_idx = out_base + (int)l * mmax + m;
        const int tor_idx = out_base + (int)lmax * mmax + (int)l * mmax + m;
        output[sph_idx] = make_float2(sph_re, sph_im);
        output[tor_idx] = make_float2(tor_re, tor_im);
    }
}

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_vector_legendre_forward_large_tma_batch2_kernel(
    __grid_constant__ const CUtensorMap comp0_tma,
    __grid_constant__ const CUtensorMap comp1_tma,
    const float* __restrict__ weight0_t,
    const float* __restrict__ weight1_t,
    float2* __restrict__ output,
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    constexpr int B_TILE = 2;
    const PackedForwardTile<TILE_L> tile = decode_packed_forward_tile<TILE_L>(blockIdx.x, lmax, mmax);
    const int m_tile = tile.m_tile * TILE_M;
    const int l_tile = tile.l_tile * TILE_L;
    const int m = m_tile + threadIdx.x;
    const int l = l_tile + threadIdx.y;
    const int batch0 = blockIdx.z * B_TILE;

    const bool valid_lm = (l < lmax) && (m < mmax);
    const bool active = valid_lm && (tile.full_triangle || (m <= l));

    __shared__ __align__(128) float2 sm_comp0[2][B_TILE][TILE_K][TILE_M];
    __shared__ __align__(128) float2 sm_comp1[2][B_TILE][TILE_K][TILE_M];
    __shared__ barrier_t bars0[B_TILE];
    __shared__ barrier_t bars1[B_TILE];

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) {
        #pragma unroll
        for (int bi = 0; bi < B_TILE; ++bi) {
            init(&bars0[bi], blockDim.x * blockDim.y);
            init(&bars1[bi], blockDim.x * blockDim.y);
        }
    }
    __syncthreads();

    const uint32_t input_tile_bytes = TILE_K * TILE_M * sizeof(float2);
    int buf = 0;

    float sph_re[B_TILE] = {0.0f, 0.0f};
    float sph_im[B_TILE] = {0.0f, 0.0f};
    float tor_re[B_TILE] = {0.0f, 0.0f};
    float tor_im[B_TILE] = {0.0f, 0.0f};
    const int w_base = (int)l * nlat * mmax + m;

    #pragma unroll
    for (int bi = 0; bi < B_TILE; ++bi) {
        const int b = batch0 + bi;
        if (b < batch_size) {
            tma_load_3d(&sm_comp0[0][bi][0][0], comp0_tma, m_tile, 0, b, bars0[bi], input_tile_bytes, tid);
            tma_load_3d(&sm_comp1[0][bi][0][0], comp1_tma, m_tile, 0, b, bars1[bi], input_tile_bytes, tid);
        }
    }
    #pragma unroll
    for (int bi = 0; bi < B_TILE; ++bi) {
        bars0[bi].wait(bars0[bi].arrive());
        bars1[bi].wait(bars1[bi].arrive());
    }

    for (int k0 = 0; k0 < nlat; k0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_k0 = k0 + TILE_K;

        if (next_k0 < nlat) {
            #pragma unroll
            for (int bi = 0; bi < B_TILE; ++bi) {
                const int b = batch0 + bi;
                if (b < batch_size) {
                    tma_load_3d(&sm_comp0[next_buf][bi][0][0], comp0_tma, m_tile, next_k0, b, bars0[bi], input_tile_bytes, tid);
                    tma_load_3d(&sm_comp1[next_buf][bi][0][0], comp1_tma, m_tile, next_k0, b, bars1[bi], input_tile_bytes, tid);
                }
            }
        }

        if (active) {
            #pragma unroll
            for (int kk = 0; kk < TILE_K; ++kk) {
                const int k_idx = k0 + kk;
                if (k_idx < nlat) {
                    const float w0 = __ldg(&weight0_t[w_base + (int)k_idx * mmax]);
                    const float w1 = __ldg(&weight1_t[w_base + (int)k_idx * mmax]);
                    #pragma unroll
                    for (int bi = 0; bi < B_TILE; ++bi) {
                        if (batch0 + bi < batch_size) {
                            const float2 c0 = sm_comp0[buf][bi][kk][threadIdx.x];
                            const float2 c1 = sm_comp1[buf][bi][kk][threadIdx.x];
                            sph_re[bi] = fmaf(w0, c0.x, sph_re[bi]);
                            sph_re[bi] = fmaf(-w1, c1.y, sph_re[bi]);
                            sph_im[bi] = fmaf(w0, c0.y, sph_im[bi]);
                            sph_im[bi] = fmaf(w1, c1.x, sph_im[bi]);
                            tor_re[bi] = fmaf(-w1, c0.y, tor_re[bi]);
                            tor_re[bi] = fmaf(-w0, c1.x, tor_re[bi]);
                            tor_im[bi] = fmaf(w1, c0.x, tor_im[bi]);
                            tor_im[bi] = fmaf(-w0, c1.y, tor_im[bi]);
                        }
                    }
                }
            }
        }

        if (next_k0 < nlat) {
            #pragma unroll
            for (int bi = 0; bi < B_TILE; ++bi) {
                bars0[bi].wait(bars0[bi].arrive());
                bars1[bi].wait(bars1[bi].arrive());
            }
        }
        buf = next_buf;
    }

    if (active) {
        #pragma unroll
        for (int bi = 0; bi < B_TILE; ++bi) {
            const int b = batch0 + bi;
            if (b < batch_size) {
                const int out_base = (int)b * 2 * lmax * mmax;
                const int sph_idx = out_base + (int)l * mmax + m;
                const int tor_idx = out_base + (int)lmax * mmax + (int)l * mmax + m;
                output[sph_idx] = make_float2(sph_re[bi], sph_im[bi]);
                output[tor_idx] = make_float2(tor_re[bi], tor_im[bi]);
            }
        }
    }
}

// --- Vector inverse TMA kernel ---

template <int TILE_L>
__launch_bounds__(TILE_M * TILE_L, 4)
__global__ void fused_vector_legendre_inverse_large_tma_kernel(
    __grid_constant__ const CUtensorMap sph_tma,
    __grid_constant__ const CUtensorMap tor_tma,
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

    __shared__ __align__(128) float2 sm_sph[2][TILE_K][TILE_M];
    __shared__ __align__(128) float2 sm_tor[2][TILE_K][TILE_M];
    __shared__ barrier_t bar0;
    __shared__ barrier_t bar1;

    const int tid = threadIdx.y * TILE_M + threadIdx.x;
    if (tid == 0) { init(&bar0, blockDim.x * blockDim.y); init(&bar1, blockDim.x * blockDim.y); }
    __syncthreads();

    const uint32_t tile_bytes = TILE_K * TILE_M * sizeof(float2);
    const int m_tile = blockIdx.x * TILE_M;
    const int m_min = m_tile;
    const int l0_start = (m_min / TILE_K) * TILE_K;
    int buf = 0;

    float comp0_re = 0.0f, comp0_im = 0.0f;
    float comp1_re = 0.0f, comp1_im = 0.0f;
    const int w_offset = (int)k * mmax + m;

    tma_load_3d(&sm_sph[0][0][0], sph_tma, m_tile, l0_start, b, bar0, tile_bytes, tid);
    tma_load_3d(&sm_tor[0][0][0], tor_tma, m_tile, l0_start, b, bar1, tile_bytes, tid);
    bar0.wait(bar0.arrive()); bar1.wait(bar1.arrive());

    for (int l0 = l0_start; l0 < lmax; l0 += TILE_K) {
        const int next_buf = 1 - buf;
        const int next_l0 = l0 + TILE_K;

        if (next_l0 < lmax) {
            tma_load_3d(&sm_sph[next_buf][0][0], sph_tma, m_tile, next_l0, b, bar0, tile_bytes, tid);
            tma_load_3d(&sm_tor[next_buf][0][0], tor_tma, m_tile, next_l0, b, bar1, tile_bytes, tid);
        }

        if (valid_out) {
            #pragma unroll
            for (int ll = 0; ll < TILE_K; ++ll) {
                const int l_idx = l0 + ll;
                const bool needed = (l_idx < lmax) && (l_idx >= m);
                const unsigned any_needed = __ballot_sync(0xFFFFFFFF, needed);
                if (any_needed == 0u) continue;

                if (needed) {
                    const float2 sph = sm_sph[buf][ll][threadIdx.x];
                    const float2 tor = sm_tor[buf][ll][threadIdx.x];
                    const float w0 = __ldg(&weight0_t[(int)l_idx * nlat * mmax + w_offset]);
                    const float w1 = __ldg(&weight1_t[(int)l_idx * nlat * mmax + w_offset]);

                    comp0_re = fmaf(w0, sph.x, comp0_re); comp0_re = fmaf(-w1, tor.y, comp0_re);
                    comp0_im = fmaf(w0, sph.y, comp0_im); comp0_im = fmaf(w1, tor.x, comp0_im);
                    comp1_re = fmaf(-w1, sph.y, comp1_re); comp1_re = fmaf(-w0, tor.x, comp1_re);
                    comp1_im = fmaf(w1, sph.x, comp1_im); comp1_im = fmaf(-w0, tor.y, comp1_im);
                }
            }
        }

        if (next_l0 < lmax) {
            bar0.wait(bar0.arrive()); bar1.wait(bar1.arrive());
        }
        buf = next_buf;
    }

    if (valid_out) {
        const int out_base = (int)b * 2 * nlat * mmax;
        output[out_base + (int)k * mmax + m] = make_float2(comp0_re, comp0_im);
        output[out_base + (int)nlat * mmax + (int)k * mmax + m] = make_float2(comp1_re, comp1_im);
    }
}

#endif // HOLYSHT_HAS_TMA

// ============================================================================
// Launch wrappers
// ============================================================================

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

    dim3 rect_grid(
        (mmax + TILE_M - 1) / TILE_M,
        (lmax + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= config.small_grid_threshold && lmax <= config.small_grid_threshold) {
        fused_legendre_forward_kernel<TILE_L><<<rect_grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
        return;
    }

    C10_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.nbytes(), stream));
    const int packed_tiles = packed_forward_tile_count<TILE_L>(lmax, mmax);
    if (packed_tiles == 0) {
        return;
    }
    dim3 packed_grid(packed_tiles, 1, batch_size);

#if HOLYSHT_HAS_TMA
    if (config.use_tma && tma_strides_aligned(mmax, sizeof(float2))) {
        auto input_tma = make_tma_desc_3d(
            input.data_ptr(),
            mmax,
            nlat,
            batch_size,
            tma_dtype_for_scalar_type(input.scalar_type()),
            sizeof(float2),
            TILE_M,
            TILE_K
        );
        if (config.tma_batch_tile >= 2 && batch_size >= 2) {
            dim3 batch_grid(packed_tiles, 1, (batch_size + 1) / 2);
            fused_legendre_forward_large_tma_batch2_kernel<TILE_L><<<batch_grid, block, 0, stream>>>(
                input_tma, weight_t.data_ptr<float>(),
                reinterpret_cast<float2*>(output.data_ptr()),
                batch_size, nlat, lmax, mmax
            );
        } else {
            fused_legendre_forward_large_tma_kernel<TILE_L><<<packed_grid, block, 0, stream>>>(
                input_tma, weight_t.data_ptr<float>(),
                reinterpret_cast<float2*>(output.data_ptr()),
                batch_size, nlat, lmax, mmax
            );
        }
        return;
    }
#endif

    fused_legendre_forward_large_kernel<TILE_L><<<packed_grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(input.data_ptr()),
        weight_t.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr()),
        batch_size, nlat, lmax, mmax
    );
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
#if HOLYSHT_HAS_TMA
    } else if (config.use_tma && tma_strides_aligned(mmax, sizeof(float2))) {
        auto tma = make_tma_desc_3d(
            input.data_ptr(),
            mmax,
            lmax,
            batch_size,
            tma_dtype_for_scalar_type(input.scalar_type()),
            sizeof(float2),
            TILE_M,
            TILE_K
        );
        fused_legendre_inverse_large_tma_kernel<TILE_L><<<grid, block, 0, stream>>>(
            tma, weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
#endif
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
    const ForwardBackendHint backend_hint,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const LaunchConfig effective_config = apply_backend_hint(config, backend_hint);
    const int batch_size = input.size(0);
    const int nlat = input.size(1);
    const int mmax = input.size(2);
    const int lmax = weight_t.size(0);

    dim3 rect_grid(
        (mmax + TILE_M - 1) / TILE_M,
        (lmax + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= effective_config.small_grid_threshold && lmax <= effective_config.small_grid_threshold) {
        fused_legendre_forward_real_kernel<scalar_t, TILE_L><<<rect_grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(),
            weight_t.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, nlat, lmax, mmax
        );
        return;
    }

    C10_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.nbytes(), stream));
    const int packed_tiles = packed_forward_tile_count<TILE_L>(lmax, mmax);
    if (packed_tiles == 0) {
        return;
    }
    dim3 packed_grid(packed_tiles, 1, batch_size);

#if HOLYSHT_HAS_TMA
    if (effective_config.use_tma && tma_strides_aligned(mmax, sizeof(scalar_t))) {
        auto input_tma = make_tma_desc_3d(
            input.data_ptr(),
            mmax,
            nlat,
            batch_size,
            tma_dtype_for_scalar_type(input.scalar_type()),
            sizeof(scalar_t),
            TILE_M,
            TILE_K
        );
        if (effective_config.tma_batch_tile >= 2 && batch_size >= 2) {
            dim3 batch_grid(packed_tiles, 1, (batch_size + 1) / 2);
            fused_legendre_forward_real_large_tma_batch2_kernel<scalar_t, TILE_L><<<batch_grid, block, 0, stream>>>(
                input_tma, weight_t.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, nlat, lmax, mmax
            );
        } else {
            fused_legendre_forward_real_large_tma_kernel<scalar_t, TILE_L><<<packed_grid, block, 0, stream>>>(
                input_tma, weight_t.data_ptr<float>(),
                output.data_ptr<float>(),
                batch_size, nlat, lmax, mmax
            );
        }
        return;
    }
#endif

    fused_legendre_forward_real_large_kernel<scalar_t, TILE_L><<<packed_grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        weight_t.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, nlat, lmax, mmax
    );
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
#if HOLYSHT_HAS_TMA
    } else if (config.use_tma && std::is_same<scalar_t, float>::value && tma_strides_aligned(mmax, sizeof(float))) {
        auto tma = make_tma_desc_3d(
            input.data_ptr(),
            mmax,
            lmax,
            batch_size,
            tma_dtype_for_scalar_type(input.scalar_type()),
            sizeof(float),
            TILE_M,
            TILE_K
        );
        fused_legendre_inverse_real_large_tma_kernel<scalar_t, TILE_L><<<grid, block, 0, stream>>>(
            tma, weight_t.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size, nlat, lmax, mmax
        );
#endif
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
    const ForwardBackendHint backend_hint,
    const LaunchConfig& config,
    const cudaStream_t stream
) {
    const LaunchConfig effective_config = apply_backend_hint(config, backend_hint);
    const int batch_size = input.size(0);
    const int nlat = input.size(2);
    const int mmax = input.size(3);
    const int lmax = weight0_t.size(0);

    dim3 rect_grid(
        (mmax + TILE_M - 1) / TILE_M,
        (lmax + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= effective_config.small_grid_threshold && lmax <= effective_config.small_grid_threshold) {
        fused_vector_legendre_forward_kernel<TILE_L><<<rect_grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight0_t.data_ptr<float>(),
            weight1_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
        return;
    }

    C10_CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.nbytes(), stream));
    const int packed_tiles = packed_forward_tile_count<TILE_L>(lmax, mmax);
    if (packed_tiles == 0) {
        return;
    }
    dim3 packed_grid(packed_tiles, 1, batch_size);

#if HOLYSHT_HAS_TMA
    // Vector input is [batch, 2, nlat, mmax] complex64. Each per-component
    // slice has batch stride 2*nlat*mmax, NOT nlat*mmax — the strided helper
    // reads the actual stride from the view so the descriptor is correct
    // (the previous implementation that assumed dim1*dim0*elem_size silently
    // misread half the data for batch_size > 1).
    if (effective_config.use_tma) {
        auto comp0 = input.select(1, 0);
        auto comp1 = input.select(1, 1);
        auto plan0 = tma_plan_from_3d_tensor(comp0, sizeof(float2), TILE_M, TILE_K);
        auto plan1 = tma_plan_from_3d_tensor(comp1, sizeof(float2), TILE_M, TILE_K);
        if (plan0.valid && plan1.valid) {
            const auto& comp0_tma = tma_cache().get(plan0);
            const auto& comp1_tma = tma_cache().get(plan1);
            if (effective_config.tma_batch_tile >= 2 && batch_size >= 2) {
                dim3 batch_grid(packed_tiles, 1, (batch_size + 1) / 2);
                fused_vector_legendre_forward_large_tma_batch2_kernel<TILE_L><<<batch_grid, block, 0, stream>>>(
                    comp0_tma, comp1_tma,
                    weight0_t.data_ptr<float>(),
                    weight1_t.data_ptr<float>(),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    batch_size, nlat, lmax, mmax
                );
            } else {
                fused_vector_legendre_forward_large_tma_kernel<TILE_L><<<packed_grid, block, 0, stream>>>(
                    comp0_tma, comp1_tma,
                    weight0_t.data_ptr<float>(),
                    weight1_t.data_ptr<float>(),
                    reinterpret_cast<float2*>(output.data_ptr()),
                    batch_size, nlat, lmax, mmax
                );
            }
            return;
        }
    }
#endif

    fused_vector_legendre_forward_large_kernel<TILE_L><<<packed_grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(input.data_ptr()),
        weight0_t.data_ptr<float>(),
        weight1_t.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr()),
        batch_size, nlat, lmax, mmax
    );
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
        return;
    }

#if HOLYSHT_HAS_TMA
    // [batch, 2, lmax, mmax] complex64 — same per-component batch-stride note
    // as launch_vector_forward.
    if (config.use_tma) {
        auto sph = input.select(1, 0);
        auto tor = input.select(1, 1);
        auto plan_sph = tma_plan_from_3d_tensor(sph, sizeof(float2), TILE_M, TILE_K);
        auto plan_tor = tma_plan_from_3d_tensor(tor, sizeof(float2), TILE_M, TILE_K);
        if (plan_sph.valid && plan_tor.valid) {
            const auto& sph_tma = tma_cache().get(plan_sph);
            const auto& tor_tma = tma_cache().get(plan_tor);
            fused_vector_legendre_inverse_large_tma_kernel<TILE_L><<<grid, block, 0, stream>>>(
                sph_tma, tor_tma,
                weight0_t.data_ptr<float>(),
                weight1_t.data_ptr<float>(),
                reinterpret_cast<float2*>(output.data_ptr()),
                batch_size, nlat, lmax, mmax
            );
            return;
        }
    }
#endif

    fused_vector_legendre_inverse_large_kernel<TILE_L><<<grid, block, 0, stream>>>(
        reinterpret_cast<const float2*>(input.data_ptr()),
        weight0_t.data_ptr<float>(),
        weight1_t.data_ptr<float>(),
        reinterpret_cast<float2*>(output.data_ptr()),
        batch_size, nlat, lmax, mmax
    );
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
    check_32bit_indexing(input.size(0), input.size(1), input.size(2));
    check_32bit_indexing(output.size(0), output.size(1), output.size(2));
    check_32bit_indexing(weight_t.size(0), weight_t.size(1), weight_t.size(2));

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
    check_32bit_indexing(input.size(0), input.size(1), input.size(2));
    check_32bit_indexing(output.size(0), output.size(1), output.size(2));
    check_32bit_indexing(weight_t.size(0), weight_t.size(1), weight_t.size(2));

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
    check_32bit_indexing(input.size(0) * 2, input.size(2), input.size(3));
    check_32bit_indexing(weight0_t.size(0), weight0_t.size(1), weight0_t.size(2));

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

void fused_legendre_forward_real(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const int64_t backend_hint
);

void fused_vector_legendre_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const int64_t backend_hint
);

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
    fused_legendre_forward_real(output, input, weight_t, static_cast<int64_t>(ForwardBackendHint::Auto));
}

void fused_legendre_forward_real(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const int64_t backend_hint
) {
    check_real_legendre_args(output, input, weight_t, false);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();
    const auto hint = normalize_backend_hint(backend_hint);

    if (input.scalar_type() == torch::kFloat32) {
        if (config.tile_l == 8) {
            launch_forward_real<float, 8>(output, input, weight_t, hint, config, stream);
        } else {
            launch_forward_real<float, 4>(output, input, weight_t, hint, config, stream);
        }
    } else {
        if (config.tile_l == 8) {
            launch_forward_real<at::BFloat16, 8>(output, input, weight_t, hint, config, stream);
        } else {
            launch_forward_real<at::BFloat16, 4>(output, input, weight_t, hint, config, stream);
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
    fused_vector_legendre_forward(output, input, weight0_t, weight1_t, static_cast<int64_t>(ForwardBackendHint::Auto));
}

void fused_vector_legendre_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const int64_t backend_hint
) {
    check_vector_legendre_args(output, input, weight0_t, weight1_t, false);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto config = select_launch_config();
    const auto hint = normalize_backend_hint(backend_hint);

    if (config.tile_l == 8) {
        launch_vector_forward<8>(output, input, weight0_t, weight1_t, hint, config, stream);
    } else {
        launch_vector_forward<4>(output, input, weight0_t, weight1_t, hint, config, stream);
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
