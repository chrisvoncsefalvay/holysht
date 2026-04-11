// HOLYSHT: Highly Optimised Legendre/Ylm/SHT
// Author: Chris von Csefalvay <chris@chrisvoncsefalvay.com>
//
// A CUDA kernel that computes the Legendre-Gauss quadrature at the heart of
// the Spherical Harmonic Transform, operating directly on complex tensors
// and fusing the real/imaginary multiplications into a single pass.
//
// Key design decisions:
//   1. Works on complex64 (interleaved float32 re/im) natively — no need for
//      view_as_real / view_as_complex conversions.
//   2. Weight matrix is stored in TRANSPOSED layout [lmax, nlat, mmax] so that
//      threads differing in m (the fastest-varying dimension) produce coalesced
//      loads for both weights AND complex input.
//   3. Each thread independently computes one output element out[b, l, m] by
//      iterating over the k (latitude) reduction — no inter-thread reductions
//      needed, eliminating warp shuffle and shared memory reduction overhead.
//   4. 2D thread blocks (TILE_M × TILE_L) map directly to output (m, l) tiles.
//
// Forward: out[b,l,m] = Σ_k  W[l,k,m] · in[b,k,m]  (complex × real→complex)
// Inverse: out[b,k,m] = Σ_l  W[l,k,m] · in[b,l,m]  (complex × real→complex)
//   (for inverse, the weight is conceptually transposed over l,k)
//
// Optimisation techniques:
//   - __launch_bounds__ for register pressure control and occupancy targeting
//   - __ldg() read-only cache path for all global memory reads
//   - Register-prefetch double buffering in large-grid tiled kernels: issues
//     the next tile's global load into registers while computing on the current
//     shared-memory tile, overlapping memory latency with FMA compute
//   - Partial loop unrolling in small-grid kernels for instruction-level parallelism
//
// Target: NVIDIA Blackwell-class GPUs, with PTX-forward-compatible builds.

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// Thread block dimensions — m is the fast (coalesced) dimension.
// The large-grid kernels reuse one input tile across multiple output rows.
constexpr int TILE_M = 32;   // Threads in wavenumber direction (warp-aligned)
constexpr int TILE_L = 8;    // Threads in output-row direction
// Block size = TILE_M * TILE_L = 256 threads

// The existing one-thread-per-output kernel wins for tiny grids because launch
// overhead dominates. For larger grids we stage the complex input in shared
// memory so the same [k, m] tile is reused across multiple output rows.
constexpr int SMALL_GRID_THRESHOLD = 128;

// =============================================================================
// Forward Legendre Transform (complex input → complex output)
// =============================================================================
// out[b, l, m] = Σ_k weight_t[l, k, m] * input[b, k, m]
//
// Grid:  (ceil(mmax / TILE_M), ceil(lmax / TILE_L), batch)
// Block: (TILE_M, TILE_L)
//
// Memory access analysis:
//   weight_t[l, k, m]: threads in x (m-dim) access consecutive addresses → COALESCED
//   input[b, k, m]:    threads in x (m-dim) access consecutive complex values → COALESCED
//   output[b, l, m]:   threads in x (m-dim) write consecutive complex values → COALESCED

__launch_bounds__(256, 4)
__global__ void fused_legendre_forward_kernel(
    const float2* __restrict__ input,     // [batch, nlat, mmax] as complex64
    const float*  __restrict__ weight_t,  // [lmax, nlat, mmax] transposed weight
    float2* __restrict__ output,          // [batch, lmax, mmax] as complex64
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int l = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || l >= lmax || b >= batch_size) return;

    // Accumulate in registers — no shared memory needed for the reduction
    float acc_re = 0.0f;
    float acc_im = 0.0f;

    // Base offsets
    const long long in_base = (long long)b * nlat * mmax + m;
    const long long w_base = (long long)l * nlat * mmax + m;

    // Main reduction loop over k (latitude)
    // Both weight and input strides by mmax per k-step — sequential for this thread,
    // but coalesced across the warp (threads differ in m, stride=1)
    #pragma unroll 4
    for (int k = 0; k < nlat; k++) {
        const float w = __ldg(&weight_t[w_base + (long long)k * mmax]);
        const float2 in_val = __ldg(&input[in_base + (long long)k * mmax]);

        // The fusion: one weight load → two FMAs
        acc_re = fmaf(w, in_val.x, acc_re);
        acc_im = fmaf(w, in_val.y, acc_im);
    }

    // Write output — coalesced across m-threads
    const long long out_idx = (long long)b * lmax * mmax + (long long)l * mmax + m;
    output[out_idx] = make_float2(acc_re, acc_im);
}


// =============================================================================
// Forward Legendre Transform (large-grid, register-prefetch double-buffered)
// =============================================================================
// Stages a [TILE_L, TILE_M] tile of the complex input in shared memory, then
// streams the weight matrix from global memory. Uses register-prefetch double
// buffering: while computing on the current shared memory tile, the next tile's
// data is loaded into thread-local registers, overlapping the ~200+ cycle global
// memory latency with the FMA compute chain.

__launch_bounds__(256, 4)
__global__ void fused_legendre_forward_large_kernel(
    const float2* __restrict__ input,     // [batch, nlat, mmax] as complex64
    const float*  __restrict__ weight_t,  // [lmax, nlat, mmax] transposed weight
    float2* __restrict__ output,          // [batch, lmax, mmax] as complex64
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

    __shared__ float2 sm_input[TILE_L][TILE_M + 1];

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    const long long in_base = ((long long)b * nlat * mmax) + m;
    const long long w_base = ((long long)l * nlat * mmax) + m;

    // Load first tile into shared memory (synchronous bootstrap)
    {
        const int k_load = threadIdx.y;
        if (can_load && k_load < nlat) {
            sm_input[threadIdx.y][threadIdx.x] = __ldg(&input[in_base + (long long)k_load * mmax]);
        } else {
            sm_input[threadIdx.y][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < nlat; k0 += TILE_L) {
        // Issue prefetch for the NEXT tile into registers.
        // This load instruction enters the memory pipeline immediately; by the
        // time the FMA compute chain below finishes, the data will (typically)
        // already reside in registers, hiding global memory latency.
        float2 prefetch;
        const int next_k = k0 + TILE_L + threadIdx.y;
        if (can_load && next_k < nlat) {
            prefetch = __ldg(&input[in_base + (long long)next_k * mmax]);
        } else {
            prefetch = make_float2(0.0f, 0.0f);
        }

        // Compute on current tile (from shared memory)
        if (active) {
            #pragma unroll
            for (int kk = 0; kk < TILE_L; kk++) {
                const int k_idx = k0 + kk;
                if (k_idx < nlat) {
                    const float w = __ldg(&weight_t[w_base + (long long)k_idx * mmax]);
                    const float2 v = sm_input[kk][threadIdx.x];
                    acc_re = fmaf(w, v.x, acc_re);
                    acc_im = fmaf(w, v.y, acc_im);
                }
            }
        }

        // Swap: write prefetched next-tile data to shared memory
        __syncthreads();
        sm_input[threadIdx.y][threadIdx.x] = prefetch;
        __syncthreads();
    }

    if (valid_out) {
        const long long out_idx = (long long)b * lmax * mmax + (long long)l * mmax + m;
        output[out_idx] = active ? make_float2(acc_re, acc_im) : make_float2(0.0f, 0.0f);
    }
}


// =============================================================================
// Inverse Legendre Transform (complex input → complex output)
// =============================================================================
// out[b, k, m] = Σ_l weight_t[l, k, m] * input[b, l, m]
//
// Grid:  (ceil(mmax / TILE_M), ceil(nlat / TILE_L), batch)
// Block: (TILE_M, TILE_L)
//
// Note: same weight layout [lmax, nlat, mmax] but now we iterate over l (reduction)
// and output k values.

__launch_bounds__(256, 4)
__global__ void fused_legendre_inverse_kernel(
    const float2* __restrict__ input,     // [batch, lmax, mmax] as complex64
    const float*  __restrict__ weight_t,  // [lmax, nlat, mmax] transposed weight
    float2* __restrict__ output,          // [batch, nlat, mmax] as complex64
    const int batch_size,
    const int nlat,
    const int lmax,
    const int mmax
) {
    const int m = blockIdx.x * TILE_M + threadIdx.x;
    const int k = blockIdx.y * TILE_L + threadIdx.y;
    const int b = blockIdx.z;

    if (m >= mmax || k >= nlat || b >= batch_size) return;

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    const long long in_base = (long long)b * lmax * mmax + m;
    // weight_t[l, k, m] — for fixed k, iterate over l
    const long long w_offset = (long long)k * mmax + m;

    #pragma unroll 4
    for (int l = 0; l < lmax; l++) {
        const float w = __ldg(&weight_t[(long long)l * nlat * mmax + w_offset]);
        const float2 in_val = __ldg(&input[in_base + (long long)l * mmax]);

        acc_re = fmaf(w, in_val.x, acc_re);
        acc_im = fmaf(w, in_val.y, acc_im);
    }

    const long long out_idx = (long long)b * nlat * mmax + (long long)k * mmax + m;
    output[out_idx] = make_float2(acc_re, acc_im);
}


// =============================================================================
// Inverse Legendre Transform (large-grid, register-prefetch double-buffered)
// =============================================================================
// Uses the same register-prefetch double buffering as the forward large kernel,
// but the reduction is over l (degree) and the output is indexed by k (latitude).

__launch_bounds__(256, 4)
__global__ void fused_legendre_inverse_large_kernel(
    const float2* __restrict__ input,     // [batch, lmax, mmax] as complex64
    const float*  __restrict__ weight_t,  // [lmax, nlat, mmax] transposed weight
    float2* __restrict__ output,          // [batch, nlat, mmax] as complex64
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

    __shared__ float2 sm_input[TILE_L][TILE_M + 1];

    float acc_re = 0.0f;
    float acc_im = 0.0f;

    const long long in_base = ((long long)b * lmax * mmax) + m;
    const long long out_idx = (long long)b * nlat * mmax + (long long)k * mmax + m;

    // Load first tile into shared memory (synchronous bootstrap)
    {
        const int l_load = threadIdx.y;
        if (can_load && l_load < lmax) {
            sm_input[threadIdx.y][threadIdx.x] = __ldg(&input[in_base + (long long)l_load * mmax]);
        } else {
            sm_input[threadIdx.y][threadIdx.x] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    for (int l0 = 0; l0 < lmax; l0 += TILE_L) {
        // Issue prefetch for the NEXT tile into registers
        float2 prefetch;
        const int next_l = l0 + TILE_L + threadIdx.y;
        if (can_load && next_l < lmax) {
            prefetch = __ldg(&input[in_base + (long long)next_l * mmax]);
        } else {
            prefetch = make_float2(0.0f, 0.0f);
        }

        // Compute on current tile (from shared memory)
        if (valid_out) {
            #pragma unroll
            for (int ll = 0; ll < TILE_L; ll++) {
                const int l_idx = l0 + ll;
                if ((l_idx < lmax) && (l_idx >= m)) {
                    const float w = __ldg(&weight_t[(long long)l_idx * nlat * mmax + (long long)k * mmax + m]);
                    const float2 v = sm_input[ll][threadIdx.x];
                    acc_re = fmaf(w, v.x, acc_re);
                    acc_im = fmaf(w, v.y, acc_im);
                }
            }
        }

        // Swap: write prefetched next-tile data to shared memory
        __syncthreads();
        sm_input[threadIdx.y][threadIdx.x] = prefetch;
        __syncthreads();
    }

    if (valid_out) {
        output[out_idx] = make_float2(acc_re, acc_im);
    }
}


// =============================================================================
// C++ wrappers
// =============================================================================

void fused_legendre_forward(
    torch::Tensor& output,          // [batch, lmax, mmax] complex64
    const torch::Tensor& input,     // [batch, nlat, mmax] complex64
    const torch::Tensor& weight_t   // [lmax, nlat, mmax] float32 (transposed)
) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && weight_t.is_cuda());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight_t.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat, "input must be complex64");
    TORCH_CHECK(output.scalar_type() == torch::kComplexFloat, "output must be complex64");
    TORCH_CHECK(weight_t.scalar_type() == torch::kFloat32, "weight must be float32");

    const int batch_size = input.size(0);
    const int nlat = input.size(1);
    const int mmax = input.size(2);
    const int lmax = weight_t.size(0);

    TORCH_CHECK(weight_t.size(1) == nlat, "weight dim 1 must match nlat");
    TORCH_CHECK(weight_t.size(2) == mmax, "weight dim 2 must match mmax");
    TORCH_CHECK(output.size(0) == batch_size);
    TORCH_CHECK(output.size(1) == lmax);
    TORCH_CHECK(output.size(2) == mmax);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (lmax + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= SMALL_GRID_THRESHOLD && lmax <= SMALL_GRID_THRESHOLD) {
        fused_legendre_forward_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_legendre_forward_large_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    }
}

void fused_legendre_inverse(
    torch::Tensor& output,          // [batch, nlat, mmax] complex64
    const torch::Tensor& input,     // [batch, lmax, mmax] complex64
    const torch::Tensor& weight_t   // [lmax, nlat, mmax] float32 (transposed)
) {
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && weight_t.is_cuda());
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous() && weight_t.is_contiguous());
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat, "input must be complex64");
    TORCH_CHECK(output.scalar_type() == torch::kComplexFloat, "output must be complex64");
    TORCH_CHECK(weight_t.scalar_type() == torch::kFloat32, "weight must be float32");

    const int batch_size = input.size(0);
    const int lmax = input.size(1);
    const int mmax = input.size(2);
    const int nlat = weight_t.size(1);

    TORCH_CHECK(weight_t.size(0) == lmax, "weight dim 0 must match lmax");
    TORCH_CHECK(weight_t.size(2) == mmax, "weight dim 2 must match mmax");
    TORCH_CHECK(output.size(0) == batch_size);
    TORCH_CHECK(output.size(1) == nlat);
    TORCH_CHECK(output.size(2) == mmax);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    dim3 grid(
        (mmax + TILE_M - 1) / TILE_M,
        (nlat + TILE_L - 1) / TILE_L,
        batch_size
    );
    dim3 block(TILE_M, TILE_L);

    if (nlat <= SMALL_GRID_THRESHOLD && lmax <= SMALL_GRID_THRESHOLD) {
        fused_legendre_inverse_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    } else {
        fused_legendre_inverse_large_kernel<<<grid, block, 0, stream>>>(
            reinterpret_cast<const float2*>(input.data_ptr()),
            weight_t.data_ptr<float>(),
            reinterpret_cast<float2*>(output.data_ptr()),
            batch_size, nlat, lmax, mmax
        );
    }
}
