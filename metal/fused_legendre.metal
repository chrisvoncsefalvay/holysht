#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Parameters
// ============================================================================

struct LegendreParams {
    uint batch_size;
    uint nlat;
    uint lmax;
    uint mmax;
};

struct PrepareIrfftParams {
    uint rows;
    uint mmax;
    uint active_mmax;
    uint full_mmax;
    bool nlon_even;
};

// ============================================================================
// Tile constants — tuned for Apple Silicon (32-wide SIMD groups)
// ============================================================================

constant uint LEGENDRE_TILE_M  = 32;
constant uint LEGENDRE_TILE_Y  = 16;  // raised from 8 for better occupancy
constant uint LEGENDRE_TILE_K  = 16;  // raised from 8 — matches TILE_Y for full thread utilization

// ============================================================================
// Direct (non-tiled) scalar kernels — float32
// ============================================================================

kernel void fused_legendre_forward_real_float(
    device const float* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint l = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || l >= params.lmax || m >= params.mmax) {
        return;
    }

    const uint out_index = ((b * params.lmax + l) * params.mmax) + m;
    if (m > l) {
        output[out_index] = 0.0f;
        return;
    }

    float acc = 0.0f;
    const uint input_base = (b * params.nlat * params.mmax) + m;
    const uint weight_base = (l * params.nlat * params.mmax) + m;
    for (uint k = 0; k < params.nlat; ++k) {
        acc = fma(weight_t[weight_base + k * params.mmax], input[input_base + k * params.mmax], acc);
    }
    output[out_index] = acc;
}

kernel void fused_legendre_inverse_real_float(
    device const float* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint k = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || k >= params.nlat || m >= params.mmax) {
        return;
    }

    float acc = 0.0f;
    const uint input_base = (b * params.lmax * params.mmax) + m;
    for (uint l = m; l < params.lmax; ++l) {
        const uint weight_index = ((l * params.nlat + k) * params.mmax) + m;
        acc = fma(weight_t[weight_index], input[input_base + l * params.mmax], acc);
    }

    output[((b * params.nlat + k) * params.mmax) + m] = acc;
}

kernel void fused_legendre_forward_complex_float(
    device const float2* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint l = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || l >= params.lmax || m >= params.mmax) {
        return;
    }

    const uint out_index = ((b * params.lmax + l) * params.mmax) + m;
    if (m > l) {
        output[out_index] = float2(0.0f, 0.0f);
        return;
    }

    float2 acc = float2(0.0f, 0.0f);
    const uint input_base = (b * params.nlat * params.mmax) + m;
    const uint weight_base = (l * params.nlat * params.mmax) + m;
    for (uint k = 0; k < params.nlat; ++k) {
        const float weight = weight_t[weight_base + k * params.mmax];
        const float2 value = input[input_base + k * params.mmax];
        acc.x = fma(weight, value.x, acc.x);
        acc.y = fma(weight, value.y, acc.y);
    }
    output[out_index] = acc;
}

kernel void fused_legendre_inverse_complex_float(
    device const float2* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint k = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || k >= params.nlat || m >= params.mmax) {
        return;
    }

    float2 acc = float2(0.0f, 0.0f);
    const uint input_base = (b * params.lmax * params.mmax) + m;
    for (uint l = m; l < params.lmax; ++l) {
        const float weight = weight_t[((l * params.nlat + k) * params.mmax) + m];
        const float2 value = input[input_base + l * params.mmax];
        acc.x = fma(weight, value.x, acc.x);
        acc.y = fma(weight, value.y, acc.y);
    }
    output[((b * params.nlat + k) * params.mmax) + m] = acc;
}

// ============================================================================
// Direct (non-tiled) scalar kernels — half precision input, float accumulation
// ============================================================================

kernel void fused_legendre_forward_real_half(
    device const half* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint l = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || l >= params.lmax || m >= params.mmax) {
        return;
    }

    const uint out_index = ((b * params.lmax + l) * params.mmax) + m;
    if (m > l) {
        output[out_index] = 0.0f;
        return;
    }

    float acc = 0.0f;
    const uint input_base = (b * params.nlat * params.mmax) + m;
    const uint weight_base = (l * params.nlat * params.mmax) + m;
    for (uint k = 0; k < params.nlat; ++k) {
        acc = fma(weight_t[weight_base + k * params.mmax], float(input[input_base + k * params.mmax]), acc);
    }
    output[out_index] = acc;
}

kernel void fused_legendre_inverse_real_half(
    device const half* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint k = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || k >= params.nlat || m >= params.mmax) {
        return;
    }

    float acc = 0.0f;
    const uint input_base = (b * params.lmax * params.mmax) + m;
    for (uint l = m; l < params.lmax; ++l) {
        const uint weight_index = ((l * params.nlat + k) * params.mmax) + m;
        acc = fma(weight_t[weight_index], float(input[input_base + l * params.mmax]), acc);
    }

    output[((b * params.nlat + k) * params.mmax) + m] = acc;
}

// ============================================================================
// Tiled scalar kernels — float32, with weight + input tiling
// ============================================================================

kernel void fused_legendre_forward_real_float_tiled(
    device const float* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint l = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const uint tile_m1 = tile_m0 + LEGENDRE_TILE_M - 1;
    const uint tile_l0 = tgid.y * LEGENDRE_TILE_Y;
    const uint tile_l1 = tile_l0 + LEGENDRE_TILE_Y - 1;
    const bool active_output = b < params.batch_size && l < params.lmax && m < params.mmax;

    // Triangular early-exit: skip entire threadgroups where all m > l
    if (tile_m0 > tile_l1) {
        if (active_output) {
            output[((b * params.lmax + l) * params.mmax) + m] = 0.0f;
        }
        return;
    }

    const bool tile_is_full_triangle = tile_l0 >= tile_m1;
    const bool active_triangle = active_output && (tile_is_full_triangle || (m <= l));

    threadgroup float input_tile[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    float acc = 0.0f;

    for (uint k0 = 0; k0 < params.nlat; k0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint k_load = k0 + tid.y;
        input_tile[tid.y][tid.x] = (m < params.mmax && b < params.batch_size && k_load < params.nlat)
            ? input[((b * params.nlat + k_load) * params.mmax) + m]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_triangle) {
            const uint weight_base = (l * params.nlat * params.mmax) + m;
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (k0 + kk) < params.nlat; ++kk) {
                acc = fma(weight_t[weight_base + (k0 + kk) * params.mmax], input_tile[kk][tid.x], acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_output) {
        output[((b * params.lmax + l) * params.mmax) + m] = active_triangle ? acc : 0.0f;
    }
}

kernel void fused_legendre_inverse_real_float_tiled(
    device const float* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint k = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const bool active_output = b < params.batch_size && k < params.nlat && m < params.mmax;

    threadgroup float input_tile[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    float acc = 0.0f;

    for (uint l0 = tile_m0; l0 < params.lmax; l0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint l_load = l0 + tid.y;
        input_tile[tid.y][tid.x] = (m < params.mmax && b < params.batch_size && l_load < params.lmax)
            ? input[((b * params.lmax + l_load) * params.mmax) + m]
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_output) {
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (l0 + kk) < params.lmax; ++kk) {
                const uint l_idx = l0 + kk;
                if (l_idx >= m) {
                    const uint weight_index = ((l_idx * params.nlat + k) * params.mmax) + m;
                    acc = fma(weight_t[weight_index], input_tile[kk][tid.x], acc);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_output) {
        output[((b * params.nlat + k) * params.mmax) + m] = acc;
    }
}

kernel void fused_legendre_forward_complex_float_tiled(
    device const float2* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint l = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const uint tile_m1 = tile_m0 + LEGENDRE_TILE_M - 1;
    const uint tile_l0 = tgid.y * LEGENDRE_TILE_Y;
    const uint tile_l1 = tile_l0 + LEGENDRE_TILE_Y - 1;
    const bool active_output = b < params.batch_size && l < params.lmax && m < params.mmax;

    if (tile_m0 > tile_l1) {
        if (active_output) {
            output[((b * params.lmax + l) * params.mmax) + m] = float2(0.0f, 0.0f);
        }
        return;
    }

    const bool tile_is_full_triangle = tile_l0 >= tile_m1;
    const bool active_triangle = active_output && (tile_is_full_triangle || (m <= l));

    threadgroup float2 input_tile[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    float2 acc = float2(0.0f, 0.0f);

    for (uint k0 = 0; k0 < params.nlat; k0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint k_load = k0 + tid.y;
        input_tile[tid.y][tid.x] = (m < params.mmax && b < params.batch_size && k_load < params.nlat)
            ? input[((b * params.nlat + k_load) * params.mmax) + m]
            : float2(0.0f, 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_triangle) {
            const uint weight_base = (l * params.nlat * params.mmax) + m;
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (k0 + kk) < params.nlat; ++kk) {
                const float weight = weight_t[weight_base + (k0 + kk) * params.mmax];
                const float2 value = input_tile[kk][tid.x];
                acc.x = fma(weight, value.x, acc.x);
                acc.y = fma(weight, value.y, acc.y);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_output) {
        output[((b * params.lmax + l) * params.mmax) + m] = active_triangle ? acc : float2(0.0f, 0.0f);
    }
}

kernel void fused_legendre_inverse_complex_float_tiled(
    device const float2* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float2* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint k = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const bool active_output = b < params.batch_size && k < params.nlat && m < params.mmax;

    threadgroup float2 input_tile[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    float2 acc = float2(0.0f, 0.0f);

    for (uint l0 = tile_m0; l0 < params.lmax; l0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint l_load = l0 + tid.y;
        input_tile[tid.y][tid.x] = (m < params.mmax && b < params.batch_size && l_load < params.lmax)
            ? input[((b * params.lmax + l_load) * params.mmax) + m]
            : float2(0.0f, 0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_output) {
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (l0 + kk) < params.lmax; ++kk) {
                const uint l_idx = l0 + kk;
                if (l_idx >= m) {
                    const float weight = weight_t[((l_idx * params.nlat + k) * params.mmax) + m];
                    const float2 value = input_tile[kk][tid.x];
                    acc.x = fma(weight, value.x, acc.x);
                    acc.y = fma(weight, value.y, acc.y);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_output) {
        output[((b * params.nlat + k) * params.mmax) + m] = acc;
    }
}

// ============================================================================
// Tiled FP16 input kernels — half input, float accumulation, float output
// ============================================================================

kernel void fused_legendre_forward_real_half_tiled(
    device const half* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint l = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const uint tile_m1 = tile_m0 + LEGENDRE_TILE_M - 1;
    const uint tile_l0 = tgid.y * LEGENDRE_TILE_Y;
    const uint tile_l1 = tile_l0 + LEGENDRE_TILE_Y - 1;
    const bool active_output = b < params.batch_size && l < params.lmax && m < params.mmax;

    if (tile_m0 > tile_l1) {
        if (active_output) {
            output[((b * params.lmax + l) * params.mmax) + m] = 0.0f;
        }
        return;
    }

    const bool tile_is_full_triangle = tile_l0 >= tile_m1;
    const bool active_triangle = active_output && (tile_is_full_triangle || (m <= l));

    // Use float tile for promoted input values
    threadgroup float input_tile[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    float acc = 0.0f;

    for (uint k0 = 0; k0 < params.nlat; k0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint k_load = k0 + tid.y;
        input_tile[tid.y][tid.x] = (m < params.mmax && b < params.batch_size && k_load < params.nlat)
            ? float(input[((b * params.nlat + k_load) * params.mmax) + m])
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_triangle) {
            const uint weight_base = (l * params.nlat * params.mmax) + m;
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (k0 + kk) < params.nlat; ++kk) {
                acc = fma(weight_t[weight_base + (k0 + kk) * params.mmax], input_tile[kk][tid.x], acc);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_output) {
        output[((b * params.lmax + l) * params.mmax) + m] = active_triangle ? acc : 0.0f;
    }
}

kernel void fused_legendre_inverse_real_half_tiled(
    device const half* input [[buffer(0)]],
    device const float* weight_t [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant LegendreParams& params [[buffer(3)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint k = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const bool active_output = b < params.batch_size && k < params.nlat && m < params.mmax;

    threadgroup float input_tile[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    float acc = 0.0f;

    for (uint l0 = tile_m0; l0 < params.lmax; l0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint l_load = l0 + tid.y;
        input_tile[tid.y][tid.x] = (m < params.mmax && b < params.batch_size && l_load < params.lmax)
            ? float(input[((b * params.lmax + l_load) * params.mmax) + m])
            : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active_output) {
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (l0 + kk) < params.lmax; ++kk) {
                const uint l_idx = l0 + kk;
                if (l_idx >= m) {
                    const uint weight_index = ((l_idx * params.nlat + k) * params.mmax) + m;
                    acc = fma(weight_t[weight_index], input_tile[kk][tid.x], acc);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (active_output) {
        output[((b * params.nlat + k) * params.mmax) + m] = acc;
    }
}

// ============================================================================
// Fused vector Legendre kernels — complex (float2) input/output
// ============================================================================

// Forward: input [B, 2, nlat, mmax] complex → output [B, 2, lmax, mmax] complex
// Fuses the sph/tor 8-einsum composition into a single kernel pass.
kernel void fused_vector_legendre_forward_complex_float(
    device const float2* input [[buffer(0)]],
    device const float* weight0_t [[buffer(1)]],
    device const float* weight1_t [[buffer(2)]],
    device float2* output [[buffer(3)]],
    constant LegendreParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint l = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || l >= params.lmax || m >= params.mmax) {
        return;
    }

    const uint out_base = b * 2 * params.lmax * params.mmax;
    const uint sph_idx = out_base + l * params.mmax + m;
    const uint tor_idx = out_base + params.lmax * params.mmax + l * params.mmax + m;

    if (m > l) {
        output[sph_idx] = float2(0.0f, 0.0f);
        output[tor_idx] = float2(0.0f, 0.0f);
        return;
    }

    float sph_re = 0.0f, sph_im = 0.0f;
    float tor_re = 0.0f, tor_im = 0.0f;

    const uint in_base = b * 2 * params.nlat * params.mmax;
    const uint w_base = l * params.nlat * params.mmax + m;

    for (uint k = 0; k < params.nlat; ++k) {
        const float2 comp0 = input[in_base + k * params.mmax + m];
        const float2 comp1 = input[in_base + params.nlat * params.mmax + k * params.mmax + m];
        const float w0 = weight0_t[w_base + k * params.mmax];
        const float w1 = weight1_t[w_base + k * params.mmax];

        sph_re = fma(w0, comp0.x, sph_re);
        sph_re = fma(-w1, comp1.y, sph_re);
        sph_im = fma(w0, comp0.y, sph_im);
        sph_im = fma(w1, comp1.x, sph_im);

        tor_re = fma(-w1, comp0.y, tor_re);
        tor_re = fma(-w0, comp1.x, tor_re);
        tor_im = fma(w1, comp0.x, tor_im);
        tor_im = fma(-w0, comp1.y, tor_im);
    }

    output[sph_idx] = float2(sph_re, sph_im);
    output[tor_idx] = float2(tor_re, tor_im);
}

// Inverse: input [B, 2, lmax, mmax] complex → output [B, 2, nlat, mmax] complex
kernel void fused_vector_legendre_inverse_complex_float(
    device const float2* input [[buffer(0)]],
    device const float* weight0_t [[buffer(1)]],
    device const float* weight1_t [[buffer(2)]],
    device float2* output [[buffer(3)]],
    constant LegendreParams& params [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint m = gid.x;
    const uint k = gid.y;
    const uint b = gid.z;

    if (b >= params.batch_size || k >= params.nlat || m >= params.mmax) {
        return;
    }

    float comp0_re = 0.0f, comp0_im = 0.0f;
    float comp1_re = 0.0f, comp1_im = 0.0f;

    const uint in_base = b * 2 * params.lmax * params.mmax;

    for (uint l = m; l < params.lmax; ++l) {
        const float2 sph = input[in_base + l * params.mmax + m];
        const float2 tor = input[in_base + params.lmax * params.mmax + l * params.mmax + m];
        const float w0 = weight0_t[l * params.nlat * params.mmax + k * params.mmax + m];
        const float w1 = weight1_t[l * params.nlat * params.mmax + k * params.mmax + m];

        comp0_re = fma(w0, sph.x, comp0_re);
        comp0_re = fma(-w1, tor.y, comp0_re);
        comp0_im = fma(w0, sph.y, comp0_im);
        comp0_im = fma(w1, tor.x, comp0_im);

        comp1_re = fma(-w1, sph.y, comp1_re);
        comp1_re = fma(-w0, tor.x, comp1_re);
        comp1_im = fma(w1, sph.x, comp1_im);
        comp1_im = fma(-w0, tor.y, comp1_im);
    }

    const uint out_base = b * 2 * params.nlat * params.mmax;
    output[out_base + k * params.mmax + m] = float2(comp0_re, comp0_im);
    output[out_base + params.nlat * params.mmax + k * params.mmax + m] = float2(comp1_re, comp1_im);
}

// Tiled vector forward with shared-memory input caching
kernel void fused_vector_legendre_forward_complex_float_tiled(
    device const float2* input [[buffer(0)]],
    device const float* weight0_t [[buffer(1)]],
    device const float* weight1_t [[buffer(2)]],
    device float2* output [[buffer(3)]],
    constant LegendreParams& params [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint l = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const uint tile_m1 = tile_m0 + LEGENDRE_TILE_M - 1;
    const uint tile_l0 = tgid.y * LEGENDRE_TILE_Y;
    const uint tile_l1 = tile_l0 + LEGENDRE_TILE_Y - 1;
    const bool valid_out = b < params.batch_size && l < params.lmax && m < params.mmax;

    if (tile_m0 > tile_l1) {
        if (valid_out) {
            const uint out_base = b * 2 * params.lmax * params.mmax;
            output[out_base + l * params.mmax + m] = float2(0.0f, 0.0f);
            output[out_base + params.lmax * params.mmax + l * params.mmax + m] = float2(0.0f, 0.0f);
        }
        return;
    }

    const bool tile_is_full_triangle = tile_l0 >= tile_m1;
    const bool active = valid_out && (tile_is_full_triangle || (m <= l));
    const bool can_load = b < params.batch_size && m < params.mmax;

    threadgroup float2 sm_comp0[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    threadgroup float2 sm_comp1[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];

    float sph_re = 0.0f, sph_im = 0.0f;
    float tor_re = 0.0f, tor_im = 0.0f;

    const uint in_base = b * 2 * params.nlat * params.mmax;
    const uint w_base = l * params.nlat * params.mmax + m;

    for (uint k0 = 0; k0 < params.nlat; k0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint k_load = k0 + tid.y;
        if (can_load && k_load < params.nlat) {
            sm_comp0[tid.y][tid.x] = input[in_base + k_load * params.mmax + m];
            sm_comp1[tid.y][tid.x] = input[in_base + params.nlat * params.mmax + k_load * params.mmax + m];
        } else {
            sm_comp0[tid.y][tid.x] = float2(0.0f, 0.0f);
            sm_comp1[tid.y][tid.x] = float2(0.0f, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (k0 + kk) < params.nlat; ++kk) {
                const float2 c0 = sm_comp0[kk][tid.x];
                const float2 c1 = sm_comp1[kk][tid.x];
                const float w0 = weight0_t[w_base + (k0 + kk) * params.mmax];
                const float w1 = weight1_t[w_base + (k0 + kk) * params.mmax];

                sph_re = fma(w0, c0.x, sph_re);
                sph_re = fma(-w1, c1.y, sph_re);
                sph_im = fma(w0, c0.y, sph_im);
                sph_im = fma(w1, c1.x, sph_im);

                tor_re = fma(-w1, c0.y, tor_re);
                tor_re = fma(-w0, c1.x, tor_re);
                tor_im = fma(w1, c0.x, tor_im);
                tor_im = fma(-w0, c1.y, tor_im);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_out) {
        const uint out_base = b * 2 * params.lmax * params.mmax;
        const uint sph_idx = out_base + l * params.mmax + m;
        const uint tor_idx = out_base + params.lmax * params.mmax + l * params.mmax + m;
        output[sph_idx] = active ? float2(sph_re, sph_im) : float2(0.0f, 0.0f);
        output[tor_idx] = active ? float2(tor_re, tor_im) : float2(0.0f, 0.0f);
    }
}

// Tiled vector inverse with shared-memory input caching
kernel void fused_vector_legendre_inverse_complex_float_tiled(
    device const float2* input [[buffer(0)]],
    device const float* weight0_t [[buffer(1)]],
    device const float* weight1_t [[buffer(2)]],
    device float2* output [[buffer(3)]],
    constant LegendreParams& params [[buffer(4)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
) {
    const uint m = tgid.x * LEGENDRE_TILE_M + tid.x;
    const uint k = tgid.y * LEGENDRE_TILE_Y + tid.y;
    const uint b = tgid.z;
    const uint tile_m0 = tgid.x * LEGENDRE_TILE_M;
    const bool valid_out = b < params.batch_size && k < params.nlat && m < params.mmax;
    const bool can_load = b < params.batch_size && m < params.mmax;

    threadgroup float2 sm_sph[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];
    threadgroup float2 sm_tor[LEGENDRE_TILE_K][LEGENDRE_TILE_M + 1];

    float comp0_re = 0.0f, comp0_im = 0.0f;
    float comp1_re = 0.0f, comp1_im = 0.0f;

    const uint in_base = b * 2 * params.lmax * params.mmax;

    for (uint l0 = tile_m0; l0 < params.lmax; l0 += LEGENDRE_TILE_K) {
        // All TILE_Y threads participate in loading (TILE_K == TILE_Y)
        const uint l_load = l0 + tid.y;
        if (can_load && l_load < params.lmax) {
            sm_sph[tid.y][tid.x] = input[in_base + l_load * params.mmax + m];
            sm_tor[tid.y][tid.x] = input[in_base + params.lmax * params.mmax + l_load * params.mmax + m];
        } else {
            sm_sph[tid.y][tid.x] = float2(0.0f, 0.0f);
            sm_tor[tid.y][tid.x] = float2(0.0f, 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (valid_out) {
            for (uint kk = 0; kk < LEGENDRE_TILE_K && (l0 + kk) < params.lmax; ++kk) {
                const uint l_idx = l0 + kk;
                if (l_idx >= m) {
                    const float2 sph = sm_sph[kk][tid.x];
                    const float2 tor = sm_tor[kk][tid.x];
                    const float w0 = weight0_t[l_idx * params.nlat * params.mmax + k * params.mmax + m];
                    const float w1 = weight1_t[l_idx * params.nlat * params.mmax + k * params.mmax + m];

                    comp0_re = fma(w0, sph.x, comp0_re);
                    comp0_re = fma(-w1, tor.y, comp0_re);
                    comp0_im = fma(w0, sph.y, comp0_im);
                    comp0_im = fma(w1, tor.x, comp0_im);

                    comp1_re = fma(-w1, sph.y, comp1_re);
                    comp1_re = fma(-w0, tor.x, comp1_re);
                    comp1_im = fma(w1, sph.x, comp1_im);
                    comp1_im = fma(-w0, tor.y, comp1_im);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (valid_out) {
        const uint out_base = b * 2 * params.nlat * params.mmax;
        output[out_base + k * params.mmax + m] = float2(comp0_re, comp0_im);
        output[out_base + params.nlat * params.mmax + k * params.mmax + m] = float2(comp1_re, comp1_im);
    }
}

// ============================================================================
// SHT helper: prepare irfft input (zero imag at DC and Nyquist)
// ============================================================================

kernel void sht_prepare_irfft_kernel(
    device float2* data [[buffer(0)]],
    constant PrepareIrfftParams& params [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint col = gid.x;
    const uint row = gid.y;

    if (row >= params.rows || col >= params.full_mmax) {
        return;
    }

    const uint idx = row * params.full_mmax + col;

    // Zero out columns beyond active_mmax
    if (col >= params.active_mmax) {
        data[idx] = float2(0.0f, 0.0f);
        return;
    }

    // DC bin: zero imaginary
    if (col == 0) {
        data[idx] = float2(data[idx].x, 0.0f);
        return;
    }

    // Nyquist bin: zero imaginary (even nlon only)
    if (params.nlon_even && col == params.full_mmax - 1) {
        data[idx] = float2(data[idx].x, 0.0f);
    }
}
