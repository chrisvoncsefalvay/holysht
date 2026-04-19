# **HOLYSHT**: highly optimised Legendre $Y_l^m$ SHT

## Summary

HOLYSHT is a production-ready GPU acceleration layer for spherical harmonic transforms (SHT) in the `torch-harmonics` ecosystem. It provides high-performance CUDA kernels for NVIDIA GPUs (Hopper/Blackwell with TMA support) and Apple Metal/MPS kernels for arm64 Macs. Rather than reimplementing `torch-harmonics`, HOLYSHT focuses on accelerating the slowest execution paths—Legendre stage computation, vector SHT composition, and inverse-FFT preparation—via backend-specific kernels. It reuses `torch-harmonics` for quadrature weight generation and remains an intentional companion library rather than a full replacement.

The project is production-proven, documented-in-progress, and includes comprehensive profiling, benchmarking, and parity-testing infrastructure suitable for low-level kernel development.

## What's included

**Core transforms:**
- `RealSHT`, `InverseRealSHT` (scalar forward/inverse)
- `RealVectorSHT`, `InverseRealVectorSHT` (vector forward/inverse)
- Explicit backward (autograd) support for all custom CUDA paths
- Mixed-precision (BF16) forward paths via CUDA real-reduction kernels

**Kernels and backends:**
- CUDA: Hopper/Blackwell TMA kernels, tensor-core variants (TF32, BF16), FMA kernels, shared-memory large-grid designs
- Metal/MPS: Native Metal compute kernels for Apple Silicon, with hybrid dispatch strategies (native kernel for small grids, fallback einsum for larger grids, tuned mid-size ranges)
- Runtime architecture-aware launch selection and backend autotuning via configurable environment variables

**Infrastructure:**
- Full benchmark suite (forward, inverse, BF16, training paths)
- Profiling helpers for `nsys`, `ncu`, and `cuobjdump` resource reporting
- Parity tests against `torch-harmonics` covering backward and non-contiguous inputs
- Local torch JIT build caching for safe day-to-day development iteration

## Performance summary

**NVIDIA GPU (GB10 / H100-class, PyTorch 2.10.0+cu130, CUDA 13.0, batch size 4):**

| Workload | 256×512 | 512×1024 |
|---|---:|---:|
| Scalar forward | **3.2x** | **2.9x** |
| Scalar inverse | ~1.8x | ~2.0x |
| Vector forward | **10.2x** | **11.4x** |
| Vector inverse | ~8.7x | **8.7x** |
| Scalar forward + backward | **2.3x** | ~2.0x |
| Vector forward + backward | **4.3x** | ~3.8x |
| BF16 scalar forward | **4.7x** | **3.0x** |
| BF16 vector forward | **10.2x** | **11.3x** |

**Apple Metal/MPS (M4 Mini, PyTorch 2.11.0, batch size 4):**
- Vector forward: **1.5x–3.25x** (wins on larger grids)
- Vector inverse: **1.4x–2.81x**
- Scalar forward: wins on smaller grids; slower on large
- Sparse `Y_n^m` synthesis: **1.1x**
- Vector forward+backward: **1.7x**

## Technical innovation

1. **Runtime backend selection**: TMA (Tensor Memory Accelerator) for Hopper/Blackwell, with tensor-core variants and FMA kernels available; autotuned on first use via configurable cache
2. **Dedicated vector kernels**: Specialized forward/inverse kernels eliminate Python-side tensor packing overhead
3. **Shared-memory design**: Large-grid FP32, BF16, and vector kernels using optimized shared-memory layouts for occupancy
4. **BF16 real reductions**: Mixed-precision forward path with float accumulation for precision
5. **CUDA-side irfft prep**: Inverse-FFT preparation offloaded to CUDA to avoid extra Python tensor passes
6. **Metal hybrid dispatch**: Kernel selection tuned per grid size for Apple Silicon
7. **Architecture detection**: Runtime SM detection (sm_80, sm_90a, sm_100, etc.) with strategy dispatch

## Configuration and profiling

**Environment variables for tuning:**
- `HOLYSHT_FORCE_BACKEND={fma,tma,tc_tf32,tc_bf16}` — lock backend selection
- `HOLYSHT_FORCE_VECTOR_STRATEGY={native_vector,stacked_real}` — override vector forward strategy
- `HOLYSHT_AUTOTUNE={0|1}` — enable/disable backend autotuning cache
- `HOLYSHT_USE_TMA={0|1}` — force TMA on/off for A/B testing
- `HOLYSHT_TMA_BATCH_TILE={1|2}` — forward batch tile size
- `HOLYSHT_MPS_SCALAR_NATIVE_MAX_NLAT` — Metal scalar kernel cutoff (default 64)
- `HOLYSHT_MPS_VECTOR_INVERSE_NATIVE_MAX_NLAT` — Metal vector-inverse cutoff (default 512)
- `HOLYSHT_ENABLE_NVTX=1` — enable NVTX profiling ranges

**Profiling scripts:**
- `scripts/profile_nsys.sh` — Nsight Systems profiling
- `scripts/profile_ncu.sh` — Nsight Compute profiling (falls back to resource reporting on restricted GPU counter access)
- `scripts/report_resources.py` — cuobjdump register/shared-memory analysis

## Practical constraints and design

- HOLYSHT still depends on `torch-harmonics` for quadrature weight generation
- Public Hugging Face distribution is forward-only; inverse and training paths remain in main source for development and regression coverage
- Default CUDA build targets `12.0+PTX` on GB10; Metal targets arm64 Macs
- Very large inverse-heavy workloads on unified-memory systems benefit from allocation caps
- Odd `mmax` shapes are padded to tile-aligned temporary spectral slabs before Legendre stage for TMA/BF16/large scalar paths when beneficial

## Intended audience

- Researchers using `torch-harmonics` who want faster SHT execution
- Neural operator workloads where SHT dominates forward or training cost
- CUDA/GPU developers seeking a small, inspectable codebase with real profiling hooks, detailed resource reporting, and documented kernel design
