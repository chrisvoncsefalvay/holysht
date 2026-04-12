# **HOLYSHT**: highly optimised Legendre $Y_l^m$ SHT

## Summary

HOLYSHT is a focused CUDA acceleration layer for spherical harmonic transforms
in the `torch-harmonics` ecosystem. It accelerates the Legendre stage, vector
SHT composition, and inverse-FFT preparation rather than replacing the whole
upstream library. For more information, [check the repo](https://github.com/chrisvoncsefalvay/holysht).

## Scope

Included:

- Scalar forward SHT
- Scalar inverse SHT
- Vector forward SHT
- Vector inverse SHT
- Explicit backward support for the custom scalar and vector kernels
- BF16 forward paths backed by CUDA real-reduction kernels
- Benchmark, profiling, and parity-test tooling


## Benchmark summary

Measured with PyTorch `2.10.0+cu130`, CUDA `13.0`, batch size
`4`, an allocation cap of `6 GiB`, and an NVIDIA GB10:

- Scalar forward: up to `4.6x`
- Scalar inverse: up to `2.0x`
- Vector forward: up to `8.7x`
- Vector inverse: up to `8.7x`
- Scalar forward + backward: `2.3x`
- Vector forward + backward: `4.3x`
- BF16 scalar forward: `1.6x`
- BF16 vector forward: `1.6x`

## Core techniques

1. Architecture-aware launch selection between `tile_l=4` and `tile_l=8`.
2. Shared-memory large-grid kernels for scalar, vector, and BF16 real paths.
3. Dedicated vector forward and inverse kernels to remove the old Python-side
   packing hot path.
4. Real-valued BF16 Legendre reductions with float accumulation.
5. Explicit autograd wrappers for scalar and vector CUDA paths.
6. CUDA-side `irfft` preparation to avoid extra Python tensor passes.
7. Local torch JIT build caching under `build/torch_extensions` for safer
   day-to-day iteration on GB10.

## Resource snapshot

From `cuobjdump --dump-resource-usage` on the local `sm_120` build:

- Scalar forward large kernel: `38` registers/thread, `3136` B shared/block,
  `256` threads/block, `6` active blocks/SM.
- Vector forward large kernel: `37` registers/thread, `5248` B shared/block,
  `256` threads/block, `6` active blocks/SM.
- BF16 forward large kernel: `34` registers/thread, `2080` B shared/block,
  `256` threads/block, `6` active blocks/SM.
- `prepare_irfft`: `19` registers/thread, no shared memory.

## Practical constraints

- HOLYSHT still depends on `torch-harmonics` for quadrature weight generation.
- The default local build targets `12.0+PTX` on GB10-class systems.
- `ncu` may require elevated GPU counter permissions; the supplied script falls
  back to `cuobjdump` resource reporting when those counters are unavailable.
- Very large inverse-heavy cases should still be run under an allocation cap on
  unified-memory systems.

## Intended audience

- Researchers already using `torch-harmonics` who want faster SHT execution.
- Neural operator workloads where SHT dominates forward or training cost.
- CUDA developers who want a small, inspectable codebase with real profiling and
  resource-report hooks rather than hand-wavy speed claims.
