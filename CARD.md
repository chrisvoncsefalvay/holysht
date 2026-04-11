# HOLYSHT Card

## Name

**HOLYSHT**: Highly Optimised Legendre/Ylm/SHT

## Summary

HOLYSHT is a focused CUDA acceleration layer for spherical harmonic transforms
in the torch-harmonics ecosystem. It targets the Legendre stage, vector SHT
composition, and inverse-FFT preparation rather than trying to replace the
entire upstream library.

## Scope

Included:

- Scalar forward SHT
- Scalar inverse SHT
- Vector forward SHT
- Vector inverse SHT
- Explicit backward support for the optimized Legendre path
- Benchmark and parity-test tooling

Intentionally excluded:

- DISCO fusion work
- Experimental side-project code
- Benchmark artefacts and stale research notes

## Benchmark Summary

Measured on April 10, 2026 with system Python, PyTorch `2.10.0+cu130`, CUDA
`13.0`, and an NVIDIA GB10:

- Scalar forward: up to `7.9x`
- Scalar inverse: up to `3.6x`
- Vector forward: up to `14.9x`
- Vector inverse: up to `13.1x`
- Scalar forward+backward: up to `4.0x`
- Vector forward+backward: up to `5.5x`
- BF16 scalar forward: up to `12.7x`

All executed correctness checks passed in the benchmark run used for this card.

## Core Techniques

1. Adaptive CUDA Legendre kernels for both small and large grids.
2. Shared-memory input tiling in the large-grid path.
3. Triangular `m <= l` skipping to avoid useless work.
4. Explicit autograd wrappers for the custom forward and inverse kernels.
5. Vector SHT composition on top of the scalar CUDA Legendre primitives.
6. CUDA-side cleanup before `irfft` to avoid extra Python-side tensor passes.

## Practical Constraints

- HOLYSHT still depends on torch-harmonics for quadrature weight generation.
- The optimized CUDA path currently targets complex64 inputs with float32
  weights, plus optional BF16 forward modes where exposed.
- Very large inverse-heavy benchmark cases should be run under an allocation cap
  on unified-memory systems.

## Intended Audience

- Researchers already using torch-harmonics who want faster SHT execution.
- Neural operator workloads where SHT is a dominant forward or training cost.
- Blackwell-class CUDA systems, especially GB10-style unified-memory machines.
