# HOLYSHT Forward Tensor-Core and Autotune Design

Date: 2026-04-18
Status: Approved in conversation, pending written-spec review
Scope: CUDA forward paths only

## Context

HOLYSHT already contains real CUDA optimization work:

- packed triangular launch tiling for large forward kernels
- TMA-backed forward kernels on Hopper/Blackwell
- batch-tiled TMA variants that reuse weights across two batch lanes
- Python-side path selection for scalar, vector, BF16, and MPS fallbacks

The current gap is that the hottest CUDA forward path still performs the math as scalar FMAs after improving the data-movement side. Launch selection is also still based on static architecture heuristics plus a few hard-coded thresholds and environment overrides.

This design addresses both gaps while keeping the project true to its current character:

- keep HOLYSHT as a native custom-kernel package rather than turning it into a wrapper around an external GEMM framework
- improve forward-only paths, matching the practical constraint that Hugging Face kernel packaging is forward-oriented
- retain the current kernels as correctness-preserving fallbacks

## Goals

1. Add a real tensor-core forward backend for large CUDA forward workloads on SM90+ and SM120-class targets.
2. Support both BF16 forward and FP32-input forward via TF32-style tensor-core execution with FP32 accumulation.
3. Replace fixed large-shape backend selection with a lightweight runtime autotuner and persistent cache.
4. Preserve the current public API and keep fallbacks intact for unsupported shapes, dtypes, and devices.
5. Update the README to say explicitly that the optimized Hugging Face kernel story is forward-only by design.

## Non-Goals

- No backward-path tensor-core work in this change.
- No inverse-kernel tensor-core work in this change.
- No MPS tensor-core work.
- No dependency on CUTLASS, cuBLASDx, or another external kernel framework for the first implementation.
- No attempt to redesign the full SHT pipeline around multi-GPU or distributed execution.

## Constraints

- Forward-only is intentional. The Hugging Face kernel packaging target is forward-oriented, so the optimized packaged path should match that reality.
- Existing scalar-FMA and TMA kernels must remain available as safe fallbacks.
- Autotuning must stay lightweight enough to run on first use without turning import or first execution into a long benchmark session.
- The implementation must coexist with existing dirty-worktree development and local JIT loading.

## Recommended Approach

Implement a native forward-only tensor-core backend inside the existing CUDA extension, centered on the real Legendre contraction. Use that backend directly for scalar real forward paths and indirectly for vector forward paths by reusing the current "stack real contractions and recombine" structure.

This is preferred over a library-backed GEMM rewrite because it fixes the actual performance critique without changing the package model or introducing a much heavier dependency story.

## Architecture Overview

### 1. New Forward Tensor-Core Backend

Add new large-grid CUDA kernels for real forward contractions:

- `fused_legendre_forward_real_large_tc_kernel`
- `fused_legendre_forward_real_large_tc_batch2_kernel`

These kernels are:

- forward-only
- CUDA-only
- enabled only for SM90+ / Hopper-and-newer style tensor-core targets
- intended for large grids where the extra staging cost is amortized

They become a third large-grid backend family beside:

- scalar FMA large kernels
- TMA-fed scalar FMA large kernels

### 2. FP32 Path Uses TF32-Style Tensor-Core Execution

For public FP32 forward paths, keep the external dtype/API behavior as FP32 while staging compute tiles into TF32-compatible fragments and accumulating in FP32.

This means:

- user-visible inputs stay `float32`
- output remains `float32` for real contractions and `complex64` after recomposition
- the tensor-core path is explicitly a TF32 compute path, not "true FP32 tensor-core math"

### 3. BF16 Path Uses Tensor Cores Natively

For BF16 forward:

- stage BF16 tiles directly into tensor-core fragments
- accumulate in FP32
- keep the public output contract unchanged

### 4. Vector Forward Reuses Real Tensor-Core Kernels

Do not solve native complex/vector tensor-core MMA first.

Instead, keep the current higher-level vector-forward structure:

- decompose vector complex input into stacked real slices
- run those real slices through the new tensor-core backend
- recombine into spheroidal and toroidal complex outputs as today

This gives most of the tensor-core benefit with much less implementation risk than a first-pass native complex MMA formulation.

### 5. Existing Complex Forward Kernels Stay as Fallbacks

The direct complex forward CUDA path remains in-tree and valid. It is not the first tensor-core landing zone.

Fallback order should be:

- tensor-core path if supported and selected
- current TMA-fed FMA path if supported and selected
- current scalar-FMA path otherwise

## Kernel Design

### Data Movement

The tensor-core kernels should stage both inputs and weights as tiles in shared memory. On Hopper/Blackwell-class devices, TMA should continue to serve as the preferred bulk-load mechanism when the layout allows it.

The point of the new backend is:

- TMA remains the data feeder
- tensor cores replace scalar FMAs as the math engine

This avoids repeating the current pattern where the input path is modernized but the inner compute loop remains a scalar register loop.

### Compute Structure

Each large forward tensor-core kernel should:

1. choose a packed triangular tile in `(l, m)` space like the current large forward kernels
2. stage an input tile and a weight tile into shared memory
3. convert or pack FP32 tiles into TF32-compatible fragments when running the FP32 public path
4. execute tensor-core MMA for the tile
5. accumulate in FP32
6. write results back in the existing output layout

### Batch-Tiled Variant

Keep the batch-2 idea from the current TMA path.

There should be a batch-tiled tensor-core variant that:

- processes two batch lanes per block
- reuses the same weight tile across both lanes
- competes directly with the existing `*_batch2` TMA/FMA kernels during autotuning

### Dtype/Shape Gating

Tensor-core kernels should initially be gated to:

- CUDA only
- forward only
- large-grid forward cases
- SM90+ only
- dtypes: BF16 directly, FP32 via TF32 staging

Small shapes should keep using the current non-tensor-core kernels because the extra staging and autotune complexity is unlikely to win there.

## Backend Selection and Autotuning

### Replace Fixed Heuristics with Runtime Selection

Current selection is largely:

- architecture heuristic
- hard-coded thresholds
- environment-variable overrides

Replace this with a lightweight runtime autotuner for large forward CUDA cases.

### Candidate Set

For each supported shape key, benchmark a small candidate set:

- existing scalar FMA large kernel
- existing TMA-fed large kernel
- new tensor-core TF32 kernel
- new tensor-core BF16 kernel
- batch-2 variants where applicable

Candidates that are invalid for the current shape/dtype/device are skipped, not treated as failures.

### Autotune Key

The persistent key should include:

- device architecture / GPU identifier
- operation kind:
  - scalar-real-forward
  - scalar-complex-forward
  - vector-real-composed-forward
- dtype mode:
  - fp32
  - bf16
- shape:
  - `nlat`
  - `lmax`
  - `mmax`
- a coarse batch bucket:
  - `1`
  - `2`
  - `3-4`
  - `5+`

The goal is to avoid tuning every possible exact batch size while still distinguishing the cases that matter for weight reuse.

### Persistent Cache

Use a small on-disk JSON cache stored by default at:

- `build/holysht_autotune_cache.json`

Required controls:

- `HOLYSHT_AUTOTUNE=0|1`
- `HOLYSHT_AUTOTUNE_REBENCH=1`
- `HOLYSHT_AUTOTUNE_CACHE_PATH=<path>`
- `HOLYSHT_FORCE_BACKEND=fma|tma|tc_tf32|tc_bf16`

Behavior:

- if a cached winner exists, use it
- if not, benchmark a short candidate list and persist the winner
- if a forced backend is set, skip tuning and use the forced choice when valid

### Guardrails

- no autotuning on tiny shapes
- no autotuning in branches where `x.requires_grad` is true during the initial forward-only implementation
- candidates that error or fail tolerance checks are blacklisted for that key
- tuning should use a small warmup/measure cycle, not an exhaustive search

## Python Integration

The Python dispatcher in `torch-ext/holysht/__init__.py` should stop relying on a single hard-coded threshold like `_tma_pad_complex_min_nlat = 512` as the main large-grid decision lever.

Instead:

- route large forward CUDA cases through an autotuned backend chooser
- keep explicit fallback rules for unsupported dtypes, devices, or grad-sensitive paths
- preserve current env-var overrides for debugging and A/B testing

The tensor-core path should first appear in:

- scalar BF16 forward
- scalar FP32 forward
- vector forward through the stacked-real composition path

## Error Handling

- If tensor-core kernels are unavailable for the target architecture, skip them cleanly and continue with the current kernels.
- If an autotune candidate throws or produces an out-of-tolerance result, record that candidate as invalid for the current key and continue evaluating other candidates.
- If the persistent autotune cache is unreadable or corrupted, fall back to in-memory tuning behavior and rewrite the cache on successful selection.
- If `HOLYSHT_FORCE_BACKEND` requests an invalid backend for the current case, fall back with a clear warning rather than hard-failing normal execution.

## Testing Strategy

### Correctness

Add tests that verify:

1. backend selection logic can choose tensor-core candidates for supported synthetic keys
2. fallback behavior remains correct when tensor-core support is unavailable
3. forced backend selection works and invalid forced selections degrade safely
4. the persistent autotune cache can be written, read, and reused
5. the tensor-core-enabled forward paths match reference tolerances for:
   - scalar FP32 forward
   - scalar BF16 forward
   - vector FP32 forward
   - vector BF16 forward

### Behavioral Tests

Add tests around:

- cache-key bucketing by batch size
- cache reuse across repeated calls
- rebench behavior when `HOLYSHT_AUTOTUNE_REBENCH=1`
- README-facing env vars and selection helpers where practical

### Performance Validation

Use the existing benchmark harness to validate:

- tensor-core TF32 beats or at least competes with TMA/FMA on the intended large forward cases
- tensor-core BF16 beats or at least competes with the current BF16 path
- autotuning converges on sensible winners for representative shapes

### Resource Validation

Extend the existing resource/profiling helpers to record tensor-core kernel variants, including:

- regs/thread
- shared memory/block
- active blocks/SM
- theoretical occupancy

## Documentation Changes

Update `README.md` to:

- describe the new tensor-core forward backend
- explain that FP32 forward on tensor cores is implemented through TF32-style execution with FP32 accumulation
- document autotune behavior and the persistent cache env vars
- say explicitly that the Hugging Face kernel packaging target is forward-only by design
- update the "next step" language so the README no longer describes tensor-core work as future tense once this lands

## Rollout Plan

1. Land tensor-core backend selection scaffolding and autotune/cache infrastructure behind conservative gating.
2. Land scalar real tensor-core kernels first.
3. Route BF16 scalar forward through the new backend.
4. Route FP32 scalar forward through TF32-backed tensor-core execution.
5. Switch vector forward to reuse the new real tensor-core backend.
6. Update benchmarks, resource reporting, tests, and README once the backend is stable.

## Risks

- Tensor-core tiling may underperform current TMA/FMA kernels on some medium-size shapes, which is why autotuning is part of the design rather than an optional follow-up.
- FP32-via-TF32 may require tighter clarity around tolerances in tests and benchmark reporting.
- Adding too many candidate kernels can make tuning noisy or slow; candidate count must stay intentionally small.

## Acceptance Criteria

This design is complete when:

1. large forward CUDA workloads can select a tensor-core backend on supported devices
2. FP32 public forward paths can execute through TF32-backed tensor-core kernels with acceptable numerical error
3. BF16 forward paths can execute through tensor-core kernels with FP32 accumulation
4. large forward backend selection is autotuned and cached persistently
5. current fallback paths remain correct
6. the README explicitly documents the forward-only Hugging Face kernel positioning
