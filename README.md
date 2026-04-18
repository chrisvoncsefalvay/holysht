# HOLYSHT: highly optimised Legendre $Y_l^m$ SHT

HOLYSHT is a focused GPU acceleration layer for spherical harmonic transforms
in the `torch-harmonics` ecosystem. It keeps the parts that proved real in
benchmarking: custom CUDA Legendre kernels, an Apple Metal/MPS backend for
arm64 Macs, dedicated vector SHT kernels, explicit autograd support,
mixed-precision forward paths, and a profiling setup that can be run safely on
a GB10 without blowing memory.

The Hugging Face kernel distribution is intentionally forward-only. That is a
deliberate packaging choice: the fastest, most tuned path is forward SHT, while
inverse and training support remain in the main source tree for local
development and regression coverage.

The package is designed as a practical companion to
[`torch-harmonics`](https://github.com/NVIDIA/torch-harmonics), not a full
reimplementation. HOLYSHT still reuses `torch-harmonics` to generate quadrature
weights, then replaces the slowest execution paths with backend-specific code
tuned for the current target.

<p align="center">
  <img src="examples/mars_weather.gif" alt="Martian weather simulation using HOLYSHT — spectral advection on a 256×512 grid with real MOLA topography" width="800">
  <br>
  <sub>Spectral advection on Mars (256×512 grid, real MOLA topography). Left: terrain + weather. Right: flow dynamics with velocity streamlines. Simulated at 300+ frames/s on an NVIDIA GB10.</sub>
</p>

## What this repo contains

- `torch-ext/holysht/`
  The public Python package with drop-in `RealSHT`, `InverseRealSHT`,
  `RealVectorSHT`, and `InverseRealVectorSHT` modules.
- `cuda/`
  CUDA kernels for scalar Legendre, vector Legendre composition, BF16 real
  reductions, and inverse-FFT preparation.
- `metal/`
  Metal kernels and the Objective-C++ MPS bridge used on Apple Silicon.
- `torch-ext/torch_binding.cpp`
  Torch extension registration for CUDA and Metal/MPS ops.
- `benchmarks/`
  End-to-end benchmarks for forward, inverse, training, and BF16 paths.
- `scripts/`
  Small profiling and resource-report helpers for `nsys`, `ncu`, and
  `cuobjdump`.
- `tests/`
  Parity tests against `torch-harmonics`, including backward and non-contiguous
  input coverage.

## Performance

Forward benchmarks were rerun on April 18, 2026 on an NVIDIA GB10 with
PyTorch `2.10.0+cu130`, CUDA `13.0`, batch size `4`, and

```bash
PYTHONPATH=torch-ext HOLYSHT_USE_TMA=1 python3 benchmarks/bench_torch_harmonics.py \
  --scenarios scalar-forward,vector-forward,bf16-forward \
  --grids 256x512,512x1024 \
  --batch-sizes 4 \
  --warmup 10 \
  --iters 30 \
  --max-alloc-gib 6
```

All `8/8` forward checks passed; the mean speedup over `torch-harmonics` was
**5.8x**.

| Workload | Grid | torch-harmonics | HOLYSHT | Speedup | MaxAbsErr |
|---|---|---:|---:|---:|---:|
| `RealSHT.forward` | `256x512` | 2.580 ms | 0.681 ms | 3.8x | `2.24e-08` |
| `RealSHT.forward` | `512x1024` | 19.420 ms | 3.453 ms | 5.6x | `1.78e-08` |
| `RealVectorSHT.forward` | `256x512` | 9.961 ms | 0.978 ms | 10.2x | `1.30e-08` |
| `RealVectorSHT.forward` | `512x1024` | 77.319 ms | 6.858 ms | 11.3x | `6.49e-09` |
| `RealSHT.forward (BF16)` | `256x512` | 2.530 ms | 0.540 ms | 4.7x | `6.95e-05` |
| `RealSHT.forward (BF16)` | `512x1024` | 19.342 ms | 6.558 ms | 3.0x | `3.45e-05` |
| `RealVectorSHT.forward (BF16)` | `256x512` | 9.959 ms | 2.049 ms | 4.9x | `3.38e-05` |
| `RealVectorSHT.forward (BF16)` | `512x1024` | 77.177 ms | 26.347 ms | 2.9x | `1.16e-05` |

The large Hopper/Blackwell forward path is now a real microkernel rather than
just "TMA for the input tile": each large forward TMA block can process two
batch lanes at once, so the weight stream is reused across multiple outputs
instead of being paid once per batch item. BF16 still goes through the native
CUDA real Legendre kernels on GB10, but now benefits from the same batch-tiled
forward microkernel. FP32 forward can also route through the TF32 WMMA kernel
when the autotuner selects `tc_tf32`. `tc_bf16` is currently an explicit
compatibility alias that falls back to the legacy real forward path until a
true BF16 tensor-core kernel lands.

Odd-`mmax` public forward shapes are still padded into tile-aligned temporary
spectral slabs before the Legendre stage when that actually helps on GB10.
TMA itself only needs 16-byte alignment, but the current forward microkernels
still prefer 8-wide `m` tiles, so the complex public path intentionally keeps
the wider pad.

The Apple Metal/MPS backend remains in-tree and validated separately on Apple
Silicon; the table above is the current GB10 forward rerun.

## Profiling and resources

The project ships a lightweight profiling path that avoids the heavyweight
`kernel-builder` workflow during normal development:

- First import builds a local torch extension under `build/torch_extensions/`.
- The loader defaults to `MAX_JOBS=1` and `TORCH_CUDA_ARCH_LIST=12.0+PTX` on
  GB10-class machines.
- `HOLYSHT_USE_TMA=0` or `1` now disables or forces the Hopper/Blackwell TMA
  path for A/B profiling.
- `HOLYSHT_TMA_BATCH_TILE=1` or `2` selects whether the large forward TMA path
  processes one or two batch lanes per block.
- `HOLYSHT_FORCE_BACKEND` can be set to `fma`, `tma`, `tc_tf32`, or
  `tc_bf16`. The benchmark runner and module forwards surface the selected
  backend in logs when this is set.
- `HOLYSHT_AUTOTUNE` is enabled by default. When it is unset or truthy, forward
  backend choices are cached in `~/.cache/holysht/holysht_autotune_cache.json`
  by default, or at the path named by `HOLYSHT_AUTOTUNE_CACHE_PATH`.
  `XDG_CACHE_HOME` shifts that default user cache root. Set
  `HOLYSHT_AUTOTUNE=0|false|no|off` to skip cache and benchmark autotuning and
  use a deterministic preference order instead (`tma`, then `fma`, then the
  first available candidate).
- `scripts/profile_nsys.sh` profiles the scalar forward case with `nsys`.
- `scripts/profile_ncu.sh` attempts `ncu`; if GPU counters are unavailable for
  the current user it falls back to `scripts/report_resources.py`.

One caveat from the current branch: public SHT workloads derived from even
`nlon` usually have `mmax = nlon / 2 + 1`, which is odd. On GB10 that means the
raw Legendre input stride is not 16-byte aligned. The public forward wrappers
handle that by padding into a tile-aligned temporary slab for vector forward,
BF16 forward, and large scalar forward. Small scalar FP32 shapes such as
`256x512` still stay on the packed non-TMA kernel because the padding overhead
was slower there.

Representative `nsys` runs on `512x1024`, batch `4`:

- `fused_legendre_forward_large_tma_batch2_kernel<8>` averaged **3.20 ms** per
  launch on `RealSHT.forward(512x1024, batch=4)`.
- `fused_vector_legendre_forward_large_tma_batch2_kernel<8>` averaged
  **6.43 ms** per launch on `RealVectorSHT.forward(512x1024, batch=4)`.

The most useful A/B on GB10 is now `HOLYSHT_TMA_BATCH_TILE=1` vs `2` on the
public forward workloads. On batch `4`, the batch-tiled microkernel wins by
reusing weights across two batch lanes:

| Public forward case | `B_TILE=1` | `B_TILE=2` | Delta |
|---|---:|---:|---:|
| `RealSHT.forward 512x1024` | 6.419 ms | 3.453 ms | 1.86x |
| `RealVectorSHT.forward 512x1024` | 12.983 ms | 6.858 ms | 1.89x |
| `RealSHT.forward (BF16) 512x1024` | 12.601 ms | 6.558 ms | 1.92x |
| `RealVectorSHT.forward (BF16) 512x1024` | 50.365 ms | 26.347 ms | 1.91x |

That is a much healthier place to be, but it is not the end of the road. The
current microkernel still scalar-loads the weight tile inside each thread; it
just amortises those loads across two batch lanes. The next Hopper/Blackwell
step is still a denser weight-staging / tensor-core formulation rather than
more input-side plumbing.

Current forward-kernel resource snapshot from
`PYTHONPATH=torch-ext python3 scripts/report_resources.py` on GB10:

| Kernel | Regs/thread | Shared/block | Block threads | Active blocks/SM | Theoretical occupancy |
|---|---:|---:|---:|---:|---:|
| Scalar forward large | 63 | 5248 B | 256 | 4 | 66.7% |
| Scalar forward large TMA | 48 | 9224 B | 256 | 5 | 83.3% |
| Scalar forward large TMA batch2 | 59 | 17424 B | 256 | 4 | 66.7% |
| Vector forward large | 64 | 9472 B | 256 | 4 | 66.7% |
| Vector forward large TMA | 48 | 17424 B | 256 | 5 | 83.3% |
| Vector forward large TMA batch2 | 24 | 33824 B | 256 | 3 | 50.0% |
| BF16 forward large | 63 | 3136 B | 256 | 4 | 66.7% |
| BF16 forward large TMA | 40 | 3080 B | 256 | 6 | 100.0% |
| BF16 forward large TMA batch2 | 40 | 5136 B | 256 | 6 | 100.0% |

The vector batch2 kernel is the clearest tradeoff: it gives up occupancy to
buy much higher weight reuse. On GB10 that trade is absolutely worth it, which
is why the measured runtime still drops from `12.983 ms` to `6.858 ms` on the
`512x1024`, batch `4` case.

## Public API

```python
import torch
from holysht import RealSHT, InverseRealSHT, RealVectorSHT

sht = RealSHT(512, 1024, grid="equiangular", norm="ortho", csphase=True).cuda()
isht = InverseRealSHT(512, 1024).cuda()

x = torch.randn(4, 512, 1024, device="cuda")
coeffs = sht(x)
x_back = isht(coeffs)

vsht = RealVectorSHT(512, 1024).cuda()
v = torch.randn(4, 2, 512, 1024, device="cuda")
v_coeffs = vsht(v)
```

The module constructors accept the same practical arguments as
`torch-harmonics`: `nlat`, `nlon`, `lmax`, `mmax`, `grid`, `norm`, and
`csphase`. HOLYSHT also exposes `dtype="bf16"` for forward real and vector
paths.

## Development notes

- The default CUDA path now uses architecture-aware launch selection with
  runtime-tunable `HOLYSHT_TILE_L` and `HOLYSHT_SMALL_GRID_THRESHOLD`
  overrides.
- The Metal path uses a hybrid dispatch policy on MPS:
  small-grid scalar transforms use the custom Metal kernel, larger scalar
  transforms fall back to the stacked einsum path, vector forward keeps the
  native real-kernel composition, and vector inverse uses the native path on
  the tuned mid-size grid range.
- `HOLYSHT_MPS_SCALAR_NATIVE_MAX_NLAT` overrides the scalar-MPS cutoff
  (`64` by default).
- `HOLYSHT_MPS_SCALAR_INVERSE_EINSUM_M_CHUNK` and
  `HOLYSHT_MPS_SCALAR_FORWARD_EINSUM_M_CHUNK` force chunked MPS fallback
  contractions over the spectral `m` dimension when unified-memory pressure is
  more important than raw latency.
- `HOLYSHT_MPS_SCALAR_INVERSE_EINSUM_CHUNK_THRESHOLD_MB` controls the automatic
  inverse-fallback chunking threshold (`1024` MiB by default). Below that size
  HOLYSHT keeps the single-shot MPS contraction.
- `HOLYSHT_MPS_VECTOR_INVERSE_NATIVE_MAX_NLAT` overrides the vector-inverse
  MPS cutoff (`512` by default).
- `HOLYSHT_MPS_VECTOR_INVERSE_NATIVE=1` forces the native Metal vector-inverse
  path for tuning experiments; `=0` forces the fallback path.
- `HOLYSHT_ENABLE_NVTX=1` enables NVTX ranges around the public hot paths.
- `HOLYSHT_USE_TMA=0` disables the Hopper/Blackwell TMA path; `=1` forces it
  on when the tensor layout satisfies the 16-byte stride requirements or when
  the public forward wrappers can cheaply pad into an aligned temporary slab.
- `HOLYSHT_TMA_BATCH_TILE=1` keeps the original one-batch-per-block TMA
  forward kernels; `=2` enables the batch-tiled forward microkernel and is the
  default on Hopper/Blackwell.
- `tc_bf16` is accepted as a backend hint for API compatibility, but it still
  degrades to the legacy real forward path until a real BF16 tensor-core
  implementation is added.
- `HOLYSHT_USE_FAST_MATH=0` disables `--use_fast_math` for the local JIT build.
- `build.toml` is pinned to `sm_120` for the kernel-builder path so it no
  longer tries to fan out across a wide architecture matrix on GB10.
- The benchmark runner still honours `HOLYSHT_MAX_ALLOC_GIB` or
  `--max-alloc-gib` to avoid unified-memory overcommit.

## Running benchmarks

```bash
PYTHONPATH=torch-ext python3 benchmarks/bench_torch_harmonics.py --quick --max-alloc-gib 6
PYTHONPATH=torch-ext python3 benchmarks/bench_torch_harmonics.py --quick --device mps
```

To profile a single case:

```bash
./scripts/profile_nsys.sh
./scripts/profile_ncu.sh
python3 scripts/report_resources.py
```

## Running tests

```bash
PYTHONPATH=torch-ext pytest tests/test_holysht.py
```

## Author

## Author

I'm [Chris von Csefalvay](chrisvoncsefalvay.com), an AI researcher specialising in post-training, and the author of _[Post-Training: A Practical Guide for
AI Engineers and Developers](https://posttraining.guide)_ (No Starch Press, 2026). I also write [Post-Slop](https://postslop.substack.com), a periodic diatribe about AI, and what it's doing for society. You can also find me on [LinkedIn](https://linkedin.com/in/chrisvoncsefalvay) and [X](https://x.com/epichrisis).

## License

MIT
