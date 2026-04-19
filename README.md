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

> Note. HOLYSHT is a prod-ready kernel running in prod in many places. This is 
> despite the fact that I am documenting the process here as we go along. I think
> more developers should document their attempts to improve their code, especially
> in low level programming, where sometimes there's a temptation to insist on an 
> air of effortless wizardry. No good kernel code is effortless. It's dead ends, 
> hours of work for another 0.1% performance and sometimes a real slog. And that's
> the beauty of it. We shouldn't be ashamed of that.

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

Forward benchmarks were run on an NVIDIA GB10 with
PyTorch `2.10.0+cu130`, CUDA `13.0`, batch size `4`, and

```bash
PYTHONPATH=torch-ext python3 benchmarks/bench_torch_harmonics.py \
  --device cuda \
  --scenarios scalar-forward,vector-forward,bf16-forward \
  --grids 256x512,512x1024 \
  --batch-sizes 4 \
  --warmup 10 \
  --iters 30 \
  --max-alloc-gib 6 \
  --force-backend auto
```


| Workload | Grid | torch-harmonics | HOLYSHT | Speedup | MaxAbsErr |
|---|---|---:|---:|---:|---:|
| `RealSHT.forward` | `256x512` | 2.581 ms | 0.803 ms | 3.2x | `2.05e-08` |
| `RealSHT.forward` | `512x1024` | 19.402 ms | 6.668 ms | 2.9x | `1.86e-08` |
| `RealVectorSHT.forward` | `256x512` | 10.102 ms | 0.991 ms | 10.2x | `7.45e-09` |
| `RealVectorSHT.forward` | `512x1024` | 77.976 ms | 6.855 ms | 11.4x | `4.69e-09` |
| `RealSHT.forward (BF16)` | `256x512` | 2.582 ms | 0.553 ms | 4.7x | `6.68e-05` |
| `RealSHT.forward (BF16)` | `512x1024` | 19.466 ms | 6.596 ms | 3.0x | `3.48e-05` |
| `RealVectorSHT.forward (BF16)` | `256x512` | 9.877 ms | 0.968 ms | 10.2x | `3.73e-09` |
| `RealVectorSHT.forward (BF16)` | `512x1024` | 77.484 ms | 6.884 ms | 11.3x | `5.63e-09` |

This is largely due to the real native-vector tensor-core backends under that
`native_vector` path:

- `tc_tf32` now dispatches to `fused_vector_legendre_forward_tf32_kernel`
- `tc_bf16` now dispatches to `fused_vector_legendre_forward_bf16_kernel`

On GB10, the second-level backend selector still prefers `tma` for the
hot public vector cases because both tensor-core variants are slower than the
native TMA kernel today.

Representative FP32 strategy/backend A/B on `512x1024`, batch `4`:

| Public vector forward case | `native_vector+tma` | `native_vector+tc_tf32` | `stacked_real+tc_tf32` | Auto |
|---|---:|---:|---:|---:|
| `RealVectorSHT.forward` | 6.859 ms | 17.606 ms | 23.459 ms | 6.855 ms |

Representative BF16 backend A/B on `512x1024`, batch `4`:

| Public vector forward case | `native_vector+tma` | `native_vector+tc_bf16` | Auto |
|---|---:|---:|---:|
| `RealVectorSHT.forward (BF16)` | 7.014 ms | 16.680 ms | 6.884 ms |


Odd-`mmax` public forward shapes are still padded into tile-aligned temporary
spectral slabs before the Legendre stage when that actually helps on GB10.
TMA itself only needs 16-byte alignment, but the current forward microkernels
still prefer 8-wide `m` tiles, so the complex public path intentionally keeps
the wider pad.

## On Metal/MPS

The non-CUDA benchmark suite was run on an M4 Mini with PyTorch `2.11.0`, Apple Metal/MPS, batch size `4`, and

```bash
PYTHONPATH=torch-ext .venv/bin/python benchmarks/bench_torch_harmonics.py \
  --quick \
  --device mps \
  --scenarios scalar-forward,scalar-inverse,roundtrip,vector-forward,vector-inverse,ynm,scalar-train,vector-train \
  --max-alloc-gib 6 \
  --output data/bench_torch_harmonics_mps.json
```

In short:

- vector forward and inverse are consistently faster, including the largest
  tested `512x1024` cases
- sparse synthesis is slightly faster
- scalar forward only wins on smaller grids
- scalar inverse, roundtrip, and scalar training are still slower on the
  larger tested MPS cases

Representative MPS results from that run:

| Workload | Grid | torch-harmonics | HOLYSHT | Speedup | MaxAbsErr |
|---|---|---:|---:|---:|---:|
| `RealSHT.forward` | `512x1024` | 15.000 ms | 20.465 ms | 0.7x | `1.69e-08` |
| `InverseRealSHT.forward` | `512x1024` | 8.844 ms | 19.963 ms | 0.4x | `1.67e-06` |
| `Roundtrip(SHT→ISHT)` | `512x1024` | 22.106 ms | 39.421 ms | 0.6x | `1.97e-06` |
| `RealVectorSHT.forward` | `512x1024` | 55.311 ms | 37.480 ms | 1.5x | `3.52e-09` |
| `InverseRealVectorSHT.forward` | `512x1024` | 56.159 ms | 39.644 ms | 1.4x | `2.32e-06` |
| `Y_n^m synthesis (ISHT, sparse)` | `128x256 (lmax=64)` | 0.323 ms | 0.282 ms | 1.1x | `0.00e+00` |
| `RealSHT.forward+backward` | `256x512` | 5.399 ms | 6.400 ms | 0.8x | `8.88e-16` |
| `RealVectorSHT.forward+backward` | `256x512` | 21.424 ms | 12.586 ms | 1.7x | `3.24e-12` |

The best Apple wins in this run were the vector paths:

- `RealVectorSHT.forward 64x128`: **3.25x**
- `InverseRealVectorSHT.forward 64x128`: **2.81x**
- `RealVectorSHT.forward+backward 256x512`: **1.70x**

The full structured results for this box are in
`data/bench_torch_harmonics_mps.json`, and `data/bench_torch_harmonics.json`.



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
- `HOLYSHT_FORCE_VECTOR_STRATEGY` can be set to `native_vector` or
  `stacked_real` to override the top-level CUDA vector-forward strategy.
- `HOLYSHT_AUTOTUNE` is enabled by default. When it is unset or truthy, forward
  backend choices are cached in `~/.cache/holysht/holysht_autotune_cache.json`
  by default, or at the path named by `HOLYSHT_AUTOTUNE_CACHE_PATH`.
  `XDG_CACHE_HOME` shifts that default user cache root. Set
  `HOLYSHT_AUTOTUNE=0|false|no|off` to skip cache and benchmark autotuning and
  use a deterministic preference order instead (`tma`, then `fma`, then the
  first available candidate).
- `scripts/profile_nsys.sh` and `scripts/profile_ncu.sh` now honour
  `HOLYSHT_PROFILE_SCENARIOS`, `HOLYSHT_PROFILE_GRIDS`,
  `HOLYSHT_PROFILE_BATCH_SIZES`, `HOLYSHT_PROFILE_WARMUP`,
  `HOLYSHT_PROFILE_ITERS`, `HOLYSHT_PROFILE_MAX_ALLOC_GIB`, and
  `HOLYSHT_PROFILE_OUTPUT_STEM`, so the Nsight helpers can target scalar,
  vector, or BF16 cases without editing the scripts.
- `scripts/profile_ncu.sh` still falls back to `scripts/report_resources.py`
  when GPU counters are unavailable for the current user.

One caveat from the current branch: public SHT workloads derived from even
`nlon` usually have `mmax = nlon / 2 + 1`, which is odd. On GB10 that means the
raw Legendre input stride is not 16-byte aligned. The public forward wrappers
handle that by padding into a tile-aligned temporary slab for vector forward,
BF16 forward, and large scalar forward. Small scalar FP32 shapes such as
`256x512` still stay on the packed non-TMA kernel because the padding overhead
was slower there.

Representative isolated HOLYSHT-only `nsys` runs on `512x1024`, batch `4`:

- Vector `native_vector+tma`
  (`HOLYSHT_FORCE_VECTOR_STRATEGY=native_vector HOLYSHT_FORCE_BACKEND=tma`):
  the hot kernel is `fused_vector_legendre_forward_large_tma_batch2_kernel<8>`,
  which takes **92.4%** of GPU kernel time with a **6.34 ms** average launch
  time across `30` launches.
- Vector `native_vector+tc_tf32`
  (`HOLYSHT_FORCE_VECTOR_STRATEGY=native_vector HOLYSHT_FORCE_BACKEND=tc_tf32`):
  the hot kernel is `fused_vector_legendre_forward_tf32_kernel`, which takes
  **97.1%** of GPU kernel time with a **17.64 ms** average launch time across
  `30` launches.

That profile is the main reason the second-level backend selector still picks
`tma` on GB10: once the public path stays in the native vector kernel family,
the remaining gap is inside the tensor-core kernel itself, not in wrapper
packing or fallback composition.

For comparison, forcing `HOLYSHT_FORCE_VECTOR_STRATEGY=stacked_real` with
`HOLYSHT_FORCE_BACKEND=tc_tf32` still drops the public
`RealVectorSHT.forward(512x1024, batch=4)` case to **23.459 ms**. So the new
native vector TF32 backend is already a real improvement over composed TF32,
but it is still nowhere near the native TMA winner on this GPU.

Current forward-kernel resource snapshot from
`PYTHONPATH=torch-ext python3 scripts/report_resources.py` on GB10:

| Kernel | Regs/thread | Shared/block | Block threads | Active blocks/SM | Theoretical occupancy |
|---|---:|---:|---:|---:|---:|
| Scalar forward large | 63 | 5248 B | 256 | 4 | 66.7% |
| Scalar forward large TMA | 48 | 9224 B | 256 | 5 | 83.3% |
| Scalar forward large TMA batch2 | 59 | 17424 B | 256 | 4 | 66.7% |
| Scalar forward TF32 WMMA | 72 | 3072 B | 32 | 28 | 58.3% |
| Vector forward large | 64 | 9472 B | 256 | 4 | 66.7% |
| Vector forward large TMA | 48 | 17424 B | 256 | 5 | 83.3% |
| Vector forward large TMA batch2 | 24 | 33824 B | 256 | 3 | 50.0% |
| Vector forward TF32 WMMA | 156 | 12288 B | 32 | 8 | 16.7% |
| Vector forward BF16 WMMA | 206 | 12288 B | 32 | 8 | 16.7% |
| BF16 forward large | 63 | 3136 B | 256 | 4 | 66.7% |
| BF16 forward large TMA | 40 | 3080 B | 256 | 6 | 100.0% |
| BF16 forward large TMA batch2 | 40 | 5136 B | 256 | 6 | 100.0% |

The updated resource table makes the current trade visible. The scalar TMA
batch2 kernel still wins public scalar forward despite its larger shared-memory
footprint, and the new vector WMMA kernels are clearly real kernels now rather
than API aliases. But on GB10 they come in at **156** and **206** regs/thread
with only **16.7%** theoretical occupancy, which lines up with the `nsys`
result that native vector `tma` still wins decisively. On this machine the real
fix was not "force tensor cores harder" but "keep public vector forward in the
native kernel family and let autotune reject the slower TC variants."

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
- `tc_tf32` and `tc_bf16` now both map to real native forward tensor-core
  backends on SM90+, but on GB10 the autotuner still prefers `tma` for the
  hot public vector workloads.
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
HOLYSHT_FORCE_BACKEND=tma HOLYSHT_PROFILE_SCENARIOS=scalar-forward \
  HOLYSHT_PROFILE_GRIDS=512x1024 HOLYSHT_PROFILE_BATCH_SIZES=4 \
  ./scripts/profile_nsys.sh
HOLYSHT_FORCE_VECTOR_STRATEGY=native_vector HOLYSHT_PROFILE_SCENARIOS=vector-forward \
  HOLYSHT_PROFILE_GRIDS=512x1024 HOLYSHT_PROFILE_BATCH_SIZES=4 \
  ./scripts/profile_nsys.sh
HOLYSHT_FORCE_BACKEND=tma ./scripts/profile_ncu.sh
python3 scripts/report_resources.py
```

## Running tests

```bash
PYTHONPATH=torch-ext pytest tests/test_holysht.py
```

## Author

I'm [Chris von Csefalvay](chrisvoncsefalvay.com), an AI researcher specialising in post-training, and the author of _[Post-Training: A Practical Guide for
AI Engineers and Developers](https://posttraining.guide)_ (No Starch Press, 2026). I also write [Post-Slop](https://postslop.substack.com), a periodic diatribe about AI, and what it's doing for society. You can also find me on [LinkedIn](https://linkedin.com/in/chrisvoncsefalvay) and [X](https://x.com/epichrisis).

## License

MIT
