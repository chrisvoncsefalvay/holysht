#!/usr/bin/env python3
"""Summarise CUDA kernel resource usage from cuobjdump output.

Author: Chris von Csefalvay
Licence: MIT
Repository: https://github.com/chrisvoncsefalvay/holysht
Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch


RESOURCE_RE = re.compile(
    r"Function (?P<name>[^:]+):\n"
    r"\s+REG:(?P<reg>\d+) STACK:(?P<stack>\d+) SHARED:(?P<shared>\d+) LOCAL:(?P<local>\d+)",
    re.MULTILINE,
)
TILE_RE = re.compile(r"Li(?P<tile>\d+)EE")


@dataclass
class KernelResource:
    name: str
    regs: int
    shared: int
    tile_l: int | None

    @property
    def block_threads(self) -> int | None:
        return None if self.tile_l is None else 32 * self.tile_l


def parse_resources(binary: Path) -> list[KernelResource]:
    output = subprocess.check_output(
        ["cuobjdump", "--dump-resource-usage", str(binary)],
        text=True,
    )
    kernels = []
    for match in RESOURCE_RE.finditer(output):
        name = match.group("name")
        tile_match = TILE_RE.search(name)
        kernels.append(
            KernelResource(
                name=name,
                regs=int(match.group("reg")),
                shared=int(match.group("shared")),
                tile_l=int(tile_match.group("tile")) if tile_match else None,
            )
        )
    return kernels


def active_blocks_per_sm(kernel: KernelResource, props) -> tuple[int | None, float | None]:
    threads = kernel.block_threads
    if threads is None:
        return None, None

    regs_per_block = kernel.regs * threads
    reg_limited = props.regs_per_multiprocessor // regs_per_block if regs_per_block else 0
    smem_limited = (
        props.shared_memory_per_multiprocessor // kernel.shared
        if kernel.shared
        else math.inf
    )
    thread_limited = props.max_threads_per_multi_processor // threads
    active = int(min(reg_limited, smem_limited, thread_limited))
    occupancy = (active * threads) / props.max_threads_per_multi_processor if active else 0.0
    return active, occupancy


def short_name(name: str) -> str:
    if "fused_legendre_forward_large_tma_batch2_kernel" in name:
        return "scalar forward large tma batch2"
    if "fused_legendre_forward_large_tma_kernel" in name:
        return "scalar forward large tma"
    if "fused_legendre_forward_large_kernel" in name:
        return "scalar forward large"
    if "fused_legendre_inverse_large_tma_kernel" in name:
        return "scalar inverse large tma"
    if "fused_legendre_inverse_large_kernel" in name:
        return "scalar inverse large"
    if "fused_vector_legendre_forward_large_tma_batch2_kernel" in name:
        return "vector forward large tma batch2"
    if "fused_vector_legendre_forward_large_tma_kernel" in name:
        return "vector forward large tma"
    if "fused_vector_legendre_forward_large_kernel" in name:
        return "vector forward large"
    if "fused_vector_legendre_inverse_large_tma_kernel" in name:
        return "vector inverse large tma"
    if "fused_vector_legendre_inverse_large_kernel" in name:
        return "vector inverse large"
    if "fused_legendre_forward_real_large_tma_batch2_kernel" in name and "BFloat16" in name:
        return "bf16 forward large tma batch2"
    if "fused_legendre_forward_real_large_tma_kernel" in name and "BFloat16" in name:
        return "bf16 forward large tma"
    if "fused_legendre_forward_real_large_kernel" in name and "BFloat16" in name:
        return "bf16 forward large"
    if "fused_legendre_forward_real_tf32_kernel" in name:
        return "scalar forward tf32 wmma"
    if "prepare_irfft_inplace" in name:
        return "prepare irfft"
    return name


def resolve_binary(binary_arg: str) -> Path:
    if binary_arg:
        binary = Path(binary_arg)
        if binary.exists():
            return binary
        raise FileNotFoundError(f"extension binary not found: {binary}")

    candidates = [
        Path("build/torch_extensions/holysht_ops_cuda.so"),
        Path("build/torch_extensions/holysht_ops.so"),
        Path("build/torch_extensions/holysht_ops_metal.so"),
    ]
    candidates.extend(sorted(Path("build/torch_extensions").glob("holysht_ops*.so")))
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "could not find a HOLYSHT extension binary under build/torch_extensions; "
        "pass --binary explicitly"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report HOLYSHT kernel resources")
    parser.add_argument(
        "--binary",
        default="",
        help="Path to the compiled extension shared object. Defaults to auto-discovery under build/torch_extensions.",
    )
    args = parser.parse_args()

    props = torch.cuda.get_device_properties(0)
    binary = resolve_binary(args.binary)
    kernels = parse_resources(binary)

    interesting = [
        kernel for kernel in kernels
        if any(
            token in kernel.name
            for token in (
                "fused_legendre_forward_large_tma_kernelILi8EE",
                "fused_legendre_forward_large_tma_batch2_kernelILi8EE",
                "fused_legendre_forward_large_kernelILi8EE",
                "fused_legendre_inverse_large_tma_kernelILi8EE",
                "fused_legendre_inverse_large_kernelILi8EE",
                "fused_vector_legendre_forward_large_tma_kernelILi8EE",
                "fused_vector_legendre_forward_large_tma_batch2_kernelILi8EE",
                "fused_vector_legendre_forward_large_kernelILi8EE",
                "fused_vector_legendre_inverse_large_tma_kernelILi8EE",
                "fused_vector_legendre_inverse_large_kernelILi8EE",
                "fused_legendre_forward_real_large_tma_kernelIN3c108BFloat16ELi8EE",
                "fused_legendre_forward_real_large_tma_batch2_kernelIN3c108BFloat16ELi8EE",
                "fused_legendre_forward_real_large_kernelIN3c108BFloat16ELi8EE",
                "fused_legendre_forward_real_tf32_kernel",
                "prepare_irfft_inplace",
            )
        )
    ]

    print(f"# resource summary for {props.name}")
    print(f"# binary: {binary}")
    print()
    print("| Kernel | Regs/thread | Shared/block (bytes) | Block threads | Active blocks/SM | Theoretical occupancy |")
    print("|---|---:|---:|---:|---:|---:|")
    for kernel in interesting:
        active_blocks, occupancy = active_blocks_per_sm(kernel, props)
        print(
            f"| {short_name(kernel.name)} | {kernel.regs} | {kernel.shared} | "
            f"{kernel.block_threads or '-'} | {active_blocks or '-'} | "
            f"{f'{occupancy * 100:.1f}%' if occupancy is not None else '-'} |"
        )


if __name__ == "__main__":
    main()
