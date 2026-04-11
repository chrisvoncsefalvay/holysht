#!/usr/bin/env bash
# HOLYSHT
# Author: Chris von Csefalvay
# Licence: MIT
# Repository: https://github.com/chrisvoncsefalvay/holysht
# Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/profiles"

mkdir -p "${OUT_DIR}"

export PYTHONPATH="${ROOT_DIR}/torch-ext${PYTHONPATH:+:${PYTHONPATH}}"
export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
export HOLYSHT_ENABLE_NVTX=1

LOG_FILE="$(mktemp)"

set +e
ncu \
  --target-processes all \
  --force-overwrite true \
  --set full \
  -o "${OUT_DIR}/holysht_scalar_forward_ncu" \
  python3 "${ROOT_DIR}/benchmarks/bench_torch_harmonics.py" \
    --quick \
    --scenarios scalar-forward \
    --grids 512x1024 \
    --batch-sizes 4 \
    --warmup 5 \
    --iters 10 \
    --max-alloc-gib 6 2>&1 | tee "${LOG_FILE}"
NCU_STATUS=${PIPESTATUS[0]}
set -e

if grep -Eq "ERR_NVGPUCTRPERM|No kernels were profiled" "${LOG_FILE}"; then
  echo
  echo "ncu counters are not available, or no kernels were captured; falling back to cuobjdump resource reporting."
  python3 "${ROOT_DIR}/scripts/report_resources.py"
  exit 0
fi

exit "${NCU_STATUS}"
