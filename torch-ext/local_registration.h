// HOLYSHT
// SPDX-License-Identifier: MIT
// Author: Chris von Csefalvay
// Repository: https://github.com/chrisvoncsefalvay/holysht
// Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
//
// Local stub for kernel-builder's registration.h
// Used during JIT compilation; the real header is provided by kernel-builder at package time.
#pragma once

#ifndef CUDA_KERNEL
#define CUDA_KERNEL
#endif

// kernel-builder provides TORCH_LIBRARY_EXPAND; for JIT builds, map to TORCH_LIBRARY
#ifndef TORCH_LIBRARY_EXPAND
#define TORCH_LIBRARY_EXPAND(name, m) TORCH_LIBRARY(name, m)
#endif

#define REGISTER_EXTENSION(name) /* noop for JIT builds */
