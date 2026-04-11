// HOLYSHT
// SPDX-License-Identifier: MIT
// Author: Chris von Csefalvay
// Repository: https://github.com/chrisvoncsefalvay/holysht
// Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
//
// Prefer kernel-builder's generated registration.h when it is available. For
// local JIT development, fall back to the lightweight stub in this repository.
#pragma once

#if __has_include(<registration.h>)
#include <registration.h>
#else
#include "local_registration.h"
#endif
