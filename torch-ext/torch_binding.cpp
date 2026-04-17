// HOLYSHT: Highly Optimised Legendre/Ylm/SHT
// SPDX-License-Identifier: MIT
// Author: Chris von Csefalvay
// Repository: https://github.com/chrisvoncsefalvay/holysht
// Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
#include <torch/library.h>
#include "registration_select.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Fused Legendre Transform
    ops.def("fused_legendre_forward(Tensor! output, Tensor input, Tensor weight_t) -> ()");
    ops.def("fused_legendre_inverse(Tensor! output, Tensor input, Tensor weight_t) -> ()");
    ops.def("fused_legendre_forward_real(Tensor! output, Tensor input, Tensor weight_t) -> ()");
    ops.def("fused_legendre_inverse_real(Tensor! output, Tensor input, Tensor weight_t) -> ()");
    ops.def("fused_vector_legendre_forward(Tensor! output, Tensor input, Tensor weight0_t, Tensor weight1_t) -> ()");
    ops.def("fused_vector_legendre_inverse(Tensor! output, Tensor input, Tensor weight0_t, Tensor weight1_t) -> ()");

    // SHT helpers
    ops.def("sht_prepare_irfft(Tensor! data, int mmax, int nlon) -> ()");

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("fused_legendre_forward", torch::kCUDA, &fused_legendre_forward);
    ops.impl("fused_legendre_inverse", torch::kCUDA, &fused_legendre_inverse);
    ops.impl("fused_legendre_forward_real", torch::kCUDA, &fused_legendre_forward_real);
    ops.impl("fused_legendre_inverse_real", torch::kCUDA, &fused_legendre_inverse_real);
    ops.impl("fused_vector_legendre_forward", torch::kCUDA, &fused_vector_legendre_forward);
    ops.impl("fused_vector_legendre_inverse", torch::kCUDA, &fused_vector_legendre_inverse);
    ops.impl("sht_prepare_irfft", torch::kCUDA, &sht_prepare_irfft);
#elif defined(METAL_KERNEL)
    ops.impl("fused_legendre_forward", torch::kMPS, &fused_legendre_forward);
    ops.impl("fused_legendre_inverse", torch::kMPS, &fused_legendre_inverse);
    ops.impl("fused_legendre_forward_real", torch::kMPS, &fused_legendre_forward_real);
    ops.impl("fused_legendre_inverse_real", torch::kMPS, &fused_legendre_inverse_real);
    ops.impl("fused_vector_legendre_forward", torch::kMPS, &fused_vector_legendre_forward);
    ops.impl("fused_vector_legendre_inverse", torch::kMPS, &fused_vector_legendre_inverse);
    ops.impl("sht_prepare_irfft", torch::kMPS, &sht_prepare_irfft);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
