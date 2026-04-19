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
    ops.def("fused_legendre_forward_real_ex(Tensor! output, Tensor input, Tensor weight_t, int backend_hint) -> ()");
    ops.def("fused_vector_legendre_forward(Tensor! output, Tensor input, Tensor weight0_t, Tensor weight1_t) -> ()");
    ops.def("fused_vector_legendre_forward_ex(Tensor! output, Tensor input, Tensor weight0_t, Tensor weight1_t, int backend_hint) -> ()");
    ops.def("fused_vector_legendre_inverse(Tensor! output, Tensor input, Tensor weight0_t, Tensor weight1_t) -> ()");
    ops.def("fused_vector_forward_pack_real(Tensor! stacked_w0, Tensor! stacked_w1, Tensor input) -> ()");
    ops.def("fused_vector_forward_recompose_real(Tensor! output, Tensor out_w0, Tensor out_w1) -> ()");

    // SHT helpers
    ops.def("sht_prepare_irfft(Tensor! data, int mmax, int nlon) -> ()");

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("fused_legendre_forward", torch::kCUDA, &fused_legendre_forward);
    ops.impl("fused_legendre_inverse", torch::kCUDA, &fused_legendre_inverse);
    ops.impl(
        "fused_legendre_forward_real",
        torch::kCUDA,
        static_cast<void (*)(torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>(&fused_legendre_forward_real)
    );
    ops.impl("fused_legendre_forward_real_ex", torch::kCUDA, &fused_legendre_forward_real_ex);
    ops.impl("fused_legendre_inverse_real", torch::kCUDA, &fused_legendre_inverse_real);
    ops.impl(
        "fused_vector_legendre_forward",
        torch::kCUDA,
        static_cast<void (*)(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>(&fused_vector_legendre_forward)
    );
    ops.impl("fused_vector_legendre_forward_ex", torch::kCUDA, &fused_vector_legendre_forward_ex);
    ops.impl("fused_vector_legendre_inverse", torch::kCUDA, &fused_vector_legendre_inverse);
    ops.impl("fused_vector_forward_pack_real", torch::kCUDA, &fused_vector_forward_pack_real);
    ops.impl("fused_vector_forward_recompose_real", torch::kCUDA, &fused_vector_forward_recompose_real);
    ops.impl("sht_prepare_irfft", torch::kCUDA, &sht_prepare_irfft);
#elif defined(METAL_KERNEL)
    ops.impl("fused_legendre_forward", torch::kMPS, &fused_legendre_forward);
    ops.impl("fused_legendre_inverse", torch::kMPS, &fused_legendre_inverse);
    ops.impl(
        "fused_legendre_forward_real",
        torch::kMPS,
        static_cast<void (*)(torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>(&fused_legendre_forward_real)
    );
    ops.impl("fused_legendre_inverse_real", torch::kMPS, &fused_legendre_inverse_real);
    ops.impl(
        "fused_vector_legendre_forward",
        torch::kMPS,
        static_cast<void (*)(torch::Tensor&, const torch::Tensor&, const torch::Tensor&, const torch::Tensor&)>(&fused_vector_legendre_forward)
    );
    ops.impl("fused_vector_legendre_inverse", torch::kMPS, &fused_vector_legendre_inverse);
    ops.impl("sht_prepare_irfft", torch::kMPS, &sht_prepare_irfft);
#endif
}

void fused_legendre_forward_real_ex(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const int64_t backend_hint
) {
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    fused_legendre_forward_real(output, input, weight_t, backend_hint);
#else
    (void)backend_hint;
    fused_legendre_forward_real(output, input, weight_t);
#endif
}

void fused_vector_legendre_forward_ex(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const int64_t backend_hint
) {
#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    fused_vector_legendre_forward(output, input, weight0_t, weight1_t, backend_hint);
#else
    (void)backend_hint;
    fused_vector_legendre_forward(output, input, weight0_t, weight1_t);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
