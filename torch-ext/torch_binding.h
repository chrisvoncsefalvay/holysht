// HOLYSHT: Highly Optimised Legendre/Ylm/SHT
// SPDX-License-Identifier: MIT
// Author: Chris von Csefalvay
// Repository: https://github.com/chrisvoncsefalvay/holysht
// Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht
#pragma once
#include <torch/all.h>

// Fused Legendre Transform (complex, transposed weight layout)
void fused_legendre_forward(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);
void fused_legendre_inverse(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);
void fused_legendre_forward_real(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);
void fused_legendre_forward_real(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const int64_t backend_hint
);
void fused_legendre_inverse_real(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);
void fused_legendre_forward_real_ex(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const int64_t backend_hint
);

// Vector Legendre composition
void fused_vector_legendre_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t
);
void fused_vector_legendre_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const int64_t backend_hint
);
void fused_vector_legendre_forward_ex(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const int64_t backend_hint
);
void fused_vector_legendre_inverse(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t
);

// SHT helpers
void sht_prepare_irfft(torch::Tensor& data, const int64_t mmax, const int64_t nlon);
