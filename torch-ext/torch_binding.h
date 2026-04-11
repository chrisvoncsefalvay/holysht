// HOLYSHT: Highly Optimised Legendre/Ylm/SHT
// Author: Chris von Csefalvay <chris@chrisvoncsefalvay.com>
#pragma once
#include <torch/all.h>

// Fused Legendre Transform (complex, transposed weight layout)
void fused_legendre_forward(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);
void fused_legendre_inverse(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);

// SHT helpers
void sht_legendre_forward_small(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);
void sht_legendre_inverse_small(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t);
void sht_prepare_irfft(torch::Tensor& data, const int mmax, const int nlon);
