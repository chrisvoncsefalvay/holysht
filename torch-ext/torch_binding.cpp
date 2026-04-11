// HOLYSHT: Highly Optimised Legendre/Ylm/SHT
// Author: Chris von Csefalvay <chris@chrisvoncsefalvay.com>
#include <torch/library.h>
#include "registration.h"
#include "torch_binding.h"

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
    // Fused Legendre Transform
    ops.def("fused_legendre_forward(Tensor! output, Tensor input, Tensor weight_t) -> ()");
    ops.def("fused_legendre_inverse(Tensor! output, Tensor input, Tensor weight_t) -> ()");

    // SHT helpers
    ops.def("sht_legendre_forward_small(Tensor! output, Tensor input, Tensor weight_t) -> ()");
    ops.def("sht_legendre_inverse_small(Tensor! output, Tensor input, Tensor weight_t) -> ()");
    ops.def("sht_prepare_irfft(Tensor! data, int mmax, int nlon) -> ()");

#if defined(CUDA_KERNEL) || defined(ROCM_KERNEL)
    ops.impl("fused_legendre_forward", torch::kCUDA, &fused_legendre_forward);
    ops.impl("fused_legendre_inverse", torch::kCUDA, &fused_legendre_inverse);
    ops.impl("sht_legendre_forward_small", torch::kCUDA, &sht_legendre_forward_small);
    ops.impl("sht_legendre_inverse_small", torch::kCUDA, &sht_legendre_inverse_small);
    ops.impl("sht_prepare_irfft", torch::kCUDA, &sht_prepare_irfft);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
