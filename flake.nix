# HOLYSHT
# Author: Chris von Csefalvay
# Licence: MIT
# Repository: https://github.com/chrisvoncsefalvay/holysht
# Hugging Face kernel: https://hf.co/chrisvoncsefalvay/holysht

{
  inputs = {
    kernel-builder.url = "github:huggingface/kernels";
  };
  outputs =
    { self, kernel-builder, ... }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
