#include <torch/torch.h>
#include <torch/mps.h>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <mutex>
#include <string>
#include <unordered_map>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#endif

#ifndef HOLYSHT_METAL_SHADER_PATH
#define HOLYSHT_METAL_SHADER_PATH nullptr
#endif

namespace {

// Threadgroup widths — tuned for Apple Silicon SIMD width (32)
constexpr NSUInteger kThreadgroupWidth = 32;
constexpr NSUInteger kDirectThreadgroupHeight = 8;    // raised from 4 for better occupancy
constexpr NSUInteger kTiledThreadgroupHeight = 16;    // raised from 8 — matches LEGENDRE_TILE_Y
constexpr NSUInteger kTiledMinReduction = 64;         // lowered from 128 — use tiled more aggressively

struct LegendreParams {
    uint32_t batch_size;
    uint32_t nlat;
    uint32_t lmax;
    uint32_t mmax;
};

struct PrepareIrfftParams {
    uint32_t rows;
    uint32_t mmax;
    uint32_t active_mmax;
    uint32_t full_mmax;
    uint32_t nlon_even;  // bool as uint32
};

static inline id<MTLBuffer> get_mtl_buffer_storage(const torch::Tensor& tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

id<MTLDevice> get_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(device != nil, "Failed to create default Metal device");
    return device;
}

id<MTLLibrary> get_library() {
    static std::once_flag once;
    static id<MTLLibrary> library = nil;
    static std::string error_message;

    std::call_once(once, [] {
        @autoreleasepool {
            NSError* error = nil;
#ifdef EMBEDDED_METALLIB_HEADER
            library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(get_device(), &error);
            if (library != nil) {
                [library retain];
            }
#else
            const char* shader_path = HOLYSHT_METAL_SHADER_PATH;
            TORCH_CHECK(shader_path != nullptr, "HOLYSHT_METAL_SHADER_PATH is not defined for the local Metal build");
            NSString* shader_path_ns = [NSString stringWithUTF8String:shader_path];
            NSString* shader_source = [NSString stringWithContentsOfFile:shader_path_ns
                                                                encoding:NSUTF8StringEncoding
                                                                   error:&error];
            if (shader_source != nil) {
                library = [get_device() newLibraryWithSource:shader_source options:nil error:&error];
            }
#endif
            if (library == nil && error != nil) {
                error_message = error.localizedDescription.UTF8String;
            }
        }
    });

    TORCH_CHECK(library != nil, "Failed to create HOLYSHT Metal library: ", error_message.empty() ? "unknown error" : error_message);
    return library;
}

id<MTLComputePipelineState> get_pipeline_state(const std::string& kernel_name) {
    static std::mutex mutex;
    static std::unordered_map<std::string, id<MTLComputePipelineState>> cache;

    std::lock_guard<std::mutex> guard(mutex);
    auto it = cache.find(kernel_name);
    if (it != cache.end()) {
        return it->second;
    }

    @autoreleasepool {
        NSError* error = nil;
        NSString* function_name = [NSString stringWithUTF8String:kernel_name.c_str()];
        id<MTLFunction> function = [get_library() newFunctionWithName:function_name];
        TORCH_CHECK(function != nil, "Failed to create Metal function for ", kernel_name);

        id<MTLComputePipelineState> pipeline =
            [get_device() newComputePipelineStateWithFunction:function error:&error];
        TORCH_CHECK(
            pipeline != nil,
            "Failed to create Metal pipeline state for ",
            kernel_name,
            ": ",
            error != nil ? error.localizedDescription.UTF8String : "unknown error"
        );

        cache.emplace(kernel_name, pipeline);
        return pipeline;
    }
}

MTLSize make_threadgroup_size(id<MTLComputePipelineState> pipeline, const NSUInteger width_cap, const NSUInteger height_cap) {
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger width = std::max<NSUInteger>(1, std::min<NSUInteger>(width_cap, max_threads));
    const NSUInteger height = std::max<NSUInteger>(1, std::min<NSUInteger>(height_cap, max_threads / width));
    return MTLSizeMake(width, height, 1);
}

NSUInteger ceil_div(const NSUInteger numerator, const NSUInteger denominator) {
    return (numerator + denominator - 1) / denominator;
}

bool should_use_tiled_kernel(const torch::Tensor& output, const torch::Tensor& input, const bool inverse) {
    if (input.size(2) < kThreadgroupWidth) {
        return false;
    }

    // Forward: reduction over nlat (input dim 1), output height is lmax (output dim 1)
    // Inverse: reduction over lmax (input dim 1), output height is nlat (output dim 1)
    const auto reduction = input.size(1);
    const auto output_height = output.size(1);
    return reduction >= kTiledMinReduction && output_height >= kTiledThreadgroupHeight;
}

// ============================================================================
// Shape checks
// ============================================================================

void check_complex_legendre_shapes(const torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t, const bool inverse) {
    TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
    TORCH_CHECK(output.device().is_mps(), "output must be an MPS tensor");
    TORCH_CHECK(weight_t.device().is_mps(), "weight_t must be an MPS tensor");
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat, "input must be complex64 on MPS");
    TORCH_CHECK(output.scalar_type() == torch::kComplexFloat, "output must be complex64 on MPS");
    TORCH_CHECK(weight_t.scalar_type() == torch::kFloat, "weight_t must be float32 on MPS");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(weight_t.is_contiguous(), "weight_t must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be rank 3");
    TORCH_CHECK(output.dim() == 3, "output must be rank 3");
    TORCH_CHECK(weight_t.dim() == 3, "weight_t must be rank 3");
    TORCH_CHECK(input.size(0) == output.size(0), "input/output batch size mismatch");
    TORCH_CHECK(weight_t.size(0) == (inverse ? input.size(1) : output.size(1)), "weight_t lmax mismatch");
    TORCH_CHECK(weight_t.size(1) == (inverse ? output.size(1) : input.size(1)), "weight_t nlat mismatch");
    TORCH_CHECK(weight_t.size(2) == input.size(2), "weight_t/input mmax mismatch");
    TORCH_CHECK(output.size(2) == input.size(2), "input/output mmax mismatch");
}

void check_real_legendre_shapes(const torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t, const bool inverse) {
    TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
    TORCH_CHECK(output.device().is_mps(), "output must be an MPS tensor");
    TORCH_CHECK(weight_t.device().is_mps(), "weight_t must be an MPS tensor");
    TORCH_CHECK(input.scalar_type() == torch::kFloat || input.scalar_type() == torch::kHalf, "input must be float32 or float16 on MPS");
    TORCH_CHECK(output.scalar_type() == torch::kFloat, "output must be float32 on MPS");
    TORCH_CHECK(weight_t.scalar_type() == torch::kFloat, "weight_t must be float32 on MPS");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(weight_t.is_contiguous(), "weight_t must be contiguous");
    TORCH_CHECK(input.dim() == 3, "input must be rank 3");
    TORCH_CHECK(output.dim() == 3, "output must be rank 3");
    TORCH_CHECK(weight_t.dim() == 3, "weight_t must be rank 3");
    TORCH_CHECK(input.size(0) == output.size(0), "input/output batch size mismatch");
    TORCH_CHECK(weight_t.size(0) == (inverse ? input.size(1) : output.size(1)), "weight_t lmax mismatch");
    TORCH_CHECK(weight_t.size(1) == (inverse ? output.size(1) : input.size(1)), "weight_t nlat mismatch");
    TORCH_CHECK(weight_t.size(2) == input.size(2), "weight_t/input mmax mismatch");
    TORCH_CHECK(output.size(2) == input.size(2), "input/output mmax mismatch");
}

void check_vector_legendre_shapes(
    const torch::Tensor& output, const torch::Tensor& input,
    const torch::Tensor& weight0_t, const torch::Tensor& weight1_t,
    const bool inverse
) {
    TORCH_CHECK(input.device().is_mps(), "input must be an MPS tensor");
    TORCH_CHECK(output.device().is_mps(), "output must be an MPS tensor");
    TORCH_CHECK(weight0_t.device().is_mps(), "weight0_t must be an MPS tensor");
    TORCH_CHECK(weight1_t.device().is_mps(), "weight1_t must be an MPS tensor");
    TORCH_CHECK(input.scalar_type() == torch::kComplexFloat, "input must be complex64");
    TORCH_CHECK(output.scalar_type() == torch::kComplexFloat, "output must be complex64");
    TORCH_CHECK(weight0_t.scalar_type() == torch::kFloat, "weight0_t must be float32");
    TORCH_CHECK(weight1_t.scalar_type() == torch::kFloat, "weight1_t must be float32");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(weight0_t.is_contiguous(), "weight0_t must be contiguous");
    TORCH_CHECK(weight1_t.is_contiguous(), "weight1_t must be contiguous");
    TORCH_CHECK(input.dim() == 4, "input must be rank 4 [B, 2, spatial, mmax]");
    TORCH_CHECK(output.dim() == 4, "output must be rank 4 [B, 2, spatial, mmax]");
    TORCH_CHECK(input.size(1) == 2 && output.size(1) == 2, "dim 1 must be 2 (sph/tor)");

    const int64_t batch_size = input.size(0);
    const int64_t mmax = input.size(3);
    TORCH_CHECK(output.size(0) == batch_size, "batch size mismatch");
    TORCH_CHECK(output.size(3) == mmax, "mmax mismatch");

    if (inverse) {
        // input: [B, 2, lmax, mmax], output: [B, 2, nlat, mmax]
        TORCH_CHECK(weight0_t.size(0) == input.size(2), "weight0_t lmax mismatch");
        TORCH_CHECK(weight0_t.size(1) == output.size(2), "weight0_t nlat mismatch");
    } else {
        // input: [B, 2, nlat, mmax], output: [B, 2, lmax, mmax]
        TORCH_CHECK(weight0_t.size(0) == output.size(2), "weight0_t lmax mismatch");
        TORCH_CHECK(weight0_t.size(1) == input.size(2), "weight0_t nlat mismatch");
    }
    TORCH_CHECK(weight0_t.size(2) == mmax, "weight0_t mmax mismatch");
    TORCH_CHECK(weight1_t.sizes() == weight0_t.sizes(), "weight0_t/weight1_t shape mismatch");
}

// ============================================================================
// Dispatch: scalar real/complex Legendre
// ============================================================================

void dispatch_legendre_kernel(
    const std::string& kernel_name,
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const bool inverse
) {
    check_real_legendre_shapes(output, input, weight_t, inverse);

    const auto params = LegendreParams{
        static_cast<uint32_t>(input.size(0)),
        static_cast<uint32_t>(inverse ? output.size(1) : input.size(1)),
        static_cast<uint32_t>(inverse ? input.size(1) : output.size(1)),
        static_cast<uint32_t>(input.size(2)),
    };

    // Select kernel variant based on dtype
    std::string dtype_suffix;
    if (input.scalar_type() == torch::kHalf) {
        dtype_suffix = "_half";
    } else {
        dtype_suffix = "_float";
    }

    // Determine the base kernel name with dtype
    // kernel_name is e.g. "fused_legendre_forward_real" — append dtype
    const std::string base_kernel = kernel_name + dtype_suffix;

    const bool use_tiled = should_use_tiled_kernel(output, input, inverse);
    const std::string selected_kernel = use_tiled ? base_kernel + "_tiled" : base_kernel;
    id<MTLComputePipelineState> pipeline = get_pipeline_state(selected_kernel);
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer != nil, "Failed to retrieve MPS command buffer");
    dispatch_queue_t serial_queue = torch::mps::get_dispatch_queue();

    dispatch_sync(serial_queue, ^() {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder != nil, "Failed to create Metal command encoder");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:get_mtl_buffer_storage(input)
                    offset:input.storage_offset() * input.element_size()
                   atIndex:0];
        [encoder setBuffer:get_mtl_buffer_storage(weight_t)
                    offset:weight_t.storage_offset() * weight_t.element_size()
                   atIndex:1];
        [encoder setBuffer:get_mtl_buffer_storage(output)
                    offset:output.storage_offset() * output.element_size()
                   atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        if (use_tiled) {
            const MTLSize threadgroup_size = MTLSizeMake(kThreadgroupWidth, kTiledThreadgroupHeight, 1);
            const MTLSize threadgroups = MTLSizeMake(
                ceil_div(params.mmax, kThreadgroupWidth),
                ceil_div(static_cast<NSUInteger>(output.size(1)), kTiledThreadgroupHeight),
                params.batch_size
            );
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroup_size];
        } else {
            const MTLSize grid_size = MTLSizeMake(params.mmax, output.size(1), params.batch_size);
            const MTLSize threadgroup_size = make_threadgroup_size(pipeline, kThreadgroupWidth, kDirectThreadgroupHeight);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }
        [encoder endEncoding];
        torch::mps::commit();
    });
}

void dispatch_legendre_complex_kernel(
    const std::string& kernel_name,
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight_t,
    const bool inverse
) {
    check_complex_legendre_shapes(output, input, weight_t, inverse);

    const auto params = LegendreParams{
        static_cast<uint32_t>(input.size(0)),
        static_cast<uint32_t>(inverse ? output.size(1) : input.size(1)),
        static_cast<uint32_t>(inverse ? input.size(1) : output.size(1)),
        static_cast<uint32_t>(input.size(2)),
    };

    const bool use_tiled = should_use_tiled_kernel(output, input, inverse);
    const std::string selected_kernel = use_tiled ? kernel_name + "_tiled" : kernel_name;
    id<MTLComputePipelineState> pipeline = get_pipeline_state(selected_kernel);
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer != nil, "Failed to retrieve MPS command buffer");
    dispatch_queue_t serial_queue = torch::mps::get_dispatch_queue();

    dispatch_sync(serial_queue, ^() {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder != nil, "Failed to create Metal command encoder");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:get_mtl_buffer_storage(input)
                    offset:input.storage_offset() * input.element_size()
                   atIndex:0];
        [encoder setBuffer:get_mtl_buffer_storage(weight_t)
                    offset:weight_t.storage_offset() * weight_t.element_size()
                   atIndex:1];
        [encoder setBuffer:get_mtl_buffer_storage(output)
                    offset:output.storage_offset() * output.element_size()
                   atIndex:2];
        [encoder setBytes:&params length:sizeof(params) atIndex:3];

        if (use_tiled) {
            const MTLSize threadgroup_size = MTLSizeMake(kThreadgroupWidth, kTiledThreadgroupHeight, 1);
            const MTLSize threadgroups = MTLSizeMake(
                ceil_div(params.mmax, kThreadgroupWidth),
                ceil_div(static_cast<NSUInteger>(output.size(1)), kTiledThreadgroupHeight),
                params.batch_size
            );
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroup_size];
        } else {
            const MTLSize grid_size = MTLSizeMake(params.mmax, output.size(1), params.batch_size);
            const MTLSize threadgroup_size = make_threadgroup_size(pipeline, kThreadgroupWidth, kDirectThreadgroupHeight);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }
        [encoder endEncoding];
        torch::mps::commit();
    });
}

// ============================================================================
// Dispatch: fused vector Legendre
// ============================================================================

void dispatch_vector_legendre_kernel(
    const std::string& kernel_name,
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t,
    const bool inverse
) {
    check_vector_legendre_shapes(output, input, weight0_t, weight1_t, inverse);

    const uint32_t batch_size = static_cast<uint32_t>(input.size(0));
    const uint32_t nlat = static_cast<uint32_t>(inverse ? output.size(2) : input.size(2));
    const uint32_t lmax = static_cast<uint32_t>(inverse ? input.size(2) : output.size(2));
    const uint32_t mmax = static_cast<uint32_t>(input.size(3));

    const auto params = LegendreParams{batch_size, nlat, lmax, mmax};

    // Determine if tiled: use reduction dimension and output spatial dimension
    const auto reduction = inverse ? lmax : nlat;
    const auto output_spatial = inverse ? nlat : lmax;
    const bool use_tiled = mmax >= kThreadgroupWidth &&
                           reduction >= kTiledMinReduction &&
                           output_spatial >= kTiledThreadgroupHeight;

    const std::string selected_kernel = use_tiled ? kernel_name + "_tiled" : kernel_name;
    id<MTLComputePipelineState> pipeline = get_pipeline_state(selected_kernel);
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer != nil, "Failed to retrieve MPS command buffer");
    dispatch_queue_t serial_queue = torch::mps::get_dispatch_queue();

    dispatch_sync(serial_queue, ^() {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder != nil, "Failed to create Metal command encoder");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:get_mtl_buffer_storage(input)
                    offset:input.storage_offset() * input.element_size()
                   atIndex:0];
        [encoder setBuffer:get_mtl_buffer_storage(weight0_t)
                    offset:weight0_t.storage_offset() * weight0_t.element_size()
                   atIndex:1];
        [encoder setBuffer:get_mtl_buffer_storage(weight1_t)
                    offset:weight1_t.storage_offset() * weight1_t.element_size()
                   atIndex:2];
        [encoder setBuffer:get_mtl_buffer_storage(output)
                    offset:output.storage_offset() * output.element_size()
                   atIndex:3];
        [encoder setBytes:&params length:sizeof(params) atIndex:4];

        if (use_tiled) {
            const MTLSize threadgroup_size = MTLSizeMake(kThreadgroupWidth, kTiledThreadgroupHeight, 1);
            const MTLSize threadgroups = MTLSizeMake(
                ceil_div(static_cast<NSUInteger>(mmax), kThreadgroupWidth),
                ceil_div(static_cast<NSUInteger>(output_spatial), kTiledThreadgroupHeight),
                batch_size
            );
            [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadgroup_size];
        } else {
            const MTLSize grid_size = MTLSizeMake(mmax, output_spatial, batch_size);
            const MTLSize threadgroup_size = make_threadgroup_size(pipeline, kThreadgroupWidth, kDirectThreadgroupHeight);
            [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        }
        [encoder endEncoding];
        torch::mps::commit();
    });
}

// ============================================================================
// Dispatch: sht_prepare_irfft
// ============================================================================

void dispatch_prepare_irfft(
    torch::Tensor& data,
    const int64_t active_mmax,
    const int64_t nlon
) {
    TORCH_CHECK(data.device().is_mps(), "data must be an MPS tensor");
    TORCH_CHECK(data.scalar_type() == torch::kComplexFloat, "data must be complex64");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    TORCH_CHECK(data.dim() == 3, "data must be rank 3 [B, rows, full_mmax]");

    const int64_t full_mmax = nlon / 2 + 1;
    const int64_t rows = data.size(0) * data.size(1);

    const auto params = PrepareIrfftParams{
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>(data.size(2)),
        static_cast<uint32_t>(active_mmax),
        static_cast<uint32_t>(full_mmax),
        static_cast<uint32_t>(nlon % 2 == 0 ? 1 : 0),
    };

    id<MTLComputePipelineState> pipeline = get_pipeline_state("sht_prepare_irfft_kernel");
    id<MTLCommandBuffer> command_buffer = torch::mps::get_command_buffer();
    TORCH_CHECK(command_buffer != nil, "Failed to retrieve MPS command buffer");
    dispatch_queue_t serial_queue = torch::mps::get_dispatch_queue();

    dispatch_sync(serial_queue, ^() {
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        TORCH_CHECK(encoder != nil, "Failed to create Metal command encoder");

        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:get_mtl_buffer_storage(data)
                    offset:data.storage_offset() * data.element_size()
                   atIndex:0];
        [encoder setBytes:&params length:sizeof(params) atIndex:1];

        const MTLSize grid_size = MTLSizeMake(full_mmax, rows, 1);
        const MTLSize threadgroup_size = make_threadgroup_size(pipeline, kThreadgroupWidth, kDirectThreadgroupHeight);
        [encoder dispatchThreads:grid_size threadsPerThreadgroup:threadgroup_size];
        [encoder endEncoding];
        torch::mps::commit();
    });
}

}  // namespace

// ============================================================================
// Public C++ API (called from torch_binding.cpp)
// ============================================================================

void fused_legendre_forward(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t) {
    dispatch_legendre_complex_kernel("fused_legendre_forward_complex_float", output, input, weight_t, false);
}

void fused_legendre_inverse(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t) {
    dispatch_legendre_complex_kernel("fused_legendre_inverse_complex_float", output, input, weight_t, true);
}

void fused_legendre_forward_real(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t) {
    dispatch_legendre_kernel("fused_legendre_forward_real", output, input, weight_t, false);
}

void fused_legendre_inverse_real(torch::Tensor& output, const torch::Tensor& input, const torch::Tensor& weight_t) {
    dispatch_legendre_kernel("fused_legendre_inverse_real", output, input, weight_t, true);
}

void fused_vector_legendre_forward(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t
) {
    dispatch_vector_legendre_kernel(
        "fused_vector_legendre_forward_complex_float",
        output, input, weight0_t, weight1_t, false
    );
}

void fused_vector_legendre_inverse(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight0_t,
    const torch::Tensor& weight1_t
) {
    dispatch_vector_legendre_kernel(
        "fused_vector_legendre_inverse_complex_float",
        output, input, weight0_t, weight1_t, true
    );
}

void sht_prepare_irfft(torch::Tensor& data, const int64_t mmax, const int64_t nlon) {
    dispatch_prepare_irfft(data, mmax, nlon);
}
