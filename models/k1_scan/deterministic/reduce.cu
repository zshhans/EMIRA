// Every output has one owner; batch accumulation order is fixed.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
__global__ void sum_batch(const float* input,float* output,int batch,int64_t width) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=width)return;
    float total=0.f;
    for(int b=0;b<batch;++b)total+=input[int64_t(b)*width+i];
    output[i]=total;
}
}

at::Tensor emira_deterministic_sum_batch(const at::Tensor& input) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous() && input.scalar_type()==at::kFloat);
    TORCH_CHECK(input.dim()>=2 && input.size(0)>0);
    const c10::cuda::CUDAGuard guard(input.device());
    auto output=at::empty(input.sizes().slice(1),input.options());
    int64_t width=output.numel();
    sum_batch<<<(width+255)/256,256,0,at::cuda::getCurrentCUDAStream().stream()>>>(
        input.data_ptr<float>(),output.data_ptr<float>(),input.size(0),width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
