// Fixed K=1 predecessor validation and route construction, entirely on device.
// Dense addressing is used only for scalar boundary messages; no full H exists.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace {
template<typename I>
__global__ void initialize(const I* input,int* source,int* inverse,int* errors,int64_t n,int length) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    int64_t s=input[i];int t=i%length;
    if(s < -1 || s>=length){atomicOr(errors,1);s=-1;}
    else if(s>=t){atomicOr(errors,2);s=-1;}
    source[i]=int(s);inverse[i]=-1;
}
__global__ void scatter_inverse(const int* source,int* inverse,int* errors,int64_t n,int length) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    int source_i=source[i];
    if(source_i>=0) {
        int t=i%length;int64_t base=i-t;
        if(atomicCAS(inverse+base+source_i,-1,t)!=-1)atomicOr(errors,4);
    }
}
__global__ void mark_boundary(const int* inverse,int* slots,int64_t n,int length,int shift) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=n)return;
    int t=i%length;
    int next=((t>>shift)+1)<<shift;
    slots[i]=inverse[i]>=next ? t : -1;
}
}

std::vector<at::Tensor> build_routes(const at::Tensor& sources) {
    TORCH_CHECK(sources.is_cuda() && sources.is_contiguous() && sources.dim()==2 && sources.size(0)>0 && sources.size(1)>0);
    TORCH_CHECK(sources.scalar_type()==at::kLong || sources.scalar_type()==at::kInt);
    TORCH_CHECK(sources.size(1)<INT_MAX);
    const c10::cuda::CUDAGuard guard(sources.device());
    auto options=sources.options().dtype(at::kInt);
    auto src=at::empty_like(sources,options),inv=at::empty_like(src),slots=at::empty_like(src);
    auto errors=at::zeros({1},options);
    int length=sources.size(1);
    int shift=length<=128 ? 7 : length<=256 ? 8 : length<=512 ? 9 : length<=1024 ? 10 : 11;
    int64_t n=sources.numel();int grid=(n+255)/256;
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    if(sources.scalar_type()==at::kLong) {
        initialize<int64_t><<<grid,256,0,stream>>>(sources.data_ptr<int64_t>(),src.data_ptr<int>(),inv.data_ptr<int>(),errors.data_ptr<int>(),n,length);
    } else {
        initialize<int><<<grid,256,0,stream>>>(sources.data_ptr<int>(),src.data_ptr<int>(),inv.data_ptr<int>(),errors.data_ptr<int>(),n,length);
    }
    scatter_inverse<<<grid,256,0,stream>>>(src.data_ptr<int>(),inv.data_ptr<int>(),errors.data_ptr<int>(),n,length);
    mark_boundary<<<grid,256,0,stream>>>(inv.data_ptr<int>(),slots.data_ptr<int>(),n,length,shift);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {src,inv,slots,errors};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("build_routes",&build_routes);}
