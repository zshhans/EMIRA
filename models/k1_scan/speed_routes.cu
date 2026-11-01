// E preparation and grouped B/C/w adjoints. All accumulators are FP32.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>

namespace {
template<typename T>
__global__ void prepare_e_kernel(const T* __restrict__ c,const int* __restrict__ inverse,
                                  const float* __restrict__ w,T* __restrict__ ce,float* __restrict__ e,int state,int length) {
    int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=length)return;
    int b=blockIdx.z;
    int64_t base=(int64_t(b)*state+blockIdx.y)*length;
    int target=inverse[int64_t(b)*length+t];
    T cv=target>=0 ? c[base+target] : T(0.f);
    ce[base+t]=cv;
    if(e!=nullptr) e[base+t]=target>=0 ? float(cv)*w[int64_t(b)*length+target] : 0.f;
}

template<typename T,int G>
__global__ void finish_kernel(const float* __restrict__ pb,const float* __restrict__ pc,
                             const float* __restrict__ pe,const int* __restrict__ src,
                             const T* __restrict__ c,const float* __restrict__ w,
                             T* __restrict__ db,T* __restrict__ dc,float* __restrict__ dw,
                             int batch,int state,int length) {
    const int t=blockIdx.x*32+threadIdx.x;
    const int b=blockIdx.z;
    const int64_t stride=int64_t(batch)*state*length;
    const int source=t<length ? src[int64_t(b)*length+t] : -1;
    const float weight=t<length ? w[int64_t(b)*length+t] : 0.f;
    float partial_w=0.f;
    for(int s=threadIdx.y;s<state;s+=8) {
        if(t<length) {
            const int64_t i=(int64_t(b)*state+s)*length+t;
            const int64_t j=i-t+source;
            float gb=0.f,gc=0.f,ge=0.f;
            #pragma unroll
            for(int group=0;group<G;++group) {
                gb+=pb[group*stride+i];
                gc+=pc[group*stride+i];
                if(source>=0) ge+=pe[group*stride+j];
            }
            db[i]=T(gb);
            dc[i]=T(gc+weight*ge);
            partial_w+=float(c[i])*ge;
        }
    }
    __shared__ float scratch[8][32];
    scratch[threadIdx.y][threadIdx.x]=partial_w;
    __syncthreads();
    if(threadIdx.y==0 && t<length) {
        float total=0.f;
        #pragma unroll
        for(int r=0;r<8;++r)total+=scratch[r][threadIdx.x];
        dw[int64_t(b)*length+t]=total;
    }
}

void check(const at::Tensor& c,const at::Tensor& index,const at::Tensor& w) {
    TORCH_CHECK(c.is_cuda() && c.is_contiguous() && c.dim()==3);
    TORCH_CHECK(index.is_cuda() && index.device()==c.device() && index.is_contiguous() && index.scalar_type()==at::kInt);
    TORCH_CHECK(index.dim()==2 && index.size(0)==c.size(0) && index.size(1)==c.size(2));
    TORCH_CHECK(w.is_cuda() && w.device()==c.device() && w.is_contiguous() && w.scalar_type()==at::kFloat && w.sizes()==index.sizes());
    TORCH_CHECK(c.scalar_type()==at::kFloat || c.scalar_type()==at::kHalf || c.scalar_type()==at::kBFloat16);
}
}

std::vector<at::Tensor> emira_prepare_e(const at::Tensor& c,const at::Tensor& inverse,const at::Tensor& w,bool training) {
    check(c,inverse,w);
    const c10::cuda::CUDAGuard guard(c.device());
    auto ce=at::empty_like(c);
    auto e=at::empty(training ? c.sizes() : at::IntArrayRef{0},c.options().dtype(at::kFloat));
    dim3 grid((c.size(2)+255)/256,c.size(1),c.size(0));
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,c.scalar_type(),"prepare_e",[&]{
        prepare_e_kernel<scalar_t><<<grid,256,0,stream>>>(c.data_ptr<scalar_t>(),inverse.data_ptr<int>(),w.data_ptr<float>(),ce.data_ptr<scalar_t>(),training ? e.data_ptr<float>() : nullptr,c.size(1),c.size(2));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {ce,e};
}

std::vector<at::Tensor> emira_finish_gradients(const at::Tensor& pb,const at::Tensor& pc,
    const at::Tensor& pe,const at::Tensor& src,const at::Tensor& c,const at::Tensor& w) {
    check(c,src,w);
    int groups=pb.size(0);
    TORCH_CHECK(groups==1 || groups==2 || groups==4 || groups==8 || groups==16);
    for(const auto& p : {pb,pc,pe}) {
        TORCH_CHECK(p.is_cuda() && p.device()==c.device() && p.is_contiguous() && p.scalar_type()==at::kFloat && p.numel()==groups*c.numel());
    }
    const c10::cuda::CUDAGuard guard(c.device());
    auto db=at::empty_like(c),dc=at::empty_like(c),dw=at::empty_like(w);
    dim3 grid((c.size(2)+31)/32,1,c.size(0)),block(32,8);
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    #define LAUNCH(G) finish_kernel<scalar_t,G><<<grid,block,0,stream>>>(pb.data_ptr<float>(),pc.data_ptr<float>(),pe.data_ptr<float>(),src.data_ptr<int>(),c.data_ptr<scalar_t>(),w.data_ptr<float>(),db.data_ptr<scalar_t>(),dc.data_ptr<scalar_t>(),dw.data_ptr<float>(),c.size(0),c.size(1),c.size(2))
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,c.scalar_type(),"finish_gradients",[&]{
        if(groups==1){LAUNCH(1);}else if(groups==2){LAUNCH(2);}else if(groups==4){LAUNCH(4);}else if(groups==8){LAUNCH(8);}else{LAUNCH(16);}
    });
    #undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {db,dc,dw};
}
