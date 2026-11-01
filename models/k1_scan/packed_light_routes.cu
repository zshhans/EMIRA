// Pretranspose only small B/C/E tensors; no state history or precision change.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
namespace {
__device__ __forceinline__ int physical(int t,int threads,int items) {
    int n=threads*items,local=t%n;
    return t/n*n+(local%items)*threads+local/items;
}
template<typename T>
__global__ void pack_kernel(const T* b,const T* c,const int* inv,const float* w,
                           T* pb,T* pc,T* pe,T* ce,int state,int length,int padded,int threads,int items) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=padded)return;
    int n=threads*items,local=p%n;
    int t=p/n*n+(local%threads)*items+local/threads;
    int batch=blockIdx.z,s=blockIdx.y;
    int64_t dst=(int64_t(batch)*state+s)*padded+p;
    T bv=T(0.f),cv=T(0.f),ev=T(0.f);float e=0.f;
    if(t<length) {
        int64_t base=(int64_t(batch)*state+s)*length;
        bv=b[base+t];cv=c[base+t];
        int target=inv[int64_t(batch)*length+t];
        if(target>=0) {ev=c[base+target];e=float(ev)*w[int64_t(batch)*length+target];}
        ce[base+t]=ev;
    }
    pb[dst]=bv;pc[dst]=cv;pe[dst]=ev;
}
template<typename T,int G>
__global__ void finish_kernel(const float* pb,const float* pc,const float* pe,
                             const int* src,const T* c,const float* w,T* db,T* dc,float* dw,
                             int batch,int state,int length,int padded,int threads,int items) {
    int t=blockIdx.x*32+threadIdx.x,b=blockIdx.z;
    int source=t<length ? src[int64_t(b)*length+t] : -1;
    int p=physical(t,threads,items),q=source>=0 ? physical(source,threads,items) : 0;
    float weight=t<length ? w[int64_t(b)*length+t] : 0.f;
    int64_t stride=int64_t(batch)*state*padded;
    float pw=0.f;
    for(int s=threadIdx.y;s<state;s+=8) if(t<length) {
        int64_t i=(int64_t(b)*state+s)*padded+p,j=i-p+q;
        float gb=0.f,gc=0.f,ge=0.f;
        #pragma unroll
        for(int g=0;g<G;++g) {
            gb+=pb[g*stride+i];gc+=pc[g*stride+i];
            if(source>=0)ge+=pe[g*stride+j];
        }
        int64_t dst=(int64_t(b)*state+s)*length+t;
        db[dst]=T(gb);dc[dst]=T(gc+weight*ge);pw+=float(c[dst])*ge;
    }
    __shared__ float scratch[8][32];
    scratch[threadIdx.y][threadIdx.x]=pw;
    __syncthreads();
    if(threadIdx.y==0 && t<length) {
        float total=0.f;
        #pragma unroll
        for(int s=0;s<8;++s)total+=scratch[s][threadIdx.x];
        dw[int64_t(b)*length+t]=total;
    }
}
void check(const at::Tensor& c,const at::Tensor& index,const at::Tensor& w,int threads,int items) {
    TORCH_CHECK(c.is_cuda() && c.is_contiguous() && c.dim()==3);
    TORCH_CHECK(index.is_cuda() && index.device()==c.device() && index.is_contiguous() && index.scalar_type()==at::kInt);
    TORCH_CHECK(index.dim()==2 && index.size(0)==c.size(0) && index.size(1)==c.size(2));
    TORCH_CHECK(w.is_cuda() && w.device()==c.device() && w.is_contiguous() && w.scalar_type()==at::kFloat && w.sizes()==index.sizes());
    TORCH_CHECK(c.scalar_type()==at::kFloat || c.scalar_type()==at::kHalf || c.scalar_type()==at::kBFloat16);
    TORCH_CHECK((threads==32 || threads==64 || threads==128 || threads==256) && (items==4 || items==8 || items==16));
}
}
std::vector<at::Tensor> emira_pack(const at::Tensor& b,const at::Tensor& c,const at::Tensor& inv,const at::Tensor& w,int threads,int items) {
    check(c,inv,w,threads,items);
    TORCH_CHECK(b.is_cuda() && b.device()==c.device() && b.is_contiguous() && b.scalar_type()==c.scalar_type() && b.sizes()==c.sizes());
    const c10::cuda::CUDAGuard guard(c.device());
    int n=threads*items,padded=((c.size(2)+n-1)/n)*n;
    auto pb=at::empty({c.size(0),c.size(1),padded},c.options()),pc=at::empty_like(pb);
    auto pe=at::empty_like(pb),ce=at::empty_like(c);
    dim3 grid((padded+255)/256,c.size(1),c.size(0));
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,c.scalar_type(),"pack",[&]{
        pack_kernel<scalar_t><<<grid,256,0,stream>>>(b.data_ptr<scalar_t>(),c.data_ptr<scalar_t>(),inv.data_ptr<int>(),w.data_ptr<float>(),pb.data_ptr<scalar_t>(),pc.data_ptr<scalar_t>(),pe.data_ptr<scalar_t>(),ce.data_ptr<scalar_t>(),c.size(1),c.size(2),padded,threads,items);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {pb,pc,pe,ce};
}
std::vector<at::Tensor> emira_unpack_gradients(const at::Tensor& pb,const at::Tensor& pc,const at::Tensor& pe,const at::Tensor& src,const at::Tensor& c,const at::Tensor& w,int threads,int items) {
    check(c,src,w,threads,items);
    int groups=pb.size(0),n=threads*items,padded=((c.size(2)+n-1)/n)*n;
    TORCH_CHECK(groups==1 || groups==2 || groups==4 || groups==8 || groups==16);
    for(const auto& p:{pb,pc,pe})
        TORCH_CHECK(p.is_cuda() && p.device()==c.device() && p.is_contiguous() && p.scalar_type()==at::kFloat && p.numel()==groups*c.size(0)*c.size(1)*padded);
    const c10::cuda::CUDAGuard guard(c.device());
    auto db=at::empty_like(c),dc=at::empty_like(c),dw=at::empty_like(w);
    dim3 grid((c.size(2)+31)/32,1,c.size(0)),block(32,8);
    auto stream=at::cuda::getCurrentCUDAStream().stream();
    #define LAUNCH(G) finish_kernel<scalar_t,G><<<grid,block,0,stream>>>(pb.data_ptr<float>(),pc.data_ptr<float>(),pe.data_ptr<float>(),src.data_ptr<int>(),c.data_ptr<scalar_t>(),w.data_ptr<float>(),db.data_ptr<scalar_t>(),dc.data_ptr<scalar_t>(),dw.data_ptr<float>(),c.size(0),c.size(1),c.size(2),padded,threads,items)
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,c.scalar_type(),"unpack",[&]{
        if(groups==1){LAUNCH(1);}else if(groups==2){LAUNCH(2);}else if(groups==4){LAUNCH(4);}else if(groups==8){LAUNCH(8);}else{LAUNCH(16);}
    });
    #undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {db,dc,dw};
}
