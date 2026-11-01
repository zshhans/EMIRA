#include "selective_scan_fwd_kernel.cuh"
template void selective_scan_fwd_cuda<at::BFloat16, float>(SSMParamsBase &params, cudaStream_t stream);
