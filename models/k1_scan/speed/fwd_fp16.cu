#include "selective_scan_fwd_kernel.cuh"
template void selective_scan_fwd_cuda<at::Half, float>(SSMParamsBase &params, cudaStream_t stream);
