#ifndef FRP_GPU_FHT_H
#define FRP_GPU_FHT_H
namespace frp {

template<typename T>
__global__ void dgpu_kernel(T *ptr, size_t l2, int nthreads) {
    // According to my benchmarks, expect computation ~12,000x as fast as CPU
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int n = 1 << l2;
    for(int i = 0; i < l2; ++i) {
        int s1 = 1 << i, s2 = s1 << 1;
        int nthreads_active = min(n >> (i + 1), nthreads);
        int npert = n / nthreads_active;
        if(tid < nthreads_active) {
            for(int j = tid * npert, e = j + npert; j != e; j += s2) {
                #pragma unroll
                for(size_t k = 0; k < s1; ++k) {
                    auto u = ptr[j + k], v = ptr[j + k + s1];
                    ptr[j + k] = u + v, ptr[j + k + s1] = u - v;
                }
            }
        }
        __syncthreads();
    }
}
} // frp
#endif /* FRP_GPU_FHT_H */
