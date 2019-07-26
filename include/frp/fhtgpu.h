#ifndef FRP_GPU_FHT_H
#define FRP_GPU_FHT_H
namespace frp {

template<typename T, bool renormalize=true>
__global__ void fht_kernel(T *ptr, size_t l2, int nthreads) {
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
    if(renormalize) {
        T mult = 1. / pow(sqrt(2.), l2);
        int npert = n / nthreads;
        #pragma unroll
        for(int i = tid * npert, e = i + npert; i < e; ++i) {
            ptr[i] *= mult;
        }
    }
}
template<typename T>
__global__ void rademacher_multiply(T *ptr, uint32_t *rvals, size_t l2, int nthreads) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t n = 1ull << l2;
    int per_thread = n / nthreads;
    int start_index = tid * per_thread, end = start_index + per_thread;
    for(int i = start_index / 32; i != end / 32; ++i) {
        auto rv = rvals[i];
        int li = i * 32;
        #pragma unroll
        for(int j = 0; j < 32; ++j) {
            auto v = ptr[li + j];
            ptr[li + j] = (rv >> j)& 1 ? -v: v;
        }
    }
}
template<typename T, bool SO renormalize=true>
__global__ void radfht_kernel(T *ptr, uint32_t *rvals, size_t l2, int nthreads) {
    // Performs both 
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int n = 1 << l2;
    int per_thread = n / nthreads;
    int start_index = tid * per_thread, end = start_index + per_thread;
    for(int i = start_index / 32; i != end / 32; ++i) {
        auto rv = rvals[i];
        int li = i * 32;
        #pragma unroll
        for(int j = 0; j < 32; ++j) {
            auto v = ptr[li + j];
            ptr[li + j] = (rv >> j)& 1 ? -v: v;
        }
    }
    for(int i = 0; i < l2; ++i) {
        int s1 = 1 << i, s2 = s1 << 1;
        int nthreads_active = min(n >> (i + 1), nthreads);
        int npert = n / nthreads_active;
        if(tid < nthreads_active) {
            #pragma unroll
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
    if(renormalize) {
        T mult = 1. / pow(sqrt(2.), l2);
        int npert = n / nthreads;
        #pragma unroll
        for(int i = tid * npert, e = i + npert; i < e; ++i) {
            ptr[i] *= mult;
        }
    }
}
} // frp
#endif /* FRP_GPU_FHT_H */
