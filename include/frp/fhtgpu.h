#ifndef FRP_GPU_FHT_H
#define FRP_GPU_FHT_H
namespace frp {

namespace detail {
// Derived from WyHash
static constexpr const uint64_t _wyp0=0xa0761d6478bd642full, _wyp1=0xe7037ed1a0b428dbull;

template<typename T>
static constexpr inline T seedind2val(T ind, T seed) {
    uint64_t oldstate = ind;
    uint64_t newstart = ind * 6364136223846793005ULL + seed;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

template<typename T>
static constexpr inline T seedind2val_lazy(T ind, T seed) {
    return (ind ^ seed) * 6364136223846793005ULL;
}


// TODO: kernel fusion between fht and random diagonal matrix multiplication from fixed seeds.

} // detail

template<typename T, bool renormalize=true, typename T2>
__global__ void grsfht_kernel(T *ptr, size_t l2, int nthreads, T theta, T2 *vals) {
    // Givens rotations-fht kernel
    // This maps pretty well to the GPU
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int n = 1 << l2;
    for(int i = 0; i < l2; ++i) {
        T theta = vals[i];
        T m1 = cos(theta), m2 = sin(theta);
        int s1 = 1 << i, s2 = s1 << 1;
        int nthreads_active = min(n >> (i + 1), nthreads);
        int npert = n / nthreads_active;
        if(tid < nthreads_active) {
            for(int j = tid * npert, e = j + npert; j != e; j += s2) {
                #pragma unroll
                for(size_t k = 0; k < s1; ++k) {
                    auto u = ptr[j + k], v = ptr[j + k + s1];
                    ptr[j + k] = u * mc - v * ms, ptr[j + k + s1] = ms * u + mc * v;
                }
            }
        }
        __syncthreads();
    }
}

template<typename T, bool renormalize=true>
__global__ void pfht_kernel(T *ptr, size_t l2, int nthreads, T theta) {
    // This maps pretty well to the GPU
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int n = 1 << l2;
    T mc = cos(theta), ms = sin(theta);
    for(int i = 0; i < l2; ++i) {
        int s1 = 1 << i, s2 = s1 << 1;
        int nthreads_active = min(n >> (i + 1), nthreads);
        int npert = n / nthreads_active;
        if(tid < nthreads_active) {
            for(int j = tid * npert, e = j + npert; j != e; j += s2) {
                #pragma unroll
                for(size_t k = 0; k < s1; ++k) {
                    auto u = ptr[j + k], v = ptr[j + k + s1];
                    ptr[j + k] = u * mc - v * ms, ptr[j + k + s1] = ms * u + mc * v;
                }
            }
        }
        __syncthreads();
    }
}

template<typename T, bool renormalize=true>
__global__ void fht_kernel(T *ptr, size_t l2, int nthreads) {
    // This maps pretty well to the GPU
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
