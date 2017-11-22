#ifndef _VEC_H__
#define _VEC_H__

#include "x86intrin.h"

namespace vec {

template<typename ValueType>
struct SIMDTypes;
    
template<>
struct SIMDTypes<float>{
#if _FEATURE_AVX512F
    using Type = __m512;
    static constexpr decltype(&_mm512_loadu_ps) load_fn =   &_mm512_loadu_ps;
    static constexpr decltype(&_mm512_storeu_ps) store_fn = &_mm512_storeu_ps;
    static constexpr decltype(&_mm512_add_ps) add_fn =     &_mm512_add_ps;
    static constexpr decltype(&_mm512_mul_ps) mul_fn =     &_mm512_mul_ps;
    static constexpr decltype(&_mm512_sub_ps) sub_fn =     &_mm512_sub_ps;
    static constexpr decltype(&_mm512_set1_ps) set1_fn =     &_mm512_set1_ps;
#elif __AVX2__
    using Type = __m256;
    static constexpr decltype(&_mm256_loadu_ps) load_fn =   &_mm256_loadu_ps;
    static constexpr decltype(&_mm256_storeu_ps) store_fn = &_mm256_storeu_ps;
    static constexpr decltype(&_mm256_add_ps) add_fn =     &_mm256_add_ps;
    static constexpr decltype(&_mm256_mul_ps) mul_fn =     &_mm256_mul_ps;
    static constexpr decltype(&_mm256_sub_ps) sub_fn =     &_mm256_sub_ps;
    static constexpr decltype(&_mm256_set1_ps) set1_fn =     &_mm256_set1_ps;
#elif __SSE2__
    using Type = __m128;
    static constexpr decltype(&_mm_loadu_ps) load_fn =   &_mm_loadu_ps;
    static constexpr decltype(&_mm_storeu_ps) store_fn = &_mm_storeu_ps;
    static constexpr decltype(&_mm_add_ps) add_fn =     &_mm_add_ps;
    static constexpr decltype(&_mm_mul_ps) mul_fn =     &_mm_mul_ps;
    static constexpr decltype(&_mm_sub_ps) sub_fn =     &_mm_sub_ps;
    static constexpr decltype(&_mm256_set1_ps) set1_fn =     &_mm_set1_ps;
#else
#error("Need at least sse2")
#endif
};

template<>
struct SIMDTypes<double>{
#if _FEATURE_AVX512F
    using Type = __m512d;
    static constexpr decltype(&_mm512_loadu_pd) load_fn =   &_mm512_loadu_pd;
    static constexpr decltype(&_mm512_storeu_pd) store_fn = &_mm512_storeu_pd;
    static constexpr decltype(&_mm512_add_pd) add_fn =     &_mm512_add_pd;
    static constexpr decltype(&_mm512_mul_pd) mul_fn =     &_mm512_mul_pd;
    static constexpr decltype(&_mm512_sub_pd) sub_fn =     &_mm512_sub_pd;
    static constexpr decltype(&_mm512_set1_pd) set1_fn =     &_mm512_set1_pd;
#elif __AVX2__
    using Type = __m256d;
    static constexpr decltype(&_mm256_loadu_pd) load_fn =   &_mm256_loadu_pd;
    static constexpr decltype(&_mm256_storeu_pd) store_fn = &_mm256_storeu_pd;
    static constexpr decltype(&_mm256_add_pd) add_fn =     &_mm256_add_pd;
    static constexpr decltype(&_mm256_mul_pd) mul_fn =     &_mm256_mul_pd;
    static constexpr decltype(&_mm256_sub_pd) sub_fn =     &_mm256_sub_pd;
    static constexpr decltype(&_mm256_set1_pd) set1_fn =     &_mm256_set1_pd;
#elif __SSE2__
    using Type = __m128d;
    static constexpr decltype(&_mm_loadu_pd) load_fn =   &_mm_loadu_pd;
    static constexpr decltype(&_mm_storeu_pd) store_fn = &_mm_storeu_pd;
    static constexpr decltype(&_mm_add_pd) add_fn =     &_mm_add_pd;
    static constexpr decltype(&_mm_mul_pd) mul_fn =     &_mm_mul_pd;
    static constexpr decltype(&_mm_sub_pd) sub_fn =     &_mm_sub_pd;
    static constexpr decltype(&_mm_set1_pd) set1_fn =     &_mm_set1_pd;
#else
#error("Need at least sse2")
#endif
};


template<typename FloatType>
void blockmul(FloatType *pos, size_t nelem, FloatType div) {
#if __AVX2__ || _FEATURE_AVX512F || __SSE2__
#pragma message("Using vectorized multiplication.")
        using SIMDType = typename vec::SIMDTypes<FloatType>::Type;
        SIMDType factor(vec::SIMDTypes<FloatType>::set1_fn(div));
        SIMDType *ptr((SIMDType *)pos);
        FloatType *end(pos + nelem);
        while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
            vec::SIMDTypes<FloatType>::store_fn((FloatType *)ptr,
                vec::SIMDTypes<FloatType>::mul_fn(factor, vec::SIMDTypes<FloatType>::load_fn((FloatType *)ptr)));
            ++ptr;
        }
        pos = (FloatType *)ptr;
        while(pos < end) *pos++ *= div;
#else
#pragma message("Only vectorized for avx2. Enjoy your serial version.")
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] *= div; // Could be vectorized.
#endif
}


} // namespace vec

#endif // #ifndef _VEC_H__
