#ifndef _VEC_H__
#define _VEC_H__

#include "x86intrin.h"

namespace vec {

template<typename ValueType>
struct SIMDTypes;

#define OP(op, suf, sz) _mm##sz##_##op##_##suf
#define decop(op, suf, sz) static constexpr decltype(&OP(op, suf, sz)) op##_fn = &OP(op, suf, sz);
#define decop512f(op) decop(op, ps, 512)
#define decop512d(op) decop(op, pd, 512)
#define decop256f(op) decop(op, ps, 256)
#define decop256d(op) decop(op, pd, 256)
#define decop128f(op) decop(op, ps, )
#define decop128d(op) decop(op, pd, )

#define declare_all(suf, sz) \
   decop(loadu, suf, sz); \
   decop(storeu, suf, sz); \
   decop(load, suf, sz); \
   decop(store, suf, sz); \
   decop(or, suf, sz); \
   decop(add, suf, sz); \
   decop(sub, suf, sz); \
   decop(mul, suf, sz); \
   decop(set1, suf, sz); \
   decop(setr, suf, sz); \
   decop(set, suf, sz); \
   decop(and, suf, sz); \
   decop(mask_and, suf, sz); \
   decop(maskz_and, suf, sz); \
   decop(maskz_andnot, suf, sz); \
   decop(mask_andnot, suf, sz); \
   decop(andnot, suf, sz); \
   decop(blendv, suf, sz); \
   /*decop(cbrt, suf, sz); \
   decop(cdfnorm, suf, sz); \
   decop(cdfnorminv, suf, sz); */ \
   decop(cmp, suf, sz); \

// cbrt and cdfnorm rely on MKL.
    
template<>
struct SIMDTypes<float>{
#if _FEATURE_AVX512F
    using Type = __m512;
    declare_all(ps, 512)
#elif __AVX2__
    using Type = __m256;
    declare_all(ps, 256)
#elif __SSE2__
    using Type = __m128;
    declare_all(ps, )
#else
#error("Need at least sse2")
#endif
};

template<>
struct SIMDTypes<double>{
#if _FEATURE_AVX512F
    using Type = __m512d;
    declare_all(pd, 512)
#elif __AVX2__
    using Type = __m256d;
    declare_all(pd, 256)
#elif __SSE2__
    using Type = __m128d;
    declare_all(pd, )
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
            vec::SIMDTypes<FloatType>::storeu_fn((FloatType *)ptr,
                vec::SIMDTypes<FloatType>::mul_fn(factor, vec::SIMDTypes<FloatType>::loadu_fn((FloatType *)ptr)));
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
