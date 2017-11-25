#ifndef _VEC_H__
#define _VEC_H__
#include "sleef/include/sleefdft.h"
#include "sleef.h"

#include "x86intrin.h"
#include <cmath>
#include <iterator>

namespace vec {

template<typename ValueType>
struct SIMDTypes;

#define OP(op, suf, sz) _mm##sz##_##op##_##suf
#define decop(op, suf, sz) static constexpr decltype(&OP(op, suf, sz)) op##_fn = &OP(op, suf, sz);

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


#define SLEEF_OP(op, suf, prec) Sleef_##op##suf##_##prec
#define dec_sleefop_prec(op, suf, prec) \
    static constexpr decltype(&SLEEF_OP(op, suf, prec)) op##_##prec##_fn = \
    &SLEEF_OP(op, suf, prec); \
    template<typename T> \
    struct apply_##op##_##prec {\
        T operator()(T val) const {return op##_##prec##_fn(val);} \
        template<typename OT>\
        OT scalar(OT val) const {return std::op(val);} \
    };


#define dec_all_precs(op, suf) \
    dec_sleefop_prec(op, suf, u35) \
    dec_sleefop_prec(op, suf, u10)


#define dec_all_trig(suf) \
   dec_all_precs(sin, suf) \
   dec_all_precs(cos, suf) \
   dec_all_precs(asin, suf) \
   dec_all_precs(acos, suf) \
   dec_all_precs(atan, suf) \
   dec_all_precs(atan2, suf) \
   dec_all_precs(cbrt, suf) \
   dec_sleefop_prec(log, suf, u10) \
   dec_sleefop_prec(log1p, suf, u10) \
   dec_sleefop_prec(expm1, suf, u10) \
   dec_sleefop_prec(exp, suf, u10) \
   dec_sleefop_prec(exp2, suf, u10) \
   /*dec_sleefop_prec(exp10, suf, u10) */ \
   dec_sleefop_prec(lgamma, suf, u10) \
   dec_sleefop_prec(tgamma, suf, u10) \
   dec_sleefop_prec(sinh, suf, u10) \
   dec_sleefop_prec(cosh, suf, u10) \
   dec_sleefop_prec(asinh, suf, u10) \
   dec_sleefop_prec(acosh, suf, u10) \
   dec_sleefop_prec(tanh, suf, u10) \
   dec_sleefop_prec(atanh, suf, u10)
    

template<>
struct SIMDTypes<float>{
#if _FEATURE_AVX512F
    using Type = __m512;
    declare_all(ps, 512)
    static const size_t ALN = 64;
    dec_all_trig(f16);
#elif __AVX2__
    using Type = __m256;
    declare_all(ps, 256)
    static const size_t ALN = 32;
    //dec_all_precs(sin, f8)
    dec_all_trig(f8);
#elif __SSE2__
    using Type = __m128;
    declare_all(ps, )
    static const size_t ALN = 16;
    dec_all_trig(f4);
#else
#error("Need at least sse2")
#endif
    static const size_t MASK = ALN - 1;
};

template<>
struct SIMDTypes<double>{
#if _FEATURE_AVX512F
    using Type = __m512d;
    declare_all(pd, 512)
    static const size_t ALN = 64;
    dec_all_trig(d8);
#elif __AVX2__
    using Type = __m256d;
    declare_all(pd, 256)
    static const size_t ALN = 32;
    dec_all_trig(d4);
#elif __SSE2__
    using Type = __m128d;
    declare_all(pd, )
    static const size_t ALN = 16;
    dec_all_trig(d2);
#else
#error("Need at least sse2")
#endif
    static const size_t MASK = ALN - 1;
};


template<typename FloatType>
void blockmul(FloatType *pos, size_t nelem, FloatType div) {
#if __AVX2__ || _FEATURE_AVX512F || __SSE2__
#pragma message("Using vectorized scalar multiplication.")
        using SIMDType = typename vec::SIMDTypes<FloatType>::Type;
        SIMDType factor(vec::SIMDTypes<FloatType>::set1_fn(div));
        SIMDType *ptr((SIMDType *)pos);
        FloatType *end(pos + nelem);
        if((uint64_t)ptr & vec::SIMDTypes<FloatType>::MASK) {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::storeu_fn((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul_fn(factor, vec::SIMDTypes<FloatType>::loadu_fn((FloatType *)ptr)));
                ++ptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::store_fn((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul_fn(factor, vec::SIMDTypes<FloatType>::load_fn((FloatType *)ptr)));
                ++ptr;
            }
        }
        pos = (FloatType *)ptr;
        while(pos < end) *pos++ *= div;
#else
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] *= div; // Could be vectorized.
#endif
}

template<typename FloatType>
void blockadd(FloatType *pos, size_t nelem, FloatType val) {
#if __AVX2__ || _FEATURE_AVX512F || __SSE2__
#pragma message("Using vectorized scalar vector addition.")
        using SIMDType = typename vec::SIMDTypes<FloatType>::Type;
        SIMDType inc(vec::SIMDTypes<FloatType>::set1_fn(val));
        SIMDType *ptr((SIMDType *)pos);
        FloatType *end(pos + nelem);
        if((uint64_t)ptr & vec::SIMDTypes<FloatType>::MASK) {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::storeu_fn((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::add_fn(inc, vec::SIMDTypes<FloatType>::loadu_fn((FloatType *)ptr)));
                ++ptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::store_fn((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::add_fn(inc, vec::SIMDTypes<FloatType>::load_fn((FloatType *)ptr)));
                ++ptr;
            }
        }
        pos = (FloatType *)ptr;
        while(pos < end) *pos++ += div;
#else
#pragma message("Enjoy your serial version.")
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) pos[i] += div; // Could be vectorized.
#endif
}

template<typename FloatType>
void vecmul(FloatType *to, const FloatType *from, size_t nelem) {
#if __AVX2__ || _FEATURE_AVX512F || __SSE2__
#pragma message("Using vectorized multiplication.")
        using SIMDType = typename vec::SIMDTypes<FloatType>::Type;
        SIMDType *ptr((SIMDType *)to), *fromptr((SIMDType *)from);
        FloatType *end(to + nelem);
        if((uint64_t)ptr & vec::SIMDTypes<FloatType>::MASK || (uint64_t)fromptr & (vec::SIMDTypes<FloatType>::MASK)) {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::storeu_fn((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul_fn(vec::SIMDTypes<FloatType>::loadu_fn((FloatType *)fromptr), vec::SIMDTypes<FloatType>::loadu_fn((FloatType *)ptr)));
                ++ptr; ++fromptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::store_fn((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul_fn(vec::SIMDTypes<FloatType>::load_fn((FloatType *)fromptr), vec::SIMDTypes<FloatType>::load_fn((FloatType *)ptr)));
                ++ptr; ++fromptr;
            }
        }
        to = (FloatType *)ptr, from = (FloatType *)fromptr;
        while(to < end) *to++ *= *from++;
#else
#pragma message("Enjoy your serial version.")
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) to[i] *= from[i]; // Could be vectorized.
#endif
}

template<typename FloatType, typename Functor>
void block_apply(FloatType *pos, size_t nelem) {
#if __AVX2__ || _FEATURE_AVX512F || __SSE2__
#pragma message("Using vectorized multiplication.")
        Functor func;
        using SIMDType = typename vec::SIMDTypes<FloatType>::Type;
        SIMDType *ptr((SIMDType *)pos);
        FloatType *end(pos + nelem);
        if((uint64_t)ptr & vec::SIMDTypes<FloatType>::MASK) {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::storeu_fn((FloatType *)ptr,
                    func(vec::SIMDTypes<FloatType>::loadu_fn((FloatType *)ptr)));
                ++ptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::store_fn((FloatType *)ptr,
                    func(vec::SIMDTypes<FloatType>::load_fn((FloatType *)ptr)));
                ++ptr;
            }
        }
        pos = (FloatType *)ptr;
        while(pos < end) *pos  = func.scalar(*pos), ++pos;
#else
#pragma message("Enjoy your serial version.")
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) to[i] *= std::sin(from[i]); // Could be vectorized.
#endif
}

template<typename Container, typename Functor>
void block_apply(Container &con) {
    if(&con[1] - &con[0] == 1) block_apply(&*std::begin(con), con.size());
    else {
        Functor func;
        for(auto &el: con) el = func.scalar(el);
    }
}

#define blocksin35 block_apply<FloatType, SIMDTypes<FloatType>::apply_sin_u35>
#define blocksin10 block_apply<FloatType, SIMDTypes<FloatType>::apply_sin_u10>
#define blocksin blocksin35
#define blockcos35 block_apply<FloatType, SIMDTypes<FloatType>::apply_cos_u35>
#define blockcos10 block_apply<FloatType, SIMDTypes<FloatType>::apply_cos_u10>
#define blockcos blockcos35

} // namespace vec
#undef OP
#undef decop

#endif // #ifndef _VEC_H__
