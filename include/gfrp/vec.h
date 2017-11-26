#ifndef _VEC_H__
#define _VEC_H__
#define NOSVML
#include "sleef/include/sleefdft.h"
#include "sleef.h"

#include "x86intrin.h"
#include <cmath>
#include <iterator>


namespace std {
    Sleef_double2 sincos(double x) {
        return Sleef_double2{std::sin(x), std::cos(x)};
    }
    Sleef_float2 sincos(float x) {
        return Sleef_float2{std::sin(x), std::cos(x)};
    }
}

namespace vec {

template<typename ValueType>
struct SIMDTypes;

#define OP(op, suf, sz) _mm##sz##_##op##_##suf
#define decop(op, suf, sz) static constexpr decltype(&OP(op, suf, sz)) op = &OP(op, suf, sz);

/* Use or separately because it's a keyword.*/

#define declare_all(suf, sz) \
   decop(loadu, suf, sz); \
   decop(storeu, suf, sz); \
   decop(load, suf, sz); \
   decop(store, suf, sz); \
   static constexpr decltype(&OP(or, suf, sz)) or_fn = &OP(or, suf, sz);\
   static constexpr decltype(&OP(and, suf, sz)) and_fn = &OP(and, suf, sz);\
   decop(add, suf, sz); \
   decop(sub, suf, sz); \
   decop(mul, suf, sz); \
   decop(set1, suf, sz); \
   decop(setr, suf, sz); \
   decop(set, suf, sz); \
   decop(mask_and, suf, sz); \
   decop(maskz_and, suf, sz); \
   decop(maskz_andnot, suf, sz); \
   decop(mask_andnot, suf, sz); \
   decop(andnot, suf, sz); \
   decop(blendv, suf, sz); \
   decop(cmp, suf, sz); \


#define SLEEF_OP(op, suf, prec, set) Sleef_##op##suf##_##prec##set
#define dec_sleefop_prec(op, suf, prec, instructset) \
    static constexpr decltype(&SLEEF_OP(op, suf, prec, instructset)) op##_##prec = \
    &SLEEF_OP(op, suf, prec, instructset); \
    struct apply_##op##_##prec {\
        template<typename... T>\
        auto operator()(T &&...args) const {return op##_##prec(std::forward<T...>(args)...);} \
        template<typename OT>\
        OT scalar(OT val) const {return std::op(val);} \
    };


#define dec_all_precs(op, suf, instructset) \
    dec_sleefop_prec(op, suf, u35, instructset) \
    dec_sleefop_prec(op, suf, u10, instructset)


#define dec_double_sz(type) using TypeDouble = Sleef_##type##_2;


#define dec_all_trig(suf, set) \
   dec_all_precs(sin, suf, set) \
   dec_all_precs(cos, suf, set) \
   dec_all_precs(asin, suf, set) \
   dec_all_precs(acos, suf, set) \
   dec_all_precs(atan, suf, set) \
   dec_all_precs(atan2, suf, set) \
   dec_all_precs(cbrt, suf, set) \
   dec_all_precs(sincos, suf, set) \
   dec_sleefop_prec(log, suf, u10, set) \
   dec_sleefop_prec(log1p, suf, u10, set) \
   dec_sleefop_prec(expm1, suf, u10, set) \
   dec_sleefop_prec(exp, suf, u10, set) \
   dec_sleefop_prec(exp2, suf, u10, set) \
   /*dec_sleefop_prec(exp10, suf, u10, set) */ \
   dec_sleefop_prec(lgamma, suf, u10, set) \
   dec_sleefop_prec(tgamma, suf, u10, set) \
   dec_sleefop_prec(sinh, suf, u10, set) \
   dec_sleefop_prec(cosh, suf, u10, set) \
   dec_sleefop_prec(asinh, suf, u10, set) \
   dec_sleefop_prec(acosh, suf, u10, set) \
   dec_sleefop_prec(tanh, suf, u10, set) \
   dec_sleefop_prec(atanh, suf, u10, set)
    

template<>
struct SIMDTypes<float>{
#if _FEATURE_AVX512F
    using Type = __m512;
    declare_all(ps, 512)
    static const size_t ALN = 64;
    dec_double_sz(__m512)
    dec_all_trig(f16, avx512f);
#elif __AVX2__
    using Type = __m256;
    declare_all(ps, 256)
    static const size_t ALN = 32;
    dec_double_sz(__m256)
    dec_all_trig(f8, avx2);
#elif __SSE2__
    using Type = __m128;
    declare_all(ps, )
    static const size_t ALN = 16;
    dec_double_sz(__m128)
    dec_all_trig(f4, sse2);
#else
#error("Need at least sse2")
#endif
    static const size_t MASK = ALN - 1;
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
};

template<>
struct SIMDTypes<double>{
#if _FEATURE_AVX512F
    using Type = __m512d;
    declare_all(pd, 512)
    static const size_t ALN = 64;
    dec_double_sz(__m512d);
    dec_all_trig(d8, avx512f);
#elif __AVX2__
    using Type = __m256d;
    declare_all(pd, 256)
    static const size_t ALN = 32;
    dec_double_sz(__m256d);
    dec_all_trig(d4, avx2);
#elif __SSE2__
    using Type = __m128d;
    declare_all(pd, )
    static const size_t ALN = 16;
    dec_double_sz(__m128d);
    dec_all_trig(d2, sse2);
#else
#error("Need at least sse2")
#endif
    static const size_t MASK = ALN - 1;
    template<typename T>
    static constexpr bool aligned(T *ptr) {
        return (reinterpret_cast<uint64_t>(ptr) & MASK) == 0;
    }
};


template<typename FloatType>
void blockmul(FloatType *pos, size_t nelem, FloatType div) {
#if __AVX2__ || _FEATURE_AVX512F || __SSE2__
#pragma message("Using vectorized scalar multiplication.")
        using SIMDType = typename vec::SIMDTypes<FloatType>::Type;
        SIMDType factor(vec::SIMDTypes<FloatType>::set1(div));
        SIMDType *ptr((SIMDType *)pos);
        FloatType *end(pos + nelem);
        if((uint64_t)ptr & vec::SIMDTypes<FloatType>::MASK) {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::storeu((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul(factor, vec::SIMDTypes<FloatType>::loadu((FloatType *)ptr)));
                ++ptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::store((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul(factor, vec::SIMDTypes<FloatType>::load((FloatType *)ptr)));
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
        SIMDType inc(vec::SIMDTypes<FloatType>::set1(val));
        SIMDType *ptr((SIMDType *)pos);
        FloatType *end(pos + nelem);
        if((uint64_t)ptr & vec::SIMDTypes<FloatType>::MASK) {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::storeu((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::add(inc, vec::SIMDTypes<FloatType>::loadu((FloatType *)ptr)));
                ++ptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::store((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::add(inc, vec::SIMDTypes<FloatType>::load((FloatType *)ptr)));
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
                vec::SIMDTypes<FloatType>::storeu((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul(vec::SIMDTypes<FloatType>::loadu((FloatType *)fromptr), vec::SIMDTypes<FloatType>::loadu((FloatType *)ptr)));
                ++ptr; ++fromptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                vec::SIMDTypes<FloatType>::store((FloatType *)ptr,
                    vec::SIMDTypes<FloatType>::mul(vec::SIMDTypes<FloatType>::load((FloatType *)fromptr), vec::SIMDTypes<FloatType>::load((FloatType *)ptr)));
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
void block_apply(FloatType *pos, size_t nelem, const Functor &func=Functor{}) {
#if __AVX2__ || _FEATURE_AVX512F || __SSE2__
#pragma message("Using vectorized multiplication.")
        using Space = typename vec::SIMDTypes<FloatType>;
        using SIMDType = typename Space::Type;
        SIMDType *ptr((SIMDType *)pos);
        FloatType *end(pos + nelem);
        if((uint64_t)ptr & Space::MASK) {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::storeu((FloatType *)ptr,
                    func(Space::loadu((FloatType *)ptr)));
                ++ptr;
            }
        } else {
            while((FloatType *)ptr < end - sizeof(SIMDType) / sizeof(FloatType)) {
                Space::store((FloatType *)ptr,
                    func(Space::load((FloatType *)ptr)));
                ++ptr;
            }
        }
        pos = (FloatType *)ptr;
        while(pos < end) *pos  = func.scalar(*pos), ++pos;
#else
#pragma message("Enjoy your serial version.")
            for(size_t i(0); i < (static_cast<size_t>(1) << nelem); ++i) to[i] *= func.scalar(to[i]); // Could be vectorized.
#endif
}

template<typename Container, typename Functor>
void block_apply(Container &con, const Functor &func=Functor{}) {
    if(&con[1] - &con[0] == 1) {
        const size_t nelem(con.size());
        block_apply(&(*std::begin(con)), nelem, func);
    }
    else {
        Functor func;
        for(auto &el: con) el = func.scalar(el);
    }
}

} // namespace vec
#undef OP
#undef decop
#undef SLEEF_OP
#undef dec_sleefop_prec
#undef dec_all_precs
#undef dec_all_trig

#endif // #ifndef _VEC_H__
