#ifndef FRP_LSH_H__
#define FRP_LSH_H__
#include "vec/vec.h"
#include "frp/util.h"


namespace frp {
using SIMDSpace = vec::SIMDTypes<uint64_t>;
using VType = typename SIMDSpace::VType;
#if _FEATURE_AVX512F
template<typename F, typename V> auto cmp_zero(V v);
template<> uint32_t cmp_zero<float> (__m512 v) {
    return _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ);
}
template<> auto cmp_zero<float> (__m512d v) {
    return _mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_GT_OQ);
}
#elif __AVX__
template<typename F, typename V> uint32_t cmp_zero(V v);
template<> uint32_t cmp_zero<float, __m256> (__m256 v) {
    return _mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ));
}
template<> uint32_t cmp_zero<double, __m256d> (__m256d v) {
    return _mm256_movemask_pd(_mm256_cmp_pd(v, _mm256_setzero_pd(), _CMP_GT_OQ));
}
#else
#error("")
#endif

template<typename FType, bool SO>
struct empty {
    template<typename...Args> empty(Args &&...args) {}
};

template<typename FType, size_t VSZ>
struct F2VType;
#if __AVX__

template<> struct F2VType<float, 32> {
    using type = __m256;
    static type load(const float *a) {
        return _mm256_loadu_ps(a);
    }
};
template<> struct F2VType<double, 32> {
    using type = __m256d;
    static type load(const double *a) {
        return _mm256_loadu_pd(a);
    }
};
#endif
#if HAS_AVX_512
template<> struct F2VType<float, 64> {
    using type = __m512;
    static type load(const float *a) {
        return _mm512_loadu_ps(a);
    }
};
template<> struct F2VType<double, 64> {
    using type = __m512d;
    static type load(const double *a) {
        return _mm512_loadu_pd(a);
    }
};
#endif

#if HAS_AVX_512
    template<typename FType>
    static constexpr int f2b(__m512d v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
    template<typename FType>
    static constexpr int f2b(__m512 v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
    //using VType = F2VType<FType, sizeof(__m512)>::type;
#endif
#if __AVX__
    template<typename FType>
    static constexpr int f2b(__m256d v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
    template<typename FType>
    static constexpr int f2b(__m256 v) {
        return cmp_zero<FType, decltype(v)>(v);
    }
    //using VType = typename F2VType<FType, sizeof(__m256)>::type;
#endif



template<typename FType=float, template<typename, bool> class Container=::blaze::DynamicVector, bool SO=blaze::rowMajor>
struct LSHasher {
    using CType = Container<FType, SO>;
    CType container_;
    template<typename... CArgs>
    LSHasher(CArgs &&...args): container_(std::forward<CArgs>(args)...) {}
    template<typename T>
    auto dot(const T &ov) const {
        return blaze::dot(container_, ov);
    }
    // TODO: Store full matrix to get hashes
    // TODO: Use structured matrices to speed up calculation (FFHT, then downsample to bins)
};



template<typename FType, bool OSO>
static uint64_t cmp2hash(const blaze::DynamicVector<FType, OSO> &c) {
    assert(c.size() <= 64);
    uint64_t ret = 0;
#if HAS_AVX_512
    static constexpr size_t COUNT = sizeof(__m512d) / sizeof(FType);
#elif __AVX__
    static constexpr size_t COUNT = sizeof(__m256d) / sizeof(FType);
#else
    static constexpr size_t COUNT = 0;
#endif
    size_t i = 0;
#if 0
//#if HAS_AVX_512 || __AVX__
    CONST_IF(COUNT) {
        using LV = F2VType<FType, sizeof(VType)>;
        for(; i < c.size() / COUNT;ret = (ret << COUNT) | cmp_zero<FType, typename LV::type>(LV::load((&c[i++ * COUNT]))));
        i *= COUNT;
    }
#endif
    for(; i < c.size(); ret = (ret << 1) | (c[i++] > 0.));
    return ret;
}
template<typename FType=float, bool SO=blaze::rowMajor>
struct MatrixLSHasher: public LSHasher<FType, ::blaze::DynamicMatrix, SO> {
    using super = LSHasher<FType, ::blaze::DynamicMatrix, SO>;
    template<typename...CArgs>
    MatrixLSHasher(CArgs &&...args): super(std::forward<CArgs>(args)...) {
        std::normal_distribution<FType> dist;
        std::mt19937_64 mt;
        for(size_t i = 0; i < this->container_.rows(); ++i) {
            auto r = row(this->container_, i);
            for(auto &e: r)
                e = dist(mt);
        }
    }
    uint64_t hash(const blaze::DynamicVector<FType, SO> &c) const {
        std::cout << this->container_ << '\n';
        std::fprintf(stderr, "size: %zu, %zu. input; %zu\n", this->container_.rows(), this->container_.columns(), c.size());
        blaze::DynamicVector<FType, SO> vec = this->container_ * c;
        return cmp2hash(vec);
    }
#if 0
    uint64_t hash(const blaze::DynamicVector<FType, !SO> &c) const {
        std::fprintf(stderr, "lolzballs !SO size: %zu, %zu. input; %zu\n", this->container_.rows(), this->container_.columns(), c.size());
        blaze::DynamicVector<FType, SO> vec = this->container_ * trans(c);
        return cmp2hash(vec);
    }
#endif
    template<bool OSO> uint64_t operator()(const blaze::DynamicVector<FType, OSO> &c) const {
        return this->hash(c);
    }
};

} // frp

#endif
