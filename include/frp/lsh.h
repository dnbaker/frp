#ifndef FRP_LSH_H__
#define FRP_LSH_H__
#include "vec/vec.h"
#include "frp/jl.h"


namespace frp {
using SIMDSpace = vec::SIMDTypes<uint64_t>;
using VType = typename SIMDSpace::VType;
template<typename F, typename V> ATTR_CONST INLINE auto cmp_zero(V v);
#if _FEATURE_AVX512F
template<> ATTR_CONST INLINE auto
cmp_zero<float> (__m512 v) {
    return _mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ);
}
template<> ATTR_CONST INLINE auto
cmp_zero<float> (__m512d v) {
    return _mm512_cmp_pd_mask(v, _mm512_setzero_pd(), _CMP_GT_OQ);
}
#elif __AVX__
template<> 
ATTR_CONST INLINE
auto cmp_zero<float, __m256> (__m256 v) {
    return _mm256_movemask_ps(_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ));
}
template<>
ATTR_CONST INLINE
auto cmp_zero<double, __m256d> (__m256d v) {
    return _mm256_movemask_pd(_mm256_cmp_pd(v, _mm256_setzero_pd(), _CMP_GT_OQ));
}
#else
#pragma message("not vectorizing signed projection hashing")
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

template<typename FType, bool SO=blaze::rowMajor, typename DistributionType=std::normal_distribution<FType>, typename...DistArgs>
blaze::DynamicMatrix<FType, SO>
generate_randproj_matrix(size_t nr, size_t ncol,
                         bool orthonormalize=true, uint64_t seed=0,
                         DistArgs &&...args)
{
    using matrix_type = blaze::DynamicMatrix<FType, SO>;
    matrix_type ret(nr, ncol);
    seed = ((seed ^ nr) * ncol) * seed;
    if(orthonormalize) {
        try {
            matrix_type r, q;
            if(ret.rows() >= ret.columns()) {
                // Randomize
                OMP_PRAGMA("omp parallel for")
                for(size_t i = 0; i < ret.rows(); ++i) {
                    blaze::RNG gen(seed + i * seed + i);
                    DistributionType dist(std::forward<DistArgs>(args)...);
                    for(auto &v: row(ret, i))
                        v = dist(gen);
                }
                // QR
                blaze::qr(ret, q, r);
                assert(ret.columns() == q.columns());
                assert(ret.rows() == q.rows());
                swap(ret, q);
            } else {
                // Generate random matrix for (C, C) and then just take the first R rows
                const auto mc = ret.columns();
                matrix_type tmp(mc, mc);
                OMP_PRAGMA("omp parallel for")
                for(size_t i = 0; i < tmp.rows(); ++i) {
                    blaze::RNG gen(seed + i * seed + i);
                    DistributionType dist(std::forward<DistArgs>(args)...);
                    for(auto &v: row(tmp, i))
                        v = dist(gen);
                }
                blaze::qr(tmp, q, r);
                ret = submatrix(q, 0, 0, ret.rows(), ret.columns());
            }
            OMP_PRAGMA("omp parallel for")
            for(size_t i = 0; i < ret.rows(); ++i)
                blaze::normalize(row(ret, i));
        } catch(const std::exception &ex) { // Orthonormalize
            std::fprintf(stderr, "failure in orthonormalization: %s\n", ex.what());
            throw;
        }
    } else {
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < nr; ++i) {
            blaze::RNG gen(seed + i);
            std::normal_distribution dist;
            for(auto &v: row(ret, i))
                v = dist(gen);
            normalize(row(ret, i));
        }
    }
    return ret;
}



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
static INLINE uint64_t cmp2hash(const blaze::DynamicVector<FType, OSO> &c, size_t n=0) {
    assert(n <= 64);
    uint64_t ret = 0;
    if(n == 0) {
        n = n;
    }
#if HAS_AVX_512
    static constexpr size_t COUNT = sizeof(__m512d) / sizeof(FType);
#elif __AVX__
    static constexpr size_t COUNT = sizeof(__m256d) / sizeof(FType);
#else
    static constexpr size_t COUNT = 0;
#endif
    size_t i = 0;
#if HAS_AVX_512 || defined(__AVX__)
    CONST_IF(COUNT) {
        using LV = F2VType<FType, sizeof(VType)>;
        for(; i < n / COUNT;ret = (ret << COUNT) | cmp_zero<FType, typename LV::type>(LV::load((&c[i++ * COUNT]))));
        i *= COUNT;
    }
#else
    for(;i + 8 <= n; i += 8) {
        ret = (ret << 8) |
              ((c[i] > 0.) << 7) | ((c[i + 1] > 0.) << 6) |
              ((c[i + 2] > 0.) << 5) | ((c[i + 3] > 0.) << 4) |
              ((c[i + 4] > 0.) << 3) | ((c[i + 5] > 0.) << 2) |
              ((c[i + 6] > 0.) << 1) | (c[i + 7] > 0.);
    }
#endif
    for(; i < n; ret = (ret << 1) | (c[i++] > 0.));
    return ret;
}

template<typename FType=float, bool SO=blaze::rowMajor, typename DistributionType=std::normal_distribution<FType>>
struct MatrixLSHasher {
    using CType = ::blaze::DynamicMatrix<FType, SO>;
    using this_type       =       MatrixLSHasher<FType, SO>;
    using const_this_type = const MatrixLSHasher<FType, SO>;
    CType container_;
    template<typename...DistArgs>
    MatrixLSHasher(size_t nr, size_t nc, bool orthonormalize=true, uint64_t seed=0,
                   DistArgs &&...args):
        container_(std::move(generate_randproj_matrix<FType, SO, DistributionType>(nr, nc, orthonormalize, seed, std::forward<DistArgs>(args)...))) {}
    auto &multiply(const blaze::DynamicVector<FType, SO> &c, blaze::DynamicVector<FType, SO> &ret) const {
        ret = this->container_ * c;
        return ret;
    }
    auto multiply(const blaze::DynamicVector<FType, SO> &c) const {
        blaze::DynamicVector<FType, SO> vec = this->container_ * c;
        return vec;
    }
    auto multiply(const blaze::DynamicVector<FType, !SO> &c) const {
        blaze::DynamicVector<FType, SO> vec = this->container_ * trans(c);
        return vec;
    }
    template<typename...Args>
    decltype(auto) project(Args &&...args) const {return multiply(std::forward<Args>(args)...);}
    template<bool OSO>
    uint64_t hash(const blaze::DynamicVector<FType, OSO> &c) const {
#if VERBOSE_AF
        std::cout << this->container_ << '\n';
#endif
        blaze::DynamicVector<FType, SO> vec = multiply(c);
        return cmp2hash(vec);
    }
    template<bool OSO>
    uint64_t operator()(const blaze::DynamicVector<FType, OSO> &c) const {
        return this->hash(c);
    }
};

template<typename FType=float, bool SO=blaze::rowMajor, typename DistributionType=std::normal_distribution<FType>>
struct FHTLSHasher {
    using this_type       =       FHTLSHasher<FType, SO>;
    using const_this_type = const FHTLSHasher<FType, SO>;
    std::vector<blaze::DynamicVector<FType, SO>> d_; // diagonal matrix
                                                     // use Matrix for case that we need more projections than we have dimensions
    std::vector<jl::OrthogonalJLTransform> jlt_;
    size_t nc_, nr_;
    template<typename...DistArgs>
    FHTLSHasher(size_t nr, size_t nc, uint64_t seed=0, unsigned nblocks=3,
                DistArgs &&...args): nc_(nc), nr_(nr) {
        unsigned njlts = (nr + nc - 1) / nc;
        jlt_.reserve(njlts);
        if(nr > nc) {
            std::mt19937_64 mt(seed + nc * nr);
            throw std::runtime_error("Not implemented: projections > dimensionality. TODO: this");
        } else {
            jlt_.emplace_back(nc, nr, seed + (nc * nr), nblocks);
            d_.emplace_back(roundup(nc), 0);
        }
        blaze::RNG gen(seed);
        DistributionType dist(std::forward<DistArgs>(args)...);
        for(auto &d: d_) 
            for(size_t i = 0; i < nc; ++i)
                d[i] = dist(gen);
    }
    auto &multiply(const blaze::DynamicVector<FType, SO> &c, blaze::DynamicVector<FType, SO> &ret) const {
        // This will change when we support more projections than input dimensions
        auto &d = d_[0];
        auto &jl = jlt_[0];
        if(ret.size() != d.size()) ret.resize(d.size());
        subvector(ret, 0, nc_) = trans(c) * subvector(d, 0, nc_);
        subvector(ret, nc_, d.size() - nc_) = 0;
        jl.transform_inplace(ret);
        return ret;
    }
    auto multiply(const blaze::DynamicVector<FType, SO> &c) const {
        blaze::DynamicVector<FType, SO> vec(d_.size());
        multiply(c, vec);
        return vec;
    }
    auto multiply(const blaze::DynamicVector<FType, !SO> &c) const {
        // This will change when we support more projections than input dimensions
        auto &d = d_[0];
        auto &jl = jlt_[0];
        blaze::DynamicVector<FType, SO> vec(d.size());
        subvector(vec, 0, nc_) = trans(c) * subvector(d, 0, nc_);
        subvector(vec, nc_, d.size() - nc_) = 0;
        jl.transform_inplace(vec);
        return vec;
    }
    template<typename...Args>
    decltype(auto) project(Args &&...args) const {return multiply(std::forward<Args>(args)...);}
    template<bool OSO>
    uint64_t hash(const blaze::DynamicVector<FType, OSO> &c) const {
#if VERBOSE_AF
        std::cout << this->container_ << '\n';
#endif
        blaze::DynamicVector<FType, SO> vec = multiply(c);
        return cmp2hash(vec, nr_);
    }
    template<bool OSO>
    uint64_t operator()(const blaze::DynamicVector<FType, OSO> &c) const {
        return this->hash(c);
    }
};

} // frp

#endif
