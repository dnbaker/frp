#ifndef _GFRP_LINALG_H__
#define _GFRP_LINALG_H__
#define _USE_MATH_DEFINES
#include <queue>
#include <cstdint>
#include <future>
#include <limits>
#include <type_traits>
#ifdef NO_BLAZE
#undef NO_BLAZE
#endif
#include "vec/vec.h"
#include "vec/stats.h"
#include "vec/welford_sd.h"
#include "x86intrin.h"
#include "frp/util.h"

#ifndef VECTOR_WIDTH
#ifdef __AVX2__
#define VECTOR_WIDTH (32ul)
#elif __SSE2__
#define VECTOR_WIDTH (16ul)
#else
#define VECTOR_WIDTH (8ul)
#endif
#endif // VECTOR_WIDTH

namespace frp { inline namespace linalg {
using std::forward;

template<class Container>
auto meanvar(const Container &c) {
    using FloatType = std::decay_t<decltype(c[0])>;
    FloatType sum(0.), varsum(0.0);
    for_each_nz(c, [&](auto, auto y) {sum += y; varsum += y * y;});
    const auto inv(static_cast<FloatType>(1)/static_cast<FloatType>(c.size()));
    varsum -= sum * sum * inv;
    varsum *= inv;
    sum *= inv;
    return std::make_pair(sum, varsum);
}
using std::runtime_error;
using std::numeric_limits;
using std::enable_if_t;
using std::is_arithmetic;

enum GramSchmitFlags: unsigned {
    FLIP = 1,
    ORTHONORMALIZE = 2,
    RESCALE_TO_GAUSSIAN = 4,
    EXECUTE_IN_PARALLEL = 8
};

template<typename MatrixKind>
void gram_schmidt(MatrixKind &b, unsigned flags=ORTHONORMALIZE);
template<typename MatrixKind1, typename MatrixKind2>
void gram_schmidt(const MatrixKind1 &a, MatrixKind2 &b, unsigned flags=RESCALE_TO_GAUSSIAN) {
    b = a;
    gram_schmidt(b, flags);
}

template<typename ValueType>
void mempluseq(ValueType *data, size_t nelem, ValueType val) {
    data += nelem * val;
    throw runtime_error("Not implemented.");
}

template<>
void mempluseq<float>(float *data, size_t nelem, float val) {

#if _FEATURE_AVX512F
#define load_fn _mm512_loadu_ps
#define store_fn _mm512_storeu_ps
#define add_fn _mm512_add_ps
    using ValType = __m512;
    const __m512 vval(__mm512_set1_ps(val));
#elif __AVX2__
#define load_fn _mm256_loadu_ps
#define store_fn _mm256_storeu_ps
#define add_fn _mm256_add_ps
    using ValType = __m256;
    const __m256 vval(_mm256_set1_ps(val));
#elif __SSE2__
#define load_fn _mm_loadu_ps
#define store_fn _mm_storeu_ps
#define add_fn _mm_add_ps
    using ValType = __m128;
    const __m128 vval(_mm_set1_ps(val));
#else
    while(nelem--) *data += val;
    return;
#endif
#if _FEATURE_AVX512F || __AVX2__ || __SSE2__
    while(nelem >= (sizeof(ValType) / sizeof(float))) {
        store_fn(data, add_fn(load_fn(data), vval)); // *data += vval
        nelem -= (sizeof(ValType) / sizeof(float)); // Move ahead and skip that many elements
        data += (sizeof(ValType) / sizeof(float));
    }
    while(nelem--) *data++ += val;
#undef load_fn
#undef add_fn
#undef store_fn
#endif
}
template<>
void mempluseq<int8_t>(int8_t *data, size_t nelem, int8_t val) {
#if _FEATURE_AVX512F
#define load_fn _mm512_loadu_si512
#define store_fn _mm512_storeu_si512
#define add_fn _mm512_add_pd
    using ValType = __m512i;
    const ValType vval(_mm512_set1_epi8(val));
#elif __AVX2__
#define load_fn _mm256_loadu_si256
#define store_fn _mm256_storeu_si256
#define add_fn _mm256_add_epi8
    using ValType = __m256i;
    const ValType vval(_mm256_set1_epi8(val));
#elif __SSE2__
#define load_fn _mm_loadu_si128
#define store_fn _mm_storeu_si128
#define add_fn _mm_add_epi8
    using ValType = __m128i;
    const ValType vval(_mm_set1_epi8(val));
#else
    while(nelem--) *data += val;
    return;
#endif
    using NumType = int8_t;
#if _FEATURE_AVX512F || __AVX2__ || __SSE2__
    while(nelem >= (sizeof(ValType) / sizeof(NumType))) {
        store_fn((ValType *)data, add_fn(load_fn((ValType *)data), vval)); // *data += vval
        nelem -= (sizeof(ValType) / sizeof(NumType)); // Move ahead and skip that many elements
        data += (sizeof(ValType) / sizeof(NumType));
    }
    while(nelem--) *data = (int8_t)(*data + val), ++data;
#undef load_fn
#undef add_fn
#undef store_fn
#endif
}
template<>
void mempluseq<int16_t>(int16_t *data, size_t nelem, int16_t val) {
#if _FEATURE_AVX512F
#define load_fn _mm512_loadu_si512
#define store_fn _mm512_storeu_si512
#define add_fn _mm512_add_epi16
    using ValType = __m512i;
    const ValType vval(_mm512_set1_epi16(val));
#elif __AVX2__
#define load_fn _mm256_loadu_si256
#define store_fn _mm256_storeu_si256
#define add_fn _mm256_add_epi16
    using ValType = __m256i;
    const ValType vval(_mm256_set1_epi16(val));
#elif __SSE2__
#define load_fn _mm_loadu_si128
#define store_fn _mm_storeu_si128
#define add_fn _mm_add_epi16
    using ValType = __m128i;
    const ValType vval(_mm_set1_epi16(val));
#else
    while(nelem--) *data += val;
    return;
#endif
    using NumType = int16_t;
#if _FEATURE_AVX512F || __AVX2__ || __SSE2__
    while(nelem >= (sizeof(ValType) / sizeof(NumType))) {
        store_fn((ValType *)data, add_fn(load_fn((ValType *)data), vval)); // *data += vval
        nelem -= (sizeof(ValType) / sizeof(NumType)); // Move ahead and skip that many elements
        data += (sizeof(ValType) / sizeof(NumType));
    }
    while(nelem--) {
        *data = (int16_t)((int16_t)(*data) + val);
         ++data;
    }
#undef load_fn
#undef add_fn
#undef store_fn
#endif
}
template<>
void mempluseq<int64_t>(int64_t *data, size_t nelem, int64_t val) {
#if _FEATURE_AVX512F
#define load_fn _mm512_loadu_si512
#define store_fn _mm512_storeu_si512
#define add_fn _mm512_add_epi64
    using ValType = __m512i;
    const ValType vval(_mm512_set1_epi64(val));
#elif __AVX2__
#define load_fn _mm256_loadu_si256
#define store_fn _mm256_storeu_si256
#define add_fn _mm256_add_epi64
    using ValType = __m256i;
    const ValType vval(_mm256_set1_epi64x(val));
#elif __SSE2__
#define load_fn _mm_loadu_si128
#define store_fn _mm_storeu_si128
#define add_fn _mm_add_epi64
    using ValType = __m128i;
    const ValType vval(_mm_set1_epi64x(val));
#else
    while(nelem--) *data += val;
    return;
#endif
    using NumType = int64_t;
#if _FEATURE_AVX512F || __AVX2__ || __SSE2__
    while(nelem >= (sizeof(ValType) / sizeof(NumType))) {
        store_fn((ValType *)data, add_fn(load_fn((ValType *)data), vval)); // *data += vval
        nelem -= (sizeof(ValType) / sizeof(NumType)); // Move ahead and skip that many elements
        data += (sizeof(ValType) / sizeof(NumType));
    }
    while(nelem--) *data++ += val;
#undef load_fn
#undef add_fn
#undef store_fn
#endif
}
template<>
void mempluseq<int32_t>(int32_t *data, size_t nelem, int32_t val) {
#if _FEATURE_AVX512F
#define load_fn _mm512_loadu_si512
#define store_fn _mm512_storeu_si512
#define add_fn _mm512_add_epi32
    using ValType = __m512i;
    const ValType vval(_mm512_set1_epi32(val));
#elif __AVX2__
#define load_fn _mm256_loadu_si256
#define store_fn _mm256_storeu_si256
#define add_fn _mm256_add_epi32
    using ValType = __m256i;
    const ValType vval(_mm256_set1_epi32(val));
#elif __SSE2__
#define load_fn _mm_loadu_si128
#define store_fn _mm_storeu_si128
#define add_fn _mm_add_epi32
    using ValType = __m128i;
    const ValType vval(_mm_set1_epi32(val));
#else
    while(nelem--) *data += val;
    return;
#endif
    using NumType = int32_t;
#if _FEATURE_AVX512F || __AVX2__ || __SSE2__
    while(nelem >= (sizeof(ValType) / sizeof(NumType))) {
        store_fn((ValType *)data, add_fn(load_fn((ValType *)data), vval)); // *data += vval
        nelem -= (sizeof(ValType) / sizeof(NumType)); // Move ahead and skip that many elements
        data += (sizeof(ValType) / sizeof(NumType));
    }
    while(nelem--) *data++ += val;
#undef load_fn
#undef add_fn
#undef store_fn
#endif
}

template<>
void mempluseq<double>(double *data, size_t nelem, double val) {
#if _FEATURE_AVX512F
#define load_fn _mm512_loadu_pd
#define store_fn _mm512_storeu_pd
#define add_fn _mm512_add_pd
    using ValType = __m512d;
    const __m512d vval(_mm512_set1_pd(val));
#elif __AVX2__
#define load_fn _mm256_loadu_pd
#define store_fn _mm256_storeu_pd
#define add_fn _mm256_add_pd
    using ValType = __m256d;
    const __m256d vval(_mm256_set1_pd(val));
#elif __SSE2__
#define load_fn _mm_loadu_pd
#define store_fn _mm_storeu_pd
#define add_fn _mm_add_pd
    using ValType = __m128d;
    const __m128d vval(_mm_set1_pd(val));
#else
    while(nelem--) *data += val;
    return;
#endif
    using NumType = double;
#if _FEATURE_AVX512F || __AVX2__ || __SSE2__
    while(nelem >= (sizeof(ValType) / sizeof(NumType))) {
        store_fn(data, add_fn(load_fn(data), vval)); // *data += vval
        nelem -= (sizeof(ValType) / sizeof(NumType)); // Move ahead and skip that many elements
        data += (sizeof(ValType) / sizeof(NumType));
    }
    while(nelem--) *data++ += val;
#undef load_fn
#undef add_fn
#undef store_fn
#endif
}


#if 0
template<typename MatrixType, typename ValueType,
         typename=enable_if_t<is_arithmetic<ValueType>::value>>
MatrixType &operator+=(MatrixType &in, ValueType val) {
    CONST_IF(blaze::IsMatrix<MatrixType>::value) {
        CONST_IF(blaze::IsSparseMatrix<MatrixType>::value) {
            for(size_t i(0); i < in.rows(); ++i) {
                for_each_nz(row(in, i), [&](auto i, auto &v) {v += val;});
#if 0
                for(auto it(in.begin(i)), eit(in.end(i)); it != eit; ++it) {
                    it->value() += val;
                }
#endif
            }
        } else {
            if(size_t(&in(0, 1) - &in(0, 0)) == 1) {
                for(size_t i(0); i < in.rows(); ++i) {
                    mempluseq(&in(i, 0), in.columns(), val);
                }
            } else {
                for(size_t i(0); i < in.columns(); ++i) {
                    mempluseq(&in(0, i), in.rows(), val);
                }
            }
        }
    } else {
        CONST_IF(blaze::IsSparseVector<MatrixType>::value) {
                for_each_nz(in, [&](auto i, auto &v) {v += val;});
        } else {
            if(size_t(&in[0] - &in[1]) == 1) {
                mempluseq(&in[0], in.size(), val);
            } else {
                for(auto &el: in) el = val;
            }
        }
    }
    return in;
}

template<typename MatrixType, typename ValueType,
         typename=enable_if_t<is_arithmetic<ValueType>::value>>
MatrixType &operator-=(MatrixType &in, ValueType val) {
    return in += -val;
}
#endif


/*
 Am I doing this RESCALE_TO_GAUSSIAN right?
 Realizations of Gort can be generated by, for example, performing a Gram-Schmidt process
 on the rows of G to obtain a set of orthonormal rows,
 and then randomly independently scaling each row so that marginally
 it has the distribution of a Gaussian vector.
 */

template<typename FloatType, bool SO>
auto &qr_gram_schmidt(const blaze::DynamicMatrix<FloatType, SO> &input,
                      blaze::DynamicMatrix<FloatType,SO> &Q, unsigned flags=ORTHONORMALIZE) {
    blaze::UpperMatrix<blaze::DynamicMatrix<FloatType,SO>> r;
    ::blaze::qr(input, Q, r);
    if(flags & ORTHONORMALIZE)
        for(size_t i(0); i < Q.rows(); ++i)
            row(Q, i) *= 1./norm(row(Q, i));
    return Q;
}

template<typename FloatType, bool SO>
auto qr_gram_schmidt(const blaze::DynamicMatrix<FloatType, SO> &input,
                     unsigned flags=ORTHONORMALIZE) {
    blaze::DynamicMatrix<FloatType,SO> ret;
    qr_gram_schmidt(input, ret, flags);
    return ret;
}

template<typename MatrixKind>
void gram_schmidt(MatrixKind &b, unsigned flags) {
    using FloatType = typename MatrixKind::ElementType;
    if(flags & FLIP) {
        blaze::DynamicVector<FloatType> inv_unorms(b.columns());
        FloatType tmp;
        for(size_t i(0), ncolumns(b.columns()); i < ncolumns; ++i) {
            auto icolumn(column(b, i));
            for(size_t j(0); j < i; ++j) {
                auto jcolumn(column(b, j));
                icolumn -= jcolumn * dot(icolumn, jcolumn) * inv_unorms[j];
            }
            if((tmp = dot(icolumn, icolumn)) == 0.0) {
            }
            inv_unorms[i] = tmp ? tmp: numeric_limits<decltype(tmp)>::max();
#if !NDEBUG
            if(tmp ==numeric_limits<decltype(tmp)>::max())
                fprintf(stderr, "Warning: norm of column %zu (0-based) is 0.0\n", i);
#endif
        }
        if(flags & ORTHONORMALIZE) {
            for(size_t i(0), ncolumns(b.columns()); i < ncolumns; ++i) {
#if !NDEBUG
                column(b, i) *= inv_unorms[i];
                std::fprintf(stderr, "Norm at %zu after renorm: %lf\n", i, norm(column(b, i)));
#else
                column(b, i) *= inv_unorms[i];
#endif
            }
        }
        if(flags & RESCALE_TO_GAUSSIAN) {
            for(size_t i = 0, ncolumns = b.columns(); i < ncolumns; ++i) {
                auto mcolumn(column(b, i));
                auto meanvarpair(meanvar(mcolumn)); // mean.first = mean, mean.second = var
                const auto invsqrt(1./std::sqrt(meanvarpair.second * static_cast<double>(b.rows())));
                mcolumn -= meanvarpair.first;
                mcolumn *= invsqrt;
            }
        }
    } else {
        blaze::DynamicVector<FloatType> inv_unorms(b.rows());
        for(size_t i(0), nrows(b.rows()); i < nrows; ++i) {
            auto irow(row(b, i));
            for(size_t j(0); j < i; ++j) {
                auto jrow(row(b, j));
                irow -= jrow * dot(irow, jrow) * inv_unorms[j];
            }
            auto tmp(1./dot(irow, irow));
            inv_unorms[i] = tmp ? tmp: numeric_limits<decltype(tmp)>::max();
        }
        if(flags & ORTHONORMALIZE)
            for(size_t i(0), nrows(b.rows()); i < nrows; ++i)
                row(b, i) *= inv_unorms[i];
        if(flags & RESCALE_TO_GAUSSIAN) {
            const size_t nrows(b.rows());
            for(size_t i = 0; i < nrows; ++i) {
                auto mrow(row(b, i));
                const auto meanvarpair(meanvar(mrow)); // mean.first = mean, mean.second = var
                mrow -= meanvarpair.first;
                mrow *= 1./std::sqrt(meanvarpair.second * static_cast<double>(b.columns()));
            }
        }
    }
}

template<typename FloatType>
constexpr inline auto ndball_surface_area(size_t nd, FloatType r) {
    // http://scipp.ucsc.edu/~haber/ph116A/volume_11.pdf
    // In LaTeX notation: $\frac{2\Pi^{\frac{n}{2}}R^{n-1}}{\Gamma(frac{n}{2})}$
    nd >>= 1;
    return 2. * std::pow(static_cast<FloatType>(M_PI), nd) / tgamma(nd) * std::pow(r, nd - 1);
}

template<typename T>
auto normalize(T &mat) {
    blaze::DynamicVector<float> averages(mat.columns()), std(mat.columns());
    for(size_t i = 0; i < mat.rows(); averages += row(mat, i++));
    averages /= mat.rows();
    for(size_t i = 0; i < mat.rows(); row(mat, i++) -= averages);
    return averages;
}
template<typename T>
auto naive_cov(const T &mat, bool by_feature=true, bool bias=true) {
    if(by_feature) {
        auto cmean = blaze::sum<blaze::columnwise>(mat) /  mat.rows();
        T cpy = mat;
        for(size_t i = 0; i < mat.rows(); ++i) {
            row(cpy, i) -= cmean;
        }
        blaze::SymmetricMatrix<T> ret = declsym(trans(cpy) * cpy);
        ret /= mat.rows() - bias;
        return ret;
    } else {
        auto cmean = blaze::sum<blaze::rowwise>(mat) /  mat.columns();
        T cpy = mat;
        for(size_t i = 0; i < mat.columns(); ++i) {
            column(cpy, i) -= cmean;
        }
        blaze::SymmetricMatrix<T> ret = declsym(cpy * trans(cpy));
        ret /= mat.columns() - bias;
        return ret;
    }
}


template<typename...Args>
auto cov(Args &&...args) {return naive_cov(std::forward<Args>(args)...);}
// Until I've benchmarked, default to the naive implementation which I've verified is correct.

template<typename T>
auto pca(const T &mat, bool by_feature=true, bool bias=true, int ncomp=-1) {
    // TODO: Use STEGR from LAPACK (https://www.netlib.org/lapack/lug/node48.html)
    //       for this if all are desired, or HEEVX if a subset are.
    // 
    // TODO: consider whitening transforms for clean-ups.
    using FType = typename T::ElementType;

    auto c = cov(mat, by_feature, bias);
    blaze::DynamicVector<FType> eigv;
    T eigenvectors;
    blaze::eigen(c, eigv, eigenvectors);
    // Sort by eigenvalue, largest to smallest
    std::vector<uint32_t> vec(eigv.size());
    std::iota(vec.begin(), vec.end(), 0u);
    std::sort(vec.begin(), vec.end(), [&eigv](auto x, auto y) {
        return eigv[x] > eigv[y];
    });
    std::fprintf(stderr, "Sorted eigenvalues\n");
    if(ncomp > 0) {
        std::fprintf(stderr, "subsampling not tested: %u\n", ncomp);
        //std::sort(&vec[0], &vec[ncomp]);
        T subset(mat.columns(), ncomp);
        std::fprintf(stderr, "Got subset\n");
        for(int i = 0; i < ncomp; ++i)
            column(subset, i) = column(eigenvectors, vec[i]);
        return std::make_pair(subset, eigv);
        //eigenvectors.resize(
    }
    else {
        return std::make_pair(eigenvectors, eigv);
    }
}


// TODO:
//      Add transformations for reductions,
//      including inner product LSH
//      and spherical transforms.

#define REQUIRE(cond, msg) do {if(!(cond)) {throw std::runtime_error(std::string("Failed requirement: " #cond) + msg);}} while(0)
template<typename FT>
struct PCAAggregator {
    static constexpr bool SO = blaze::rowMajor;
    using SymMat = blaze::SymmetricMatrix<blaze::DynamicMatrix<FT, SO>>;
    SymMat mat_;
    stats::OnlineVectorSD<blaze::DynamicVector<FT, SO>> mean_estimator_;
    size_t n_;
    std::unique_ptr<blaze::DynamicMatrix<FT, SO>> eigvec_;
    std::unique_ptr<blaze::DynamicVector<FT, SO>> eigval_;
    size_t nvs_;
    static constexpr bool is_sparse = blaze::IsSparseMatrix_v<decltype(mat_)>;

    PCAAggregator(size_t from, size_t to=0 /* ncomp */):
        mat_(from),
        mean_estimator_(from),
        nvs_(to ? to: size_t(-1))
    {
    }
    PCAAggregator(PCAAggregator &&o) = default;
    PCAAggregator(const PCAAggregator &o) = delete;
    template<bool aligned, bool padded>
    void add(const blaze::CustomMatrix<FT, aligned, padded, SO> &o) {
        REQUIRE(o.columns() == mat_.columns(), "must have matching # columns");
        assert((trans(o) * o).rows() == mat_.columns());
        std::future<void> fut = std::async(std::launch::async, [&]() {
            for(size_t i = 0; i < mat_.rows(); ++i)
                this->add(row(o, i));
        });
        mat_ += declsym(trans(o) * o);
        // TODO: map/reduce computation
        fut.get();
    }
    void add(const blaze::DynamicMatrix<FT, SO> &o) {
        REQUIRE(o.columns() == mat_.columns(), "must have matching # columns");
        assert((trans(o) * o).rows() == mat_.columns());
        std::future<void> fut = std::async(std::launch::async, [&]() {
            for(size_t i = 0; i < mat_.rows(); ++i)
                this->add(row(o, i));
        });
        mat_ += declsym(trans(o) * o);
        // TODO: map/reduce computation
        fut.get();
    }
    template<typename T>
    void add(const T &x) {
        if constexpr(blaze::TransposeFlag_v<T> == blaze::columnVector) {
            mean_estimator_.add(x);
            mat_ += x * trans(x);
        } else {
            mean_estimator_.add(trans(x));
            mat_ += trans(x) * x;
        }
        ++n_;
    }
    template<typename T>
    auto project(const T &x) const {
        REQUIRE(ready(), "must be ready");
        return eigvec_ * x;
    }
    bool ready() const {
        return eigvec_.get() && eigval_.get();
    }
    void finalize() {
        if(!n_) throw std::runtime_error("Can't finalize nothing");
        eigvec_.reset(new blaze::DynamicMatrix<FT, SO>(mat_.rows(), mat_.columns()));
        eigval_.reset(new blaze::DynamicVector<FT, SO>(mat_.rows(), mat_.columns()));
        auto &vecs = *eigvec_;
        auto &vals = *eigval_;
        blaze::DynamicMatrix<FT, SO> mat =
               (1. / (n_ > 1 ? n_ - 1: n_) * mat_) // XX^T / (n - 1)
                -
                mean_estimator_.mean() * trans(mean_estimator_.mean()); // muXmuXT
        blaze::eigen(mat, vals, vecs);
        blaze::DynamicVector<uint32_t> indices(vals.size());
        std::iota(indices.begin(), indices.end(), 0u);
        std::sort(indices.begin(), indices.end(), [&](auto x, auto y) {return vals[x] > vals[y];});
        auto rrows = std::min(nvs_, mat.rows());
        blaze::DynamicMatrix<FT, SO> ret(rrows, mat.columns());
        blaze::DynamicVector<FT, SO> retvals(rrows);
        for(auto i = 0u; i < rrows; ++i) {
            retvals[i] = vals[indices[i]];
            row(ret, i) = vecs[indices[i]];
        }
        std::swap(ret, vecs);
        std::swap(retvals, vals);
    }
};

}} // namespace frp::linalg

#endif // #ifnef _GFRP_LINALG_H__
