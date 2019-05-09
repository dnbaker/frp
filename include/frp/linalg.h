#ifndef _GFRP_LINALG_H__
#define _GFRP_LINALG_H__
#define _USE_MATH_DEFINES
#include <queue>
#include "frp/util.h"
#include "vec/vec.h"
#include "x86intrin.h"

#ifndef VECTOR_WIDTH
#ifdef __AVX2__
#define VECTOR_WIDTH (32ul)
#elif __SSE2__
#define VECTOR_WIDTH (16ul)
#else
#define VECTOR_WIDTH (8ul)
#endif
#endif // VECTOR_WIDTH

namespace frp { namespace linalg {

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


template<typename MatrixType, typename ValueType,
         typename=enable_if_t<is_arithmetic<ValueType>::value>>
MatrixType &operator+=(MatrixType &in, ValueType val) {
    if constexpr(blaze::IsMatrix<MatrixType>::value) {
        if constexpr(blaze::IsSparseMatrix<MatrixType>::value) {
            for(size_t i(0); i < in.rows(); ++i) {
                for(auto it(in.begin(i)), eit(in.end(i)); it != eit; ++it) {
                    it->value() += val;
                }
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
        if constexpr(blaze::IsSparseVector<MatrixType>::value) {
            for(auto it(in.begin()), eit(in.end()); it != eit; ++it) it->value() += val;
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
        blaze::SymmetricMatrix<T> ret = trans(cpy) * cpy;
        ret /= mat.rows() - bias;
        return ret;
    } else {
        auto cmean = blaze::sum<blaze::rowwise>(mat) /  mat.columns();
        T cpy = mat;
        for(size_t i = 0; i < mat.columns(); ++i) {
            column(cpy, i) -= cmean;
        }
        blaze::SymmetricMatrix<T> ret = cpy * trans(cpy);
        ret /= mat.columns() - bias;
        return ret;
    }
}


template<typename...Args>
auto cov(Args &&...args) {return naive_cov(std::forward<Args>(args)...);}
// Until I've benchmarked, default to the naive implementation which I've verified is correct.

template<typename T>
auto pca(const T &mat, bool by_feature=true, bool bias=true, int ncomp=-1) {
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
        std::sort(&vec[0], &vec[ncomp]);
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

}} // namespace frp::linalg

#endif // #ifnef _GFRP_LINALG_H__
