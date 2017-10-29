#ifndef _GFRP_LINALG_H__
#define _GFRP_LINALG_H__
#define _USE_MATH_DEFINES
#include "gfrp/util.h"

namespace gfrp { namespace linalg {

template<typename MatrixKind1, typename MatrixKind2>
void gram_schmidt(const MatrixKind1 &a, MatrixKind2 &b, int flags) {
    b = a;
    gram_schmidt(b, flags);
}

enum GramSchmitFlags {
    FLIP = 1,
    ORTHONORMALIZE = 2,
    RESCALE_TO_GAUSSIAN = 4
};


template<typename MatrixKind>
void gram_schmidt(MatrixKind &b, int flags=(FLIP & ORTHONORMALIZE)) {
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
            inv_unorms[i] = tmp ? tmp: std::numeric_limits<decltype(tmp)>::max();
#if !NDEBUG
            if(tmp ==std::numeric_limits<decltype(tmp)>::max())
                std::fprintf(stderr, "Warning: norm of column %zu (0-based) is 0.0\n", i);
#endif
        }
        if(flags & ORTHONORMALIZE)
            for(size_t i(0), ncolumns(b.columns()); i < ncolumns; ++i)
                column(b, i) *= inv_unorms[i];
        if(flags & RESCALE_TO_GAUSSIAN) {
            #pragma omp parallel
            for(size_t i = 0, ncolumns = b.columns(); i < ncolumns; ++i) {
                auto mcolumn(column(b, i));
                auto meanvarpair(meanvar(mcolumn)); // mean.first = mean, mean.second = var
                const auto invsqrt(1./std::sqrt(meanvarpair.second));
                for(auto &el: mcolumn) el = (el - meanvarpair.first) * invsqrt;
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
            inv_unorms[i] = tmp ? tmp: std::numeric_limits<decltype(tmp)>::max();
        }
        if(flags & ORTHONORMALIZE)
            for(size_t i(0), nrows(b.rows()); i < nrows; ++i)
                row(b, i) *= inv_unorms[i];
        if(flags & RESCALE_TO_GAUSSIAN) {
            #pragma omp parallel
            for(size_t i = 0, nrows = b.rows(); i < nrows; ++i) {
                auto mrow(row(b, i));
                auto meanvarpair(meanvar(mrow)); // mean.first = mean, mean.second = var
                for(auto &el: mrow) el -= meanvarpair.first;
                mrow *= 1./std::sqrt(meanvarpair.second);
            }
        }
    }
}

template<typename MatrixKind>
decltype(auto) frobnorm(const MatrixKind &mat) {
    using FloatType = typename MatrixKind::ElementType;
    FloatType ret(0.);
    for(size_t i(0); i < mat.rows(); ++i)
        ret += dot(row(mat, i), row(mat, i));
    return ret;
}

template<typename FloatType>
constexpr inline auto ndball_surface_area(std::size_t nd, FloatType r) {
    // http://scipp.ucsc.edu/~haber/ph116A/volume_11.pdf
    // In LaTeX notation: $\frac{2\Pi^{\frac{n}{2}}R^{n-1}}{\Gamma(frac{n}{2})}$
    nd >>= 1;
    return 2. * std::pow(static_cast<FloatType>(M_PI), nd) / tgamma(nd) * std::pow(r, nd - 1);
}


}} // namespace gfrp::linalg

#endif // #ifnef _GFRP_LINALG_H__
