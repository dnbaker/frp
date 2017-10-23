#ifndef _GFRP_LINALG_H__
#define _GFRP_LINALG_H__
#define _USE_MATH_DEFINES
#include "gfrp/util.h"

namespace gfrp { namespace linalg {

template<typename MatrixKind1, typename MatrixKind2>
void gram_schmidt(const MatrixKind1 &a, MatrixKind2 &b, bool orthonormalize=false, bool flip=false) {
    b = a;
    gram_schmidt(b, orthonormalize);
}

template<typename MatrixKind>
void gram_schmidt(MatrixKind &b, bool orthonormalize=false, bool flip=false) {
    if(flip) {
        blaze::DynamicVector<typename MatrixKind::ElementType> inv_unorms(b.columns());
        for(size_t i(0), ncolumns(b.columns()); i < ncolumns; ++i) {
            auto icolumn(column(b, i));
            for(size_t j(0); j < i; ++j) {
                auto jcolumn(column(b, j));
                icolumn -= jcolumn * dot(icolumn, jcolumn) * inv_unorms[j];
            }
            auto tmp(1./dot(icolumn, icolumn));
            inv_unorms[i] = (tmp != static_cast<decltype(tmp)>(0.)) ? tmp
                                                                    : 0.;
        }
        if(orthonormalize)
            for(size_t i(0), ncolumns(b.columns()); i < ncolumns; ++i)
                column(b, i) *= inv_unorms[i];
    } else {
        blaze::DynamicVector<typename MatrixKind::ElementType> inv_unorms(b.rows());
        for(size_t i(0), nrows(b.rows()); i < nrows; ++i) {
            auto irow(row(b, i));
            for(size_t j(0); j < i; ++j) {
                auto jrow(row(b, j));
                irow -= jrow * dot(irow, jrow) * inv_unorms[j];
            }
            auto tmp(1./dot(irow, irow));
            inv_unorms[i] = (tmp != static_cast<decltype(tmp)>(0.)) ? tmp
                                                                    : 0.;
        }
        if(orthonormalize)
            for(size_t i(0), nrows(b.rows()); i < nrows; ++i)
                row(b, i) *= inv_unorms[i];
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
