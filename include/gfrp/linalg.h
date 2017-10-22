#ifndef _GFRP_LINALG_H__
#define _GFRP_LINALG_H__
#include "blaze/Math.h"
#include "kspp/ks.h"

namespace gfrp { namespace linalg {

template<typename MatrixKind1, typename MatrixKind2>
void gram_schmidt(const MatrixKind1 &a, MatrixKind2 &b, bool orthonormalize=false) {
    if(a.rows() != b.rows() || a.columns() != b.columns())
        throw std::runtime_error(
            ks::sprintf("Expected a and b to have the same dimensions. (a: %zu, %zu) (b: %zu, %zu).\n",
                        a.rows(), a.columns(), b.rows(), b.columns()).data());
    gram_schmidt(b, orthonormalize);
}

template<typename MatrixKind>
void gram_schmidt(MatrixKind &b, bool orthonormalize=false) {
    blaze::DynamicVector<typename MatrixKind::ElementType> inv_unorms(b.rows());
    auto ivit(std::begin(inv_unorms));
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


}} // namespace gfrp::linalg

#endif // #ifnef _GFRP_LINALG_H__
