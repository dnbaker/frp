#pragma once
#ifndef NN_MAT_H__
#define NN_MAT_H__
#include <cstdint>
#include "blaze/Math.h"
#include "frp/linalg.h"

namespace frp {
namespace graph {
using std::uint32_t;
using u32 = std::uint32_t;

#if 0
template<typename FT, typename IT>
struct inddist: public std::pair<FT, IT> {
    static_assert(std::is_floating_point<FT>::value, "FT must be floating point");
    static_assert(std::is_integral<IT>::value, "IT must be integral");
    template<typename...Args> inddist(Args &&...args) std::pair<FT, IT>(std::forward<Args>(args)...) {}
};
template<typename FT, typename IT>
struct idpq: std::priority_queue<inddist<FT, IT>> {
    auto &getc() {return this->c;}
    const auto &getc() const {return this->c;}
};
#endif

template<typename MT, bool SO, typename Functor>
auto dm2laplacian(const blaze::Matrix<MT, SO> &distance_matrix, const Functor &func=Functor()) {
    // Requires that distance_matrix is symmetric. It's okay if it isn't exactly due to floating point errors.
    using FT = typename blaze::Matrix<MT, SO>::ElementType_t;
    const size_t nr = rows(distance_matrix);
    assert(rows(distance_matrix) == columns(distance_matrix));
    blaze::SymmetricMatrix<blaze::DynamicMatrix<FT>> ret(nr);
    for(size_t i = 0, e = rows(distance_matrix); i < e; ++i) {
        auto r = row(distance_matrix, i);
        auto rr = row(ret);
        func(rr, r, i);
        rr[i] = 0.;
        rr[i] = -sum(rr);
    }
    return ret;
}

template<typename MT>
auto laplacian_embedding(const MT &laplacian, unsigned k, bool normalize=false) {
    using FT = typename MT::ElementType_t;
    blaze::DynamicVector<FT> eigv;
    blaze::DynamicMatrix<FT> eigenvectors;
    blaze::eigen(laplacian, eigv, eigenvectors);
    blaze::DynamicMatrix<FT> firstk(eigenvectors.rows(), k);
    // LAPACK returns eigenvectors in reverse order because... it does
    for(size_t i = 0; i < k; ++i)
        column(firstk, i) = column(eigenvectors, k - i - 1);
    if(normalize) {
        OMP_PRAGMA("omp parallel for")
        for(size_t i = 0; i < firstk.rows(); ++i)
            row(firstk, i) /= l2Norm(row(firstk, i));
    }
    return firstk;
}

// The smallest non-null eigenvectors of the unnormalized Laplacian approximate the RatioCut minimization criterion,and
// The smallest non-null eigenvectors of therandom-walkLaplacianapproximate the NCut criterion.

} // graph
} // frp

#endif /* NN_MAT_H__ */
