#ifndef _GFRP_H__
#define _GFRP_H__
#include "blaze/Math.h"
#include "gfrp/include/rand.h"

// Performs transformations on matrices and vectors.

namespace gfrp { namespace tx {

#ifndef FLOAT_TYPE
#  define FLOAT_TYPE double
#endif

#define get_at(val, ind) (val & (1uL << ind))
#define get_bool(val, ind) (!!get_at(val, ind))

#define process_el(el, rval, c) do {el = vals[get_bool(rval, c++)];if(c & 64) rval = rs(), c = 0;} while(0)

// Multiplies each entry by a Rademacher-distributed random variable.
template<template<typename, bool> typename MatrixKind, typename FloatType, bool StorageType>
void rademacher(MatrixKind<FloatType, StorageType> &mat, rng::RandTwister &rs) {
    static const FloatType vals[]{-1, 1};
    auto rval(rs());
    char c(0);
    if constexpr(blaze::IsMatrix<MatrixKind<FloatType, StorageType>>::value) {
        if constexpr(StorageType == blaze::rowMajor) {
            for(std::size_t i(0), e(mat.rows()); i < e; ++i) {
                auto mrow(row(mat, i));
                for(auto &el: mrow) {
                    process_el(el, rval, c);
                }
            }
        } else {
            for(std::size_t i(0), e(mat.rows()); i < e; ++i) {
                auto mcol(column(mat, i));
                for(auto &el: mcol) {
                    process_el(el, rval, c);
                }
            }
        }
    } else {
        for(auto &el: mat) process_el(el, rval, c);
    }
}

template<typename MatrixKind>
MatrixKind make_rademacher(std::size_t n, rng::RandTwister &rs) {
    MatrixKind ret(n, n);
    rademacher(ret, rs);
    return ret;
}

template<typename VectorKind>
VectorKind make_rademacher(std::size_t n, rng::RandTwister &rs) {
    VectorKind ret(n);
    rademacher(ret, rs);
    return ret;
}


template<template<typename, bool> typename MatrixKind, typename FloatType, bool StorageType>
void rademacher(MatrixKind<FloatType, StorageType> &mat) {
    rademacher(mat, rng::random_twist);
}

#undef process_el
#undef get_at
#undef get_bool

}} //namespace gfrp::tx

#endif  // GFRP_H
