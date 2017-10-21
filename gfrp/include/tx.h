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


// Multiplies each entry by a Rademacher-distributed random variable.
template<template<typename, bool> typename MatrixKind, typename FloatType, bool StorageType>
void rademacher(MatrixKind<FloatType, StorageType> &mat, rng::RandTwister &rs) {
    static const FloatType vals[]{-1, 1};
    auto rval(rs());
    char c(0);
    if constexpr(StorageType == blaze::rowMajor) {
        for(std::size_t i(0), e(mat.rows()); i < e; ++i) {
            auto mrow(row(mat, i));
            for(auto &el: mrow) {
                el *= vals[get_bool(rval, c++)];
                if(c & 64) rval = rs(), c = 0;
            }
        }
    } else {
        for(std::size_t i(0), e(mat.rows()); i < e; ++i) {
            auto mcol(column(mat, i));
            for(auto &el: mcol) {
                el *= vals[get_bool(rval, c++)];
                if(c & 64) rval = rs(), c = 0;
            }
        }
    }
}

template<typename VectorKind>
void rademacher(VectorKind &vec, rng::RandTwister &rs) {
    using FloatType = typename VectorKind::ElementType;
    static const FloatType vals[]{-1, 1};
    auto rval(rs());
    char c(0);
    for(auto &el: vec) {
        el *= vals[get_bool(rval, c++)];
        if(c & 64) rval = rs(), c = 0;
    }
}

template<template<typename, bool> typename MatrixKind, typename FloatType, bool StorageType>
void rademacher(MatrixKind<FloatType, StorageType> &mat) {
    rademacher(mat, rng::random_twist);
}


}} //namespace gfrp::tx

#endif  // GFRP_H
