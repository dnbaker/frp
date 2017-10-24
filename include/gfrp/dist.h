#ifndef _GFRP_DIST_H__
#define _GFRP__DIST_H__
#include <random>
#include "gfrp/rand.h"
#include "gfrp/linalg.h"

namespace gfrp { namespace dist {

// Fill a matrix with distributions. Contains utilities for filling
// vectors with C++ std distributions as well as Rademacher.

template<typename Container, template<typename> typename Distribution, typename... DistArgs>
void sample_fill(Container &con, DistArgs &&... args) {
    using FloatType = std::decay_t<decltype(con[0])>;
    std::mt19937_64 mt;
    std::normal_distribution<FloatType> dist(std::forward<DistArgs>(args)...);
    for(auto &el: con) el = dist(mt);
}
void random_fill(uint64_t *data, uint64_t len) {
    std::mt19937_64 mt;
    for(uint64_t i(!len); i < len; ++i) data[i] = mt();
}

#define DEFINE_DIST_FILL(type, name) \
    template<typename Container, typename...Args> \
    void name##_fill(Container &con, Args &&... args) { \
        sample_fill<Container, type, Args...>(con, std::forward<Args>(args)...); \
    }

DEFINE_DIST_FILL(std::normal_distribution, gaussian)
DEFINE_DIST_FILL(std::cauchy_distribution, cauchy)
DEFINE_DIST_FILL(std::chi_squared_distribution, chisq)
DEFINE_DIST_FILL(std::lognormal_distribution, lognormal)
DEFINE_DIST_FILL(std::extreme_value_distribution, extreme_value)
DEFINE_DIST_FILL(std::weibull_distribution, weibull)
    
#define get_bool(val, ind) (!!(rval & (1uL << (ind))))
#define process_el(el, rval, c) do {\
        el = vals[get_bool(rval, c++)];\
        if(c & 64) rval = rs(), c = 0; \
    } while(0)

// Fills each entry by a Rademacher-distributed random variable.
template<template<typename, bool> typename MatrixKind, typename FloatType, bool StorageType,
         typename=std::enable_if_t<std::is_arithmetic<FloatType>::value>>
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
#undef process_el
#undef get_bool

template<template<typename, bool> typename MatrixKind, typename FloatType, bool StorageType>
void rademacher(MatrixKind<FloatType, StorageType> &mat) {
    rademacher(mat, rng::tsrandom_twist);
}

template<typename MatrixKind>
MatrixKind make_rademacher(std::size_t n, rng::RandTwister &rs) {
    MatrixKind ret(n, n);
    rademacher(ret, rs);
    return ret;
}

template<typename MatrixKind>
MatrixKind make_rademacher(std::size_t n) {
    return make_rademacher<MatrixKind>(n, rng::tsrandom_twist);
}

}}

#endif // #ifndef _GFRP_DIST_H__
