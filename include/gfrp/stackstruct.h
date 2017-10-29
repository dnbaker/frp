#ifndef _GFRP_STACKSTRUCT_H__
#define _GFRP_STACKSTRUCT_H__
#include "gfrp/util.h"
#include "FFHT/fht.h"

namespace gfrp {

template<template<typename, bool> typename VecType, typename FloatType, bool VectorOrientation>
void fht(VecType<FloatType, VectorOrientation> &vec) {
    throw std::runtime_error("NotImplemented.");
}

template<typename Container>
struct is_dense_single {
    static constexpr bool value = blaze::IsDenseVector<Container>::value || blaze::IsDenseMatrix<Container>::value;
};

#if 0
template<typename... Containers>
struct is_dense {
    static constexpr bool value = is_dense_single<Containers>::value && ...;
};
#endif
template<typename C1, typename C2>
struct is_dense {
    static constexpr bool value = is_dense_single<C1>::value && is_dense_single<C2>::value;
};


template<typename VecType1, typename VecType2>
void fht(const VecType1 &in, VecType2 &out) {
    static_assert(std::is_same<typename VecType1::ElementType, typename VecType2::ElementType>::value, "Input vectors must have the same type.");
    if constexpr(is_dense<VecType1, VecType2>::value) {
        fast_copy(&out[0], &in[0], sizeof(in[0]) * out.size());
        fht(&out[0], log2_64(out.size()));
        return;
    } else {
        if constexpr(blaze::TransposeFlag<VecType1>::value == blaze::TransposeFlag<VecType2>::value) {
            if(out.size() == in.size()) {
                out = in;
                fht(out);
            } else throw std::runtime_error("NotImplemented.");
        } else {
            if(out.size() == in.size()) {
                out = transpose(in);
                fht(out);
            } else throw std::runtime_error("NotImplemented.");
        }
    }
}

template<typename FloatType>
struct HadamardBlock {
    size_t n_;
    size_t pow2up_;
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
        throw std::runtime_error("NotImplemented.");
    }

    template<typename OutVector>
    void apply(OutVector &out) {
        if constexpr(blaze::IsSparseVector<OutVector>::value || blaze::IsSparseMatrix<OutVector>::value) {
            throw std::runtime_error("Fast Hadamard transform not implemented for sparse vectors yet.");
        }
        if(out.size() & (out.size() - 1) == 0) {
            fht(&out[0], log2_64(out.size()));
            return;
        }
        throw std::runtime_error("NotImplemented.");
    }
    HadamardBlock(size_t n): n_(n), pow2up_(roundup64(n_)) {}
};


} // namespace gfrp

#endif // #ifndef _GFRP_STACKSTRUCT_H__
