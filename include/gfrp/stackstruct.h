#ifndef _GFRP_STACKSTRUCT_H__
#define _GFRP_STACKSTRUCT_H__
#include "gfrp/util.h"
#include "FFHT/fht.h"

namespace gfrp {

template<typename FloatType>
struct HadamardBlock {
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
        throw std::runtime_error("NotImplemented.");
    }

    template<typename OutVector>
    void apply(OutVector &out) {
        if(out.size() & (out.size() - 1)) {
            throw std::runtime_error("Error: out.size() should be power of 2. (This can be adjusted in the future.");
        }
        if constexpr(blaze::IsSparseVector<OutVector>::value || blaze::IsSparseMatrix<OutVector>::value) {
            throw std::runtime_error("Fast Hadamard transform not implemented for sparse vectors yet.");
        }
        fht(&out[0], log2_64(out.size()));
    }
};


} // namespace gfrp

#endif // #ifndef _GFRP_STACKSTRUCT_H__
