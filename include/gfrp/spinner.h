#ifndef _GFRP_SPINNER_H__
#define _GFRP_SPINNER_H__
#include "gfrp/compactrad.h"
#include "gfrp/linalg.h"
#include "FFHT/fht.h"

namespace gfrp {

template<typename FloatType, typename StructMat, typename DiagMat, typename=std::enable_if_t<std::is_floating_point<FloatType>::value>>
struct SDBlock {
    StructMat s_;
    DiagMat   d_;
    //size_t k_, n_, m_;
    // n_: dimension of data we're projecting up or down.
    // k_: ultimate size of the space to which we're projecting.
    // m_: The size of 
    template<typename... DiagArgs>
    SDBlock(StructMat &&s, DiagArgs &&... args): s_(std::move(s)), d_(std::forward<DiagArgs>(args)...)
    {
        assert(s_.size() == d_.size());
    }
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
    }
};

template<typename FloatType>
struct HadamardBlock {
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
    }
};

template<>
struct HadamardBlock<float> {
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        throw std::runtime_error("Not Implemented");
    }
};

template<>
struct HadamardBlock<double> {
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        throw std::runtime_error("Not Implemented");
    }
};

template<typename SizeType=size_t, typename Container=blaze::DynamicVector<SizeType>>
class Shuffler {
    Container shuffler_;
public:
    Shuffler(size_t n): shuffler_(make_shuffled<Container>(n)) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        throw std::runtime_error("Not Implemented");
    }
};

template<typename... Blocks>
class SpinBlockTransformer {
    // This variadic template allows me to mix various kinds of blocks, so long as they perform 
    size_t k_, n_, m_;
    std::tuple<Blocks...> blocks_;
public:
    SpinBlockTransformer(size_t k, size_t n, size_t m, Blocks &&... blocks):
        k_(k), n_(n), m_(m), blocks_(std::forward(blocks)...) {}

// Template magic for unrolling from the back.
    template<typename InVector, typename OutVector, size_t Index>
    struct ApplicationStruct {
        void operator()(const InVector &in, OutVector &out) const {
            std::get<Index - 1>(blocks_).apply(in, out);
            ApplicationStruct<InVector, OutVector, Index - 1> as;
            as(in, out);
        }
    };
    template<typename InVector, typename OutVector>
    struct ApplicationStruct<InVector, OutVector, 1> {
        void operator()(const InVector &in, OutVector &out) const {
            std::get<0>(blocks_).apply(in, out);
        }
    };
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector out) {
        ApplicationStruct<InVector, OutVector, std::tuple_size<decltype(blocks_)>::value> as;
        as(in, out);
    }
};

} // namespace gfrp

#endif // #ifndef _GFRP_SPINNER_H__
