#ifndef _GFRP_SPINNER_H__
#define _GFRP_SPINNER_H__
#include "gfrp/compactrad.h"
#include "gfrp/linalg.h"
#include "FFHT/fht.h"

namespace gfrp {
/*
 *
 * TODO: Think about how this should be done. Leave this alone for now.
 * Does 'apply' need 1 or two matrices to run on?
 *
 *
 *
 */

template<typename FloatType, typename StructMat, typename DiagMat, typename=std::enable_if_t<std::is_floating_point<FloatType>::value>>
struct SDBlock {
    StructMat s_;
    DiagMat   d_;
    //size_t k_, n_, m_;
    // n_: dimension of data we're projecting up or down.
    // k_: ultimate size of the space to which we're projecting.
    // m_: The size of 
    SDBlock(StructMat &&s, DiagMat &&d): s_(std::forward<StructMat>(s)), d_(std::forward<DiagMat>(d))
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

template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector>
struct ScalingBlock {
    using VectorType = VectorKind<FloatType, VectorOrientation>;
    VectorType vec_;
    template<typename...Args>
    ScalingBlock(Args &&...args): vec_(std::forward<Args>(args)...) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        throw std::runtime_error("Not Implemented");
    }
    template<typename Vector>
    void apply(Vector &out) const {
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
        //The naive approach is double memory.
    }
};

template<typename T> class TD;

template<typename SizeType=size_t, typename RNG=aes::AesCtr>
class OnlineShuffler {
    //Provides reproducible shuffling by re-generating a random sequence for shuffling an array.
    //This 
    using ResultType = typename RNG::result_type;
    //TD<ResultType> thing;
    const uint64_t seed_;
    RNG             rng_;
public:
    explicit OnlineShuffler(ResultType seed): seed_{seed}, rng_(seed) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        //The naive approach is double memory.
    }
    template<typename Vector>
    void apply(Vector &vec) const {
        rng_.seed(seed_);
        std::shuffle(std::begin(vec), std::end(vec), rng_);
        //The naive approach is double memory.
    }
};

template<typename... Blocks>
class SpinBlockTransformer {
    // This variadic template allows me to mix various kinds of blocks, so long as they perform
    size_t k_, n_, m_;
    std::tuple<Blocks...> blocks_;
public:
    SpinBlockTransformer(size_t k, size_t n, size_t m, Blocks &&... blocks):
        k_(k), n_(n), m_(m), blocks_(blocks...) {}

    SpinBlockTransformer(size_t k, size_t n, size_t m, std::tuple<Blocks...> &&blocks):
        k_(k), n_(n), m_(m), blocks_(std::move(blocks)) {}

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
