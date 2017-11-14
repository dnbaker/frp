#ifndef _GFRP_SPINNER_H__
#define _GFRP_SPINNER_H__
#include "gfrp/compact.h"
#include "gfrp/linalg.h"
#include "gfrp/stackstruct.h"
#include "gfrp/sample.h"
#include "FFHT/fht.h"
#include <array>
#include <functional>

namespace gfrp {
/*
 *
 * TODO:
 *
 * 0. Add full-size JL transform and OJL transform matrices. This is straight-forward for the JLs. But maybe not for random projection.
 *    What does it mean to rescale them after Gram-Schmidt? To unit variance and zero mean? Or does it mean to multiply some other way?
 *
 * 1. Think about how this should be done.
 * Can any of this be precomputed?
 * Could inserting the Rademacher matrix access insertion into a F*T application save time?
 * What normalization constant to we need in front? Should that be decomposed into the Gaussian scaler?
 *   1. Note that mixtures of these are arbitrary function approximators -- how to ...?
 *
 * 2. Note that these can be applied to fast LSH algorithms.
 * 3. Don't lose sight of the fact that these can be inserted into neural networks to replace fully-connected layers.
 *   1. How do we update these parameters? What are the parameters?
 *
 *
 *
 */

template<typename StructMat, typename DiagMat>
struct SDBlock {
    StructMat s_;
    DiagMat   d_;
    //size_t k_, n_, m_;
    // n_: dimension of data we're projecting up or down.
    // k_: ultimate size of the space to which we're projecting.
    // m_: The size of each sub-block
    SDBlock(StructMat &&s, DiagMat &&d): s_(forward<StructMat>(s)), d_(forward<DiagMat>(d))
    {
    }
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) {
        if(out.size() == in.size()) {
            out = in;
            apply(out);
        }
    }
    template<typename OutVector>
    void apply(OutVector &out) {
        d_.apply(out); // Element-wise multiplication.
        s_.apply(out); // Structured-matrix multiplication.
    }
};


template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector>
struct ScalingBlock {
public:
    using VectorType = VectorKind<FloatType, VectorOrientation>;
private:
    VectorType vec_;
public:
    template<typename...Args>
    ScalingBlock(Args &&...args): vec_(forward<Args>(args)...) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        if(out.size() != in.siz()) throw std::runtime_error("NotImplementedError");
        out = in;
        apply(out);
    }
    template<typename Vector>
    void apply(Vector &out) const {
        out *= vec_;
    }
};

template<typename FloatType, typename=enable_if_t<is_arithmetic<FloatType>::value>>
struct AdditionBlock {
    const FloatType v_;
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        throw runtime_error("Not Implemented");
    }
    template<typename Vector>
    void apply(Vector &out) const {
        out += v_;
    }
    AdditionBlock(FloatType val): v_(val) {}
};

template<typename FloatType, typename=enable_if_t<is_arithmetic<FloatType>::value>>
struct ProductBlock {
    const FloatType v_;
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        if(in.size() == out.size()) {
            out = in;
            apply<OutVector>(out);
        } else {
            throw runtime_error("Not Implemented");
        }
    }
    template<typename Vector>
    void apply(Vector &out) const {
        out *= v_;
    }
    ProductBlock(FloatType val): v_(val) {}
};

template<typename FloatType, typename=enable_if_t<is_arithmetic<FloatType>::value>>
class FastFoodGaussianProductBlock {
    const FloatType v_;
public:
    FastFoodGaussianProductBlock(FloatType sigma): v_(1./sigma) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        if(in.size() == out.size()) {
            out = in;
            apply<OutVector>(out);
        } else {
            throw runtime_error("Not Implemented");
        }
    }
    template<typename Vector>
    void apply(Vector &out) const {
        const auto tmp(1./std::sqrt(out.size()) * v_);
        out *= tmp;
    }
};

template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector, typename RNG=aes::AesCtr<uint64_t>>
class GaussianScalingBlock: public ScalingBlock<FloatType, VectorOrientation, VectorKind> {
    // This might need a rescaling.
    using VectorType = VectorKind<FloatType, VectorOrientation>;
    VectorType vec_;
public:
    template<typename...Args>
    GaussianScalingBlock(uint64_t seed=0, FloatType mean=0., FloatType var=1., Args &&...args): vec_(forward<Args>(args)...) {
        gaussian_fill(vec_, seed, mean, var);
    }
};

template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector, typename RNG=aes::AesCtr<uint64_t>>
class UnitGaussianScalingBlock: public ScalingBlock<FloatType, VectorOrientation, VectorKind> {
    // This might need a rescaling.
    using VectorType = VectorKind<FloatType, VectorOrientation>;
    VectorType vec_;
public:
    template<typename...Args>
    UnitGaussianScalingBlock(uint64_t seed=0, Args &&...args): vec_(forward<Args>(args)...) {
        assert(vec_.size());
        unit_gaussian_fill(vec_, seed);
    }
};

class HadamardRademacherSDBlock: public SDBlock<HadamardBlock, CompactRademacher> {
public:
    using RademType = CompactRademacher;
    using HadamType = HadamardBlock;
    using SDType    = SDBlock<HadamType, RademType>;
    using size_type = typename RademType::size_type;
    HadamardRademacherSDBlock(size_type n=0, size_type seed=0):
        SDType(HadamType(), RademType(n, seed)) {}
    void resize(size_type newsize) {
        if(newsize & (newsize - 1))
            std::fprintf(stderr, "[W:%s] Resizing HR block to new size of %zu (from %zu, rounded up %zu)\n",
                         __PRETTY_FUNCTION__, (size_t)roundup(newsize), SDType::d_.size(), newsize);
        newsize = roundup(newsize);
        SDType::s_.resize(newsize);
        SDType::d_.resize(newsize);
    }
    void seed(size_type seed) {
        SDType::s_.seed(seed);
        SDType::d_.seed(seed);
    }
    template<typename VecType>
    void apply(VecType &in) const {
        SDType::d_.apply(in);
        SDType::s_.apply(in);
    }
    template<typename FloatType>
    void apply(FloatType *in) const {
        const size_t l2s(log2_64(SDType::d_.size()));
        SDType::d_.apply(in);
        SDType::s_.apply(in,  l2s);
    }
};


template<typename SizeType=uint32_t, typename RNG=aes::AesCtr<SizeType>>
class OnlineShuffler {
    //Provides reproducible shuffling by re-generating a random sequence for shuffling an array.
    //This
    using ResultType = typename RNG::result_type;
    const ResultType seed_;
    RNG               rng_;
public:
    explicit OnlineShuffler(ResultType seed=0): seed_{seed}, rng_(seed) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        fprintf(stderr, "[W:%s] OnlineShuffler can only shuffle from arrays of different sizes by sampling.\n");
        const auto isz(in.size());
        if(isz == out.size()) {
            out = in;
            apply<OutVector>(out);
        } else if(isz > out.size()) {
            std::fprintf(stderr, "Warning: This means we're just subsampling\n");
            unordered_set<uint64_t> indices;
            indices.reserve(out.size());
            while(indices.size() < out.size()) indices.insert(fastrange64(rng_(), isz));
            auto it(out.begin());
            for(const auto index: indices) *it++ = in[index]; // Could consider a sorted map for quicker iteration/cache coherence.
        } else {
            std::fprintf(stderr, "Warning: This means we're subsampling with replacement and not caring because sizes are mismatched.\n");
            for(auto it(out.begin()), eit(out.end()); it != eit;)
                *it++ = in[fastrange64(rng_(), isz)];
        }
        //The naive approach is double memory.
    }
    template<typename Vector>
    void apply(Vector &vec) const {
        rng_.seed(seed_);
        for(ResultType i(vec.size() - 1); i > 1; --i) {
            std::swap(vec[i - 1], fastrange(rng_(), i));
        }
    }
};


template<typename... Blocks>
class SpinBlockTransformer {
    // This variadic template allows me to mix various kinds of blocks, so long as they perform operations
    size_t k_, n_, m_;
    std::tuple<Blocks...> blocks_;
    static constexpr size_t NBLOCKS = std::tuple_size<decltype(blocks_)>::value;
public:
    SpinBlockTransformer(size_t k, size_t n, size_t m, Blocks &&... blocks):
        k_(k), n_(n), m_(m), blocks_(blocks...) {}

    SpinBlockTransformer(size_t k, size_t n, size_t m, std::tuple<Blocks...> &&blocks):
        k_(k), n_(n), m_(m), blocks_(std::move(blocks)) {}

// Template magic for unrolling from the back.
    template<typename OutVector, size_t Index>
    struct ApplicationStruct {
        void operator()(OutVector &out) const {
            std::get<Index - 1>(blocks_).apply(out);
            ApplicationStruct<OutVector, Index - 1>()(out);
        }
    };
    template<typename OutVector>
    struct ApplicationStruct<OutVector, 1> {
        void operator()(OutVector &out) const {
            std::get<0>(blocks_).apply(out);
        }
    };
    // Use it: last block is applied from in to out (as it needs to consume input).
    // All prior blocks, in reverse order, are applied in-place on out.
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector out) {
        std::get<NBLOCKS - 1>().apply(in, out);
        ApplicationStruct<OutVector, NBLOCKS - 1> as;
        as(in, out);
    }
};

} // namespace gfrp

#endif // #ifndef _GFRP_SPINNER_H__
