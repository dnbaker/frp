#ifndef _GFRP_SPINNER_H__
#define _GFRP_SPINNER_H__
#include "fastrange.h"
#include "frp/compact.h"
#include "frp/linalg.h"
#include "frp/stackstruct.h"
#include "frp/sample.h"
#include "frp/util.h"
#include "frp/tx.h"
#include "FFHT/fht.h"
#include "boost/math/special_functions/detail/igamma_inverse.hpp"
#include <array>
#include <functional>

namespace frp {
template<typename T>
inline T fastrange(T, T) {}
template<> inline int fastrange<int>(int w, int p) {return fastrangeint(w, p);}
template<> inline uint32_t fastrange<uint32_t>(uint32_t w, uint32_t p) {return fastrange32(w, p);}
template<> inline uint64_t fastrange<uint64_t>(uint64_t w, uint64_t p) {return fastrange64(w, p);}
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
protected:
    using VectorType = VectorKind<FloatType, VectorOrientation>;
    VectorType vec_;
public:
    template<typename...Args>
    ScalingBlock(Args &&...args): vec_(forward<Args>(args)...) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        if(out.size() != in.size()) throw std::runtime_error("NotImplementedError");
        out = in;
        apply(out);
    }
    template<typename Vector>
    void apply(Vector &out) const {
#if VERBOSE
        std::cerr << "Applying scaling block with norm " << vec_norm() << ":\n";
        pv(out); std::cerr << '\n';
#endif
        if constexpr(blaze::TransposeFlag<VectorType>::value != blaze::TransposeFlag<Vector>::value) {
            if(&out[1] - &out[0] != 1) throw std::runtime_error("Can't use vectorized approach. Change your code so you can.");
            vec::vecmul(&out[0], &vec_[0], out.size());
        }
        else out *= vec_;
#if VERBOSE
        std::cerr << "Applied scaling block with norm " << vec_norm() << ":\n";
        pv(out); std::cerr << '\n';
#endif
    }
    FloatType vec_norm() const {return norm(vec_);}
    size_t size() const {return vec_.size();}
    void rescale(FloatType val) {
        vec_ *= val;
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

template<typename FloatType, typename=enable_if_t<is_floating_point<FloatType>::value>>
class FastFoodGaussianProductBlock {
    const FloatType sigma_;
public:
    FastFoodGaussianProductBlock(FloatType sigma): sigma_(sigma) {}
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
        out *= 1. / (sigma_ * std::sqrt(out.size()));
    }
    size_t size() const {return -1;}
};

template<typename FloatType, typename=enable_if_t<is_floating_point<FloatType>::value>>
class SORFProductBlock {
    const FloatType sigma_;
public:
    SORFProductBlock(FloatType sigma): sigma_(sigma) {}
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
        //static_assert(std::is_same<std::decay_t<decltype(*std::begin(out))>, FloatType>::value, "Output vector must have the same type as the block type.");
        out *= (std::sqrt(FloatType(out.size())) / sigma_);
    }
    size_t size() const {return -1;}
};

template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector>
class RandomGaussianScalingBlock: public ScalingBlock<FloatType, VectorOrientation, VectorKind> {
    using VectorType = VectorKind<FloatType, VectorOrientation>;
public:
    template<typename...Args>
    RandomGaussianScalingBlock(uint64_t seed, Args &&...args): ScalingBlock<FloatType, VectorOrientation, VectorKind>(forward<Args>(args)...) {
        //std::fprintf(stderr, "[%s] Size of scaling block: %zu\n", __PRETTY_FUNCTION__, vec_.size());
        unit_gaussian_fill(ScalingBlock<FloatType, VectorOrientation, VectorKind>::vec_, seed);
    }
};
template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector, bool high_prec=true>
class RandomGammaIncInvScalingBlock: public ScalingBlock<FloatType, VectorOrientation, VectorKind> {
    using VectorType = VectorKind<FloatType, VectorOrientation>;
public:
    template<typename...Args>
    RandomGammaIncInvScalingBlock(uint64_t seed, Args &&...args): ScalingBlock<FloatType, VectorOrientation, VectorKind>(forward<Args>(args)...) {
        uniform_fill(ScalingBlock<FloatType, VectorOrientation, VectorKind>::vec_, seed, 0, 1);
        const FloatType val(this->vec_.size());
        using Space = vec::SIMDTypes<FloatType>;
        using PackedType = typename Space::Type;
        PackedType el, *ptr((PackedType *)&this->vec_[0]);
        PackedType two(Space::set1(2.));
        if constexpr(high_prec) {
            for(size_t i(this->vec_.size() / Space::COUNT); --i;) {
                el = Space::load((FloatType *)&ptr[i]);
                for(u32 j(0); j < Space::COUNT; ++j) el[j] = boost::math::gamma_p_inv(val, el[j]);
                el = Space::mul(el, two);
                el = Space::sqrt_u05(el);
                Space::store((FloatType *)&ptr[i], el);
            }
        } else {
            for(size_t i(this->vec_.size() / Space::Count); --i;) {
                --i;
                el = Space::load((FloatType *)&ptr[i - 1]);
                for(u32 j(0); j < Space::COUNT; ++j) el[j] = boost::math::gamma_p_inv(val, el[j]);
                el = Space::mul(el, two);
                el = Space::sqrt_u35(el);
                Space::store((FloatType *)&ptr[i - 1], el);
            }
        }
    }
};

template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector>
class RandomChiScalingBlock: public ScalingBlock<FloatType, VectorOrientation, VectorKind> {
    using VectorType = VectorKind<FloatType, VectorOrientation>;
    using ScalingBlock<FloatType, VectorOrientation, VectorKind>::vec_;
public:
    template<typename...Args>
    RandomChiScalingBlock(uint64_t seed, Args &&...args): ScalingBlock<FloatType, VectorOrientation, VectorKind>(forward<Args>(args)...) {
        using SqrtStruct = typename vec::SIMDTypes<FloatType>::apply_sqrt_u05;
        chisq_fill(vec_, seed);
        vec::block_apply(vec_, SqrtStruct());
    }
};

template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector, typename RNG=aes::AesCtr<uint64_t>>
class GaussianScalingBlock: public ScalingBlock<FloatType, VectorOrientation, VectorKind> {
    // This might need a rescaling.
    using VectorType = VectorKind<FloatType, VectorOrientation>;
    using ScalingBlock<FloatType, VectorOrientation, VectorKind>::vec_;
public:
    template<typename...Args>
    GaussianScalingBlock(uint64_t seed=0, FloatType mean=0., FloatType var=1., Args &&...args):
            ScalingBlock<FloatType, VectorOrientation, VectorKind>(forward<Args>(args)...) {
        gaussian_fill(vec_, seed, mean, var);
    }
};

template<typename FloatType, bool VectorOrientation=blaze::columnVector, template<typename, bool> typename VectorKind=blaze::DynamicVector, typename RNG=aes::AesCtr<uint64_t>>
class UnitGaussianScalingBlock: public ScalingBlock<FloatType, VectorOrientation, VectorKind> {
    // This might need a rescaling.
    using VectorType = VectorKind<FloatType, VectorOrientation>;
public:
    template<typename...Args>
    UnitGaussianScalingBlock(uint64_t seed, Args &&...args): ScalingBlock<FloatType, VectorOrientation, VectorKind>(forward<Args>(args)...) {
        unit_gaussian_fill(this->vec_, seed);
    }
    UnitGaussianScalingBlock(): UnitGaussianScalingBlock(uint64_t(0)) {}
};

template<typename RademacherType>
class HRBlock: public SDBlock<HadamardBlock, RademacherType> {
public:
    using RademType = RademacherType;
    using HadamType = HadamardBlock;
    using SDType    = SDBlock<HadamType, RademType>;
    using size_type = typename RademType::size_type;
    HRBlock(size_type n=0, size_type seed=0):
        SDType(HadamType(), RademType(roundup(n), seed)) {}
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

using HadamardRademacherSDBlock = HRBlock<CompactRademacher>;


template<typename SizeType=uint32_t, typename RNG=aes::AesCtr<SizeType>>
class OnlineShuffler {
    //Provides reproducible shuffling by re-generating a random sequence for shuffling an array.
    //This
    using ResultType = typename RNG::result_type;
    const ResultType seed_;
    mutable RNG       rng_;
    static_assert(std::is_same<ResultType, SizeType>::value, "Must have same type");
public:
    explicit OnlineShuffler(ResultType size=0, ResultType seed=0): seed_{seed ^ size}, rng_(seed) {}
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        fprintf(stderr, "[W:%s] OnlineShuffler can only shuffle from arrays of different sizes by sampling.\n", __PRETTY_FUNCTION__);
        const auto isz(in.size());
        if(isz == out.size()) {
            out = in;
            apply<OutVector>(out);
        } else if(isz > out.size()) {
            std::fprintf(stderr, "Warning: This means we're just subsampling\n");
            unordered_set<uint64_t> indices;
            indices.reserve(out.size());
            while(indices.size() < out.size()) indices.insert(fastrange<SizeType>(rng_(), isz));
            auto it(out.begin());
            for(const auto index: indices) *it++ = in[index]; // Could consider a sorted map for quicker iteration/cache coherence.
        } else {
            std::fprintf(stderr, "Warning: This means we're subsampling with replacement and not caring because sizes are mismatched.\n");
            for(auto it(out.begin()), eit(out.end()); it != eit;)
                *it++ = in[fastrange<SizeType>(rng_(), isz)];
        }
        //The naive approach is double memory.
    }
    template<typename Vector>
    void apply(Vector &vec) const {
        using std::swap;
        rng_.seed(seed_);
        for(auto i(vec.size()); i > 1; --i) {
#if 0
            auto rd(rng_());
            rd = fastrange<SizeType>(rd, (ResultType)i);
            auto tmp(vec[i-1]);
            static_assert(std::is_same<decay_t<decltype(rd)>, typename RNG::result_type>::value, "This really should work.");
            swap(vec[i-1], vec[rd]);
#endif
            swap(vec[i-1], vec[fastrange<SizeType>(rng_(), i)]);
        }
    }
    size_t size() const {return -1;}
};

template<typename SizeType=uint32_t>
class PrecomputedShuffler {
    //Provides reproducible shuffling by re-generating a random sequence for shuffling an array.
    std::vector<SizeType> indices_;
public:
    PrecomputedShuffler(SizeType size, SizeType seed): indices_(size) {
        aes::AesCtr<SizeType> gen(seed);
        for(SizeType i(size); i > 1; --i) indices_[i - 1] = fastrange<SizeType>(gen(), i);
        //std::fprintf(stderr, "PrecomputedShuffler: \n");
        //pv(indices_);
    }
    template<typename Vector>
    void apply(Vector &vec) const {
        for(SizeType i(vec.size() - 1); i > 1; --i) {
#if 0
            using Type = decay_t<decltype(vec[0])>;
            Type tmp(vec[i]);
            vec[indices_[i]
#endif
            std::swap(vec[i], vec[indices_[i]]);
        }
    }
    template<typename Vector1, typename Vector2>
    void apply(const Vector1 &in, Vector2 &out) const {
        out = in;
        apply(out);
    }
};

template<typename SizeType=uint32_t>
class LutShuffler {
    //Provides reproducible shuffling by re-generating a random sequence for shuffling an array.
    std::vector<SizeType> indices_;
public:
    LutShuffler(SizeType size, SizeType seed): indices_(make_shuffled<std::vector<SizeType>>(seed, size)) {}
    template<typename Vector>
    void apply(Vector &vec) const {
        blaze::DynamicVector<decay_t<decltype(vec[0])>, TransposeFlag<Vector>::value> tmp(vec.size());
        tmp = vec;
        apply(tmp, vec);
    }
    template<typename Vector1, typename Vector2>
    void apply(const Vector1 &in, Vector2 &out) const {
        for(SizeType i(0); i < in.size(); ++i) {
            out[i] = in[indices_[i]];
        }
    }
};


template<typename... Blocks>
class SpinBlockTransformer {
    // This variadic template allows me to mix various kinds of blocks, so long as they perform operations
    std::tuple<Blocks...> blocks_;
    static constexpr size_t NBLOCKS = std::tuple_size<decltype(blocks_)>::value;
public:
    SpinBlockTransformer(Blocks &&... blocks):
        blocks_(blocks...)
    {
        // std::fprintf(stderr, "[%s] Made with tuple constructor.\n", __PRETTY_FUNCTION__);
    }

    SpinBlockTransformer(std::tuple<Blocks...> &&blocks):
        blocks_(std::move(blocks))
    {
        // std::fprintf(stderr, "[%s] Made with tuple constructor.\n", __PRETTY_FUNCTION__);
    }

    // Template magic for unrolling from the back.
    template<typename OutVector, size_t Index>
    struct ApplicationStruct {
        const SpinBlockTransformer &ref_;
        ApplicationStruct(const SpinBlockTransformer &ref): ref_(ref) {}
        void operator()(OutVector &out) const {
            std::get<Index - 1>(ref_.blocks_).apply(out);
            ApplicationStruct<OutVector, Index - 1> as(ref_);
            if constexpr(Index - 1 == 4) {
                //TD<decay_t<decltype(std::get<Index - 1>(ref_.blocks_))>> td;
            }
#if !NDEBUG
        try {
#endif
            as(out);
#if !NDEBUG
        } catch (std::invalid_argument &ex) {
            std::fprintf(stderr, "what: %s. Index: %zu. out size: %zu\n", ex.what(), Index, out.size());
            throw(ex);
        }
#endif
            //std::cerr << "after " << Index - 1 << ": " << out << '\n';
        }
    };
    template<typename OutVector>
    struct ApplicationStruct<OutVector, 1> {
        const SpinBlockTransformer &ref_;
        ApplicationStruct(const SpinBlockTransformer &ref): ref_(ref) {}
        void operator()(OutVector &out) const {
            std::get<0>(ref_.blocks_).apply(out);
        }
    };
    // Use it: last block is applied from in to out (as it needs to consume input).
    // All prior blocks, in reverse order, are applied in-place on out.
    template<typename InVector, typename OutVector>
    void apply(const InVector &in, OutVector &out) const {
        //std::fprintf(stderr, "Trying to apply %zu, %zu\n", in.size(), out.size());
        std::get<NBLOCKS - 1>(blocks_).apply(in, out);
        ApplicationStruct<OutVector, NBLOCKS - 1> as(*this);
        as(out);
    }
    template<typename OutVector>
    void apply(OutVector &out) const {
        //std::fprintf(stderr, "[%s:%d] Trying to apply on size %zu\n", __PRETTY_FUNCTION__, __LINE__, out.size());
        std::get<NBLOCKS - 1>(blocks_).apply(out);
        ApplicationStruct<OutVector, NBLOCKS - 1> as(*this);
        //std::fprintf(stderr, "[%s:%d] Applied on size %zu\n", __PRETTY_FUNCTION__, __LINE__, out.size());
        as(out);
    }
    auto &get_tuple() {return blocks_;}
};

} // namespace frp

#endif // #ifndef _GFRP_SPINNER_H__
