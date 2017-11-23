#ifndef _GFRP_KERNEL_H__
#define _GFRP_KERNEL_H__
#include "gfrp/spinner.h"

namespace gfrp {

namespace ff {

struct GaussianFinalizer {
    template<typename VecType>
    void apply(VecType &in) const {
        if((in.size() & (in.size() - 1))) std::fprintf(stderr, "in.size() [%zu] is not a power of 2.\n", in.size()), exit(1);
#if ADD_RANDOM_NOISE
        boost::random::uniform_real_distribution<decay_t<decltype(in[0])>> dist(0, 2 * M_PI);
        aes::AesCtr<uint64_t> gen;
        for(auto &el: in) el += dist(gen);
#endif
        in = cos(in);
        /* This can be accelerated using SLEEF.
           Sleef_sincosf4_u35, u10, u05 (sse), or 8 for avx2 or 16 for avx512
           The great thing about sleef is that it does not require the use of intel-only materials.
           This could be a nice addition to Blaze downstream.
        */
    }
};


template<typename FloatType>
class FastFoodKernelBlock {
    size_t final_output_size_; // This is twice the size passed to the Hadamard transforms
    using RandomScalingBlock = RandomChiScalingBlock<FloatType>;
    using SizeType = uint32_t;
    using Shuffler = LutShuffler<SizeType>;
    using SpinTransformer =
        SpinBlockTransformer<FastFoodGaussianProductBlock<FloatType>,
                             RandomScalingBlock, HadamardBlock,
                             UnitGaussianScalingBlock<FloatType>, Shuffler, HadamardBlock,
                             CompactRademacher>;
    SpinTransformer tx_;

public:
    using float_type = FloatType;
    using GaussianMatrixType = UnitGaussianScalingBlock<FloatType>;
    FastFoodKernelBlock(size_t size, FloatType sigma=1., uint64_t seed=-1, bool renorm=true):
        final_output_size_(size),
        tx_(
            std::make_tuple(FastFoodGaussianProductBlock<FloatType>(sigma),
                   RandomScalingBlock(seed + seed * seed - size * size, size),
                   HadamardBlock(size, renorm),
                   GaussianMatrixType(seed * seed, size),
                   Shuffler(size, seed),
                   HadamardBlock(size, renorm),
                   CompactRademacher(size, (seed ^ (size * size)) + seed)))
    {
        if(final_output_size_ & (final_output_size_ - 1))
            throw std::runtime_error((std::string(__PRETTY_FUNCTION__) + "'s size should be a power of two.").data());
        std::get<RandomScalingBlock>(tx_.get_tuple()).rescale(1./std::sqrt(std::get<GaussianMatrixType>(tx_.get_tuple()).vec_norm()));
    }
    size_t transform_size() const {return final_output_size_;}
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        if(out.size() != final_output_size_) {
            fprintf(stderr, "Warning: Output size was wrong (%zu, not %zu). Resizing\n", out.size(), final_output_size_);
        }
        if(roundup(in.size()) != transform_size()) throw std::runtime_error("ZOMG");
        blaze::reset(out);

        subvector(out, 0, in.size()) = in;
        //auto half_vector(subvector(out, 0, transform_size()));
        //std::fprintf(stderr, "half vector is size %zu out of out size %zu\n", half_vector.size(), out.size());
        tx_.apply(out);
#if 0
        tmp += '[';
        for(const auto el: out) tmp.sprintf("%e,", el);
        tmp.back() = ']';
        std::fprintf(stderr, "After copying input vector to output vector and apply: %s\n", tmp.data());
#endif
    }
};

template<typename KernelBlock,
         typename Finalizer=GaussianFinalizer>
class Kernel {
    std::vector<KernelBlock> blocks_;
    Finalizer             finalizer_;

public:
    using FloatType = typename KernelBlock::float_type;
    template<typename... Args>
    Kernel(size_t stacked_size, size_t input_size,
           FloatType sigma, uint64_t seed,
           Args &&... args):
        finalizer_(std::forward<Args>(args)...)
    {
        size_t input_ru = roundup(input_size);
        stacked_size = std::max(stacked_size, input_ru);
        if(stacked_size % input_ru)
            stacked_size = input_ru - (stacked_size % input_ru);
        if(stacked_size % input_ru) std::fprintf(stderr, "Stacked size is not evenly divisible.\n"), exit(1);
        size_t nblocks = (stacked_size) / input_ru;
        aes::AesCtr gen(seed);
        while(blocks_.size() < nblocks) {
            blocks_.emplace_back(input_ru, sigma, gen());
        }
    }
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        size_t in_rounded(roundup(in.size()));
        if(out.size() != (blocks_.size()) * in_rounded) {
            if constexpr(blaze::IsView<OutputType>::value) {
                throw std::runtime_error(ks::sprintf("[%s] Resizing out block from %zu to %zu to match %zu input and %zu rounded up input.\n",
                                                     __PRETTY_FUNCTION__, out.size(), (blocks_.size() << 1) * in_rounded, in.size(), (size_t)roundup(in.size())).data());
            } else {
                std::fprintf(stderr, "Resizing out block from %zu to %zu to match %zu input and %zu rounded up input.\n",
                             out.size(), (blocks_.size() << 1) * in_rounded, in.size(), (size_t)roundup(in.size()));
                out.resize((blocks_.size() << 1) * in_rounded);
            }
        }
        //in_rounded <<= 1; // To account for the doubling for the sin/cos entry for each random projection.
#if 0
        #pragma omp parallel for
#endif
        for(size_t i = 0; i < blocks_.size(); ++i) {
            auto sv(subvector(out, in_rounded * i, in_rounded));
            blocks_[i].apply(sv, in);
            finalizer_.apply(sv);
        }
    }
};

} // namespace ff


} // namespace gfrp

#endif // #ifndef _GFRP_KERNEL_H__
