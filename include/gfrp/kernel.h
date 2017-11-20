#ifndef _GFRP_KERNEL_H__
#define _GFRP_KERNEL_H__
#include "gfrp/spinner.h"

namespace gfrp {

namespace ff {

struct GaussianFinalizer {
    template<typename VecType>
    void apply(VecType &in) {
        assert((in.size() & (in.size() - 1)) == 0);
        for(uint32_t i(in.size()>>1); i; --i) in[(i<<1) - 1] = (in[(i-1)<<1] = in[i - 1]) + 0.5;
        in = cos(in);
    }
};


template<typename FloatType, typename RandomScalingBlock=RandomGaussianScalingBlock<FloatType>,
         typename GaussianMatrixType=UnitGaussianScalingBlock<FloatType>,
         typename FirstSBlockType=HadamardBlock, typename LastSBlockType=FirstSBlockType>
class KernelBlock {
    size_t final_output_size_; // This is twice the size passed to the Hadamard transforms
    using SpinTransformer =
        SpinBlockTransformer<FastFoodGaussianProductBlock<FloatType>,
                             RandomGaussianScalingBlock<FloatType>, LastSBlockType,
                             GaussianMatrixType, OnlineShuffler<size_t>, FirstSBlockType,
                             CompactRademacher>;
    SpinTransformer tx_;

public:
    using float_type = FloatType;
    KernelBlock(size_t size, FloatType sigma=1., uint64_t seed=-1):
        final_output_size_(size),
        tx_(
            std::make_tuple(FastFoodGaussianProductBlock<FloatType>(sigma),
                   RandomGaussianScalingBlock<FloatType>(1., seed + seed * seed - size * size, transform_size()),
                   LastSBlockType(transform_size()),
                   GaussianMatrixType(seed * seed),
                   OnlineShuffler<size_t>(seed),
                   FirstSBlockType(transform_size()),
                   CompactRademacher(transform_size(), (seed ^ (size * size)) + seed)))
    {
        if(final_output_size_ & (final_output_size_ - 1)) throw std::runtime_error("GaussianKernel's size should be a power of two.");
        std::get<1>(tx_.get_tuple()).rescale(std::get<3>(tx_.get_tuple()));
    }
    size_t transform_size() const {return final_output_size_ >> 1;}
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        if(out.size() != final_output_size_) {
            fprintf(stderr, "Warning: Output size was wrong (%zu, not %zu). Resizing\n", out.size(), final_output_size_);
        }
        if(roundup(in.size()) != transform_size()) throw std::runtime_error("ZOMG");
        blaze::reset(out);
        subvector(out, 0, in.size()) = in; // Copy input to output space.

        auto half_vector(subvector(out, 0, transform_size()));
        tx_.apply(half_vector);   
    }
};

template<typename KernelBlock,
         typename Finalizer=GaussianFinalizer>
class Kernel {
    std::vector<KernelBlock> blocks_;
    Finalizer        finalizer_;

public:
    using FloatType = typename KernelBlock::float_type;
    template<typename... Args>
    Kernel(size_t stacked_size, size_t input_size,
           FloatType sigma, uint64_t seed,
           Args &&... args):
        finalizer_(std::forward<Args>(args)...)
    {
        stacked_size    = roundup(stacked_size);
        size_t input_ru = roundup(input_size);
        assert(stacked_size >= input_ru << 1);
        size_t nblocks = (stacked_size >> 1) / input_ru + !!((stacked_size >> 1) % input_ru);
        aes::AesCtr gen(seed);
        while(blocks_.size() < nblocks) {
            blocks_.emplace_back(input_ru, sigma, gen());
        }
    }
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        if(double(out.size() >> 1) / blocks_.size() != double(in.size())) {
            throw std::runtime_error("Unexpected sizes. (out: %zu. in: %zu)\n", out.size(), in.size());
        }
        size_t inru(roundup(in.size()));
#ifdef USE_OPENMP
        #pragma omp parallel for
#endif
        for(size_t i = 0; i < blocks_.size(); ++i) {
            auto sv(subvector(out, inru * i, inru));
            blocks_[i].apply(sv, in);
            finalizer_.apply(sv);
        }
    }
};

} // namespace ff


} // namespace gfrp

#endif // #ifndef _GFRP_KERNEL_H__
