#ifndef _GFRP_KERNEL_H__
#define _GFRP_KERNEL_H__
#include "frp/spinner.h"

namespace frp {

namespace kernel {


struct GaussianFinalizer {
private:
    uint32_t use_lowprec_:1;
public:
    GaussianFinalizer(bool use_low_precision=false): use_lowprec_(use_low_precision) {}
    void set_use_lowprec(bool use_lowprec) {use_lowprec_ = use_lowprec;}

    template<typename VecType>
    void apply(VecType &in) const {
        if((in.size() & (in.size() - 1))) std::fprintf(stderr, "in.size() [%zu] is not a power of 2.\n", in.size()), exit(1);
        using FloatType = typename std::decay_t<decltype(in[0])>;
        using SIMDType  = vec::SIMDTypes<FloatType>;
        using VT = typename SIMDType::Type;
        using DT = typename SIMDType::TypeDouble;
        static const size_t ratio(sizeof(VT) / sizeof(FloatType));
        DT dest;
        VT *srcptr((VT *)&in[0]);
        if(use_lowprec_) {
            if constexpr(IS_BLAZE(VecType)) {
                for(u32 i((in.size() >> 1) / ratio); i;) {
                    dest = SIMDType::sincos_u35(SIMDType::load((FloatType *)&srcptr[i - 1]));
                    SIMDType::store((FloatType *)&srcptr[(i << 1) - 1], dest.y);
                    SIMDType::store((FloatType *)&srcptr[--i << 1], dest.x);
                }
            } else {
                if(SIMDType::aligned(srcptr)) {
                    for(u32 i((in.size() >> 1) / ratio); i;) {
                        dest = SIMDType::sincos_u35(SIMDType::load((FloatType *)&srcptr[i - 1]));
                        SIMDType::store((FloatType *)&srcptr[(i << 1) - 1], dest.y);
                        SIMDType::store((FloatType *)&srcptr[--i << 1], dest.x);
                    }
                } else {
                    for(u32 i((in.size() >> 1) / ratio); i;) {
                        dest = SIMDType::sincos_u35(SIMDType::loadu((FloatType *)&srcptr[i - 1]));
                        SIMDType::storeu((FloatType *)&srcptr[(i << 1) - 1], dest.y);
                        SIMDType::storeu((FloatType *)&srcptr[--i << 1], dest.x);
                    }
                }
            }
        } else {
            if constexpr(IS_BLAZE(VecType)) {
                for(u32 i((in.size() >> 1) / ratio); i;) {
                    dest = SIMDType::sincos_u10(SIMDType::load((FloatType *)&srcptr[i - 1]));
                    SIMDType::store((FloatType *)&srcptr[(i << 1) - 1], dest.y);
                    SIMDType::store((FloatType *)&srcptr[--i << 1], dest.x);
                }
            } else {
                if(SIMDType::aligned(srcptr)) {
                    for(u32 i((in.size() >> 1) / ratio); i;) {
                        dest = SIMDType::sincos_u10(SIMDType::load((FloatType *)&srcptr[i - 1]));
                        SIMDType::store((FloatType *)&srcptr[(i << 1) - 1], dest.y);
                        SIMDType::store((FloatType *)&srcptr[--i << 1], dest.x);
                    }
                } else {
                    for(u32 i((in.size() >> 1) / ratio); i;) {
                        dest = SIMDType::sincos_u10(SIMDType::loadu((FloatType *)&srcptr[i - 1]));
                        SIMDType::storeu((FloatType *)&srcptr[(i << 1) - 1], dest.y);
                        SIMDType::storeu((FloatType *)&srcptr[--i << 1], dest.x);
                    }
                }
            }
        }
    }
};


namespace ff {

template<typename FloatType, typename RademType=CompactRademacher>
class KernelBlock {
protected:
    size_t final_output_size_; // This is twice the size passed to the Hadamard transforms
    using RandomScalingBlock = RandomChiScalingBlock<FloatType>;
    using SizeType = uint32_t;
    using Shuffler = LutShuffler<SizeType>;
    using SpinTransformer =
        SpinBlockTransformer<FastFoodGaussianProductBlock<FloatType>,
                             RandomScalingBlock, HadamardBlock,
                             UnitGaussianScalingBlock<FloatType>, Shuffler, HadamardBlock,
                             RademType>;
    SpinTransformer tx_;

public:
    using float_type = FloatType;
    using GaussianMatrixType = UnitGaussianScalingBlock<FloatType>;
    KernelBlock(size_t size, uint64_t seed=-1, FloatType sigma=1., bool renorm=true):
        final_output_size_(size),
        tx_(
            std::make_tuple(FastFoodGaussianProductBlock<FloatType>(sigma),
                   RandomScalingBlock(seed + seed * seed - size * size, size),
                   HadamardBlock(size, renorm),
                   GaussianMatrixType(seed * seed, size),
                   Shuffler(size, seed),
                   HadamardBlock(size, renorm),
                   RademType(size, (seed ^ (size * size)) + seed)))
    {
        if(final_output_size_ & (final_output_size_ - 1))
            throw std::runtime_error((std::string(__PRETTY_FUNCTION__) + "'s size should be a power of two.").data());
        auto &rsbref(std::get<RandomScalingBlock>(tx_.get_tuple()));
        auto &gmref(std::get<GaussianMatrixType>(tx_.get_tuple()));
        rsbref.rescale(float_type(size)/std::sqrt(gmref.vec_norm()));
    }
    size_t transform_size() const {return final_output_size_;}
#if 0
    auto       &rsbref()       {return std::get<RandomScalingBlock>(tx_.get_tuple());}
    const auto &rsbref() const {return std::get<RandomScalingBlock>(tx_.get_tuple());}
#endif
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        if(out.size() != final_output_size_) {
            fprintf(stderr, "Warning: Output size was wrong (%zu, not %zu). Resizing\n", out.size(), final_output_size_);
        }
        if(roundup(in.size()) != transform_size()) throw std::runtime_error("ZOMG");
        blaze::reset(out);

        subvector(out, 0, in.size()) = in;
        auto half_vector(subvector(out, 0, transform_size()));
        tx_.apply(half_vector);
    }
};

} // namespace ff

namespace sorf {

template<typename FloatType, typename RademType=PRNRademacher>
class KernelBlock {
protected:
    const size_t final_output_size_;
    SORFProductBlock<FloatType>                        sorf_;
    std::vector<std::pair<HadamardBlock, RademType>> blocks_;
public:
    using float_type = FloatType;
    KernelBlock(size_t size, uint64_t seed=-1,
                FloatType sigma=1., size_t nblocks=3):
                    final_output_size_(size), sorf_(sigma) {
        if(nblocks == 0) throw std::runtime_error("Need more than 0 blocks for sorf::KernelBlock. (Recommended: 3.)");
        while(blocks_.size() < nblocks)
            blocks_.emplace_back(std::make_pair(HadamardBlock(),
                                 RademType(size, seed++)));
    }
    size_t transform_size() const {return final_output_size_;}
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        if(out.size() != final_output_size_) {
            char buf[128];
            std::sprintf(buf, "Warning: Output size was wrong (%zu, not %zu). Resizing\n", out.size(), final_output_size_);
            throw std::runtime_error(buf);
        }
        if(roundup(in.size()) != transform_size()) throw std::runtime_error("ZOMG");
        blaze::reset(out);
        subvector(out, 0, in.size()) = in;
        std::fprintf(stderr, "Applying sorf::KernelBlock\n");
        for(auto &pair: blocks_) pair.second.apply(out), pair.first.apply(out);
        sorf_.apply(out);
    }
};

template<typename FloatType, typename RademType=CompactRademacher>
class ChiKernelBlock: public KernelBlock<FloatType, RademType> {
    RandomChiScalingBlock<FloatType> rcsb_;
public:
    ChiKernelBlock(size_t size, uint64_t seed=-1,
                   FloatType sigma=1., size_t nblocks=3): KernelBlock<FloatType, RademType>(size, seed, sigma, nblocks), rcsb_(seed * seed + seed - 1, size)
    {

    }
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        KernelBlock<FloatType, RademType>::apply(out, in);
        rcsb_.apply(out);
    }
};

} // namespace sorf

namespace orf {

namespace detail {
template<typename FloatType>
auto make_q(size_t size, FloatType sigma, uint64_t seed=0) {
    blaze::DynamicMatrix<FloatType> randg(size, size);
    for(size_t i(0); i < size; ++i) {
        auto rrow(row(randg, i));
        unit_gaussian_fill(rrow, seed++);
    }
    auto ret(linalg::qr_gram_schmidt(randg, 0));
    blaze::DynamicVector<FloatType> SV(size);
    chisq_fill(SV, seed++);
    SV = sqrt(SV);
    blaze::DiagonalMatrix<blaze::DynamicMatrix<FloatType>> S(size);
    for(size_t i(0); i < SV.size(); ++i) S(i,i) = SV[i];
    ret = S * ret;
    ret *= 1./sigma;
    return ret;
}
}

template<typename FloatType, typename RademType=CompactRademacher>
class KernelBlock {
protected:
    const size_t         final_output_size_;
    blaze::DynamicMatrix<FloatType> matrix_;
public:
    using float_type = FloatType;
    KernelBlock(size_t size, uint64_t seed=-1,
                FloatType sigma=1.):
        final_output_size_(size), matrix_(detail::make_q(size, sigma, seed)) {}
    size_t transform_size() const {return final_output_size_;}
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        if(out.size() != final_output_size_) {
            fprintf(stderr, "Warning: Output size was wrong (%zu, not %zu). Resizing\n", out.size(), final_output_size_);
        }
        std::fprintf(stderr, "Applying orf::KernelBlock\n");
        if constexpr(blaze::TransposeFlag<InputType>::value) {
            out = trans(matrix_ * trans(in));
        } else {
            out = matrix_ * in;
        }
    }
};

} // namespace orf

template<typename KernelBlock,
         typename Finalizer=GaussianFinalizer>
class Kernel {
    std::vector<KernelBlock> blocks_;
    Finalizer             finalizer_;
public:
    using FloatType = typename KernelBlock::float_type;
#ifdef SIGMA_RESCALE
    const FloatType sigma_;
#endif

    template<typename... Args>
    Kernel(size_t stacked_size, size_t input_size,
           uint64_t seed,
           Args &&... args)
#ifdef SIGMA_RESCALE
        : sigma_(sigma)
#endif
    {
        size_t input_ru = roundup(input_size);
        stacked_size = std::max(stacked_size, input_ru);
        if(stacked_size % input_ru)
            stacked_size = input_ru - (stacked_size % input_ru);
        if(stacked_size % input_ru) std::fprintf(stderr, "Stacked size is not evenly divisible.\n"), exit(1);
        size_t nblocks = (stacked_size) / input_ru;
        aes::AesCtr gen(seed);
        while(blocks_.size() < nblocks) {
            blocks_.emplace_back(input_ru, gen(), std::forward<Args>(args)...);
        }
    }

    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) {
        size_t in_rounded(roundup(in.size()));
        if(out.size() != (blocks_.size() << 1) * in_rounded) {
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
            auto sv(subvector(out, (in_rounded << 1) * i, in_rounded));
            blocks_[i].apply(sv, in);
            finalizer_.apply(sv);
        }
        vec::blockmul(out, 1./std::sqrt(static_cast<FloatType>(out.size() >> 1)));
        // TODO: add this multiplication to the finalizer to avoid a second RAM pass-through.
#ifdef SIGMA_RESCALE
        vec::blockmul(out, sigma_ / std::sqrt(std::sqrt(in.size())));
#endif
    }
};

template<typename FloatType, typename RademType>
using FastFoodKernelBlock = ff::KernelBlock<FloatType, RademType>;
template<typename FloatType, typename RademType>
using SORFKernelBlock = sorf::KernelBlock<FloatType, RademType>;

} // namespace kernel



} // namespace frp

#endif // #ifndef _GFRP_KERNEL_H__
