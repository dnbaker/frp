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
#if 1
        using SIMDType  = vec::SIMDTypes<FloatType>;
        using VT = typename SIMDType::Type;
        using DT = typename SIMDType::TypeDouble;
        static const size_t ratio(sizeof(VT) / sizeof(FloatType));
        DT dest;
        VT *srcptr((VT *)&in[0]);
        if(use_lowprec_) {
            CONST_IF(IS_CONTIGUOUS_UNCOMPRESSED_BLAZE(VecType)) {
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
            CONST_IF(IS_CONTIGUOUS_UNCOMPRESSED_BLAZE(VecType)) {
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
#else
        auto subv(subvector(in, 0, in.size() >> 1));
        std::uniform_real_distribution<FloatType> dist;
        aes::AesCtr<std::uint64_t> gen(in.size());
        for(size_t i(0); i < subv.size(); ++i)
            subv[i] += dist(gen);
        blaze::DynamicVector<FloatType> sinv = trans(subv);
        sinv = sin(sinv);
        blaze::DynamicVector<FloatType> cosv = trans(subv);
        cosv = cos(cosv);
        auto subv2(subvector(in, in.size() >> 1, in.size() >> 1));
        subv = trans(cosv);
        subv2 *= 0.;
#endif
    }
};
struct BoringFinalizer {
private:
    uint64_t seed_;
    uint32_t use_lowprec_:1;
public:
    BoringFinalizer(size_t seed=0, bool use_low_precision=false): seed_(seed), use_lowprec_(use_low_precision) {}
    void set_use_lowprec(bool use_lowprec) {use_lowprec_ = use_lowprec;}

    template<typename VecType>
    void apply(VecType &in) const {
        using FloatType = std::decay_t<decltype(*in.begin())>;
        auto subv(subvector(in, 0, in.size() >> 1));
        std::uniform_real_distribution<FloatType> dist;
        aes::AesCtr<std::uint64_t> gen(std::hash<uint64_t>()(in.size() + seed_));
        for(size_t i(0); i < subv.size(); ++i)
            subv[i] += dist(gen);
        blaze::DynamicVector<FloatType> sinv = sin(trans(subv));
        blaze::DynamicVector<FloatType> cosv = cos(trans(subv));
        auto subv2(subvector(in, in.size() >> 1, in.size() >> 1));
        subv = trans(cosv);
        subv2 = trans(sinv);
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
        if(final_output_size_ & (final_output_size_ - 1)) {
            const auto msg = std::string(__PRETTY_FUNCTION__) + "'s size should be a power of two.\n";
            std::cerr << msg;
            throw std::runtime_error(msg);
        }
        auto &rsbref(std::get<RandomScalingBlock>(tx_.get_tuple()));
        auto &gmref(std::get<GaussianMatrixType>(tx_.get_tuple()));
        rsbref.rescale(float_type(size)/std::sqrt(gmref.vec_norm()));
    }
    size_t transform_size() const {return final_output_size_;}
#if 0
    auto       &rsbref()       {return std::get<RandomScalingBlock>(tx_.get_tuple());}
    const auto &rsbref() const {return std::get<RandomScalingBlock>(tx_.get_tuple());}
#endif
    template<typename OutputType>
    void apply(OutputType &out, size_t nelem) const {
        if(out.size() != final_output_size_) {
            fprintf(stderr, "[%s:%d:%s] Warning: Output size was wrong (%zu, not %zu). Resizing\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, out.size(), final_output_size_);
        }
        blaze::reset(subvector(out, nelem, out.size() - nelem));
        auto half_vector(subvector(out, 0, transform_size()));
        tx_.apply(half_vector);
    }
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) const {
        if(out.size() != final_output_size_) {
            fprintf(stderr, "[%s:%d:%s] Warning: Output size was wrong (%zu, not %zu). Resizing\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, out.size(), final_output_size_);
        }
        if(roundup(in.size()) != transform_size()) {::std::cerr << "ZOMG error in " << __PRETTY_FUNCTION__ << " at line " << __LINE__ <<'\n'; throw std::runtime_error("ZOMG");}
        if(&out[0] != &in[0]) {
            subvector(out, 0, in.size()) = in;
        }
        blaze::reset(subvector(out, in.size(), out.size() - in.size()));
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
        if(nblocks == 0) {
            const char *s = "Need more than 0 blocks for sorf::KernelBlock. (Recommended: 3.)\n";
            ::std::cerr << s; throw std::runtime_error(s);
        }
        while(blocks_.size() < nblocks)
            blocks_.emplace_back(std::make_pair(HadamardBlock(),
                                 RademType(size, seed++)));
    }
    size_t transform_size() const {return final_output_size_;}
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) const {
        if(out.size() != final_output_size_) {
            char buf[1024];
            std::sprintf(buf, "[%s:%d:%s] Warning: Output size was wrong (%zu, not %zu). Resizing\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, out.size(), final_output_size_);
            ::std::cerr << buf;
            throw std::runtime_error(buf);
        }
        if(roundup(in.size()) != transform_size()) {::std::cerr << "ZOMG error in " << __PRETTY_FUNCTION__ << " at line " << __LINE__ <<'\n'; throw std::runtime_error("ZOMG");}
        blaze::reset(out);
        subvector(out, 0, in.size()) = in;
        //std::fprintf(stderr, "Applying sorf::KernelBlock\n");
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
    void apply(OutputType &out, const InputType &in) const {
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
    unit_gaussian_fill(randg, seed);
    auto ret(linalg::qr_gram_schmidt(randg, linalg::ORTHONORMALIZE));
    blaze::DynamicVector<FloatType> SV(size);
    chisq_fill(SV, seed++);
    SV = sqrt(SV);
    blaze::DiagonalMatrix<blaze::DynamicMatrix<FloatType>> S(size);
    for(size_t i(0); i < SV.size(); ++i) S(i,i) = SV[i];
    ret = S * ret;
    ret *= 1./sigma;
    return ret;
}
} // namespace detail

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
    void apply(OutputType &out, const InputType &in) const {
        if(out.size() != final_output_size_) {
            char buf[512];
            std::sprintf(buf, "[%s:%d:%s] Warning: Output size was wrong (%zu, not %zu). Resizing\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, out.size(), final_output_size_);
            ::std::cerr << buf;
            throw std::runtime_error(buf);
        }
        // std::fprintf(stderr, "Applying orf::KernelBlock\n");
        if constexpr(blaze::TransposeFlag<InputType>::value) {
            out = trans(matrix_ * trans(in));
        } else {
            out = matrix_ * in;
        }
    }
};

} // namespace orf
namespace rf {

template<typename FloatType, typename RademType=CompactRademacher>
class KernelBlock {
protected:
    const size_t         final_output_size_;
    blaze::DynamicMatrix<FloatType> matrix_;
public:
    using float_type = FloatType;
    KernelBlock(size_t size, uint64_t seed=-1,
                FloatType sigma=1.):
        final_output_size_(size), matrix_(size, size) {
#if 0
        for(size_t i(0); i < matrix_.rows(); ++i) {
            auto mrow(row(matrix_, i));
            unit_gaussian_fill(mrow, seed++);
        }
#else
        unit_gaussian_fill(matrix_, seed);
#endif
        matrix_ *= 1./sigma;
    }
    size_t transform_size() const {return final_output_size_;}
    template<typename InputType, typename OutputType>
    void apply(OutputType &out, const InputType &in) const {
        if(out.size() != final_output_size_) {
            char buf[512];
            std::sprintf(buf, "[%s:%d:%s] Warning: Output size was wrong (%zu, not %zu). Resizing\n", __FILE__, __LINE__, __PRETTY_FUNCTION__, out.size(), final_output_size_);
            ::std::cerr << buf;
            throw std::runtime_error(buf);
        }
        // std::fprintf(stderr, "Applying rf::KernelBlock\n");
        if constexpr(blaze::TransposeFlag<InputType>::value) {
            out = trans(matrix_ * trans(in));
        } else {
            out = matrix_ * in;
        }
    }
};

} // namespace rf

template<typename KernelBlock,
         typename Finalizer=GaussianFinalizer>
class Kernel {
    std::vector<KernelBlock> blocks_;
    Finalizer             finalizer_;
    const size_t              indim_;
    size_t                   outdim_;
public:
    using FloatType = typename KernelBlock::float_type;
#ifdef SIGMA_RESCALE
    const FloatType sigma_;
#endif

    template<typename... Args>
    Kernel(size_t stacked_size, size_t input_size,
           uint64_t seed,
           Args &&... args): indim_(input_size)
#ifdef SIGMA_RESCALE
        , sigma_(sigma)
#endif
    {
        size_t input_ru = roundup(input_size);
        stacked_size = std::max(stacked_size, input_ru);
        if(stacked_size % input_ru)
            stacked_size = input_ru - (stacked_size % input_ru);
        outdim_ = stacked_size;
        aes::AesCtr<uint64_t> gen(seed);
        for(size_t nblocks = stacked_size / input_ru;
            blocks_.size() < nblocks;
            blocks_.emplace_back(input_ru, gen(), std::forward<Args>(args)...));
    }

    size_t nblocks() const {return blocks_.size();}
    size_t indim() const {return indim_;}
    size_t outdim() const {return outdim_;}

    template<typename OutputType>
    void apply(OutputType &out, size_t nelem) const {
        size_t in_rounded(roundup(nelem));
        blaze::DynamicVector<FloatType> tmp(nelem);
        tmp = ::blaze::subvector(out, 0, nelem);
        if(out.size() != (blocks_.size() << 1) * in_rounded) {
            if constexpr(blaze::IsView<OutputType>::value) {
                auto ks(ks::sprintf("[%s] Wanted to resize out block from %zu to %zu to match %zu input and %zu rounded up input.\n",
                                    __PRETTY_FUNCTION__, out.size(), (blocks_.size() << 1) * in_rounded, nelem, static_cast<size_t>(roundup(nelem))));
                ks.write(stderr);
                throw std::runtime_error(ks.data());
            } else {
                std::fprintf(stderr, "Resizing out block from %zu to %zu to match %zu input and %zu rounded up input.\n",
                             out.size(), (blocks_.size() << 1) * in_rounded, nelem, (size_t)roundup(nelem));
                out.resize((blocks_.size() << 1) * in_rounded);
            }
        }
        for(size_t i = 0; i < blocks_.size(); ++i) {
            auto sv(subvector(out, (in_rounded << 1) * i, in_rounded));
            blocks_[i].apply(sv, tmp);
            finalizer_.apply(sv);
        }
    }
    template<typename InputType, typename OutputType, typename=std::enable_if_t<!std::is_arithmetic_v<InputType>>>
    void apply(OutputType &out, const InputType &in) const {
        size_t in_rounded(roundup(in.size()));
        if(out.size() != (blocks_.size() << 1) * in_rounded) {
            if constexpr(blaze::IsView<OutputType>::value) {
                char buf[2048];
                std::sprintf(buf, "[%s] Wanted to resize out block from %zu to %zu to match %zu input and %zu rounded up input.\n",
                                    __PRETTY_FUNCTION__, out.size(), (blocks_.size() << 1) * in_rounded, in.size(), static_cast<size_t>(roundup(in.size())));
                ::std::cerr << buf;
                throw std::runtime_error(buf);
            } else {
                std::fprintf(stderr, "Resizing out block from %zu to %zu to match %zu input and %zu rounded up input.\n",
                             out.size(), (blocks_.size() << 1) * in_rounded, in.size(), (size_t)roundup(in.size()));
                out.resize((blocks_.size() << 1) * in_rounded);
            }
        }
#if 0
        #pragma omp parallel for
#endif
        for(size_t i = 0; i < blocks_.size(); ++i) {
            auto sv(subvector(out, (in_rounded << 1) * i, in_rounded));
            blocks_[i].apply(sv, in);
            finalizer_.apply(sv);
        }

#ifdef SIGMA_RESCALE
#define MULVAL (std::sqrt(2. / static_cast<FloatType>(out.size() >> 1)) * sigma_ / std::sqrt(std::sqrt(in.size())))
#else
#define MULVAL (std::sqrt(2. / static_cast<FloatType>(out.size() >> 1)))
#endif
        vec::blockmul(out, MULVAL);
#undef MULVAL
        // TODO: add this multiplication to the finalizer to avoid a second RAM pass-through.
    }
};

template<typename FloatType, typename RademType>
using FastFoodKernelBlock = ff::KernelBlock<FloatType, RademType>;
template<typename FloatType, typename RademType>
using SORFKernelBlock = sorf::KernelBlock<FloatType, RademType>;

} // namespace kernel



} // namespace frp

#endif // #ifndef _GFRP_KERNEL_H__
