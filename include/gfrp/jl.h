#ifndef _JL_H__
#define _JL_H__
#include <random>
#include "gfrp/spinner.h"

namespace gfrp {

template<typename MatrixType>
class JLTransform  {
    using FloatType = typename MatrixType::ElementType;
    const size_t m_, n_;
    MatrixType matrix_;
public:
    JLTransform(size_t m, size_t n):
        m_{m}, n_{n}, matrix_(m, n)  {
        if(m_ >= n_) fprintf(stderr, "Warning: JLTransform has to reduce dimensionality.");
    }
    template<typename RNG, typename Distribution>
    void fill(RNG &rng, Distribution &dist, bool orthogonalize=true) {
        for(size_t i(0); i < m_; ++i)
            for(size_t j(0); j < n_; ++j)
                matrix_(i, j) = dist(rng);
        if(orthogonalize) {
            linalg::gram_schmidt(matrix_, linalg::RESCALE_TO_GAUSSIAN);
        }
        matrix_ *= 1. / std::sqrt(static_cast<double>(m_));
    }
    void fill(uint64_t seed, bool orthogonalize=true) {
        aes::AesCtr rng(seed);
        boost::random::detail::unit_normal_distribution<FloatType> dist;
        fill(rng, dist, orthogonalize);
    }
    template<typename InVec, typename OutVec>
    void apply(const InVec &in, OutVec out) {
        assert(out.size() == m_);
        assert(in.size() == n_);
        out = matrix_ * in;
    }
    auto size() const {return matrix_.rows() * matrix_.columns();}
};

template<size_t nblocks=3, typename SizeType=size_t, bool OverrideBlockCount=false>
class OJLTransform {
    SizeType from_, to_;
    std::array<SizeType, nblocks> seeds_;
    //CachedSubsampler<std::unordered_set> sampler_;
public:
    using size_type = SizeType;
    HadamardRademacherSDBlock blocks_[nblocks];
    static_assert((nblocks & 1) || OverrideBlockCount, "Using an even number of blocks results in provably worse performance."
                                                       "You probably don't want to do this. If you're sure, change the last [fourth] template argument to true.");
    
    OJLTransform(size_t from, size_t to, std::array<SizeType, nblocks> &&seeds):
        seeds_{seeds}
    {
        resize(from, to);
    }
    OJLTransform(size_t from, size_t to, size_type seedseed):
        OJLTransform(from, to, aes::seed_to_array<size_type, nblocks>(seedseed))
    {
    }
    void resize(size_type newfrom, size_type newto) {
        //std::fprintf(stderr, "Resizing from %zu to %zu (rounded up %zu)\n", from_, roundup(newfrom), newfrom);
        newfrom = roundup(newfrom);
        resize_from(newfrom);
        resize_to(newto);
    }
    size_t from_size() const {return from_;}
    size_t to_size()   const {return to_;}
    void reseed_impl() {
    }
    void reseed(std::array<SizeType, nblocks> &&seeds) {
        seeds_ = std::move(seeds);
        reseed(seeds_);
    }
    void reseed(const std::array<SizeType, nblocks> &seeds) {
        seeds_ = seeds;
        resize_from(from_);
    }
    void reseed(size_type newseed) {
        reseed(aes::seed_to_array<size_type, nblocks>(newseed));
    }
    void resize_from(size_type newfrom) {
        from_ = newfrom;
        for(size_type i(0); i < nblocks; ++i) {
            blocks_[i].seed(seeds_[i]);
            blocks_[i].resize(from_);
        }
    }
    void resize_to(size_type newto) {
        to_ = newto;
    }
    template<typename Vec1, typename Vec2>
    void transform(const Vec1 &in, Vec2 &out) const {
        Vec2 tmp(in); // Copy.
        transform_inplace(tmp);
        out = subvector(tmp, 0, to_); // Copy result out.
    }
    template<typename Vec1, typename=std::enable_if_t<blaze::IsVector<Vec1>::value>>
    void transform_inplace(Vec1 &in) const {
        for(auto it(std::rbegin(blocks_)), eit(std::rend(blocks_)); it != eit; ++it) {
            it->apply(in);
        }
    }
    template<typename FloatType, typename=std::enable_if_t<std::is_floating_point<FloatType>::value>>
    void transform_inplace(FloatType *in) const {
        for(auto it(std::rbegin(blocks_)), eit(std::rend(blocks_)); it != eit; ++it) {
            it->apply(in);
        }
    }
    // Downstream application has to subsample itself.
    // Optionally add a (potentially scaled?) Guassian multiplication layer.
};

template<size_t nblocks=3, typename SizeType=size_t, bool OverrideBlockCount=false>
using COJLT = OJLTransform<nblocks, SizeType, OverrideBlockCount>;

} // namespace gfrp

#endif // #ifndef _JL_H__
