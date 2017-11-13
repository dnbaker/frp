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
        std::fprintf(stderr, "Creating transform of from %zu to %zu and seed %zu\n", from, to, (size_t)seedseed);
        for(auto &seed: seeds_) std::cerr << "Seed: " << seed << '\n';
    }
    void resize(size_type newfrom, size_type newto) {
        resize_from(newfrom);
        resize_to(newto);
    }
    void reseed_impl() {
        for(size_type i(0); i < nblocks; ++i) {
            blocks_[i].resize(from_);
            blocks_[i].seed(seeds_[i]);
        }
    }
    void reseed(std::array<SizeType, nblocks> &&seeds) {
        seeds_ = std::move(seeds);
        reseed_impl();
    }
    void reseed(const std::array<SizeType, nblocks> &seeds) {
        seeds_ = seeds;
        reseed_impl();
    }
    void reseed(size_type newseed) {
        seeds_ = aes::seed_to_array<size_type, nblocks>(newseed);
        reseed_impl();
    }
    void resize_from(size_type newfrom) {
        from_ = newfrom;
        reseed_impl();
    }
    void resize_to(size_type newto) {
        to_ = newto;
    }
    template<typename Vec1, typename Vec2>
    void transform(const Vec1 &in, Vec2 &out) {
        Vec2 tmp(in); // Copy.
        transform_inplace(tmp);
        out = subvector(tmp, 0, to_); // Copy result out.
    }
    template<typename Vec1>
    void transform_inplace(Vec1 &in) {
        for(auto it(std::rbegin(blocks_)), eit(std::rend(blocks_)); it != eit; ++it) {
            it->apply(in);
        }
    }
    template<typename FloatType>
    void transform_inplace(FloatType *in) {
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
