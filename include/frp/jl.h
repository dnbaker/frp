#ifndef _JL_H__
#define _JL_H__
#include <random>
#include "frp/spinner.h"

namespace frp {

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

class OrthogonalJLTransform {
    size_t from_, to_;
    std::vector<HadamardRademacherSDBlock> blocks_;
    std::vector<uint64_t> seeds_;
public:
    using size_type = uint64_t;

    OrthogonalJLTransform(size_t from, size_t to, uint64_t seed, size_t nblocks=3): from_(roundup(from)), to_(to)
    {
        aes::AesCtr<uint64_t> gen(seed);
        while(seeds_.size() < nblocks) seeds_.push_back(gen());
        for(const auto seed: seeds_) blocks_.emplace_back(from, seed);
    }
    void resize(size_type newfrom, size_type newto) {
        //std::fprintf(stderr, "Resizing from %zu to %zu (rounded up %zu)\n", from_, roundup(newfrom), newfrom);
        newfrom = roundup(newfrom);
        resize_from(newfrom);
        resize_to(newto);
    }
    size_t from_size() const {return from_;}
    size_t to_size()   const {return to_;}
    void reseed(size_type newseed) {
        seeds_.clear();
        aes::AesCtr<uint64_t> gen(newseed);
        while(seeds_.size() < nblocks()) seeds_.push_back(gen());
    }
    void resize_from(size_type newfrom) {
        from_ = newfrom;
        for(size_type i(0); i < nblocks(); ++i) {
            blocks_[i].seed(seeds_[i]);
            blocks_[i].resize(from_);
        }
    }
    void resize_to(size_type newto) {
        to_ = newto;
    }
    size_t nblocks() const {return blocks_.size();}
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
        in *= std::sqrt(static_cast<double>(from_) / to_);
    }
    template<typename FloatType, typename=std::enable_if_t<std::is_floating_point<FloatType>::value>>
    void transform_inplace(FloatType *in) const {
        for(auto it(std::rbegin(blocks_)), eit(std::rend(blocks_)); it != eit; (it++)->apply(in)); // Apply transforms
        // Renormalize.
        using SType = typename vec::SIMDTypes<FloatType>;
        const FloatType *end(in + to_);
        const typename SType::ValueType vmul = SType::set1(std::sqrt(static_cast<FloatType>(from_) / to_));
        if(SType::aligned(in)) {
            do {
                SType::store(in, SType::mul(SType::load(in), vmul));
            } while(++in < end);
        } else {
            do {
                SType::storeu(in, SType::mul(SType::loadu(in), vmul));
            } while(++in < end);
        }
    }
    // Downstream application has to subsample itself.
    // Optionally add a (potentially scaled?) Guassian multiplication layer.
};

using OJLTransform = OrthogonalJLTransform;
using OJLT = OJLTransform;

} // namespace frp

#endif // #ifndef _JL_H__
