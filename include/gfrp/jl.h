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

} // namespace gfrp
#endif
