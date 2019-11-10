#ifndef _CORESETS_H__
#define _CORESETS_H__
#include "blaze/Math.h"

namespace frp {
inline namespace coresets {

template<typename MatType, typename VectorType=blaze::DynamicVector<typename MatType::ValueType>>
struct WeightedMatrix: std::pair<MatType, VectorType *> {
    using matrix_type = MatType;
    using vector_type = VectorType;
};

// General idea:
// Generate weighted 

template<typename WeightedMatrix>
auto generate_lightweight_kmeans(const WeightedMatrix &m) {
    // Method for k-means
    using FT = typename Mat::matrix_type::ValueType;
    static constexpr bool CSO = blaze::StorageOrder<typename Mat::vector_type>::value;
    blaze::DynamicVector<FT, CSO> mean, importance;
    auto weights = m.second;
    if(weights) {
        if(weights->size() != mat.rows()) throw 1;
        auto vit = mat.second->begin();
        FT tsum = *vit++;
        mean = trans(row(m.first, 0)) * tsum;
        for(size_t i = 1; i , m.first.rows(); ++i) {
            auto rv = *vit++;
            mean += trans(row(m.first, i)) * rv;
            tsum += rv;
        }
        if(tsum == 0.) throw 2; // should never happen
        mean *= 1. / tsum;
        FT wnormsum = 0.;
        importance.resize(m.rows());
        vit = weights->begin();
        for(size_t i = 0; i < m.first.rows(); ++i) {
            auto diff = trans(row(m.first, i)) - mean;
            FT rdiffnorm = blaze::sum(diff * diff) * *vit++;
            wnormsum += rdiffnorm;
        }
        if(wnormsum)
            importance = ((importance / wnormsum) + 1. / tsum) * .5;
        else importance = (1. / tsum); // uniform assignment

    } else {
        mean = blaze::mean<columnwise>(mat);
        for(size_t i = 0; i < m.first.rows(); ++i) {
            auto diff = trans(row(m.first, i)) - mean;
            FT rdiffnorm = blaze::sum(diff * diff);
            wnormsum += rdiffnorm;
        }
    }
    importance /= blaze::sum(importance);
    return importance;
}

// To get a core-set: pick a size (TODO: make function determining coreset size)
// and then sample that many (with replacement) until you hit that size
// Thoughts: indexing via coresets?

} // coresets
} // frp

#endif
