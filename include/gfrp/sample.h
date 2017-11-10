#ifndef _GFRP_SAMPLE_H__
#define _GFRP_SAMPLE_H__
#include <unordered_set>
#include "gfrp/util.h"

namespace gfrp {
//TODO: make precomputed subsample indices for JL transforms and other squashings.
enum SubsampleStrategy {
    FIRST_M = 0,
    RANDOM_NO_REPLACEMENT = 1,
    RANDOM_NO_REPLACEMENT_HASH_SET = 2,
    RANDOM_NO_REPLACEMENT_VEC = 3,
    RANDOM_W_REPLACEMENT = 4
};

template<template <typename> typename SetContainer=std::unordered_set, typename SizeType=unsigned>
auto random_set_in_range(SizeType n, SizeType range, uint64_t seed=0) {
    if(n > range) throw std::logic_error(ks::sprintf("n (%zu) mod range (%zu) is imposcerous.", n, range).data());
    aes::AesCtr<SizeType> gen;
    if(seed == 0) seed = (n * range);
    SetContainer<SizeType> ret;
    ret.reserve(n);
    while(ret.size() < n) ret.insert(fastrange(n, range));
    return ret;
}

template<typename FullVector, typename SmallVector>
void subsample(const FullVector &in, SmallVector &out, SubsampleStrategy strat, uint64_t seed) {
    std::fprintf(stderr, "Warning: This always regenates the indices to copy over. The seed must be set the same every time.\n"
                         "You can make and save a reordering if you want to keep it the same later.\n");
    static_assert(is_same<decay_t<decltype(in[0])>, decay_t<decltype(out[0])>>::value,
                  "Vectors must have the same float type.");
    // For n <100, a straight vector is faster
    if(strat == RANDOM_NO_REPLACEMENT)
        strat = out.size() > 100 ? RANDOM_NO_REPLACEMENT_HASH_SET: RANDOM_NO_REPLACEMENT_VEC;
    switch(strat) {
        case FIRST_M:
            auto sv(subvector(in, 0, out.size()));
            out = sv;
        break;
        case RANDOM_NO_REPLACEMENT_HASH_SET:
        {
            auto indices(random_set_in_range<unsigned>(out.size(), in.size()));
            unsigned ind(0);
            for(const auto el: indices) out[ind++] = in[el];
        }
        break;
        case RANDOM_NO_REPLACEMENT_VEC: {
            aes::AesCtr<uint32_t> gen;
            std::vector<unsigned> indices;
            while(indices.size() < out.size()) {
                const auto tmp(fastrange(gen(), in.size()));
                if(std::find(std::begin(indices), std::end(indices), tmp) == std::end(indices)) {
                    indices.push_back(tmp);
                }
            }
            for(unsigned i(0); i < out.size(); ++i) out[i] = in[indices[i]];
        }
        break;
        case RANDOM_W_REPLACEMENT:
        {
            aes::AesCtr<uint32_t> gen;
            for(auto &el: out) {
                el = in[fastrange(gen(), out.size())];
            }
        }
    }
}
template<typename FullVector, typename OutVector=FullVector>
OutVector subsample(const FullVector &in, SubsampleStrategy strat, uint64_t seed, size_t outsz) {
    OutVector out(outsz);
    static_assert(is_same<decay_t<decltype(in[0])>, decay_t<decltype(out[0])>>::value,
                  "Vectors must have the same float type.");
    subsample(in, out, strat, seed);
    return out;
}

template<template <typename> typename SetContainer=std::unordered_set, typename SizeType=unsigned>
class CachedSubsampler {

    const SizeType in_;
    std::vector<SizeType> indices;

public:
    CachedSubsampler(SizeType in, SizeType out, SizeType seed=0): in_(in) {
        auto idxset(random_set_in_range<SetContainer, SizeType>(out, in, seed));
        indices = std::vector<SizeType>(std::begin(idxset), std::end(idxset));
        std::sort(indices.begin(), indices.end()); // For better memory access pattern.
    }
    template<typename Vec1, typename Vec2>
    void apply(const Vec1 &in, Vec2 &out) {
        assert(out.size() == indices.size());
        auto oit(out.begin());
        for(const auto ind: indices) *oit++ = in[ind];
    }

    // TODO: Add the cached subsampler.
};

} // namespace gfrp

#endif // _GFRP_SAMPLE_H__
