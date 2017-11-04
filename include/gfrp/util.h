#ifndef _GFRP_UTIL_H__
#define _GFRP_UTIL_H__
#include <cstdlib>
#include <cmath>
#include <limits>
#include <climits>
#include <type_traits>
#include <tuple>
#include <cstdint>
#include <experimental/filesystem>
#include "kspp/ks.h"
#include "FFHT/fast_copy.h"
#include "blaze/Math.h"

#ifndef FLOAT_TYPE
#define FLOAT_TYPE double
#endif

namespace gfrp {
using std::uint64_t;
using std::uint32_t;
using std::uint16_t;
using std::uint8_t;
using std::int64_t;
using std::int32_t;
using std::int16_t;
using std::int8_t;
using blaze::DynamicVector;
using blaze::DynamicMatrix;
using std::size_t;


inline constexpr uint64_t roundup(uint64_t x) {
    x--;
    x |= x >> 1; x |= x >> 2; x |= x >> 4; x |= x >> 8; x |= x >> 16; x |= x >> 32;
    return ++x;
}

inline constexpr int log2_64(uint64_t value)
{
    // https://stackoverflow.com/questions/11376288/fast-computing-of-log2-for-64-bit-integers
    const int tab64[64] = {
        63,  0, 58,  1, 59, 47, 53,  2,
        60, 39, 48, 27, 54, 33, 42,  3,
        61, 51, 37, 40, 49, 18, 28, 20,
        55, 30, 34, 11, 43, 14, 22,  4,
        62, 57, 46, 52, 38, 26, 32, 41,
        50, 36, 17, 19, 29, 10, 13, 21,
        56, 45, 25, 31, 35, 16,  9, 12,
        44, 24, 15,  8, 23,  7,  6,  5
    };
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    value |= value >> 32;
    return tab64[((uint64_t)((value - (value >> 1))*0x07EDD5E59A4E28C2)) >> 58];
}

template<class Container>
auto mean(const Container &c) {
    using FloatType = std::decay_t<decltype(c[0])>;
    FloatType sum(0.);
    if constexpr(blaze::IsSparseVector<Container>::value || blaze::IsSparseVector<Container>::value) {
        for(const auto entry: c) sum += entry.value();
    } else {
        for(const auto entry: c) sum += entry;
    }
    sum /= c.size();
    return sum;
}

template<class Container>
auto sum(const Container &c) {
    return std::accumulate(c.begin(), c.end(), static_cast<std::decay_t<decltype(*c.begin())>>(0));
}

template<class Container>
auto meanvar(const Container &c) {
    using FloatType = std::decay_t<decltype(c[0])>;
    FloatType sum(0.), varsum(0.0);
    if constexpr(blaze::IsSparseVector<Container>::value || blaze::IsSparseVector<Container>::value) {
        for(const auto entry: c) sum += entry.value(), varsum += entry.value() * entry.value();
    } else {
        for(const auto entry: c) sum += entry, varsum += entry * entry;
    }
    const auto inv(static_cast<FloatType>(1)/static_cast<FloatType>(c.size()));
    varsum -= sum * sum * inv;
    varsum *= inv;
    sum *= inv;
    return std::make_pair(sum, varsum);
}

} // namespace gfrp

#endif
