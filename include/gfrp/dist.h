#ifndef _GFRP_DIST_H__
#define _GFRP_DIST_H__
#include <random>
#include "gfrp/rand.h"
#include "gfrp/linalg.h"

namespace gfrp {

// Fill a matrix with distributions. Contains utilities for filling
// vectors with C++ std distributions as well as Rademacher.

template<typename Container, template<typename> typename Distribution, typename RNG=aes::AesCtr<uint64_t>, typename... DistArgs>
void sample_fill(Container &con, uint64_t seed, DistArgs &&... args) {
    using FloatType = std::decay_t<decltype(*std::begin(con))>;
    RNG gen(seed);
    Distribution<FloatType> dist(forward<DistArgs>(args)...);
    for(auto &el: con) el = dist(gen);
}


template<typename RNG=aes::AesCtr<uint64_t>>
void random_fill(uint64_t *data, uint64_t len, uint64_t seed=0) {
    for(RNG gen(seed); len; data[--len] = gen());
}

#define DEFINE_DIST_FILL(type, name) \
    template<typename Container, typename RNG=aes::AesCtr<uint64_t>, typename...Args> \
    void name##_fill(Container &con, uint64_t seed, Args &&... args) { \
        sample_fill<Container, type, RNG, Args...>(con, seed, forward<Args>(args)...); \
    }\
    struct name##_fill_struct {\
        template<typename Container, typename RNG=aes::AesCtr<uint64_t>, typename...Args>\
        void operator()(Container &con, uint64_t seed, Args &&... args) const {\
            name##_fill<Container, RNG, Args...>(con, seed, forward<Args>(args)...);\
        }\
    };

template<typename FloatType>
class unit_normal: public boost::random::detail::unit_normal_distribution<FloatType> {
public:
    void reset() {}
};

DEFINE_DIST_FILL(boost::normal_distribution, gaussian)
DEFINE_DIST_FILL(unit_normal, unit_gaussian)
DEFINE_DIST_FILL(boost::cauchy_distribution, cauchy)
DEFINE_DIST_FILL(boost::random::chi_squared_distribution, chisq)
DEFINE_DIST_FILL(boost::lognormal_distribution, lognormal)
DEFINE_DIST_FILL(boost::random::extreme_value_distribution, extreme_value)
DEFINE_DIST_FILL(boost::random::weibull_distribution, weibull)

}

#endif // #ifndef _GFRP_DIST_H__
