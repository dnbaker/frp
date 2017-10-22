#ifndef _GFRP_DIST_H__
#define _GFRP__DIST_H__
#include <random>
#include "gfrp/rand.h"

namespace gfrp { namespace dist {

template<typename Container, template<typename> typename Distribution, typename... DistArgs>
void sample_fill(Container &con, DistArgs &&... args) {
    using FloatType = std::decay_t<decltype(con[0])>;
    std::mt19937_64 mt;
    std::normal_distribution<FloatType> dist(std::forward<DistArgs>(args)...);
    for(auto &el: con) el = dist(mt);
}

#define DEFINE_DIST_FILL(type, name) \
    template<typename Container, typename...Args> \
    void name##_fill(Container &con, Args &&... args) { \
        sample_fill<Container, type, Args...>(con, std::forward<Args>(args)...); \
    }

DEFINE_DIST_FILL(std::normal_distribution, gaussian)
DEFINE_DIST_FILL(std::cauchy_distribution, cauchy)
DEFINE_DIST_FILL(std::chi_squared_distribution, chisq)
DEFINE_DIST_FILL(std::lognormal_distribution, lognormal)
DEFINE_DIST_FILL(std::extreme_value_distribution, extreme_value)
DEFINE_DIST_FILL(std::weibull_distribution, weibull)
    

}}

#endif // #ifndef _GFRP_DIST_H__
