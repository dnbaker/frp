#include <random>
#include "gfrp/gfrp.h"
#include "random/include/boost/random/normal_distribution.hpp"

template<typename... Types>                                                        
using unormd = boost::random::detail::unit_normal_distribution<Types...>;

int main(int argc, char *argv[]) {
    std::size_t size(argc <= 1 ? 1 << 16: std::strtoull(argv[1], 0, 10)),
                niter(argc <= 2 ? 1000: std::strtoull(argv[2], 0, 10));
    size = roundup64(size);
    blaze::DynamicVector<double> dps(size);
    aes::AesCtr aes(0);
    unormd<double> vals;
    for(auto &el: dps) el = vals(aes);
    std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
    //gfrp::fht(&dps[0], log2_64(size));
    std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
}
