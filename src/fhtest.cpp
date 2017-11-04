#include <random>
#include "gfrp/gfrp.h"
#include "fftwrapper/fftwrapper.h"
#include "random/include/boost/random/normal_distribution.hpp"
using namespace gfrp;

template<typename... Types>                                                        
using unormd = boost::random::detail::unit_normal_distribution<Types...>;

int main(int argc, char *argv[]) {
    std::size_t size(argc <= 1 ? 1 << 16: std::strtoull(argv[1], 0, 10)),
                niter(argc <= 2 ? 1000: std::strtoull(argv[2], 0, 10));
    size = roundup64(size);
    blaze::DynamicVector<float> dps(size);
    blaze::DynamicVector<float> dpsout(size);
    float *a(&dps[0]), *b(&dpsout[0]);
    aes::AesCtr aes(0);
    unormd<float> vals;
    for(auto &el: dps) el = vals(aes);
    std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
    gfrp::fht(dps);
    std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
    blaze::DynamicVector<float> sizes(niter);
    for(auto &el: sizes) {
        fht(&dps[0], log2_64(size));
        dps *= 1./std::sqrt(size);
        std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
        el = gfrp::sum(dps);
    }
    std::cerr << sizes << '\n';
    std::cerr << "now fft\n";
    dpsout = dps;
    DCTBlock<float> dcblock((int)size, dps.data(), dps.data());
    std::cerr << 
    dcblock.execute(dps.data(), dps.data());
    dcblock.execute(dps.data(), dps.data());
}
