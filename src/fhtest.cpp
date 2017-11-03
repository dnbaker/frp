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
    blaze::DynamicVector<double> dps(size);
    aes::AesCtr aes(0);
    unormd<double> vals;
    for(auto &el: dps) el = vals(aes);
    std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
    gfrp::fht(dps);
    std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
    blaze::DynamicVector<double> sizes(niter);
    for(auto &el: sizes) {
        fht(&dps[0], log2_64(size));
        dps *= 1./std::sqrt(size);
        std::cerr << "Sum: " << gfrp::sum(dps) << '\n';
        el = gfrp::sum(dps);
    }
    std::cerr << sizes << '\n';
    fft::FFTWDispatcher<float> disp(size);
    disp.make_plan(&dps[0], &dps[0]);
    disp.run(&dps[0], &dps[0]);
    for(size_t i(0); i < size; ++i) {
        disp.run(&dps[0], &dps[0]);
        std::cerr << dps << '\n';
        sizes[i] = gfrp::sum(dps);
        std::cerr << "ratio: " << sizes[i] / sizes[i == 0 ? i : i - 1] << '\n';
    }
    std::cerr << sizes;
}
